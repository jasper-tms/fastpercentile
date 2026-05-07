#!/usr/bin/env python3
"""
Memory-bandwidth-bound percentile for small-integer arrays.

For `int8`, `uint8`, `int16`, and `uint16` inputs of any shape this
module computes one or more percentiles in a single parallel pass
over the data with results matching `numpy.percentile` (default
'linear' / method 7 interpolation).

The idea
--------
For an N-element array with at most B distinct values (B = 256 for
8-bit, 65536 for 16-bit), a single linear scan into a B-bin
histogram captures all the information we need to find any rank.
This is the same shape of work as `np.max` -- one pass, branchless
inner loop -- so it runs at memory bandwidth.  After the scan we
walk the cumulative histogram to locate the bins containing each
requested rank; that walk is O(B + n_percentiles) which is
negligible.

Public API
----------
`percentile(arr, q, n_threads=None)`
    Compute one or more percentiles.  Mirrors `numpy.percentile`.

`histogram(arr, n_threads=None)`
    Build a parallel histogram of `arr`.
"""
import numpy as np
from numba import njit, prange, get_num_threads
from typing import Sequence, Union

__all__ = ['percentile', 'histogram', 'warmup']


# --------------------------------------------------------------- #
# Parallel histograms.  Each thread fills a private table; we then
# reduce.  Bin index for signed types is `value + offset` so that
# the binning is monotonic in the original value.
# --------------------------------------------------------------- #


@njit(cache=True, parallel=True, boundscheck=False)
def _hist_u8(arr: np.ndarray,
             n_threads: int) -> np.ndarray:
    """
    Parallel histogram for uint8.
    """
    n_bins = 256
    n = arr.shape[0]
    local = np.zeros((n_threads, n_bins), dtype=np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        loc = local[t]
        for i in range(start, end):
            loc[arr[i]] += 1
    out = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(n_bins):
            out[b] += local[t, b]
    return out


@njit(cache=True, parallel=True, boundscheck=False)
def _hist_i8(arr: np.ndarray,
             n_threads: int) -> np.ndarray:
    """
    Parallel histogram for int8 (offset = 128).
    """
    n_bins = 256
    n = arr.shape[0]
    local = np.zeros((n_threads, n_bins), dtype=np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        loc = local[t]
        for i in range(start, end):
            loc[arr[i] + 128] += 1
    out = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(n_bins):
            out[b] += local[t, b]
    return out


@njit(cache=True, parallel=True, boundscheck=False)
def _hist_u16(arr: np.ndarray,
              n_threads: int) -> np.ndarray:
    """
    Parallel histogram for uint16.
    """
    n_bins = 65536
    n = arr.shape[0]
    local = np.zeros((n_threads, n_bins), dtype=np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        loc = local[t]
        for i in range(start, end):
            loc[arr[i]] += 1
    out = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(n_bins):
            out[b] += local[t, b]
    return out


@njit(cache=True, parallel=True, boundscheck=False)
def _hist_i16(arr: np.ndarray,
              n_threads: int) -> np.ndarray:
    """
    Parallel histogram for int16 (offset = 32768).
    """
    n_bins = 65536
    n = arr.shape[0]
    local = np.zeros((n_threads, n_bins), dtype=np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        loc = local[t]
        for i in range(start, end):
            loc[arr[i] + 32768] += 1
    out = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(n_bins):
            out[b] += local[t, b]
    return out


# --------------------------------------------------------------- #
# Rank-walking from a finished histogram.  Caller must sort the
# percentiles ascending so we can scan the cumulative count once.
# --------------------------------------------------------------- #


@njit(cache=True, boundscheck=False)
def _ranks_from_hist(hist: np.ndarray,
                     ranks_lo: np.ndarray,
                     ranks_hi: np.ndarray,
                     fracs: np.ndarray,
                     offset: int) -> np.ndarray:
    """
    Walk the cumulative histogram to find the bin holding each
    integer rank, then linearly interpolate.

    Parameters
    ----------
    hist : int64 array
        Histogram counts.
    ranks_lo, ranks_hi : int64 arrays of equal length
        0-indexed integer ranks bracketing each query.  Both must
        be sorted ascending.
    fracs : float64 array
        Fractional position between `ranks_lo` and `ranks_hi`.
    offset : int
        Subtracted from bin indices to recover original values.

    Returns
    -------
    float64 array of interpolated percentile values.
    """
    n_q = ranks_lo.shape[0]
    n_bins = hist.shape[0]
    out = np.empty(n_q, dtype=np.float64)

    cum = np.int64(0)
    bin_idx = 0
    for q in range(n_q):
        target_lo = ranks_lo[q]
        while bin_idx < n_bins and cum + hist[bin_idx] <= target_lo:
            cum += hist[bin_idx]
            bin_idx += 1
        bin_lo = bin_idx

        target_hi = ranks_hi[q]
        cum_hi = cum
        bin_idx_hi = bin_idx
        while bin_idx_hi < n_bins and cum_hi + hist[bin_idx_hi] <= target_hi:
            cum_hi += hist[bin_idx_hi]
            bin_idx_hi += 1
        bin_hi = bin_idx_hi

        val_lo = float(bin_lo - offset)
        val_hi = float(bin_hi - offset)
        out[q] = val_lo + fracs[q] * (val_hi - val_lo)
    return out


# --------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------- #


_DTYPE_DISPATCH = {
    np.dtype('uint8'):  (_hist_u8, 0),
    np.dtype('int8'):   (_hist_i8, 128),
    np.dtype('uint16'): (_hist_u16, 0),
    np.dtype('int16'):  (_hist_i16, 32768),
}


def _as_flat_view(arr: np.ndarray) -> np.ndarray:
    """
    Return a 1D view of `arr` without copying when possible.

    Order does not matter for histograms, so for either C- or
    F-contiguous input we just walk the raw memory.  For
    arbitrarily-strided input we fall back to a copy.
    """
    if arr.flags.c_contiguous:
        return arr.reshape(-1)
    if arr.flags.f_contiguous:
        return arr.ravel(order='F')
    return np.ascontiguousarray(arr).ravel()


def histogram(arr: np.ndarray,
              n_threads: Union[int, None] = None) -> np.ndarray:
    """
    Build a parallel histogram of `arr`.

    Parameters
    ----------
    arr : np.ndarray of int8/uint8/int16/uint16
        Any shape; treated as a flat sequence of values.
    n_threads : int, optional
        Number of parallel histogram threads.  Defaults to
        `numba.get_num_threads()`.

    Returns
    -------
    np.ndarray of int64
        Length 256 for 8-bit input, 65536 for 16-bit input.  The
        bin index for value `v` is `v + offset`, where offset is 0
        for unsigned dtypes and `2 ** (bits - 1)` for signed
        dtypes.
    """
    if n_threads is None:
        n_threads = get_num_threads()
    arr_flat = _as_flat_view(arr)
    try:
        hist_fn, _ = _DTYPE_DISPATCH[arr_flat.dtype]
    except KeyError:
        raise TypeError(
            'fastpercentile only supports int8/uint8/int16/uint16, '
            'got ' + str(arr_flat.dtype))
    return hist_fn(arr_flat, n_threads)


def percentile(arr: np.ndarray,
               q: Union[float, Sequence[float], np.ndarray],
               n_threads: Union[int, None] = None
               ) -> Union[float, np.ndarray]:
    """
    Compute one or more percentiles of `arr` via a parallel
    histogram and a cumulative-count walk.

    Matches `numpy.percentile(arr, q)` with the default 'linear'
    interpolation method to within float rounding (typically exact
    for integer inputs).

    Parameters
    ----------
    arr : np.ndarray of int8/uint8/int16/uint16
        Any shape; treated as a flat sequence of values.
    q : float or sequence of floats in [0, 100]
        Percentile(s) to compute.
    n_threads : int, optional
        Number of parallel histogram threads.  Defaults to
        `numba.get_num_threads()`.

    Returns
    -------
    float or np.ndarray of float64
        Scalar if `q` is scalar, else an array shaped like
        `np.atleast_1d(q)`.
    """
    if n_threads is None:
        n_threads = get_num_threads()

    arr_flat = _as_flat_view(arr)
    n_total = arr_flat.size
    if n_total == 0:
        raise ValueError('percentile of empty array is undefined')

    try:
        hist_fn, offset = _DTYPE_DISPATCH[arr_flat.dtype]
    except KeyError:
        raise TypeError(
            'fastpercentile only supports int8/uint8/int16/uint16, '
            'got ' + str(arr_flat.dtype))

    hist = hist_fn(arr_flat, n_threads)

    q_in = np.asarray(q, dtype=np.float64)
    q_was_scalar = (q_in.ndim == 0)
    q_arr = np.atleast_1d(q_in)
    if (q_arr < 0).any() or (q_arr > 100).any():
        raise ValueError('percentiles must lie in [0, 100]')

    # Sort percentiles ascending so we can scan the cumulative
    # histogram once; remember the inverse permutation.
    order = np.argsort(q_arr, kind='stable')
    q_sorted = q_arr[order]

    exact_rank = (n_total - 1) * q_sorted / 100.0
    ranks_lo = np.floor(exact_rank).astype(np.int64)
    ranks_hi = np.minimum(ranks_lo + 1, n_total - 1)
    fracs = exact_rank - ranks_lo

    sorted_out = _ranks_from_hist(hist, ranks_lo, ranks_hi, fracs, offset)

    out = np.empty_like(sorted_out)
    out[order] = sorted_out
    if q_was_scalar:
        return float(out[0])
    return out


def warmup() -> None:
    """
    Trigger JIT compilation for all four dtype paths so the first
    real call has no compile latency.
    """
    for dtype in (np.uint8, np.int8, np.uint16, np.int16):
        tiny = np.zeros(8, dtype=dtype)
        percentile(tiny, [0.0, 50.0, 100.0])
