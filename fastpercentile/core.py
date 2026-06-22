#!/usr/bin/env python3
"""
Memory-bandwidth-bound percentile for small-integer arrays.

For `int8`, `uint8`, `int16`, and `uint16` inputs of any shape this
module computes one or more percentiles in a single parallel pass
over the data with results matching `numpy.percentile` (default
'linear' / method 7 interpolation).

For `int32`, `uint32`, `int64`, and `uint64` inputs a direct
histogram over every possible value is not feasible (it would need
`2 ** 32` or `2 ** 64` bins).  Instead we use a radix refinement:
the first pass histograms only the top 16 bits of each value, which
tells us which coarse bucket each requested percentile falls in.
Subsequent passes re-scan the array but only count the next 16 bits
of the few elements inside those buckets, narrowing the answer 16
bits at a time until the exact value is pinned.  This costs one pass
per 16-bit "digit" (two passes for 32-bit input, four for 64-bit)
and keeps the auxiliary memory at the same fixed 65536-bin scale.

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

For wider integers we split each value into 16-bit digits (most
significant first) and resolve those digits one pass at a time, as
described above.

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
# Radix histograms for 32- and 64-bit integers.  We never allocate
# 2 ** 32 bins; instead every pass histograms a single 16-bit
# "digit" of an order-preserving unsigned key.
#
# The key is the raw bits reinterpreted as unsigned, XORed with
# `flip` -- 0 for unsigned dtypes and the sign bit (2 ** (bits - 1))
# for signed dtypes.  Flipping the sign bit maps the signed range
# onto the unsigned range while preserving numeric order, so the
# usual cumulative-count walk works on the key and we convert back
# to the original value only at the very end.
# --------------------------------------------------------------- #


@njit(cache=True, parallel=True, boundscheck=False)
def _hist_digit_coarse(arr_u: np.ndarray,
                       n_threads: int,
                       flip: np.uint64,
                       shift: np.uint64) -> np.ndarray:
    """
    Parallel histogram of one 16-bit digit of the unsigned key,
    over every element.  Used for the first (most significant) pass.
    """
    n_bins = 65536
    n = arr_u.shape[0]
    mask = np.uint64(0xFFFF)
    local = np.zeros((n_threads, n_bins), dtype=np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        loc = local[t]
        for i in range(start, end):
            key = np.uint64(arr_u[i]) ^ flip
            digit = (key >> shift) & mask
            loc[digit] += 1
    out = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(n_bins):
            out[b] += local[t, b]
    return out


@njit(cache=True, parallel=True, boundscheck=False)
def _hist_digit_refine(arr_u: np.ndarray,
                       n_threads: int,
                       flip: np.uint64,
                       shift: np.uint64,
                       prefixes: np.ndarray) -> np.ndarray:
    """
    Parallel histogram of the 16-bit digit at `shift`, but only for
    elements whose higher bits (`key >> (shift + 16)`) match one of
    the sorted `prefixes`.  Each matching prefix gets its own
    histogram row, so a single pass refines every target bucket at
    once.
    """
    n_bins = 65536
    n_slots = prefixes.shape[0]
    n = arr_u.shape[0]
    mask = np.uint64(0xFFFF)
    prefix_shift = shift + np.uint64(16)
    local = np.zeros((n_threads, n_slots, n_bins), dtype=np.int64)
    chunk = (n + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        loc = local[t]
        for i in range(start, end):
            key = np.uint64(arr_u[i]) ^ flip
            prefix = key >> prefix_shift
            # Binary search for `prefix` in the sorted prefix list.
            lo = 0
            hi = n_slots
            while lo < hi:
                mid = (lo + hi) // 2
                if prefixes[mid] < prefix:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < n_slots and prefixes[lo] == prefix:
                digit = (key >> shift) & mask
                loc[lo, digit] += 1
    out = np.zeros((n_slots, n_bins), dtype=np.int64)
    for t in range(n_threads):
        for s in range(n_slots):
            for b in range(n_bins):
                out[s, b] += local[t, s, b]
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


# Wide integer dtypes resolved by radix refinement.  Each entry is
# (unsigned view dtype, sign-bit flip, number of 16-bit digits).
_WIDE_DISPATCH = {
    np.dtype('uint32'): (np.uint32, 0,            2),
    np.dtype('int32'):  (np.uint32, 1 << 31,      2),
    np.dtype('uint64'): (np.uint64, 0,            4),
    np.dtype('int64'):  (np.uint64, 1 << 63,      4),
}

_SUPPORTED_DTYPE_MESSAGE = (
    'fastpercentile supports int8/uint8/int16/uint16 (single pass) and '
    'int32/uint32/int64/uint64 (radix refinement), got ')


def _key_to_value(key: int,
                  flip: int,
                  bits: int) -> float:
    """
    Convert a fully resolved unsigned key back to the numeric value
    of the original dtype, as a float.

    Parameters
    ----------
    key : int
        The order-preserving unsigned key (all digits resolved).
    flip : int
        The sign-bit flip used to build the key: 0 for unsigned
        dtypes, `2 ** (bits - 1)` for signed dtypes.
    bits : int
        Width of the original dtype in bits.

    Returns
    -------
    float
    """
    unsigned_value = key ^ flip
    if flip == 0:
        return float(unsigned_value)
    if unsigned_value >= (1 << (bits - 1)):
        return float(unsigned_value - (1 << bits))
    return float(unsigned_value)


def _resolve_wide(arr_u: np.ndarray,
                  n_total: int,
                  ranks_lo: np.ndarray,
                  ranks_hi: np.ndarray,
                  fracs: np.ndarray,
                  flip: int,
                  bits: int,
                  n_digits: int,
                  n_threads: int) -> np.ndarray:
    """
    Resolve percentiles for a wide-integer array by radix refinement.

    The first pass histograms the most significant 16-bit digit of
    every element's key; each subsequent pass histograms the next
    digit, but only for the few coarse buckets that still contain a
    requested rank.  After `n_digits` passes every requested order
    statistic is pinned to an exact value.

    Parameters
    ----------
    arr_u : np.ndarray
        The input reinterpreted as an unsigned integer array of the
        same width.
    n_total : int
        Total number of elements.
    ranks_lo, ranks_hi : int64 arrays of equal length
        0-indexed integer ranks bracketing each query, in the sorted
        order produced by the caller.
    fracs : float64 array
        Fractional position between `ranks_lo` and `ranks_hi`.
    flip : int
        Sign-bit flip for the key (see `_key_to_value`).
    bits : int
        Width of the original dtype in bits.
    n_digits : int
        Number of 16-bit digits to resolve (`bits // 16`).
    n_threads : int
        Number of parallel threads.

    Returns
    -------
    float64 array of interpolated percentile values, aligned with
    `ranks_lo`.
    """
    flip_u = np.uint64(flip)

    # The distinct ranks we must pin down: each percentile needs the
    # order statistic just below and just above it (often the same
    # bucket, occasionally adjacent ones).  Deduplicate so shared
    # buckets are refined only once.
    target_ranks = np.unique(np.concatenate([ranks_lo, ranks_hi]))
    n_targets = target_ranks.shape[0]

    # `resolved[j]` accumulates the high bits of target j's key as a
    # number equal to `key >> shift`; `base[j]` is the count of all
    # elements whose key is strictly below target j's current bucket.
    resolved = np.zeros(n_targets, dtype=np.uint64)
    base = np.zeros(n_targets, dtype=np.int64)

    shift = 16 * (n_digits - 1)

    # First (coarse) pass over every element.
    hist = _hist_digit_coarse(arr_u, n_threads, flip_u, np.uint64(shift))
    cum = np.cumsum(hist)
    bins = np.searchsorted(cum, target_ranks, side='right')
    resolved[:] = bins.astype(np.uint64)
    base[:] = cum[bins] - hist[bins]
    shift -= 16

    # Refinement passes for the remaining digits.
    while shift >= 0:
        prefixes = np.unique(resolved)
        hists = _hist_digit_refine(
            arr_u, n_threads, flip_u, np.uint64(shift), prefixes)
        cums = np.cumsum(hists, axis=1)
        for j in range(n_targets):
            slot = int(np.searchsorted(prefixes, resolved[j]))
            local_rank = int(target_ranks[j] - base[j])
            digit = int(np.searchsorted(cums[slot], local_rank, side='right'))
            cum_before = int(cums[slot, digit] - hists[slot, digit])
            resolved[j] = (resolved[j] << np.uint64(16)) | np.uint64(digit)
            base[j] += cum_before
        shift -= 16

    # Map each resolved rank to its value, then interpolate.
    value_by_rank = {
        int(target_ranks[j]): _key_to_value(int(resolved[j]), flip, bits)
        for j in range(n_targets)
    }
    out = np.empty(ranks_lo.shape[0], dtype=np.float64)
    for k in range(ranks_lo.shape[0]):
        value_lo = value_by_rank[int(ranks_lo[k])]
        value_hi = value_by_rank[int(ranks_hi[k])]
        out[k] = value_lo + fracs[k] * (value_hi - value_lo)
    return out


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

    Only the small-integer dtypes (int8/uint8/int16/uint16) have a
    direct full histogram; for 32/64-bit integers a single histogram
    over every value is not feasible, so use `percentile` instead
    (which resolves the values it needs by radix refinement).

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
    if arr_flat.dtype in _WIDE_DISPATCH:
        raise TypeError(
            'a full histogram is not feasible for 32/64-bit integers '
            '(' + str(arr_flat.dtype) + '); use percentile() instead, '
            'which resolves only the values it needs')
    try:
        hist_fn, _ = _DTYPE_DISPATCH[arr_flat.dtype]
    except KeyError:
        raise TypeError(_SUPPORTED_DTYPE_MESSAGE + str(arr_flat.dtype))
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

    For int8/uint8/int16/uint16 this is a single parallel histogram
    pass; for int32/uint32/int64/uint64 it is a radix refinement of
    two (32-bit) or four (64-bit) passes.

    Parameters
    ----------
    arr : np.ndarray of int8/uint8/int16/uint16/int32/uint32/
            int64/uint64
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

    dtype = arr_flat.dtype
    if dtype not in _DTYPE_DISPATCH and dtype not in _WIDE_DISPATCH:
        raise TypeError(_SUPPORTED_DTYPE_MESSAGE + str(dtype))

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

    if dtype in _DTYPE_DISPATCH:
        hist_fn, offset = _DTYPE_DISPATCH[dtype]
        hist = hist_fn(arr_flat, n_threads)
        sorted_out = _ranks_from_hist(hist, ranks_lo, ranks_hi, fracs, offset)
    else:
        view_dtype, flip, n_digits = _WIDE_DISPATCH[dtype]
        bits = dtype.itemsize * 8
        arr_u = arr_flat.view(view_dtype)
        sorted_out = _resolve_wide(
            arr_u, n_total, ranks_lo, ranks_hi, fracs,
            flip, bits, n_digits, n_threads)

    out = np.empty_like(sorted_out)
    out[order] = sorted_out
    if q_was_scalar:
        return float(out[0])
    return out


def warmup() -> None:
    """
    Trigger JIT compilation for every dtype path so the first real
    call has no compile latency.  This covers the single-pass small
    dtypes as well as the coarse and refinement kernels used by the
    32- and 64-bit radix paths.
    """
    for dtype in (np.uint8, np.int8, np.uint16, np.int16,
                  np.uint32, np.int32, np.uint64, np.int64):
        tiny = np.zeros(8, dtype=dtype)
        percentile(tiny, [0.0, 50.0, 100.0])
