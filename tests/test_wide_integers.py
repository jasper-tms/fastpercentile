#!/usr/bin/env python3
"""
Correctness tests for the 32- and 64-bit integer radix-refinement
path of `fastpercentile.percentile`.

For these dtypes a direct full histogram is infeasible, so the
percentile is resolved 16 bits at a time over multiple passes.  Each
test compares against `numpy.percentile` with its default 'linear'
interpolation.
"""
import numpy as np
import pytest

import fastpercentile

from conftest import DEFAULT_QS


WIDE_DTYPES = [np.uint32, np.int32, np.uint64, np.int64]
WIDE_IDS = [d.__name__ for d in WIDE_DTYPES]


@pytest.fixture(params=WIDE_DTYPES, ids=WIDE_IDS)
def wide_dtype(request):
    """
    Iterate over each supported wide-integer dtype.
    """
    return np.dtype(request.param)


def test_matches_numpy_full_range(warm_jit, wide_dtype):
    """
    Percentiles over values spanning the entire dtype range should
    match `np.percentile` to within float64 rounding.

    For 64-bit inputs the true order statistics can exceed 2 ** 53
    and so are not exactly representable as float64; `np.percentile`
    has the same limitation, so we allow a tiny relative tolerance
    rather than demanding bit-exactness.
    """
    info = np.iinfo(wide_dtype)
    rng = np.random.default_rng(0)
    arr = rng.integers(info.min, info.max, size=200_003, dtype=wide_dtype)
    expected = np.percentile(arr, DEFAULT_QS)
    got = fastpercentile.percentile(arr, DEFAULT_QS)
    assert got.shape == expected.shape
    assert np.allclose(expected, got, rtol=1e-12, atol=0)


def test_matches_numpy_small_range_exact(warm_jit, wide_dtype):
    """
    Over a value range comfortably inside float64's exact-integer
    regime, results should match `np.percentile` exactly.
    """
    info = np.iinfo(wide_dtype)
    low = -100_000 if info.min < 0 else 0
    rng = np.random.default_rng(1)
    arr = rng.integers(low, 100_000, size=200_003, dtype=wide_dtype)
    expected = np.percentile(arr, DEFAULT_QS)
    got = fastpercentile.percentile(arr, DEFAULT_QS)
    assert np.max(np.abs(expected - got)) == 0.0


def test_boundary_straddle(warm_jit):
    """
    When the two order statistics bracketing a percentile fall in
    different coarse (top-16-bit) buckets, both buckets must be
    refined.  These values are chosen so that the bracketing ranks
    for several percentiles land in distinct buckets.
    """
    values = np.array([0x0001_0000, 0x0002_0005,
                       0x0003_0007, 0x0004_000A], dtype=np.uint32)
    arr = np.repeat(values, 25)
    qs = [0, 1, 25, 49, 50, 51, 75, 99, 100]
    expected = np.percentile(arr, qs)
    got = fastpercentile.percentile(arr, qs)
    assert np.max(np.abs(expected - got)) == 0.0


def test_low_16_bits_matter(warm_jit):
    """
    Values that share their top 16 bits but differ in the low 16
    bits must still be resolved exactly -- this is the whole point
    of the refinement passes.
    """
    base = np.uint32(0x00AB_0000)
    arr = (base + np.arange(50_000, dtype=np.uint32)).astype(np.uint32)
    expected = np.percentile(arr, DEFAULT_QS)
    got = fastpercentile.percentile(arr, DEFAULT_QS)
    assert np.max(np.abs(expected - got)) == 0.0


def test_int64_spans_sign_boundary(warm_jit):
    """
    Signed 64-bit input straddling zero must order negatives before
    positives (the sign-bit-flip key must preserve numeric order).
    """
    rng = np.random.default_rng(2)
    arr = rng.integers(-(2 ** 40), 2 ** 40, size=100_003, dtype=np.int64)
    expected = np.percentile(arr, DEFAULT_QS)
    got = fastpercentile.percentile(arr, DEFAULT_QS)
    assert np.max(np.abs(expected - got)) == 0.0


def test_fortran_contiguous_int64(warm_jit):
    """
    Fortran-contiguous wide-integer input is handled as a no-copy
    view, same as the small-dtype path.
    """
    rng = np.random.default_rng(3)
    arr = np.asfortranarray(
        rng.integers(0, 1_000_000, size=(30, 40, 50), dtype=np.int64))
    assert arr.flags.f_contiguous and not arr.flags.c_contiguous
    expected = np.percentile(arr, [25, 50, 75])
    got = fastpercentile.percentile(arr, [25, 50, 75])
    assert np.max(np.abs(expected - got)) == 0.0


def test_strided_view_uint64(warm_jit):
    """
    A non-contiguous slice should still produce the right answer via
    the fallback copy.
    """
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 1_000_000, size=(300, 300), dtype=np.uint64)
    sliced = arr[::2, ::3]
    assert not sliced.flags.c_contiguous and not sliced.flags.f_contiguous
    expected = np.percentile(sliced, [10, 50, 90])
    got = fastpercentile.percentile(sliced, [10, 50, 90])
    assert np.max(np.abs(expected - got)) == 0.0


def test_scalar_q_returns_float(warm_jit):
    """
    A scalar `q` returns a Python float for wide dtypes too.
    """
    arr = np.arange(1000, dtype=np.int64) * 100_000
    got = fastpercentile.percentile(arr, 50)
    assert isinstance(got, float)
    assert got == float(np.percentile(arr, 50))


def test_unsorted_q_returns_in_input_order(warm_jit):
    """
    Output order follows input `q` order, not the internal sort.
    """
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 2 ** 31, size=50_003, dtype=np.int64)
    qs = [99.0, 1.0, 50.0]
    expected = np.percentile(arr, qs)
    got = fastpercentile.percentile(arr, qs)
    assert np.allclose(expected, got, rtol=1e-12)


def test_n_threads_argument(warm_jit, wide_dtype):
    """
    Restricting `n_threads` must not change the result.
    """
    rng = np.random.default_rng(6)
    arr = rng.integers(0, 2 ** 31, size=80_003, dtype=wide_dtype)
    expected = fastpercentile.percentile(arr, DEFAULT_QS)
    got = fastpercentile.percentile(arr, DEFAULT_QS, n_threads=1)
    assert np.allclose(expected, got, rtol=1e-12)


def test_all_identical(warm_jit, wide_dtype):
    """
    Every percentile of a constant array is that constant.
    """
    arr = np.full(1000, 123_456, dtype=wide_dtype)
    got = fastpercentile.percentile(arr, [0, 50, 100])
    assert np.all(got == 123_456.0)


def test_single_element(warm_jit, wide_dtype):
    """
    A one-element array returns that element for every percentile.
    """
    arr = np.array([777], dtype=wide_dtype)
    got = fastpercentile.percentile(arr, [0, 25, 50, 100])
    assert np.all(got == 777.0)


def test_histogram_rejects_wide_dtype(wide_dtype):
    """
    `histogram` has no feasible output for 32/64-bit integers and
    should say so explicitly.
    """
    with pytest.raises(TypeError, match='not feasible'):
        fastpercentile.histogram(np.zeros(10, dtype=wide_dtype))
