#!/usr/bin/env python3
"""
Correctness tests for `fastpercentile.percentile` and
`fastpercentile.histogram`.
"""
import numpy as np
import pytest

import fastpercentile

from conftest import DEFAULT_QS


def test_matches_numpy_1d(warm_jit, random_1d):
    """
    Multi-percentile call should match `np.percentile` exactly on
    integer inputs.
    """
    expected = np.percentile(random_1d, DEFAULT_QS)
    got = fastpercentile.percentile(random_1d, DEFAULT_QS)
    assert isinstance(got, np.ndarray)
    assert got.shape == expected.shape
    assert np.max(np.abs(expected - got)) < 1e-9


def test_scalar_q_returns_scalar(warm_jit, random_1d):
    """
    Passing a scalar `q` should return a Python float, matching
    numpy's behaviour.
    """
    got = fastpercentile.percentile(random_1d, 50)
    assert isinstance(got, float)
    expected = float(np.percentile(random_1d, 50))
    assert abs(got - expected) < 1e-9


def test_unsorted_q_returns_in_input_order(warm_jit, random_1d):
    """
    The returned array should be in the order of the input `q`,
    not the sorted order we use internally.
    """
    qs = [99.0, 1.0, 50.0]
    expected = np.percentile(random_1d, qs)
    got = fastpercentile.percentile(random_1d, qs)
    assert np.allclose(expected, got, atol=1e-9)


def test_median_matches_numpy(warm_jit, random_1d):
    """
    `median` is the 50th percentile and should match `np.median`,
    returning a Python float.
    """
    got = fastpercentile.median(random_1d)
    assert isinstance(got, float)
    assert abs(got - float(np.median(random_1d))) < 1e-9


def test_median_even_length_averages_middle_pair(warm_jit):
    """
    For an even-length array the median is the average of the two
    middle values, matching `np.median` (and giving a non-integer
    result when those two values differ by an odd amount).
    """
    arr = np.array([10, 20, 30, 41], dtype=np.uint16)
    assert fastpercentile.median(arr) == 25.0
    assert fastpercentile.median(arr) == float(np.median(arr))


def test_endpoints():
    """
    Percentile 0 should equal the min, 100 should equal the max.
    """
    arr = np.arange(1000, dtype=np.uint16)
    lo, hi = fastpercentile.percentile(arr, [0, 100])
    assert lo == 0.0
    assert hi == 999.0


def test_handles_fortran_contiguous():
    """
    Fortran-contiguous input must not trigger a multi-GB ravel
    copy; we just walk the raw memory.
    """
    arr = np.asfortranarray(
        np.random.default_rng(1).integers(
            0, 65536, size=(20, 30, 40), dtype=np.uint16))
    assert arr.flags.f_contiguous and not arr.flags.c_contiguous
    expected = np.percentile(arr, [25, 50, 75])
    got = fastpercentile.percentile(arr, [25, 50, 75])
    assert np.allclose(expected, got, atol=1e-9)


def test_handles_strided_view():
    """
    A non-contiguous slice should still produce the right answer
    (via fallback copy).
    """
    arr = np.random.default_rng(2).integers(
        0, 65536, size=(100, 100), dtype=np.uint16)
    sliced = arr[::2, ::3]
    assert not sliced.flags.c_contiguous
    assert not sliced.flags.f_contiguous
    expected = np.percentile(sliced, [10, 90])
    got = fastpercentile.percentile(sliced, [10, 90])
    assert np.allclose(expected, got, atol=1e-9)


def test_unsupported_dtype_raises():
    """
    Floats should error explicitly rather than silently producing
    garbage.  Integer dtypes (including 32/64-bit) are all supported.
    """
    with pytest.raises(TypeError, match='int8/uint8/int16/uint16'):
        fastpercentile.percentile(np.zeros(10, dtype=np.float32), 50)
    with pytest.raises(TypeError):
        fastpercentile.percentile(np.zeros(10, dtype=np.float64), 50)


def test_q_out_of_range_raises():
    """
    Percentiles outside [0, 100] should error.
    """
    arr = np.arange(100, dtype=np.uint8)
    with pytest.raises(ValueError):
        fastpercentile.percentile(arr, [-1, 50])
    with pytest.raises(ValueError):
        fastpercentile.percentile(arr, [50, 101])


def test_empty_raises():
    """
    Percentile of an empty array is undefined.
    """
    with pytest.raises(ValueError):
        fastpercentile.percentile(np.array([], dtype=np.uint8), 50)


def test_histogram_counts_match_bincount(random_1d, dtype):
    """
    `histogram` should produce counts identical to `np.bincount`,
    after accounting for the signed-dtype offset.
    """
    info = np.iinfo(dtype)
    h = fastpercentile.histogram(random_1d)
    expected_n_bins = 2 ** (info.bits)
    assert h.shape == (expected_n_bins,)
    if dtype.kind == 'u':
        ref = np.bincount(random_1d.astype(np.int64),
                          minlength=expected_n_bins)
    else:
        offset = 2 ** (info.bits - 1)
        ref = np.bincount(random_1d.astype(np.int64) + offset,
                          minlength=expected_n_bins)
    assert np.array_equal(h, ref)
    assert h.sum() == random_1d.size


def test_n_threads_argument(random_1d):
    """
    Restricting `n_threads` should produce identical results.
    """
    expected = fastpercentile.percentile(random_1d, DEFAULT_QS)
    got = fastpercentile.percentile(random_1d, DEFAULT_QS, n_threads=1)
    assert np.allclose(expected, got, atol=1e-9)
