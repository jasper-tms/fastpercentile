#!/usr/bin/env python3
"""
Correctness tests for float32/float64 support in
`fastpercentile.percentile`/`median` (NaN-propagating, matching
`numpy.percentile`/`numpy.median`) and `nanpercentile`/`nanmedian`
(NaN-ignoring, matching `numpy.nanpercentile`/`numpy.nanmedian`).

Floats are resolved by mapping each value's raw bits to an
order-preserving unsigned key and running the same radix refinement
used for wide integers, so the same value is recovered bit-exactly and
the linear interpolation matches numpy to within float rounding.
"""
import warnings

import numpy as np
import pytest

import fastpercentile

from conftest import DEFAULT_QS


FLOAT_DTYPES = [np.float32, np.float64]

# Sizes that exercise both the small-array sort fast path and the
# full radix refinement (whose crossover is 65536 * digits elements).
FLAT_SIZES = [20, 5000, 200_003]

# Shapes that exercise both axis branches: the within-slice loop (few
# groups, or large slices) and the group-parallel sort kernel (many
# small groups).
AXIS_SHAPES = [(13, 11, 9), (600, 7), (7, 600), (100, 100)]


@pytest.fixture(params=FLOAT_DTYPES, ids=[d.__name__ for d in FLOAT_DTYPES])
def float_dtype(request):
    """
    Iterate over each supported floating-point dtype.
    """
    return np.dtype(request.param)


def _spread(rng, shape, dtype):
    """
    Random floats spanning several orders of magnitude, plus the
    signed zeros, so the bit-ordering transform is exercised across
    exponents and signs.

    Infinities are deliberately left out of these close-match fixtures:
    interpolating to or from infinity is ill-defined and even numpy is
    self-inconsistent there (`numpy.median` returns the exact middle
    element while `numpy.percentile` can return NaN).  Infinity
    handling has its own focused test below.
    """
    base = rng.standard_normal(shape).astype(dtype)
    scale = (10.0 ** rng.integers(-20, 20, size=shape)).astype(dtype)
    values = (base * scale).astype(dtype)
    flat = values.reshape(-1)
    specials = np.array([0.0, -0.0], dtype=dtype)
    flat[:specials.size] = specials
    return values


def _all_axes(ndim):
    axes = [None] + list(range(ndim)) + [-1]
    if ndim >= 2:
        axes.append((0, ndim - 1))
        axes.append(tuple(range(ndim)))
    return axes


@pytest.mark.parametrize('size', FLAT_SIZES)
def test_flat_matches_numpy(warm_jit, float_dtype, size):
    """
    Flat percentiles match `numpy.percentile` for both the sort fast
    path (small arrays) and the radix path (large arrays), across a
    wide range of magnitudes and signs.
    """
    rng = np.random.default_rng(0)
    arr = _spread(rng, size, float_dtype)
    reference = arr.astype(np.float64)
    for q in (50, DEFAULT_QS, [99.0, 1.0, 50.0]):
        expected = np.percentile(reference, q)
        got = np.asarray(fastpercentile.percentile(arr, q))
        assert got.shape == np.asarray(expected).shape, (size, q)
        # Interpolated answers can be huge; compare with a relative
        # tolerance keyed to the data scale.
        assert np.allclose(got, expected, rtol=1e-6,
                           atol=1e-6 * (abs(reference).max() + 1)), (size, q)


def test_flat_recovers_exact_value(warm_jit, float_dtype):
    """
    A percentile that lands exactly on a data point should come back
    bit-exact (the radix refinement resolves the full bit pattern), so
    e.g. the maximum and minimum are reproduced exactly.
    """
    rng = np.random.default_rng(1)
    arr = _spread(rng, 200_003, float_dtype)
    got_min, got_max = fastpercentile.percentile(arr, [0.0, 100.0])
    assert got_min == float(arr.min())
    assert got_max == float(arr.max())


@pytest.mark.parametrize('shape', AXIS_SHAPES)
@pytest.mark.parametrize('keepdims', [False, True])
def test_axis_matches_numpy(warm_jit, float_dtype, shape, keepdims):
    """
    Percentiles along every axis (and axis tuple) match
    `numpy.percentile`, with and without `keepdims`.
    """
    rng = np.random.default_rng(2)
    arr = _spread(rng, shape, float_dtype)
    reference = arr.astype(np.float64)
    scale = abs(reference).max() + 1
    for q in (50, [25.0, 50.0, 99.0]):
        for axis in _all_axes(arr.ndim):
            expected = np.percentile(reference, q, axis=axis,
                                     keepdims=keepdims)
            got = np.asarray(fastpercentile.percentile(
                arr, q, axis=axis, keepdims=keepdims))
            assert got.shape == np.asarray(expected).shape, (q, axis, keepdims)
            assert np.allclose(got, expected, rtol=1e-6,
                               atol=1e-6 * scale), (q, axis, keepdims)


def test_median_matches_numpy(warm_jit, float_dtype):
    """
    `median` forwards `axis`/`keepdims` and matches `numpy.median`.
    """
    rng = np.random.default_rng(3)
    arr = _spread(rng, (5, 7, 3), float_dtype)
    reference = arr.astype(np.float64)
    scale = abs(reference).max() + 1
    for axis in (None, 0, 1, 2, (0, 2)):
        for keepdims in (False, True):
            expected = np.median(reference, axis=axis, keepdims=keepdims)
            got = np.asarray(
                fastpercentile.median(arr, axis=axis, keepdims=keepdims))
            assert got.shape == np.asarray(expected).shape, (axis, keepdims)
            assert np.allclose(got, expected, rtol=1e-6,
                               atol=1e-6 * scale), (axis, keepdims)


def test_scalar_axis_none_returns_python_float(warm_jit, float_dtype):
    """
    A scalar `q` over the whole array returns a plain Python float.
    """
    arr = np.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=float_dtype)
    result = fastpercentile.percentile(arr, 50)
    assert isinstance(result, float)
    assert result == 3.0


# --------------------------------------------------------------- #
# NaN handling: percentile propagates, nanpercentile ignores.
# --------------------------------------------------------------- #


@pytest.mark.parametrize('size', FLAT_SIZES)
def test_percentile_propagates_nan(warm_jit, float_dtype, size):
    """
    Any NaN in the data makes `percentile` return NaN, matching
    `numpy.percentile`, on both the sort and radix paths.
    """
    rng = np.random.default_rng(4)
    arr = _spread(rng, size, float_dtype)
    arr.reshape(-1)[size // 2] = np.nan
    got = np.asarray(fastpercentile.percentile(arr, [25.0, 50.0, 99.0]))
    assert np.isnan(got).all()


@pytest.mark.parametrize('size', FLAT_SIZES)
def test_nanpercentile_ignores_nan(warm_jit, float_dtype, size):
    """
    `nanpercentile` drops NaNs and matches `numpy.nanpercentile` on
    both paths.
    """
    rng = np.random.default_rng(5)
    arr = _spread(rng, size, float_dtype)
    flat = arr.reshape(-1)
    flat[rng.random(size) < 0.1] = np.nan
    reference = arr.astype(np.float64)
    scale = np.nanmax(abs(reference)) + 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = np.nanpercentile(reference, DEFAULT_QS)
    got = np.asarray(fastpercentile.nanpercentile(arr, DEFAULT_QS))
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6 * scale)


def test_nanpercentile_axis_matches_numpy(warm_jit, float_dtype):
    """
    `nanpercentile` over an axis matches numpy when each slice carries
    a different number of NaNs, in both axis branches (few large
    groups and many small groups).
    """
    rng = np.random.default_rng(6)
    for shape in [(50, 40), (5000, 12)]:
        arr = _spread(rng, shape, float_dtype)
        arr[rng.random(shape) < 0.2] = np.nan
        reference = arr.astype(np.float64)
        scale = np.nanmax(abs(reference)) + 1
        for axis in (0, 1):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                expected = np.nanpercentile(reference, [10.0, 50.0, 90.0],
                                            axis=axis)
            got = np.asarray(fastpercentile.nanpercentile(
                arr, [10.0, 50.0, 90.0], axis=axis))
            assert got.shape == expected.shape, (shape, axis)
            assert np.allclose(got, expected, rtol=1e-6,
                               atol=1e-6 * scale), (shape, axis)


def test_nanmedian_matches_numpy(warm_jit, float_dtype):
    """
    `nanmedian` matches `numpy.nanmedian`.
    """
    rng = np.random.default_rng(7)
    arr = _spread(rng, (6, 8), float_dtype)
    arr[rng.random((6, 8)) < 0.25] = np.nan
    reference = arr.astype(np.float64)
    scale = np.nanmax(abs(reference)) + 1
    for axis in (None, 0, 1):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected = np.nanmedian(reference, axis=axis)
        got = np.asarray(fastpercentile.nanmedian(arr, axis=axis))
        assert np.allclose(got, expected, rtol=1e-6,
                           atol=1e-6 * scale, equal_nan=True), axis


def test_all_nan_slice_is_nan(warm_jit, float_dtype):
    """
    A wholly-NaN population yields NaN from `nanpercentile`, both flat
    and per slice (mixed with valid slices), matching numpy's value.
    """
    flat = np.array([np.nan, np.nan, np.nan], dtype=float_dtype)
    assert np.isnan(fastpercentile.nanpercentile(flat, 50))

    arr = np.array([[np.nan, np.nan, np.nan],
                    [1.0, 2.0, 3.0],
                    [np.nan, 5.0, np.nan]], dtype=float_dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = np.nanpercentile(arr.astype(np.float64), 50, axis=1)
    got = np.asarray(fastpercentile.nanpercentile(arr, 50, axis=1))
    assert np.allclose(got, expected, equal_nan=True)


def test_infinity_ordering(warm_jit, float_dtype):
    """
    Infinities order correctly: the maximum/minimum and any
    integral-rank order statistic come back exactly, even though
    interpolating across an infinite endpoint is left undefined.
    """
    rng = np.random.default_rng(9)
    arr = _spread(rng, 4001, float_dtype)
    arr[7] = np.inf
    arr[9] = -np.inf
    # 0th and 100th percentiles are -inf and +inf.
    low, high = fastpercentile.percentile(arr, [0.0, 100.0])
    assert low == -np.inf
    assert high == np.inf
    # An integral-rank percentile returns the exact order statistic
    # (no interpolation), so it matches sorting the data directly.
    srt = np.sort(arr.astype(np.float64))
    n = srt.size
    for q in (25.0, 50.0, 75.0):
        rank = (n - 1) * q / 100.0
        if rank == int(rank):
            got = fastpercentile.percentile(arr, q)
            assert got == srt[int(rank)]


def test_infinity_integral_rank_beats_numpy_nan(warm_jit, float_dtype):
    """
    Like the narrow-signed-integer overflow case, we sidestep a numpy
    quirk: `numpy.percentile` interpolates even on an integral rank and
    so turns an infinite neighbor into NaN (0 * inf), whereas we return
    the exact order statistic.
    """
    arr = np.array([0.0, -0.0, np.inf], dtype=float_dtype)
    got = fastpercentile.percentile(arr, 50)
    assert got == 0.0
    # numpy.median agrees with us here (it also returns the exact
    # middle element); numpy.percentile does not.
    assert got == np.median(arr.astype(np.float64))


def test_nanpercentile_on_integers_matches_percentile(warm_jit):
    """
    Integers can't be NaN, so `nanpercentile` on integer input is just
    `percentile`.
    """
    rng = np.random.default_rng(8)
    arr = rng.integers(0, 1000, size=5000, dtype=np.int32)
    expected = np.percentile(arr.astype(np.float64), [25, 50, 75])
    got = fastpercentile.nanpercentile(arr, [25, 50, 75])
    assert np.allclose(got, expected)
