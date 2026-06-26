#!/usr/bin/env python3
"""
Correctness tests for the `axis=` and `keepdims=` arguments of
`fastpercentile.percentile` and `fastpercentile.median`.

The reference is always `numpy.percentile`/`numpy.median` computed on
a float64 copy of the input.  Computing the reference in float64
sidesteps a numpy quirk where `numpy.percentile` on narrow signed
integer dtypes can overflow during its linear interpolation and return
a value outside the data range; fastpercentile interpolates in float64
and so stays correct (and matches the float64 reference) in those
cases.
"""
import numpy as np
import pytest

import fastpercentile

from conftest import DEFAULT_QS


# Shapes chosen to exercise both branches of the axis dispatch: the
# within-slice histogram loop (few groups, or large slices) and the
# group-parallel sort kernel (many small groups).
AXIS_SHAPES = [(13, 11, 9), (600, 7), (7, 600), (100, 100)]


def _all_axes(ndim):
    """
    A representative set of axis arguments for an `ndim`-d array:
    every single axis (including a negative one), a couple of tuples,
    and the all-axes tuple.
    """
    axes = [None] + list(range(ndim)) + [-1]
    if ndim >= 2:
        axes.append((0, ndim - 1))
        axes.append(tuple(range(ndim)))
    return axes


@pytest.mark.parametrize('shape', AXIS_SHAPES)
@pytest.mark.parametrize('keepdims', [False, True])
def test_axis_matches_numpy(warm_jit, dtype, shape, keepdims):
    """
    Percentiles along every axis (and axis tuple) match
    `numpy.percentile` on a float64 copy, for both scalar and
    multi-percentile `q`, with and without `keepdims`.
    """
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    arr = rng.integers(info.min, min(info.max, 10 ** 6) + 1,
                       size=shape, dtype=dtype)
    reference = arr.astype(np.float64)
    for q in (50, DEFAULT_QS, [99.0, 1.0, 50.0]):
        for axis in _all_axes(arr.ndim):
            expected = np.percentile(reference, q, axis=axis,
                                     keepdims=keepdims)
            got = fastpercentile.percentile(arr, q, axis=axis,
                                            keepdims=keepdims)
            got = np.asarray(got)
            assert got.shape == expected.shape, (q, axis, keepdims)
            assert np.allclose(got, expected, atol=1e-6), (q, axis, keepdims)


def test_axis_signed_narrow_avoids_numpy_overflow(warm_jit):
    """
    On narrow signed dtypes `numpy.percentile` can overflow during
    interpolation and return a value outside the data range; ours
    interpolates in float64 and stays within range, matching the
    float64 reference.
    """
    arr = np.array([[89, 27, 63, -110, 46, 115, -104]], dtype=np.int8)
    got = fastpercentile.percentile(arr, 25, axis=1)
    reference = np.percentile(arr.astype(np.float64), 25, axis=1)
    assert np.allclose(got, reference)
    # The true 25th percentile is well inside [-110, 115]; numpy's
    # int8 result famously is not.
    assert arr.min() <= got[0] <= arr.max()


def test_axis_keepdims_broadcasts(warm_jit):
    """
    With `keepdims=True` the result should broadcast against the
    input, matching numpy's reduced-to-one-dim shape.
    """
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 1000, size=(4, 6, 8), dtype=np.uint16)
    got = fastpercentile.percentile(arr, 50, axis=1, keepdims=True)
    assert got.shape == (4, 1, 8)
    # Broadcasting against the input must not raise.
    np.broadcast_shapes(got.shape, arr.shape)


def test_axis_wide_integer(warm_jit):
    """
    The axis path also serves the 32/64-bit dtypes resolved by the
    radix path in the flat case.
    """
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 2 ** 40, size=(50, 30), dtype=np.uint64)
    for axis in (0, 1):
        expected = np.percentile(arr.astype(np.float64), [25, 50, 75],
                                 axis=axis)
        got = fastpercentile.percentile(arr, [25, 50, 75], axis=axis)
        assert np.allclose(got, expected, atol=1e-6)


def test_median_axis_matches_numpy(warm_jit):
    """
    `median` should forward `axis` and `keepdims` and match
    `numpy.median`.
    """
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 1000, size=(5, 7, 3), dtype=np.uint16)
    reference = arr.astype(np.float64)
    for axis in (None, 0, 1, 2, (0, 2)):
        for keepdims in (False, True):
            expected = np.median(reference, axis=axis, keepdims=keepdims)
            got = np.asarray(
                fastpercentile.median(arr, axis=axis, keepdims=keepdims))
            assert got.shape == np.asarray(expected).shape, (axis, keepdims)
            assert np.allclose(got, expected, atol=1e-6), (axis, keepdims)


def test_keepdims_axis_none(warm_jit):
    """
    `keepdims=True` with `axis=None` collapses every axis to length
    one, matching numpy.
    """
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 1000, size=(4, 6), dtype=np.uint16)
    reference = arr.astype(np.float64)
    got = fastpercentile.percentile(arr, [25, 75], axis=None, keepdims=True)
    expected = np.percentile(reference, [25, 75], axis=None, keepdims=True)
    assert got.shape == expected.shape == (2, 1, 1)
    assert np.allclose(got, expected, atol=1e-6)


def test_unsorted_q_axis_returns_input_order(warm_jit):
    """
    The percentile axis should follow the order of the input `q`,
    not the sorted order used internally, in the axis path too.
    """
    rng = np.random.default_rng(6)
    arr = rng.integers(0, 1000, size=(40, 9), dtype=np.uint16)
    qs = [99.0, 1.0, 50.0]
    expected = np.percentile(arr.astype(np.float64), qs, axis=1)
    got = fastpercentile.percentile(arr, qs, axis=1)
    assert np.allclose(got, expected, atol=1e-6)


def test_out_of_range_axis_raises(warm_jit):
    """
    An axis outside the array's dimensionality should raise, as in
    numpy.
    """
    # numpy raises AxisError, which subclasses both ValueError and
    # IndexError; catch those so the test is numpy-version-agnostic.
    arr = np.zeros((4, 5), dtype=np.uint16)
    with pytest.raises((ValueError, IndexError)):
        fastpercentile.percentile(arr, 50, axis=2)
    with pytest.raises((ValueError, IndexError)):
        fastpercentile.percentile(arr, 50, axis=-3)


def test_repeated_axis_raises(warm_jit):
    """
    Repeating an axis in the tuple should raise, as in numpy.
    """
    arr = np.zeros((4, 5, 6), dtype=np.uint16)
    with pytest.raises(ValueError):
        fastpercentile.percentile(arr, 50, axis=(1, 1))


def test_empty_reduced_axis_raises(warm_jit):
    """
    Reducing over an axis of length zero is undefined.
    """
    arr = np.zeros((4, 0), dtype=np.uint16)
    with pytest.raises(ValueError):
        fastpercentile.percentile(arr, 50, axis=1)
