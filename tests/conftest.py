"""Shared pytest fixtures for fastpercentile tests."""

import numpy as np
import pytest


SUPPORTED_DTYPES = [np.uint8, np.int8, np.uint16, np.int16]
DEFAULT_QS = [0.0, 0.1, 1.0, 25.0, 50.0, 75.0, 99.0, 99.9, 100.0]


@pytest.fixture(scope='session')
def warm_jit():
    """
    Trigger numba compilation for all dtype paths once so test
    timings are not dominated by first-call JIT cost.
    """
    import fastpercentile
    fastpercentile.warmup()


@pytest.fixture(params=SUPPORTED_DTYPES,
                ids=[d.__name__ for d in SUPPORTED_DTYPES])
def dtype(request):
    """
    Iterate over each supported small-integer dtype.
    """
    return np.dtype(request.param)


@pytest.fixture
def random_1d(dtype):
    """
    A 1D random array spanning the full value range of `dtype`.
    """
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    return rng.integers(info.min, info.max + 1,
                        size=200_003, dtype=dtype)
