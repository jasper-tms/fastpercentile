# fastpercentile: Memory-bandwidth-bound percentile for small-integer arrays

[![Tests](https://github.com/jasper-tms/fastpercentile/actions/workflows/tests.yml/badge.svg)](https://github.com/jasper-tms/fastpercentile/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/fastpercentile)](https://pypi.org/project/fastpercentile/)
[![License](https://img.shields.io/github/license/jasper-tms/fastpercentile)](https://github.com/jasper-tms/fastpercentile/blob/main/LICENSE)

`np.percentile` is O(n) but in practice has a brutal constant factor — on a 1.7-billion-element `uint16` volume it takes ~22 s, vs ~0.15 s for `np.max`. There is no good reason for percentile to be so much slower than max: both can be done in a single pass over the array.

For small-integer dtypes (`int8`, `uint8`, `int16`, `uint16`) the data only takes one of at most 65 536 distinct values, so a single parallel pass into a histogram captures everything you need to compute any percentile. After the histogram is built, walking the cumulative count to find the bin holding each requested rank costs essentially nothing.

This package implements that, in a few hundred lines of `numba`. On a 32-thread workstation it runs at DRAM bandwidth — about as fast as `np.max`, and ~300× faster than `np.percentile`:

```
np.max         : 0.148 s
fastpercentile : 0.072 s   <- four percentiles in one pass
np.percentile  : 22.2  s
```

Auxiliary memory is ~16 MB regardless of input size (32 threads × one 65 536-bin local table each, plus a final reduced histogram), so it adds no measurable RAM pressure on top of the input.


### Usage

```python
import numpy as np
import fastpercentile

arr = np.random.randint(0, 65536, size=(305, 96, 69, 846), dtype=np.uint16)

# A scalar percentile
p99 = fastpercentile.percentile(arr, 99)

# Multiple percentiles in a single pass over the data
p1, p50, p99, p99_9 = fastpercentile.percentile(arr, [1, 50, 99, 99.9])

# Or just grab the histogram if you want to do something else with it
hist = fastpercentile.histogram(arr)  # length 65536 for uint16
```

Results match `numpy.percentile(arr, q)` with the default `'linear'` interpolation method (typically exact for integer inputs).


### Supported dtypes

`int8`, `uint8`, `int16`, `uint16`. Floats and 32/64-bit integers are not supported because a direct histogram is not feasible for them — for those, use `numpy.percentile` or `bottleneck.nanpercentile`.


### Memory layout

A 4D array loaded by `pynrrd` is typically Fortran-contiguous. `fastpercentile` walks raw memory, so it does not care about C vs F order — both are handled as a no-copy view. Arbitrarily-strided arrays fall back to a copy.


### Installation

**Option 1:** `pip install` from PyPI:

    pip install fastpercentile

**Option 2:** `pip install` directly from GitHub:

    pip install git+https://github.com/jasper-tms/fastpercentile.git

**Option 3:** First `git clone` this repo and then `pip install` it from your clone:

    cd ~/repos
    git clone https://github.com/jasper-tms/fastpercentile.git
    cd fastpercentile
    pip install '.[dev]'


### Notes on threading

`fastpercentile` uses every logical core on the machine by default (via `numba.get_num_threads()`). To limit it for a particular call, pass `n_threads=N`; to set it globally, use `numba.set_num_threads(N)` or the `NUMBA_NUM_THREADS` environment variable. On most systems the workload saturates DRAM bandwidth around `nproc / 2` threads, so reserving a few cores for the rest of the machine costs little throughput.
