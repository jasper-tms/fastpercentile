# fastpercentile: Memory-bandwidth-bound percentile for small-integer arrays

[![Tests](https://github.com/jasper-tms/fastpercentile/actions/workflows/tests.yml/badge.svg)](https://github.com/jasper-tms/fastpercentile/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/fastpercentile)](https://pypi.org/project/fastpercentile/)
[![License](https://img.shields.io/github/license/jasper-tms/fastpercentile)](https://github.com/jasper-tms/fastpercentile/blob/main/LICENSE)

`np.percentile` is O(n) but in practice has a brutal constant factor — on a 1.7-billion-element `uint16` volume it takes ~14 s, vs ~0.12 s for `np.max`. There is no good reason for percentile to be so much slower than max: both can be done in a single pass over the array.

For small-integer dtypes (`int8`, `uint8`, `int16`, `uint16`) the data only takes one of at most 65536 distinct values, so a single parallel pass into a histogram captures everything you need to compute any percentile. After the histogram is built, walking the cumulative count to find the bin holding each requested rank costs essentially nothing.

This package implements that, in a few hundred lines of `numba`. On a 32-thread workstation it runs at DRAM bandwidth — about as fast as `np.max`, and over 100× faster than `np.percentile`:

```python
import time
import numpy as np
import fastpercentile

# A 1.7-billion-element uint16 volume (~3.4 GB; lower the size if you have less RAM)
arr = np.random.randint(0, 65536, size=1_700_000_000, dtype=np.uint16)
fastpercentile.percentile(arr, 50)  # warm up the numba JIT (it compiles on first call)

qs = [1, 50, 99, 99.9]

start = time.perf_counter()
arr.max()
print(f'np.max         : {time.perf_counter() - start:6.3f} s')

start = time.perf_counter()
fastpercentile.percentile(arr, qs)
print(f'fastpercentile : {time.perf_counter() - start:6.3f} s')

start = time.perf_counter()
np.percentile(arr, qs)
print(f'np.percentile  : {time.perf_counter() - start:6.3f} s')
```

```
np.max         :  0.120 s
fastpercentile :  0.121 s   <- four percentiles in one pass
np.percentile  : 14.043 s
```

Additional memory usage is only ~16 MB regardless of input size (32 threads × one 65536-bin local table each, plus a final reduced histogram), so it adds no measurable RAM pressure on top of the input.


### Usage

```python
import numpy as np
import fastpercentile

arr = np.random.randint(0, 65536, size=(305, 96, 69, 846), dtype=np.uint16)

# A scalar percentile
p99 = fastpercentile.percentile(arr, 99)

# The median (50th percentile)
m = fastpercentile.median(arr)

# Multiple percentiles in a single pass over the data
p1, p50, p99, p99_9 = fastpercentile.percentile(arr, [1, 50, 99, 99.9])

# Or just grab the histogram if you want to do something else with it.
# (Only works for 8-bit and 16-bit values because 32-bit and 64-bit
# histograms don't fit in memory.)
hist = fastpercentile.histogram(arr)  # length 65536 for uint16
```

Results match `numpy.percentile(arr, q)` with the default `'linear'` interpolation method (typically exact for integer inputs).


### Supported dtypes

All integer dtypes: `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`. Floats are not supported — for those, use `numpy.percentile` or `bottleneck.nanpercentile`.

8- and 16-bit integers use the single-pass histogram described above. For 32- and 64-bit integers a direct histogram is infeasible (it would need `2**32` or `2**64` 8-byte bins — 32 GB or ~150 exabytes), so they use a **radix refinement** instead: the first pass histograms only the top 16 bits of each value, which localizes each requested percentile to a coarse bucket; later passes re-scan the array but only count the next 16 bits of the few elements inside those buckets, narrowing the answer 16 bits at a time. This costs one pass per 16-bit digit — two for 32-bit input, four for 64-bit — and keeps auxiliary memory at the same fixed 65536-bin scale. It's still far faster than `numpy.percentile`, just with a 2× or 4× larger constant than the 16-bit path. (For 64-bit values above `2**53` the result carries the same float64 rounding `numpy.percentile` does.)

`fastpercentile.histogram()` only returns a full histogram for the 8- and 16-bit dtypes; for wider integers a full histogram isn't feasible, so call `percentile()` directly.


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
