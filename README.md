# fastpercentile: Memory-bandwidth-bound percentile for small-integer arrays

[![Tests](https://github.com/jasper-tms/fastpercentile/actions/workflows/tests.yml/badge.svg)](https://github.com/jasper-tms/fastpercentile/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/fastpercentile)](https://pypi.org/project/fastpercentile/)
[![License](https://img.shields.io/github/license/jasper-tms/fastpercentile)](https://github.com/jasper-tms/fastpercentile/blob/main/LICENSE)

There's no reason why median and percentile calculations should be any slower than a `np.max()` call, and yet, on a 1-billion-element numpy array:

```
np.max                    :  0.080 seconds
np.median                 :  5.529 seconds
np.percentile             :  8.878 seconds
```

This package provides optimized versions of median and percentile calculations on numpy arrays, giving ~100× faster speeds than `np.percentile`:

```
fastpercentile.median     :  0.083 seconds
fastpercentile.percentile :  0.084 seconds
```

<details>
<summary>Click to see Python code that you can run yourself to compare the speeds on your computer</summary>

```python
import time
import numpy as np
import fastpercentile

# A 1-billion-element uint16 volume (takes up ~2 GB of RAM)
arr = np.random.randint(0, 65536, size=1_000_000_000, dtype=np.uint16)
qs = [1, 50, 99, 99.9]
fastpercentile.median(arr)  # Compile (happens once on first call) before measuring runtimes

commands_to_run = """
np.max(arr)
np.median(arr)
np.percentile(arr, qs)
fastpercentile.median(arr)
fastpercentile.percentile(arr, qs)
"""

for command in commands_to_run.strip().splitlines():
    start = time.perf_counter()
    result = eval(command)
    print(f'{command.split("(")[0]:26s}: {time.perf_counter() - start:6.3f} seconds, result {result}')
```

</details>

### Algorithm

For small-integer dtypes (`int8`, `uint8`, `int16`, `uint16`) the data only takes one of at most 65536 distinct values, so a single parallel pass of counting how many times each value occurs (that is, building a histogram of value occurrences) gives everything needed to compute any percentile. After the histogram is built, walking the cumulative count to find the bin holding each requested rank costs essentially nothing. The whole thing is a few hundred lines of `numba`, and additional memory usage is only ~16 MB regardless of input size (32 threads × one 65536-bin local table each, plus a final reduced histogram), so it adds no measurable RAM pressure on top of the input.

<details>
<summary>Click to read how we run this algorithm 2 or 4 times in a row to handle 32-bit or 64-bit integers, respectively, without using much additional memory.</summary>

For 32- and 64-bit integers, a direct histogram over all possible values is infeasible (it would need `2**32` or `2**64` 8-byte bins — 32 GB or ~150 exabytes), so they use a **radix refinement** instead: the first pass computes histograms on only the top 16 bits of each value, which localizes each requested percentile to a coarse bucket; later passes re-scan the array but only count the next 16 bits of the few elements inside those buckets, narrowing the answer 16 bits at a time. This costs one pass per 16-bit digit — two for 32-bit input, four for 64-bit — and keeps auxiliary memory at the same fixed 65536-bin scale. 32-bit arrays are ~4× slower and 64-bit arrays are ~16× slower to process than 16-bit data, which is still much faster than `np.percentile`. (Note that for 64-bit values above `2**53`, the result carries the same float64 rounding error that `numpy.percentile` also has.)

</details>

Floats are not supported — for those, use `numpy.percentile` or `bottleneck.nanpercentile`.


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



### Installation

**Option 1:** `pip install` from PyPI:

    pip install fastpercentile

**Option 2:** `pip install` directly from GitHub:

    pip install git+https://github.com/jasper-tms/fastpercentile.git

**Option 3:** First `git clone` this repo and then `pip install` it from your clone:

    cd ~/repos
    git clone https://github.com/jasper-tms/fastpercentile.git
    cd fastpercentile
    pip install .


### Notes on threading

`fastpercentile` uses every logical core on the machine by default (via `numba.get_num_threads()`). To limit it for a particular call, pass `n_threads=N`; to set it globally, use `numba.set_num_threads(N)` or the `NUMBA_NUM_THREADS` environment variable. On most systems the workload saturates DRAM bandwidth around `nproc / 2` threads, so reserving a few cores for the rest of the machine costs little throughput.
