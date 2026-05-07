#!/usr/bin/env python3
"""
Benchmark `fastpercentile.percentile` against `np.percentile` and
`np.max` on a large nrrd volume.

Run:
    python tests/benchmarks/benchmark_volume.py [path/to/file.nrrd]
"""
import os
import sys
import time

import numpy as np

import fastpercentile


DEFAULT_NRRD = os.path.expanduser(
    '~/Desktop/SCAPE-local/260316_AN19B004_invivo/j2_run3/'
    'j2_run3_green_demotioned.nrrd')


def time_call(fn, *args, repeats=3, **kwargs):
    best = float('inf')
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best, result


def main():
    import nrrd
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NRRD
    if not os.path.exists(path):
        print('NRRD file not found at ' + path)
        sys.exit(1)

    print('Loading ' + path)
    t0 = time.perf_counter()
    data, _ = nrrd.read(path)
    print('  loaded shape={} dtype={} in {:.2f}s'.format(
        data.shape, data.dtype, time.perf_counter() - t0))

    fastpercentile.warmup()

    qs = [1.0, 50.0, 99.0, 99.9]
    print('\n--- timings (best of 3) ---')

    t_max, max_val = time_call(np.max, data)
    print('np.max          : {:6.3f}s   -> max={}'.format(t_max, max_val))

    t_fast, fast_vals = time_call(fastpercentile.percentile, data, qs)
    print('fastpercentile  : {:6.3f}s   -> {}'.format(
        t_fast, dict(zip(qs, fast_vals))))

    print('np.percentile (slow, 1 repeat) ...')
    t0 = time.perf_counter()
    np_vals = np.percentile(data, qs)
    t_np = time.perf_counter() - t0
    print('np.percentile   : {:6.3f}s   -> {}'.format(
        t_np, dict(zip(qs, np_vals))))

    diffs = np.abs(np_vals - fast_vals)
    if not np.all(diffs < 1e-9):
        raise AssertionError('fastpercentile disagrees with np.percentile')

    print('\n--- summary ---')
    print('  fast / np.max        = {:.2f}x'.format(t_fast / t_max))
    print('  np.percentile / fast = {:.1f}x speedup'.format(t_np / t_fast))


if __name__ == '__main__':
    main()
