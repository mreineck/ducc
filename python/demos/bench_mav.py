# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2026 Philipp Arras
# Author: Philipp Arras

"""Benchmark of the axis handling in mav_apply and flexible_mav_apply (via
Python functions that are thin wrappers around them).

To track the effect of a change, run the script before and after it:

    python3 bench_mav.py --save before.json
    # rebuild ducc0
    python3 bench_mav.py --compare before.json
"""

import os

os.environ.setdefault("DUCC0_PIN_DISTANCE", "1")
os.environ.setdefault("DUCC0_PIN_OFFSET", "0")

import argparse
import json
from time import perf_counter

import numpy as np

import ducc0

rng = np.random.default_rng(42)


def timeit(func, budget=0.3, min_samples=10):
    """Median runtime of func() in seconds."""
    func()  # warmup
    t0 = perf_counter()
    func()
    inner = max(1, int(2e-3/max(perf_counter()-t0, 1e-9)))  # samples >= ~2 ms
    samples = []
    tstart = perf_counter()
    while len(samples) < min_samples or perf_counter()-tstart < budget:
        t0 = perf_counter()
        for _ in range(inner):
            func()
        samples.append((perf_counter()-t0)/inner)
    return float(np.median(samples))


def padded(nrows, ncols, pad):
    """Random (nrows, ncols) array with gaps of `pad` elements between rows,
    so that its axes cannot be merged."""
    return rng.random((nrows, ncols+pad))[:, :ncols]


def padded3(n0, n1, n2, pad):
    """Random (n0, n1, n2) array with gaps of `pad` elements between rows and
    of one row between the (n1, n2) slices, so that no axes can be merged."""
    return rng.random((n0, n1+1, n2+pad))[:, :n1, :n2]


def copy_case(inp):
    out = np.empty(inp.shape)
    def setup(nthreads):
        func = lambda: ducc0.misc.transpose(inp, out, nthreads)
        check = lambda: np.array_equal(out, inp)
        return func, check
    return setup


def reduce_case(a, b, c):
    ref = 0.5*np.sum((a-b)**2*c)
    def setup(nthreads):
        res = [None]
        def func():
            res[0] = ducc0.misc.experimental.LogUnnormalizedGaussProbability(
                a, b, c, nthreads)
        check = lambda: abs(res[0]-ref) <= 1e-10*abs(ref)
        return func, check
    return setup


hpbase = ducc0.healpix.Healpix_Base(1024, "RING")


def padded_pix(shape, pad):
    """Random pixel indices of the given shape (2 or 3 axes), with gaps of
    `pad` entries along the last axis and of one row along the second-to-last
    axis, so that no axes can be merged."""
    full = rng.integers(0, hpbase.npix(),
                        shape[:-2]+(shape[-2]+1, shape[-1]+pad))
    return full[..., :shape[-2], :shape[-1]]


def pix2ang_case(pix):
    ref = hpbase.pix2ang(pix, 1)
    def setup(nthreads):
        res = [None]
        def func():
            res[0] = hpbase.pix2ang(pix, nthreads)
        check = lambda: np.array_equal(res[0], ref)
        return func, check
    return setup


def cases():
    # Can a short axis 0 be merged with the other axes?
    yield ("transpose 4x2048x2048 (swap last 2 axes)", True,
           lambda: copy_case(rng.random((4, 2048, 2048)).transpose(0, 2, 1)))
    yield ("transpose 3x1000x1000 (swap last 2 axes)", True,
           lambda: copy_case(rng.random((3, 1000, 1000)).transpose(0, 2, 1)))
    yield ("copy 4x4194304 (padded rows)", True,
           lambda: copy_case(padded(4, 1 << 22, 8)))
    yield ("reduce 3x1048576 (padded rows)", True,
           lambda: reduce_case(*(padded(3, 1 << 20, 8) for _ in range(3))))

    # Several short leading axes
    yield ("transpose 2x2x2048x2048 (swap last 2 axes)", True,
           lambda: copy_case(rng.random((2, 3, 2048, 2048))[:, :2]
                             .transpose(0, 1, 3, 2)))
    yield ("copy 2x2x4194304 (padded rows)", True,
           lambda: copy_case(padded3(2, 2, 1 << 22, 8)))
    yield ("reduce 2x2x1048576 (padded rows)", True,
           lambda: reduce_case(*(padded3(2, 2, 1 << 20, 8) for _ in range(3))))

    # Leading axes of intermediate length: axis 0 alone is too short for
    # 8*nthreads work items at 64 threads (512) and/or 16 threads (128)
    yield ("copy 24x262144 (padded rows)", True,
           lambda: copy_case(padded(24, 1 << 18, 8)))
    yield ("copy 100x262144 (padded rows)", True,
           lambda: copy_case(padded(100, 1 << 18, 8)))
    yield ("copy 24x8x65536 (padded rows)", True,
           lambda: copy_case(padded3(24, 8, 1 << 16, 8)))
    yield ("copy 100x4x32768 (padded rows)", True,
           lambda: copy_case(padded3(100, 4, 1 << 15, 8)))
    yield ("copy 200x8x4096 (padded rows)", True,
           lambda: copy_case(padded3(200, 8, 4096, 8)))
    yield ("reduce 48x4x65536 (padded rows)", True,
           lambda: reduce_case(*(padded3(48, 4, 1 << 16, 8)
                                 for _ in range(3))))
    yield ("transpose 48x512x512 (swap last 2 axes)", True,
           lambda: copy_case(rng.random((48, 512, 512)).transpose(0, 2, 1)))
    yield ("transpose 100x256x256 (swap last 2 axes)", True,
           lambda: copy_case(rng.random((100, 256, 256)).transpose(0, 2, 1)))
    yield ("flexible pix2ang 24x8x65536 (padded rows)", True,
           lambda: pix2ang_case(padded_pix((24, 8, 1 << 16), 8)))

    # Compute bound, with a number of rows just above F*nthreads for
    # nthreads=16 (17, 33, 65, 129) and nthreads=64 (65, 129, 257, 513), which
    # is the worst case for the load balance if only axis 0 is distributed
    for nrows in (17, 33, 65, 129, 257, 513):
        yield (f"flexible pix2ang {nrows}x16384 (padded rows)", True,
               lambda nrows=nrows: pix2ang_case(padded_pix((nrows, 1 << 14), 8)))

    # flexible_mav_apply (compute bound)
    yield ("flexible pix2ang 4x1048576 (padded rows)", True,
           lambda: pix2ang_case(padded_pix((4, 1 << 20), 8)))
    yield ("flexible pix2ang 2x2x1048576 (padded rows)", True,
           lambda: pix2ang_case(padded_pix((2, 2, 1 << 20), 8)))
    yield ("flexible pix2ang 4194304 (contiguous, control)", True,
           lambda: pix2ang_case(rng.integers(0, hpbase.npix(), 1 << 22)))

    # Controls with a long axis 0
    yield ("transpose 4000x4000", True,
           lambda: copy_case(rng.random((4000, 4000)).T))
    yield ("copy 4096x4096 (padded rows)", True,
           lambda: copy_case(padded(4096, 4096, 8)))

    # Broadcast (stride-0) input along the contiguous axis of the others
    yield ("copy 2^24 from scalar (stride 0)", True,
           lambda: copy_case(np.broadcast_to(rng.random(1), (1 << 24,))))

    # Per-row overhead (via short, non-mergeable rows)
    yield ("copy 32768x2 (short rows, in cache)", False,
           lambda: copy_case(padded(32768, 2, 2)))
    yield ("copy 16384x4 (short rows, in cache)", False,
           lambda: copy_case(padded(16384, 4, 4)))
    yield ("copy 4194304x4 (short rows, main memory)", False,
           lambda: copy_case(padded(1 << 22, 4, 4)))
    yield ("reduce 16384x4 (short rows, in cache)", False,
           lambda: reduce_case(*(padded(16384, 4, 4) for _ in range(3))))
    yield ("reduce 4194304x4 (short rows, main memory)", False,
           lambda: reduce_case(*(padded(1 << 22, 4, 4) for _ in range(3))))

    # Broadcast input along the last axis
    def bcast_last(nrows, ncols):
        return np.broadcast_to(rng.random((nrows, 1)), (nrows, ncols))
    yield ("copy 512x256 (input stride 0 on last axis)", False,
           lambda: copy_case(bcast_last(512, 256)))
    yield ("reduce 16384x64 (one input stride 0 on last axis)", False,
           lambda: reduce_case(rng.random((16384, 64)), rng.random((16384, 64)),
                               bcast_last(16384, 64)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    nthreads_default = min(16, ducc0.misc.available_hardware_threads())
    parser.add_argument("--nthreads", type=int, nargs="+",
                        default=[nthreads_default],
                        help="thread counts for the parallel cases")
    parser.add_argument("--save", help="write results to this JSON file")
    parser.add_argument("--compare",
                        help="compare with results in this JSON file")
    parser.add_argument("--filter", default="",
                        help="only run cases whose name contains this string")
    args = parser.parse_args()

    ducc0.misc.preallocate_memory(3.)
    print(f"ducc0 {ducc0.__version__} from {ducc0.__file__}")
    print(f"DUCC0_PIN_DISTANCE={os.environ['DUCC0_PIN_DISTANCE']} "
          f"DUCC0_PIN_OFFSET={os.environ['DUCC0_PIN_OFFSET']}")
    old = {}
    if args.compare:
        with open(args.compare) as f:
            old = json.load(f)

    header = f"{'case':52s} {'nthr':>4s} {'time [ms]':>10s}"
    if old:
        header += f" {'before [ms]':>12s} {'speedup':>8s}"
    print(header)
    results = {}
    for name, parallel, make in cases():
        if args.filter not in name:
            continue
        setup = make()
        for nthreads in (args.nthreads if parallel else [1]):
            func, check = setup(nthreads)
            t = timeit(func)
            if not check():
                raise RuntimeError(f"wrong result: {name}, "
                                   f"nthreads={nthreads}")
            key = f"{name} | nthreads={nthreads}"
            results[key] = t
            line = f"{name:52s} {nthreads:4d} {t*1e3:10.3f}"
            if key in old:
                line += f" {old[key]*1e3:12.3f} {old[key]/t:7.2f}x"
            print(line, flush=True)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
