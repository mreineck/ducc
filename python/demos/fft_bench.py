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
# Copyright(C) 2019-2021 Max-Planck-Society


import numpy as np
import ducc0
from time import time
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)


def measure_fftw(a, nrepeat, nthr, flags=('FFTW_MEASURE',)):
    import pyfftw
    f1 = pyfftw.empty_aligned(a.shape, dtype=a.dtype)
    f2 = pyfftw.empty_aligned(a.shape, dtype=a.dtype)
    fftw = pyfftw.FFTW(f1, f2, flags=flags, axes=range(a.ndim), threads=nthr)
    f1[()] = a
    times = []
    for i in range(nrepeat):
        t0 = time()
        fftw()
        t1 = time()
        times.append(t1-t0)
    return times, f2


def measure_fftw_est(a, nrepeat, nthr):
    return measure_fftw(a, nrepeat, nthr, flags=('FFTW_ESTIMATE',))


def measure_fftw_np_interface(a, nrepeat, nthr):
    import pyfftw
    pyfftw.interfaces.cache.enable()
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = pyfftw.interfaces.numpy_fft.fftn(a)
        t1 = time()
        del b
        times.append(t1-t0)
    return times, b


def measure_duccfft(a, nrepeat, nthr):
    times = []
    b = a.copy()
    for i in range(nrepeat):
        t0 = time()
        b = ducc0.fft.c2c(a, out=b, forward=True, nthreads=nthr)
        t1 = time()
        times.append(t1-t0)
    return times, b


# ducc0.fft, avoiding critical array strides and using in-place transforms.
# This is probably the most performant mode for ducc0 with multi-D transforms.
def measure_duccfft_noncrit_inplace(a, nrepeat, nthr):
    times = []
    work = ducc0.misc.make_noncritical(a.copy())
    for i in range(nrepeat):
        work[()] = a
        t0 = time()
        work = ducc0.fft.c2c(work, out=work, forward=True, nthreads=nthr)
        t1 = time()
        times.append(t1-t0)
    return times, work


def measure_scipy_fftpack(a, nrepeat, nthr):
    import scipy.fftpack
    times = []
    if nthr != 1:
        raise NotImplementedError("scipy.fftpack does not support multiple threads")
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = scipy.fftpack.fftn(a)
        t1 = time()
        times.append (t1-t0)
    return times, b


def measure_scipy_fft(a, nrepeat, nthr):
    import scipy.fft
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = scipy.fft.fftn(a, workers=nthr)
        t1 = time()
        times.append(t1-t0)
    return times, b


def measure_numpy_fft(a, nrepeat, nthr):
    if nthr != 1:
        raise NotImplementedError("numpy.fft does not support multiple threads")
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = np.fft.fftn(a)
        t1 = time()
        times.append(t1-t0)
    return times, b


def measure_mkl_fft(a, nrepeat, nthr):
    import os
    os.environ['OMP_NUM_THREADS'] = str(nthr)
    import mkl_fft
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = mkl_fft.fftn(a)
        t1 = time()
        times.append(t1-t0)
    return times, b


def bench_nd(ndim, nmax, nthr, ntry, tp, funcs, nrepeat, ttl="", filename="",
             nice_sizes=True):
    print("{}D, type {}, max extent is {}:".format(ndim, tp, nmax))
    results = [[] for i in range(len(funcs))]
    for n in range(ntry):
        shp = rng.integers(nmax//3, nmax+1, ndim)
        if nice_sizes:
            shp = np.array([ducc0.fft.good_size(sz) for sz in shp])
        print("  {0:4d}/{1}: shape={2} ...".format(n, ntry, shp), end=" ", flush=True)
        a = (rng.random(shp)-0.5 + 1j*(rng.random(shp)-0.5)).astype(tp)
        output = []
        for func, res in zip(funcs, results):
            tmp = func(a, nrepeat, nthr)
            res.append(np.average(tmp[0]))
            output.append(tmp[1])
        print("{0:5.2e}/{1:5.2e} = {2:5.2f}  L2 error={3}".format(results[0][n], results[1][n], results[0][n]/results[1][n], ducc0.misc.l2error(output[0], output[1])))
    results = np.array(results)
    plt.title("{}: {}D, {}, max_extent={}".format(
        ttl, ndim, str(tp), nmax))
    plt.xlabel("time ratio")
    plt.ylabel("counts")
    plt.hist(results[0, :]/results[1, :], bins="auto")
    if filename != "":
        plt.savefig(filename)
    plt.show()
    plt.close()


funcs = (measure_duccfft_noncrit_inplace, measure_fftw)
ttl = "duccfft/FFTW"
ntry = 10
nthr = 1
nice_sizes = True
limits = [8192, 2048, 256]
#limits = [524288, 8192, 512]
bench_nd(1, limits[0], nthr, ntry, "c16", funcs, 10, ttl, "1d.png", nice_sizes)
bench_nd(2, limits[1], nthr, ntry, "c16", funcs, 10, ttl, "2d.png", nice_sizes)
bench_nd(3, limits[2], nthr, ntry, "c16", funcs, 10, ttl, "3d.png", nice_sizes)
bench_nd(1, limits[0], nthr, ntry, "c8", funcs, 10, ttl, "1d_single.png", nice_sizes)
bench_nd(2, limits[1], nthr, ntry, "c8", funcs, 10, ttl, "2d_single.png", nice_sizes)
bench_nd(3, limits[2], nthr, ntry, "c8", funcs, 10, ttl, "3d_single.png", nice_sizes)
