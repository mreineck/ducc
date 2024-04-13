#!/usr/bin/env python3

import numpy as np
import numba
from numba import cuda
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
import math


# cuda.detect()

# Disable profiling of everything to have less cluttered output
cuda.profile_stop()

# Check cuFFT vs scipy fft in single and double precision
shape = 300, 100
a = cp.random.random(shape).astype(cp.complex64)
with scipy.fft.set_backend(cufft):
    b = scipy.fft.fft(a)  # equivalent to cufft.fft(a)
np.testing.assert_array_almost_equal(cp.asnumpy(b), scipy.fft.fft(cp.asnumpy(a)), decimal=5)

a = cp.random.random(shape).astype(cp.complex128)
with scipy.fft.set_backend(cufft):
    b = scipy.fft.fft(a)  # equivalent to cufft.fft(a)
np.testing.assert_array_almost_equal(cp.asnumpy(b), scipy.fft.fft(cp.asnumpy(a)), decimal=8)



# Integrate cuFFT into numba
@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

an_array = b
ref = cp.asnumpy(an_array) + 1
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
with cuda.profiling():
    increment_a_2D_array[blockspergrid, threadsperblock](an_array)

np.testing.assert_array_almost_equal(cp.asnumpy(an_array), ref)


# Automatic vectorization
SQRT_TWOPI = np.float32(math.sqrt(2 * math.pi))

@numba.vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian(x, x0, sigma):
    return math.exp(-((x - x0) / sigma)**2 / 2) / SQRT_TWOPI / sigma
x = np.linspace(-3, 3, 10000, dtype=np.float32)
g = gaussian(x, 0, 1)  # 1D result

x2d = x.reshape((100,100))
g2d = gaussian(x2d, 0, 1) # 2D result
