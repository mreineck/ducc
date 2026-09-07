"""Negative typing checks for rejected DUCC calls."""

import numpy as np
from numpy.typing import NDArray

import ducc0.fft
import ducc0.misc.experimental
import ducc0.sht
import ducc0.wgridder

map32: NDArray[np.float32] = np.empty((1, 2, 2), dtype=np.float32)
alm64: NDArray[np.complex128] = np.empty((1, 3), dtype=np.complex128)
uvw32: NDArray[np.float32] = np.empty((1, 3), dtype=np.float32)
freq64: NDArray[np.float64] = np.empty(1, dtype=np.float64)
vis32: NDArray[np.complex64] = np.empty((1, 1), dtype=np.complex64)
complex64: NDArray[np.complex64] = np.empty(2, dtype=np.complex64)
f32: NDArray[np.float32] = np.empty(2, dtype=np.float32)

# SHT map and alm precision must match.
ducc0.sht.analysis_2d(map=map32, spin=0, lmax=8, geometry="GL", alm=alm64)  # type: ignore

# Wgridder coordinates require float64.
ducc0.wgridder.vis2dirty(uvw=uvw32, freq=freq64, vis=vis32, npix_x=32, npix_y=32, pixsize_x=0.01, pixsize_y=0.01, epsilon=1e-6)  # type: ignore

# FFT axes are integer indices.
ducc0.fft.c2c(a=complex64, axes=[0.0])  # type: ignore

# mul_conj requires a complex second argument.
ducc0.misc.experimental.mul_conj(complex64, f32)  # type: ignore
