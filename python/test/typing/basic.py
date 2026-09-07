"""Installed-wheel typing checks for the public DUCC modules."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

import ducc0
import ducc0.fft
import ducc0.healpix
import ducc0.misc
import ducc0.misc.experimental
import ducc0.nufft
import ducc0.nufft.experimental
import ducc0.pointingprovider
import ducc0.sht
import ducc0.sht.experimental
import ducc0.totalconvolve
import ducc0.wgridder

from ducc0 import misc, sht

f32: npt.NDArray[np.float32] = np.empty(4, dtype=np.float32)
f64: npt.NDArray[np.float64] = np.empty(4, dtype=np.float64)
c32: npt.NDArray[np.complex64] = np.empty(4, dtype=np.complex64)
c64: npt.NDArray[np.complex128] = np.empty(4, dtype=np.complex128)
i32: npt.NDArray[np.int32] = np.empty(4, dtype=np.int32)
i64: npt.NDArray[np.int64] = np.empty(4, dtype=np.int64)

assert_type(ducc0.fft.c2c(f32), npt.NDArray[np.complex64])
assert_type(ducc0.fft.c2c(c64), npt.NDArray[np.complex128])
assert_type(ducc0.fft.r2c(f64), npt.NDArray[np.complex128])
fft_result = ducc0.fft.c2c(f32)
assert_type(ducc0.fft.c2r(fft_result), npt.NDArray[np.float32])
assert_type(ducc0.fft.dct(f64, 2), npt.NDArray[np.float64])

assert_type(misc.GL_thetas(64), npt.NDArray[np.float64])
assert_type(misc.GL_weights(64, 128), npt.NDArray[np.float64])
assert_type(misc.quat2ptg(f32), npt.NDArray[np.float32])
assert_type(misc.experimental.mul_conj(f32, c32), npt.NDArray[np.complex64])
assert_type(misc.experimental.mul_conj(f32, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.mul_conj(f64, c32), npt.NDArray[np.complex128])
assert_type(misc.experimental.mul_conj(f64, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.mul_conj(c32, c32), npt.NDArray[np.complex64])
assert_type(misc.experimental.mul_conj(c32, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.mul_conj(c64, c32), npt.NDArray[np.complex128])
assert_type(misc.experimental.mul_conj(c64, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.div_conj(f32, c32), npt.NDArray[np.complex64])
assert_type(misc.experimental.div_conj(f32, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.div_conj(f64, c32), npt.NDArray[np.complex128])
assert_type(misc.experimental.div_conj(f64, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.div_conj(c32, c32), npt.NDArray[np.complex64])
assert_type(misc.experimental.div_conj(c32, c64), npt.NDArray[np.complex128])
assert_type(misc.experimental.div_conj(c64, c32), npt.NDArray[np.complex128])
assert_type(misc.experimental.div_conj(c64, c64), npt.NDArray[np.complex128])

healpix = ducc0.healpix.Healpix_Base(1, "RING")
assert_type(healpix.scheme(), Literal["RING", "NEST"])
assert_type(healpix.pix2ang(i64), npt.NDArray[np.float64])
assert_type(healpix.pix2ang(i32), npt.NDArray[np.float64])
assert_type(healpix.pix2vec(i32), npt.NDArray[np.float64])
assert_type(healpix.query_disc(f32, 0.5), npt.NDArray[np.int64])
assert_type(ducc0.healpix.ang2vec(f32), npt.NDArray[np.float64])

coords32: npt.NDArray[np.float32] = np.empty((4, 1), dtype=np.float32)
coords64: npt.NDArray[np.float64] = np.empty((4, 1), dtype=np.float64)
assert_type(
    ducc0.nufft.u2nu(
        grid=c32, coord=coords32, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex64],
)
assert_type(
    ducc0.nufft.u2nu(
        grid=c32, coord=coords64, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex64],
)
assert_type(
    ducc0.nufft.u2nu(
        grid=c64, coord=coords32, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex128],
)
assert_type(
    ducc0.nufft.u2nu(
        grid=c64, coord=coords64, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex128],
)
assert_type(
    ducc0.nufft.nu2u(
        points=c32, coord=coords32, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex64],
)
assert_type(
    ducc0.nufft.nu2u(
        points=c32, coord=coords64, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex64],
)
assert_type(
    ducc0.nufft.nu2u(
        points=c64, coord=coords32, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex128],
)
assert_type(
    ducc0.nufft.nu2u(
        points=c64, coord=coords64, forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex128],
)
assert_type(
    ducc0.nufft.experimental.nu2nu(
        points_in=c32, coord_in=coords32, coord_out=coords32,
        forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex64],
)
assert_type(
    ducc0.nufft.experimental.nu2nu(
        points_in=c32, coord_in=coords64, coord_out=coords64,
        forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex64],
)
assert_type(
    ducc0.nufft.experimental.nu2nu(
        points_in=c64, coord_in=coords32, coord_out=coords32,
        forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex128],
)
assert_type(
    ducc0.nufft.experimental.nu2nu(
        points_in=c64, coord_in=coords64, coord_out=coords64,
        forward=True, epsilon=1e-5
    ),
    npt.NDArray[np.complex128],
)

provider_input: npt.NDArray[np.float64] = np.empty((2, 4), dtype=np.float64)
provider = ducc0.pointingprovider.PointingProvider(0.0, 1.0, provider_input)
rotation: npt.NDArray[np.float64] = np.empty(4, dtype=np.float64)
provider_out32: npt.NDArray[np.float32] = np.empty((2, 4), dtype=np.float32)
provider_out64: npt.NDArray[np.float64] = np.empty((2, 4), dtype=np.float64)
assert_type(
    provider.get_rotated_quaternions(0.0, 1.0, rotation, 2),
    npt.NDArray[np.float64],
)
assert_type(
    provider.get_rotated_quaternions(
        0.0, 1.0, rotation, out=provider_out32
    ),
    npt.NDArray[np.float32],
)
assert_type(
    provider.get_rotated_quaternions(
        0.0, 1.0, rotation, out=provider_out64
    ),
    npt.NDArray[np.float64],
)

alm32 = sht.analysis_2d(map=f32, spin=0, lmax=1, geometry="GL")
alm64 = sht.analysis_2d(map=f64, spin=0, lmax=1, geometry="GL")
assert_type(alm32, npt.NDArray[np.complex64])
assert_type(alm64, npt.NDArray[np.complex128])
assert_type(sht.synthesis_2d(alm=alm32, spin=0, lmax=1, geometry="GL"), npt.NDArray[np.float32])
assert_type(sht.synthesis_2d(alm=alm64, spin=0, lmax=1, geometry="GL"), npt.NDArray[np.float64])
assert_type(sht.get_gridweights("GL", 64), npt.NDArray[np.float64])
sht.experimental.alm2flm(alm64, 0)

uvw: npt.NDArray[np.float64] = np.empty((1, 3), dtype=np.float64)
freq: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
vis32: npt.NDArray[np.complex64] = np.empty((1, 1), dtype=np.complex64)
vis64: npt.NDArray[np.complex128] = np.empty((1, 1), dtype=np.complex128)
dirty32: npt.NDArray[np.float32] = ducc0.wgridder.vis2dirty(
    uvw=uvw, freq=freq, vis=vis32, npix_x=4, npix_y=4,
    pixsize_x=1.0, pixsize_y=1.0, epsilon=1e-5,
)
dirty64: npt.NDArray[np.float64] = ducc0.wgridder.vis2dirty(
    uvw=uvw, freq=freq, vis=vis64, npix_x=4, npix_y=4,
    pixsize_x=1.0, pixsize_y=1.0, epsilon=1e-5,
)
assert_type(dirty32, npt.NDArray[np.float32])
assert_type(dirty64, npt.NDArray[np.float64])
assert_type(
    ducc0.wgridder.dirty2vis(
        uvw=uvw, freq=freq, dirty=dirty32,
        pixsize_x=1.0, pixsize_y=1.0, epsilon=1e-5,
    ),
    npt.NDArray[np.complex64],
)

convolver = ducc0.totalconvolve.ConvolverPlan(1, 1, epsilon=1e-5)
assert_type(convolver.Ntheta(), int)
interpolator = ducc0.totalconvolve.Interpolator(1, 1, 1, epsilon=1e-5)
assert_type(interpolator.interpol(f64), npt.NDArray[np.complex128])
