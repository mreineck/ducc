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
# Copyright(C) 2026 Max-Planck-Society

"""Regression tests for bugs found in the 2026-09 bug hunt.

Every test below documents one confirmed bug and asserts the *correct*
behavior. All of them are marked xfail(strict=True) while the bug is
present, so:

  * running the file shows each test as XFAIL, which proves the bug is
    still there (each failure was verified against ducc0 0.41.1);
  * once a bug is fixed, its test turns into XPASS and pytest reports a
    failure, telling you to drop the xfail marker and keep the test as
    a permanent regression test.
"""

import numpy as np
import pytest

import ducc0

pmp = pytest.mark.parametrize
xfail_because = lambda reason: pytest.mark.xfail(strict=True, reason=reason)


@xfail_because("misc.special_add_at never validates index values and writes "
               "out of bounds (misc_pymod.cc:165); should raise like np.add.at")
def test_special_add_at_out_of_bounds():
    # `out` is a view into a larger buffer; index 20 lies outside out.shape.
    # Correct behavior: exception. Buggy behavior: silent OOB write, which
    # corrupts buf[30] (deterministic, no segfault needed to demonstrate).
    buf = np.zeros(64)
    out = buf[10:14]
    with pytest.raises(Exception):
        ducc0.misc.special_add_at(out, 0, np.array([20], np.int64),
                                  np.array([7.0]))
    assert buf[30] == 0.0


@xfail_because("nufft.u2nu: the grid dimensionality check is dead code "
               "(nufft_pymod.cc:58); a 1-D grid with 2-D coords is silently "
               "reshaped to (1, n) instead of raising")
def test_u2nu_grid_dimension_check_dead():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-.5, .5, (10, 2))
    grid1d = rng.standard_normal(16).astype(np.complex128)
    with pytest.raises(Exception):
        ducc0.nufft.u2nu(coord=coords, grid=grid1d, forward=True, epsilon=1e-8)


@xfail_because("sht.sharpjob_d.map2alm has no HEALPix branch and fails deep "
               "in analysis_2d with a misleading error (sht_pymod.cc:2162); "
               "either support it or reject it with a clear message")
def test_map2alm_healpix_geometry():
    job = ducc0.sht.sharpjob_d()
    job.set_healpix_geometry(2)
    job.set_triangular_alm_info(4, 4)
    with pytest.raises(Exception, match="(?i).*(healpix|not supported).*"):
        job.map2alm(np.zeros(48, dtype=np.float64))


@xfail_because("misc.l2error scalar overload returns NaN for two zeros; its "
               "docstring (and the array overload) promise 0.0 "
               "(misc_pymod.cc:527)")
def test_l2error_scalar_zero():
    assert ducc0.misc.l2error(0.0, 0.0) == 0.0


@xfail_because("sht.experimental.alm2leg documents mval/mstart as uint64 "
               "but the binding requires int64 (sht_pymod.cc:428 vs :175)")
def test_alm2leg_mval_dtype_as_documented():
    lmax = 4
    alm = np.zeros((lmax+1)*(lmax+1), dtype=np.complex128)
    theta = np.linspace(.1, np.pi-.1, 5)
    mval = np.arange(lmax+1, dtype=np.uint64)
    mstart = np.arange(lmax+1, dtype=np.int64)*10
    leg = ducc0.sht.experimental.alm2leg(
        alm=alm, lmax=lmax, theta=theta, spin=0, mval=mval, mstart=mstart)
    assert leg is not None


@xfail_because("totalconvolve Interpolator.interpol return shape is the "
               "opposite of the documented one (totalconvolve_pymod.cc:744)")
def test_interpol_return_shape_as_documented():
    lmax, kmax, ncomp = 8, 4, 3
    ptg = np.random.default_rng(0).random((100, 3))
    ptg[:, 0] *= np.pi
    ptg[:, 1:] *= 2*np.pi
    inter = ducc0.totalconvolve.Interpolator(lmax, kmax, ncomp,
                                             epsilon=1e-10, nthreads=1)
    res = inter.interpol(ptg)
    # docstring: "n2 is either 1 (if separate=True was used in the
    # constructor) or the second dimension of the input slm and blm arrays"
    assert res.shape == (1, 100)


@xfail_because("totalconvolve Interpolator.getSlm beam/return shapes are "
               "transposed relative to the docstring "
               "(totalconvolve_pymod.cc:784)")
def test_getslm_shapes_as_documented():
    lmax, kmax, ncomp = 6, 4, 3
    inter = ducc0.totalconvolve.Interpolator(lmax, kmax, ncomp,
                                             epsilon=1e-10, nthreads=1)
    beam = np.zeros((ncomp, (kmax+1)**2), dtype=np.complex128)
    out = inter.getSlm(beam)
    # docstring claims beam is (nalm_beam, nbeam) and the return value is
    # (nalm_sky, nbeam); reality is (nbeam, nalm) for both
    assert out.shape == ((lmax+1)**2, ncomp)


@xfail_because("wgridder.experimental.dirty2vis_bda docstrings describe "
               "(nrows, nchan) 2-D vis arrays but the API is 1-D "
               "(wgridder_pymod.cc:220)")
def test_dirty2vis_bda_return_shape_as_documented():
    rng = np.random.default_rng(0)
    nrow = 4
    uvw = rng.uniform(-1, 1, (nrow, 3))
    fl_id = np.zeros(nrow, dtype=np.uint64)
    fl_nfreqs = np.array([3], dtype=np.uint64)
    fl_freqs = np.linspace(1e8, 1.1e8, 3)
    res = ducc0.wgridder.experimental.dirty2vis_bda(
        uvw=uvw, freqlist_id=fl_id, freqlist_nfreqs=fl_nfreqs,
        freqlist_freqs=fl_freqs, dirty=np.zeros((16, 16)),
        pixsize_x=.002, pixsize_y=.002, epsilon=1e-6)
    assert res.shape == (nrow, 3)


@xfail_because("misc.vdot docstring accepts scalars but the binding takes "
               "only arrays (misc_pymod.cc:58 vs :110)")
def test_vdot_scalar_as_documented():
    assert ducc0.misc.vdot(3.0, 4.0) == 12.0


@xfail_because("misc.make_noncritical docstring accepts integer dtypes but "
               "the dispatcher rejects them (misc_pymod.cc:606)")
def test_make_noncritical_int_as_documented():
    arr = np.arange(8, dtype=np.int64)
    res = ducc0.misc.make_noncritical(arr)
    assert res.dtype == np.int64
    assert np.array_equal(res, arr)


@xfail_because("fft.r2r_fftpack docstring says N is 'the length of axis' "
               "but the normalization uses the product over all transformed "
               "axes (fft_pymod.cc:644 vs :205)")
def test_r2r_fftpack_multi_axis_norm_as_documented():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 8))
    r0 = ducc0.fft.r2r_fftpack(a, (0, 1), True, True, 0)
    r2 = ducc0.fft.r2r_fftpack(a, (0, 1), True, True, 2)
    # documented: N is "the length of `axis`", i.e. a single axis length;
    # actual: N is the product of the lengths of all transformed axes
    assert np.allclose(r2, r0/6)


@xfail_because("sigma_min/sigma_max docstrings demand 1.2<=sigma_min<"
               "sigma_max<=2.5, but the actual (and defaulted) range is "
               "1.1..2.6 (sht/totalconvolve/nufft pymods)")
def test_sigma_range_as_documented():
    # nufft's own defaults (1.19, 2.51) violate the documented range
    # "1.2 <= sigma_min < sigma_max <= 2.5"; a call with the defaults must
    # therefore fail if the documentation were right
    rng = np.random.default_rng(0)
    coords = rng.uniform(-.5, .5, (10, 2))
    grid = rng.standard_normal((8, 8)).astype(np.complex128)
    with pytest.raises(Exception):
        ducc0.nufft.u2nu(coord=coords, grid=grid, forward=True, epsilon=1e-8,
                         sigma_min=1.19, sigma_max=2.51)


@xfail_because("sht.rotate_alm docstring promises output 'same dtype and "
               "shape as alm', but with mmax_out<mmax_in the output is "
               "shorter (sht_pymod.cc:137)")
def test_rotate_alm_output_shape_as_documented():
    lmax, mmax_in, mmax_out = 6, 6, 2
    nalm_in = ((mmax_in+1)*(mmax_in+2))//2 + (mmax_in+1)*(lmax-mmax_in)
    alm = np.zeros(nalm_in, dtype=np.complex128)
    out = ducc0.sht.rotate_alm(alm, lmax, 0., 0., 0.,
                               mmax_in=mmax_in, mmax_out=mmax_out)
    # docstring claims the return value has the "same dtype and shape as alm",
    # but ncoeff_out = 18 != ncoeff_in = 28 (the same docstring's own formula)
    assert out.shape == alm.shape
