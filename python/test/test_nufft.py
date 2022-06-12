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
# Copyright(C) 2020-2022 Max-Planck-Society

from itertools import product

import ducc0
import ducc0

try:
    import finufft
    have_finufft = True
except ImportError:
    have_finufft = False
import numpy as np
import pytest
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize


# def explicit_nufft(uvw, freq, ms, nxdirty, nydirty, xpixsize, ypixsize):
    # x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
                       # indexing='ij')
    # x *= xpixsize
    # y *= ypixsize
    # res = np.zeros((nxdirty, nydirty))
    # eps = x**2+y**2
    # if apply_w:
        # nm1 = -eps/(np.sqrt(1.-eps)+1.)
        # n = nm1+1
    # else:
        # nm1 = 0.
        # n = 1.
    # for row in range(ms.shape[0]):
        # for chan in range(ms.shape[1]):
            # if mask is not None and mask[row, chan] == 0:
                # continue
            # phase = (freq[chan]/SPEEDOFLIGHT *
                     # (x*uvw[row, 0] + y*uvw[row, 1] - uvw[row, 2]*nm1))
            # if wgt is None:
                # res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
            # else:
                # res += (ms[row, chan]*wgt[row, chan]
                        # * np.exp(2j*np.pi*phase)).real
    # return res/n


# def with_finufft(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize,
                 # mask, epsilon):
    # u = np.outer(uvw[:, 0], freq)*(xpixsize/SPEEDOFLIGHT)*2*np.pi
    # v = np.outer(uvw[:, 1], freq)*(ypixsize/SPEEDOFLIGHT)*2*np.pi
    # if wgt is not None:
        # ms = ms*wgt
    # if mask is not None:
        # ms = ms*mask
    # eps = epsilon/10  # Apparently finufft measures epsilon differently
    # # Plan on the fly
    # res0 = finufft.nufft2d1(u.ravel(), v.ravel(), ms.ravel(), (nxdirty,
                            # nydirty), eps=eps).real
    # # Plan beforehand
    # plan = finufft.Plan(1, (nxdirty, nydirty), eps=eps)
    # plan.setpts(u.ravel(), v.ravel())
    # res1 = plan.execute(ms.ravel()).real
    # np.testing.assert_allclose(res0, res1)
    # return res0


@pmp('nx', [20, 21, 250, 257])
@pmp('ny', [21, 32, 250, 257])
@pmp("npoints", (1, 37, 1000))
@pmp("epsilon", (1e-1, 1e-3, 3e-5, 2e-13))
@pmp("forward", (True, False))
@pmp("singleprec", (True, False))
@pmp("nthreads", (1, 2, 7))
def test_nufft_2d(nx, ny, npoints, epsilon, forward, singleprec, nthreads):
    if singleprec and epsilon < 1e-6:
        pytest.skip()
    rng = np.random.default_rng(42)
    uvw = (rng.random((npoints, 2))-0.5)*2*np.pi
    ms = rng.random(npoints)-0.5 + 1j*(rng.random(npoints)-0.5)
    dirty = rng.random((nx, ny))-0.5
    dirty = dirty +  1j*(rng.random((nx, ny))-0.5)
    nu = nv = 0
    if singleprec:
        ms = ms.astype("c8")
        dirty = dirty.astype("c8")

    def check(d2, m2):
        ref = max(ducc0.misc.vdot(ms, ms).real, ducc0.misc.vdot(m2, m2).real,
                  ducc0.misc.vdot(dirty, dirty).real, ducc0.misc.vdot(d2, d2).real)
        tol = 3e-5*ref if singleprec else 2e-13*ref
        assert_allclose(ducc0.misc.vdot(ms, m2), ducc0.misc.vdot(d2, dirty), rtol=tol)

    dirty2 = np.empty((nx,ny), dtype=dirty.dtype)
    dirty2 = ducc0.nufft.nu2u(points=ms, coord=uvw, forward=forward, epsilon=epsilon, nthreads=nthreads, out=dirty2).astype("c16")
    ms2 = ducc0.nufft.u2nu(grid=dirty, coord=uvw, forward=not forward, epsilon=epsilon, nthreads=nthreads).astype("c16")
    check(dirty2, ms2)

    if have_finufft and not singleprec:
        comp = finufft.nufft2d2(uvw[:,0], uvw[:,1], dirty, nthreads=nthreads,eps=epsilon,isign=1 if forward else -1)
        if comp.ndim==0:
            comp=np.array([comp[()]])
        assert_allclose(ducc0.misc.l2error(ms2,comp), 0, atol=10*epsilon)

@pmp('nx', [20, 21, 64])
@pmp('ny', [21, 32, 64])
@pmp('nz', [22, 35, 64])
@pmp("npoints", (1, 37, 1000))
@pmp("epsilon", (1e-5, 1e-3, 3e-5, 2e-13))
@pmp("forward", (True, False))
@pmp("singleprec", (True, False))
@pmp("nthreads", (1, 2, 7))
def test_nufft_3d(nx, ny, nz, npoints, epsilon, forward, singleprec, nthreads):
    if singleprec and epsilon < 1e-6:
        pytest.skip()
    rng = np.random.default_rng(42)
    uvw = (rng.random((npoints, 3))-0.5)*2*np.pi
    ms = rng.random(npoints)-0.5 + 1j*(rng.random(npoints)-0.5)
    dirty = rng.random((nx, ny, nz))-0.5
    dirty = dirty +  1j*(rng.random((nx, ny, nz))-0.5)
    nu = nv = nw = 0
    if singleprec:
        ms = ms.astype("c8")
        dirty = dirty.astype("c8")

    def check(d2, m2):
        ref = max(ducc0.misc.vdot(ms, ms).real, ducc0.misc.vdot(m2, m2).real,
                  ducc0.misc.vdot(dirty, dirty).real, ducc0.misc.vdot(d2, d2).real)
        tol = 3e-5*ref if singleprec else 2e-13*ref
        assert_allclose(ducc0.misc.vdot(ms, m2), ducc0.misc.vdot(d2, dirty), rtol=tol)

    dirty2 = np.empty((nx,ny,nz), dtype=dirty.dtype)
    dirty2 = ducc0.nufft.nu2u(points=ms, coord=uvw, forward=forward, epsilon=epsilon, nthreads=nthreads, out=dirty2, verbosity=0).astype("c16")
    ms2 = ducc0.nufft.u2nu(grid=dirty, coord=uvw, forward=not forward, epsilon=epsilon, nthreads=nthreads, verbosity=0).astype("c16")
    check(dirty2, ms2)

    if have_finufft and not singleprec:
        comp = finufft.nufft3d2(uvw[:,0], uvw[:,1], uvw[:,2], dirty, nthreads=nthreads,eps=epsilon,isign=1 if forward else -1)
        if comp.ndim==0:
            comp=np.array([comp[()]])
        assert_allclose(ducc0.misc.l2error(ms2,comp), 0, atol=10*epsilon)
