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
# Copyright(C) 2020-2021 Max-Planck-Society

from itertools import product

import ducc0.wgridder as ng
import ducc0
from ducc0.misc import vdot
import numpy as np
import pytest
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize
SPEEDOFLIGHT = 299792458.


@pmp('nx', [(10, 1), (30, 3), (128, 2)])
@pmp('ny', [(12, 1), (128, 2), (250, 5)])
@pmp("nrow", (1, 2, 27))
@pmp("nchan", (1, 5))
@pmp("epsilon", (1e-1, 1e-3, 3e-5, 2e-13))
@pmp("singleprec", (True, False))
@pmp("wstacking", (True, False))
@pmp("use_wgt", (True, False))
@pmp("use_mask", (False, True))
@pmp("nthreads", (1,))
def test_adjointness_ms2dirty(nx, ny, nrow, nchan, epsilon,
                              singleprec, wstacking, use_wgt, nthreads,
                              use_mask):
    (nxdirty, nxfacets), (nydirty, nyfacets) = nx, ny
    if singleprec and epsilon < 1e-6:
        pytest.skip()

    if wstacking or use_wgt:
        pytest.skip()

    if nxfacets != 1 or nyfacets != 1:
        pytest.skip()

    rng = np.random.default_rng(42)
    pixsizex = np.pi/180/60/nxdirty*0.2398
    pixsizey = np.pi/180/60/nxdirty
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (rng.random((nrow, 3))-0.5)/(pixsizey*f0/SPEEDOFLIGHT)
    ms = rng.random((nrow, nchan))-0.5 + 1j*(rng.random((nrow, nchan))-0.5)
    wgt = rng.uniform(0.9, 1.1, (nrow, nchan)) if use_wgt else None
    mask = (rng.uniform(0, 1, (nrow, nchan)) > 0.5).astype(np.uint8) \
        if use_mask else None
    dirty = rng.random((nxdirty, nydirty))-0.5
    nu = nv = 0
    if singleprec:
        ms = ms.astype("c8")
        dirty = dirty.astype("f4")
        if wgt is not None:
            wgt = wgt.astype("f4")

    kwargs = dict(uvw=uvw, freq=freq, dirty=dirty, wgt=wgt, pixsize_x=pixsizex, pixsize_y=pixsizey,
                  epsilon=epsilon, do_wgridding=wstacking, verbosity=0, mask=mask)

    ms0 = ng.experimental.dirty2vis(**kwargs, gpu=False).astype("c16")
    ms1 = ng.experimental.dirty2vis(**kwargs, gpu=True).astype("c16")

    assert_allclose(ducc0.misc.l2error(ms0, ms1), 0, atol=epsilon)
