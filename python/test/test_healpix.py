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


import ducc0.healpix as ph
import numpy as np
import math
import pytest
from numpy.testing import assert_equal

pmp = pytest.mark.parametrize


def list2fixture(lst):
    @pytest.fixture(params=lst)
    def myfixture(request):
        return request.param

    return myfixture


pow2 = [1 << shift for shift in range(29)]
nonpow2 = [i+7 for i in pow2]
nside_nest = list2fixture(pow2)
nside_ring = list2fixture(pow2+nonpow2)

vlen = list2fixture([1, 10, 100, 1000, 10000])
ftype = list2fixture([np.float32, np.float64])
itype = list2fixture([np.int32, np.int64])


def random_ptg(rng, vlen):
    res = np.empty((vlen, 2), dtype=np.float64)
    res[:, 0] = np.arccos((rng.random(vlen)-0.5)*2)
#    res[:, 0] = math.pi*rng.random(vlen)
    res[:, 1] = rng.random(vlen)*2*math.pi
    return res


def test_pixangpix_nest(vlen, nside_nest, itype):
    if itype == np.int32 and nside_nest > 8192:
        return
    base = ph.Healpix_Base(nside_nest, "NEST")
    rng = np.random.default_rng(42)
    inp = rng.integers(low=0, high=12*nside_nest*nside_nest-1, size=vlen)
    out = base.ang2pix(base.pix2ang(inp.astype(itype)))
    assert_equal(inp, out)


def test_pixangpix_ring(vlen, nside_ring, itype):
    if itype == np.int32 and nside_ring > 8192:
        return
    base = ph.Healpix_Base(nside_ring, "RING")
    rng = np.random.default_rng(42)
    inp = rng.integers(low=0, high=12*nside_ring*nside_ring-1, size=vlen)
    out = base.ang2pix(base.pix2ang(inp))
    assert_equal(inp, out)


def test_vecpixvec_nest(vlen, nside_nest, ftype):
    if ftype == np.float32 and nside_nest > 8192:
        return
    base = ph.Healpix_Base(nside_nest, "NEST")
    rng = np.random.default_rng(42)
    inp = ph.ang2vec(random_ptg(rng, vlen).astype(ftype)).astype(ftype)
    out = base.pix2vec(base.vec2pix(inp)).astype(ftype)
    assert_equal(np.all(ph.v_angle(inp, out) < base.max_pixrad()), True)


def test_vecpixvec_ring(vlen, nside_ring, ftype):
    if ftype == np.float32 and nside_ring > 8192:
        return
    base = ph.Healpix_Base(nside_ring, "RING")
    rng = np.random.default_rng(42)
    inp = ph.ang2vec(random_ptg(rng, vlen).astype(ftype)).astype(ftype)
    out = base.pix2vec(base.vec2pix(inp)).astype(ftype)
    assert_equal(np.all(ph.v_angle(inp, out) < base.max_pixrad()), True)


def test_ringnestring(vlen, nside_nest, itype):
    if itype == np.int32 and nside_nest > 8192:
        return
    base = ph.Healpix_Base(nside_nest, "NEST")
    rng = np.random.default_rng(42)
    inp = rng.integers(low=0, high=12*nside_nest*nside_nest-1, size=vlen)
    out = base.ring2nest(base.nest2ring(inp.astype(itype)).astype(itype))
    assert_equal(np.all(out == inp), True)


def test_vecangvec(vlen, ftype):
    rng = np.random.default_rng(42)
    inp = random_ptg(rng, vlen).astype(ftype)
    out = ph.vec2ang(ph.ang2vec(inp))
    assert_equal(np.all(np.abs(out-inp) < 1e-14), True)
