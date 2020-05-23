import cxxbase1.pyHealpix as ph
import numpy as np
import math
import pytest
from numpy.testing import assert_equal, assert_allclose

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


def random_ptg(vlen):
    res = np.empty((vlen, 2), dtype=np.float64)
    res[:, 0] = np.arccos((np.random.random_sample(vlen)-0.5)*2)
#    res[:, 0] = math.pi*np.random.random_sample(vlen)
    res[:, 1] = np.random.random_sample(vlen)*2*math.pi
    return res


def test_pixangpix_nest(vlen, nside_nest):
    base = ph.Healpix_Base(nside_nest, "NEST")
    inp = np.random.randint(low=0, high=12*nside_nest*nside_nest-1, size=vlen)
    out = base.ang2pix(base.pix2ang(inp))
    assert_equal(inp, out)


def test_pixangpix_ring(vlen, nside_ring):
    base = ph.Healpix_Base(nside_ring, "RING")
    inp = np.random.randint(low=0, high=12*nside_ring*nside_ring-1, size=vlen)
    out = base.ang2pix(base.pix2ang(inp))
    assert_equal(inp, out)


def test_vecpixvec_nest(vlen, nside_nest):
    base = ph.Healpix_Base(nside_nest, "NEST")
    inp = ph.ang2vec(random_ptg(vlen))
    out = base.pix2vec(base.vec2pix(inp))
    assert_equal(np.all(ph.v_angle(inp, out) < base.max_pixrad()), True)


def test_vecpixvec_ring(vlen, nside_ring):
    base = ph.Healpix_Base(nside_ring, "RING")
    inp = ph.ang2vec(random_ptg(vlen))
    out = base.pix2vec(base.vec2pix(inp))
    assert_equal(np.all(ph.v_angle(inp, out) < base.max_pixrad()), True)


def test_ringnestring(vlen, nside_nest):
    base = ph.Healpix_Base(nside_nest, "NEST")
    inp = np.random.randint(low=0, high=12*nside_nest*nside_nest-1, size=vlen)
    out = base.ring2nest(base.nest2ring(inp))
    assert_equal(np.all(out == inp), True)


def test_vecangvec(vlen):
    inp = random_ptg(vlen)
    out = ph.vec2ang(ph.ang2vec(inp))
    assert_equal(np.all(np.abs(out-inp) < 1e-14), True)
