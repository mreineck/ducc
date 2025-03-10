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
# Copyright(C) 2025 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras


import ducc0
import numpy as np
import pytest
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize


@pmp("shape", ([43], [654, 23], [32, 3, 11]))
@pmp("dtype_cov", (np.float32, np.float64))
@pmp("cplx", (False, True))
@pmp("broadcast", (False, True))
@pmp("nthreads", (1, 2))
def test_gaussenergy(shape, dtype_cov, cplx, broadcast, nthreads):
    rng = np.random.default_rng(42)

    a = rng.uniform(-.5, .5, shape).astype(dtype_cov)
    b = rng.uniform(-.5, .5, shape).astype(dtype_cov)
    c = rng.uniform(-.5, .5, shape).astype(dtype_cov)
    if cplx:
        a = a + 1j*rng.uniform(-.5, .5, shape).astype(dtype_cov)
        b = b + 1j*rng.uniform(-.5, .5, shape).astype(dtype_cov)
    if broadcast:
        a = np.broadcast_to(a[2:3], b.shape)
    res = ducc0.misc.experimental.LogUnnormalizedGaussProbability(a, b, c, nthreads)
    ref = 0.5*ducc0.misc.vdot((a-b)*c, a-b).real
    rtol = 1e-5 if dtype_cov == np.float32 else 1e-12
    assert_allclose(res, ref, rtol=rtol)

    res, deriv = ducc0.misc.experimental.LogUnnormalizedGaussProbabilityWithDeriv(a, b, c, nthreads=nthreads)
    assert_allclose(res, ref, rtol=rtol)
    assert_allclose(deriv, (a-b)*c, rtol=rtol)
