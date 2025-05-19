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
from ducc0.misc import special_add_at
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


@pmp("a_shape, axis, index, b, expected",
     [
        # Repeated index: accumulate at same position
        ((3,), 0, np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]), np.array([0.0, 6.0, 0.0])),
        # Index in reversed order
        ((3,), 0, np.array([2, 1, 0]), np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])),
        # All zeros in index (accumulate to one bin)
        ((3,), 0, np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]), np.array([6.0, 0.0, 0.0])),
        # All same index in 2D along axis 0
        ((3, 2), 0, np.array([1, 1, 1]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[0, 0], [9, 12], [0, 0]])),
        # Mixed and repeated index in 2D
        ((4, 2), 0, np.array([1, 0, 1]), np.array([[10, 20], [30, 40], [50, 60]]), np.array([[30, 40], [60, 80], [0, 0], [0, 0]])),
        # Index skipping bins
        ((5,), 0, np.array([0, 2, 4]), np.array([5.0, 10.0, 15.0]), np.array([5.0, 0.0, 10.0, 0.0, 15.0])),
        # Broadcasting pattern in axis 1
        ((2, 4), 1, np.array([1, 1, 2]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0, 3, 3, 0], [0, 9, 6, 0]])),
        # Using every bin more than once (with overlaps)
        ((4,), 0, np.array([1, 2, 1, 3]), np.array([2.0, 4.0, 6.0, 8.0]), np.array([0.0, 8.0, 4.0, 8.0])),
     ]
)
@pmp("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_special_add_at_creative(a_shape, axis, index, b, expected, dtype):
    b, expected = b.astype(dtype), expected.astype(dtype)
    a = np.zeros(a_shape, dtype=b.dtype)
    out = special_add_at(a, axis=axis, index=index, b=b)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_special_add_at_complex(dtype):
    a = np.zeros((3,), dtype=dtype)
    b = np.array([1+2j, 3+4j, 5+6j], dtype=dtype)

    index = np.array([0, 1, 2], dtype=np.int32)
    expected = np.array([1+2j, 3+4j, 5+6j], dtype=dtype)
    out = special_add_at(a.copy(), axis=0, index=index, b=b)
    np.testing.assert_array_almost_equal(out, expected)

    index = np.array([0, 1, 1], dtype=np.int32)
    expected = np.array([1+2j, 3+4j + 5+6j, 0], dtype=dtype)
    out = special_add_at(a.copy(), axis=0, index=index, b=b)
    np.testing.assert_array_almost_equal(out, expected)
