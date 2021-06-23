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


import numpy as np
import pytest
from numpy.testing import assert_
import ducc0.sht

pmp = pytest.mark.parametrize


def _assert_close(a, b, epsilon):
    err = ducc0.misc.l2error(a, b)
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_(err < epsilon)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(rng, lmax, mmax, ncomp):
    res = rng.uniform(-1., 1., (nalm(lmax, mmax), ncomp)) \
     + 1j*rng.uniform(-1., 1., (nalm(lmax, mmax), ncomp))
    # make a_lm with m==0 real-valued
    res[0:lmax+1, :].imag = 0.
    return res


@pmp("lmax", tuple(range(70)))
@pmp("nthreads", (0,1,2))
def test_rotation(lmax, nthreads):
    rng = np.random.default_rng(42)
    phi, theta, psi = rng.uniform(-2*np.pi, 2*np.pi, (3,))

    alm = random_alm(rng, lmax, lmax, 1)[:,0]
    alm2 = ducc0.sht.rotate_alm(alm, lmax, phi, theta, psi, nthreads)
    alm2 = ducc0.sht.rotate_alm(alm2, lmax, -psi, -theta, -phi, nthreads)
    _assert_close(alm, alm2, 1e-12)
    alm = alm.astype(np.complex64)
    alm2 = ducc0.sht.rotate_alm(alm, lmax, phi, theta, psi, nthreads)
    alm2 = ducc0.sht.rotate_alm(alm2, lmax, -psi, -theta, -phi, nthreads)
    _assert_close(alm, alm2, 1e-6)
