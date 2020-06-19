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
# Copyright(C) 2020 Max-Planck-Society


import ducc0.pointingprovider as pp
# import pyfftw
import numpy as np
import pytest
from numpy.testing import assert_


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def _assert_close(a, b, epsilon):
    err = _l2error(a, b)
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_(err < epsilon)


pmp = pytest.mark.parametrize

@pmp("size", (10, 37, 1000))
@pmp("t0", (-45.3, 0, 10))
@pmp("freq", (1, 1.3e-7, 3e10))
def testp1(size, t0, freq):
    rng = np.random.default_rng(42)
    quat = rng.uniform(-.5, .5, (size,4))
    prov = pp.PointingProvider(t0, freq, quat)
    rquat = np.array([1., 0., 0., 0.])  # a non-rotating quaternion
    quat2 = prov.get_rotated_quaternions(t0, freq, rquat, size)
    nquat = quat/np.sqrt(np.sum(quat**2,axis=1,keepdims=True))
    # adjust signs
    nquat = nquat * np.sign(nquat[:,0]).reshape(-1,1)
    quat2 = quat2 * np.sign(quat2[:,0]).reshape(-1,1)
    _assert_close(quat2, nquat, 1e-13)
