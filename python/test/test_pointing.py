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


import ducc0
import ducc0.pointingprovider as pp
# import pyfftw
import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose

ducc0.misc.resize_thread_pool(2)

pmp = pytest.mark.parametrize


@pmp("size", (2, 10, 37, 1000))
@pmp("t0", (-45.3, 0, 10))
@pmp("freq", (1, 1.3e-7, 3e10))
@pmp("nthreads", (1, 2))
def testp1(size, t0, freq, nthreads):
    rng = np.random.default_rng(42)
    quat = rng.uniform(-.5, .5, (size, 4))
    prov = pp.PointingProvider(t0, freq, quat, nthreads)
    rquat = np.array([0., 0., 0., 1.])  # a non-rotating quaternion
    quat2 = prov.get_rotated_quaternions(t0, freq, rquat, size)
    quat3 = prov.get_rotated_quaternions(t0, freq, rquat, size, rot_left=False)
    nquat = quat/np.sqrt(np.sum(quat**2, axis=1, keepdims=True))
    # adjust signs
    nquat = nquat * np.sign(nquat[:, 0]).reshape(-1, 1)
    quat2 = quat2 * np.sign(quat2[:, 0]).reshape(-1, 1)
    quat3 = quat3 * np.sign(quat3[:, 0]).reshape(-1, 1)
    assert_allclose(ducc0.misc.l2error(quat2, nquat), 0, atol=1e-13)
    assert_allclose(ducc0.misc.l2error(quat3, nquat), 0, atol=1e-13)


def testp2():
    rng = np.random.default_rng(42)
    t01, f1, size1 = 0., 1., 200
    quat1 = rng.uniform(-.5, .5, (size1, 4))
    prov = pp.PointingProvider(t01, f1, quat1)
    rquat = rng.uniform(-.5, .5, (4,))
    t02, f2, size2 = 3.7, 10.2, 300
    quat2 = prov.get_rotated_quaternions(t02, f2, rquat, size2)
    quat3 = np.empty((size2, 4), dtype=np.float64)
    quat3 = prov.get_rotated_quaternions(t02, f2, rquat, out=quat3)
    quat4 = np.empty((size2, 4), dtype=np.float64)
    quat4 = prov.get_rotated_quaternions(t02, f2, rquat, rot_left=False,
                                         out=quat4)
    assert_((quat2 == quat3).all(), "problem")
    quat2 *= np.sign(quat2[:, 0]).reshape((-1, 1))
    quat4 *= np.sign(quat4[:, 0]).reshape((-1, 1))

    try:
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
    except:
        pytest.skip()
    times1 = t01 + 1./f1*np.arange(size1)
    r1 = R.from_quat(quat1)
    rrquat = R.from_quat(rquat)
    slerp = Slerp(times1, r1)
    times2 = t02 + 1./f2*np.arange(size2)
    r2 = rrquat*slerp(times2)
    squat2 = r2.as_quat()
    squat2 *= np.sign(squat2[:, 0]).reshape((-1, 1))
    assert_allclose(ducc0.misc.l2error(quat2, squat2), 0, atol=1e-13)
    r3 = slerp(times2)*rrquat
    squat3 = r3.as_quat()
    squat3 *= np.sign(squat3[:, 0]).reshape((-1, 1))
    assert_allclose(ducc0.misc.l2error(quat4, squat3), 0, atol=1e-13)


def testp3():  # test periodicity
    rng = np.random.default_rng(42)
    t01, f1, size1 = 0., 1., 20
    quat1 = rng.uniform(-.5, .5, (size1, 4))
    prov = pp.PointingProvider(t01, f1, quat1)
    rquat = rng.uniform(-.5, .5, (4,))
    t02, f2, size2 = 3.7, 10.2, 30000
    quat2 = prov.get_rotated_quaternions(t02, f2, rquat, size2)
    t03 = t02 + 2*size1/f1
    quat3 = prov.get_rotated_quaternions(t03, f2, rquat, size2)
    assert_allclose(ducc0.misc.l2error(quat2, quat3), 0., atol=1e-13)
    t04 = t02 - 5*size1/f1
    quat4 = prov.get_rotated_quaternions(t04, f2, rquat, size2)
    assert_allclose(ducc0.misc.l2error(quat2, quat4), 0., atol=1e-13)


def test_quat_euler():
    nsamp = 10000

    euler = np.empty((nsamp,3))

    # theta
    euler[:,0] = np.random.uniform(low=-3*np.pi, high=5*np.pi, size=nsamp)
    # phi
    euler[:,1] = np.random.uniform(low=-5*np.pi, high=8*np.pi, size=nsamp)
    # psi
    euler[:,2] = np.random.uniform(low=-5*np.pi, high=8*np.pi, size=nsamp)

    euler2 = ducc0.misc.quat2ptg(ducc0.misc.ptg2quat(euler))

    # Normalize angle ranges
    euler = (euler + 8*np.pi)%(2*np.pi)
    mask = euler[:,0]>np.pi
    euler[:,0] = np.where(mask,2*np.pi-euler[:,0],euler[:,0])
    euler[:,1] = np.where(mask,(euler[:,1]+np.pi)%(2*np.pi),euler[:,1])
    euler[:,2] = np.where(mask,(euler[:,2]+np.pi)%(2*np.pi),euler[:,2])

    euler2[:,1:3] = (euler2[:,1:3] + 8*np.pi)%(2*np.pi)

    assert_allclose(ducc0.misc.l2error(euler, euler2), 0, atol=1e-13)
