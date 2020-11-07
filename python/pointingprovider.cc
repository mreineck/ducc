/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/math/quaternion.h"

namespace ducc0 {

namespace detail_pymodule_pointingprovider {

using namespace std;

namespace py = pybind11;

template<typename T> class PointingProvider
  {
  private:
    double t0_, freq_;
    vector<quaternion_t<T>> quat_;
    vector<T> rangle, rxsin;
    vector<bool> rotflip;

  public:
    PointingProvider(double t0, double freq, const mav<T,2> &quat)
      : t0_(t0), freq_(freq), quat_(quat.shape(0)), rangle(quat.shape(0)),
        rxsin(quat.shape(0)), rotflip(quat.shape(0))
      {
      MR_assert(quat.shape(0)>=2, "need at least 2 quaternions");
      MR_assert(quat.shape(1)==4, "need 4 entries in quaternion");
      quat_[0] = quaternion_t<T>(quat(0,0), quat(0,1), quat(0,2), quat(0,3)).normalized();
      for (size_t m=0; m<quat_.size()-1; ++m)
        {
        quat_[m+1] = quaternion_t<T>(quat(m+1,0), quat(m+1,1), quat(m+1,2), quat(m+1,3)).normalized();
        quaternion_t<T> delta(quat_[m+1]*quat_[m].conj());
        rotflip[m]=false;
        if (delta.w < 0.)
          { rotflip[m]=true; delta.flip(); }
        auto [v, omega] = delta.toAxisAngle();
        rangle[m]=omega*.5;
        rxsin[m]=1./sin(rangle[m]);
        }
      }

    void get_rotated_quaternions(double t0, double freq, const mav<T,1> &rot,
      mav<T,2> &out, bool rot_left)
      {
      MR_assert(rot.shape(0)==4, "need 4 entries in quaternion");
      auto rot_ = quaternion_t<T>(rot(0), rot(1), rot(2), rot(3)).normalized();
      MR_assert(out.shape(1)==4, "need 4 entries in quaternion");
      double ofs = (t0-t0_)*freq_;
      for (size_t i=0; i<out.shape(0); ++i)
        {
        double fi = ofs + (i/freq)*freq_;
        MR_assert((fi>=0) && fi<=(quat_.size()-1+1e-7), "time outside available range");
        size_t idx = size_t(fi);
        idx = min(idx, quat_.size()-2);
        double frac = fi-idx;
        double omega = rangle[idx];
        double xsin = rxsin[idx];
        double w1 = sin((1.-frac)*omega)*xsin,
               w2 = sin(frac*omega)*xsin;
        if (rotflip[idx]) w1=-w1;
        const quaternion_t<T> &q1(quat_[idx]), &q2(quat_[idx+1]);
        quaternion_t<T> q(w1*q1.x + w2*q2.x,
                          w1*q1.y + w2*q2.y,
                          w1*q1.z + w2*q2.z,
                          w1*q1.w + w2*q2.w);
        q = rot_left ? rot_*q : q*rot_;
        out.v(i,0) = q.x;
        out.v(i,1) = q.y;
        out.v(i,2) = q.z;
        out.v(i,3) = q.w;
        }
      }
  };

template<typename T> class PyPointingProvider: public PointingProvider<T>
  {
  protected:
    using PointingProvider<T>::get_rotated_quaternions;

  public:
    PyPointingProvider(double t0, double freq, const py::array &quat)
      : PointingProvider<T>(t0, freq, to_mav<T,2>(quat)) {}

    py::array pyget_rotated_quaternions_out(double t0, double freq,
      const py::array &quat, bool rot_left, py::array &out)
      {
      auto res2 = to_mav<T,2>(out,true);
      auto quat2 = to_mav<T,1>(quat);
      get_rotated_quaternions(t0, freq, quat2, res2, rot_left);
      return move(out);
      }
    py::array pyget_rotated_quaternions(double t0, double freq,
      const py::array &quat, size_t nval, bool rot_left)
      {
      auto res = make_Pyarr<T>({nval,4});
      return pyget_rotated_quaternions_out(t0, freq, quat, rot_left, res);
      }
  };

const char *pointingprovider_DS = R"""(
Functionality for converting satellite orientations to detector orientations
at a different frequency
)""";

const char *PointingProvider_init_DS = R"""(
Creates a PointingProvider object from a starting time, a sampling frequency
and a list of orientation quaternions.

Parameters
----------
t0 : float
    the time of the first sample
    This is arbitrary and just provides a reference to the starting times
    of the requested detector orientations.
freq : float
    the frequency at which the provided satellite orientations are sampled
quat : numpy.ndarray((nval, 4), dtype=numpy.float64)
    the satellite orientation quaternions. Components are expecetd in the order
    (x, y, z, w). The quaternions need not be normalized.

Returns
-------
PointongProvider : the constructed object
)""";

const char *get_rotated_quaternions_DS = R"""(
Produces quaternions started at the requested time, sampled at the requested
frequency, which are rotated relative to the satellite orientation according to
a provided quaternion.

Parameters
----------
t0 : float
    the time of the first output sample
    This must use the same reference system as the time passed to the
    constructor.
freq : float
    the frequency at which the output orientations should be sampled
rot : numpy.ndarray((4,), dtype=numpy.float64)
    A single rotation quaternion describing the rotation from the satellite to
    the detector reference system. Components are expecetd in the order
    (x, y, z, w). The quaternion need not be normalized.
nval : int
    the number of requested quaternions
rot_left : bool (optional, default=True)
    if True, the rotation quaternion is multiplied from the left side,
    otherwise from the right.

Returns
-------
numpy.ndarray((nval, 4), dtype=numpy.float64) : the output quaternions
    The quaternions are normalized and in the order (x, y, z, w)
)""";

const char *get_rotated_quaternions2_DS = R"""(
Produces quaternions started at the requested time, sampled at the requested
frequency, which are rotated relative to the satellite orientation according to
a provided quaternion.

Parameters
----------
t0 : float
    the time of the first output sample
    This must use the same reference system as the time passed to the
    constructor.
freq : float
    the frequency at which the output orientations should be sampled
rot : numpy.ndarray((4,), dtype=numpy.float64)
    A single rotation quaternion describing the rotation from the satellite to
    the detector reference system. Components are expecetd in the order
    (x, y, z, w). The quaternion need not be normalized.
rot_left : bool (optional, default=True)
    if True, the rotation quaternion is multiplied from the left side,
    otherwise from the right.
out : numpy.ndarray((nval, 4), dtype=numpy.float64)
    the array to put the computed quaternions into

Returns
-------
numpy.ndarray((nval, 4), dtype=numpy.float64) : the output quaternions
    The quaternions are normalized and in the order (x, y, z, w)
    This is identical to the provided "out" array.
)""";

void add_pointingprovider(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("pointingprovider");
  m.doc() = pointingprovider_DS;

  using pp_d = PyPointingProvider<double>;
  py::class_<pp_d>(m, "PointingProvider")
    .def(py::init<double, double, const py::array &>(),
         PointingProvider_init_DS, "t0"_a, "freq"_a, "quat"_a)
    .def ("get_rotated_quaternions", &pp_d::pyget_rotated_quaternions,
       get_rotated_quaternions_DS,"t0"_a, "freq"_a, "rot"_a, "nval"_a,
       "rot_left"_a=true)
    .def ("get_rotated_quaternions", &pp_d::pyget_rotated_quaternions_out,
       get_rotated_quaternions2_DS,"t0"_a, "freq"_a, "rot"_a,
       "rot_left"_a=true, "out"_a);
  }

}

using detail_pymodule_pointingprovider::add_pointingprovider;

}
