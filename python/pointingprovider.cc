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
      mav<T,2> &out)
      {
      MR_assert(rot.shape(0)==4, "need 4 entries in quaternion");
      auto rot_ = quaternion_t<T>(rot(0), rot(1), rot(2), rot(3)).normalized();
      MR_assert(out.shape(1)==4, "need 4 entries in quaternion");
      for (size_t i=0; i<out.shape(0); ++i)
        {
        double t = t0+i/freq;
        double fi = (t-t0_)*freq_;
        MR_assert((fi>=0) && fi<quat_.size(), "time outside available range");
        size_t idx = size_t(fi);
        double frac = fi-idx;
        double omega = rangle[idx];
        double xsin = rxsin[idx];
        double w1 = sin((1.-frac)*omega)*xsin,
               w2 = sin(frac*omega)*xsin;
        if (rotflip[idx]) w1=-w1;
        const quaternion_t<T> &q1(quat_[idx]), &q2(quat_[idx+1]);
        quaternion_t<T> q(w1*q1.w + w2*q2.w,
                          w1*q1.x + w2*q2.x,
                          w1*q1.y + w2*q2.y,
                          w1*q1.z + w2*q2.z);
        q *= rot_;
        out.v(i,0) = q.w;
        out.v(i,1) = q.x;
        out.v(i,2) = q.y;
        out.v(i,3) = q.z;
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

    py::array pyget_rotated_quaternions(double t0, double freq,
      const py::array &quat, size_t nval)
      {
      auto res = make_Pyarr<T>({nval,4});
      auto res2 = to_mav<T,2>(res,true);
      auto quat2 = to_mav<T,1>(quat);
      get_rotated_quaternions(t0, freq, quat2, res2);
      return res;
      }
  };

void add_pointingprovider(py::module &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("pointingprovider");

//  m.doc() = totalconvolve_DS;

  using pp_d = PyPointingProvider<double>;
  py::class_<pp_d>(m, "PointingProvider")
    .def(py::init<double, double, const py::array &>(), "t0"_a, "freq"_a, "quat"_a)
    .def ("get_rotated_quaternions", &pp_d::pyget_rotated_quaternions, "t0"_a, "freq"_a, "quat"_a, "nval"_a);
  }

}

using detail_pymodule_pointingprovider::add_pointingprovider;

}
