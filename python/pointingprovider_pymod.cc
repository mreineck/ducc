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
 *  Copyright (C) 2020-2024 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/infra/threading.h"
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/math/quaternion.h"

namespace ducc0 {

namespace detail_pymodule_pointingprovider {

using namespace std;

// the next line is necessary to address some sloppy name choices in AdaptiveCpp
using std::min, std::max;

namespace py = pybind11;

template<typename T> class PointingProvider
  {
  private:
    double t0_, freq_;
    size_t nquat;
    vector<quaternion_t<T>> quat_;
    vector<T> rangle, rxsin;
    vector<bool> rotflip;
    size_t nthreads;

  public:
    PointingProvider(double t0, double freq, const cmav<T,2> &quat, size_t nthreads_=1)
      : t0_(t0), freq_(freq), nquat(quat.shape(0)), quat_(nquat+1), rangle(nquat),
        rxsin(nquat), rotflip(nquat), nthreads(nthreads_)
      {
      MR_assert(nquat>=2, "need at least 2 quaternions");
      MR_assert(quat.shape(1)==4, "need 4 entries in quaternion");
      quat_[0] = quaternion_t<T>(quat(0,0), quat(0,1), quat(0,2), quat(0,3)).normalized();
      for (size_t m=0; m<nquat; ++m)
        {
        size_t mp1 = (m+1==nquat) ? 0 : m+1;
        quat_[m+1] = quaternion_t<T>(quat(mp1,0), quat(mp1,1), quat(mp1,2), quat(mp1,3)).normalized();
        quaternion_t<T> delta(quat_[m+1]*quat_[m].conj());
        rotflip[m]=false;
        if (delta.w < 0.)
          { rotflip[m]=true; delta.flip(); }
        auto [v, omega] = delta.toAxisAngle();
        rangle[m]=omega*.5;
        rxsin[m]=1./sin(rangle[m]);
        }
      }

    template<typename T2> void get_rotated_quaternions(double t0, double freq,
      const cmav<T,1> &rot, const vmav<T2,2> &out, bool rot_left)
      {
      double tmp = fmod(t0-t0_, nquat*freq_);
      if (tmp<0) tmp += nquat*freq_;
      t0 = t0_ + tmp;
      MR_assert(rot.shape(0)==4, "need 4 entries in quaternion");
      MR_assert(out.shape(1)==4, "need 4 entries in quaternion");
      double ofs = (t0-t0_)*freq_;
      double fratio = freq_/freq;
      execParallel(out.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        size_t i=lo;
        auto rot_ = quaternion_t<T>(rot(0), rot(1), rot(2), rot(3)).normalized();
// FIXME: SIMD section disabled until it supports the small-omega scenario
#if 0
        using Tsimd = native_simd<T>;
        auto rots_ = quaternion_t<Tsimd>(rot_.x, rot_.y, rot_.z, rot_.w);
        quaternion_t<Tsimd> q1s(0,0,0,0), q2s(0,0,0,0);
        constexpr size_t vlen = Tsimd::size();
        array<size_t,vlen> idx;
        for (; i+vlen-1<hi; i+=vlen)
          {
          Tsimd fi, frac, omega, xsin, w1, w2;
          for (size_t ii = 0; ii<vlen; ++ii)
            {
            fi[ii] = ofs + (i+ii)*fratio;
            MR_assert(fi[ii]>=0, "time before start of available range");
            idx[ii] = size_t(fi[ii]);
            if (idx[ii]>=nquat)
              idx[ii] %= nquat;
            frac[ii] = fi[ii]-floor(fi[ii]);
            omega[ii] = rangle[idx[ii]];
            xsin[ii] = rxsin[idx[ii]];
            }
          w1 = sin((1.-frac)*omega)*xsin;
          w2 = sin(frac*omega)*xsin;
          for (size_t ii=0; ii<vlen; ++ii)
            {
            if (rotflip[idx[ii]]) w1[ii]=-w1[ii];
            q1s.x[ii] = quat_[idx[ii]].x;
            q1s.y[ii] = quat_[idx[ii]].y;
            q1s.z[ii] = quat_[idx[ii]].z;
            q1s.w[ii] = quat_[idx[ii]].w;
            q2s.x[ii] = quat_[idx[ii]+1].x;
            q2s.y[ii] = quat_[idx[ii]+1].y;
            q2s.z[ii] = quat_[idx[ii]+1].z;
            q2s.w[ii] = quat_[idx[ii]+1].w;
            }
          quaternion_t<Tsimd> q(w1*q1s.x + w2*q2s.x,
                                w1*q1s.y + w2*q2s.y,
                                w1*q1s.z + w2*q2s.z,
                                w1*q1s.w + w2*q2s.w);
          q = rot_left ? rots_*q : q*rots_;
          for (size_t ii=0; ii<vlen; ++ii)
            {
            out(i+ii,0) = T2(q.x[ii]);
            out(i+ii,1) = T2(q.y[ii]);
            out(i+ii,2) = T2(q.z[ii]);
            out(i+ii,3) = T2(q.w[ii]);
            }
          }
#endif
        for (; i<hi; ++i)
          {
          double fi = ofs + i*fratio;
          MR_assert(fi>=0, "time before start of available range");
          size_t idx = size_t(fi);
          if (idx>=nquat)
            idx %= nquat;
          double frac = fi-floor(fi);
          double omega = rangle[idx];
          double w1, w2;
          if (abs(omega)>1e-12)
            {
            double xsin = rxsin[idx];
            w1 = sin((1.-frac)*omega)*xsin;
            w2 = sin(frac*omega)*xsin;
            }
          else
            {  // appoximate sin(x)==x
            w1 = 1.-frac;
            w2 = frac;
            }
          if (rotflip[idx]) w1=-w1;
          const quaternion_t<T> &q1(quat_[idx]), &q2(quat_[idx+1]);
          quaternion_t<T> q(w1*q1.x + w2*q2.x,
                            w1*q1.y + w2*q2.y,
                            w1*q1.z + w2*q2.z,
                            w1*q1.w + w2*q2.w);
          q = rot_left ? rot_*q : q*rot_;
          out(i,0) = T2(q.x);
          out(i,1) = T2(q.y);
          out(i,2) = T2(q.z);
          out(i,3) = T2(q.w);
          }
        });
      }
  };

template<typename T> class PyPointingProvider: public PointingProvider<T>
  {
  protected:
    using PointingProvider<T>::get_rotated_quaternions;

  public:
    PyPointingProvider(double t0, double freq, const py::array &quat, size_t nthreads_=1)
      : PointingProvider<T>(t0, freq, to_cmav<T,2>(quat), nthreads_) {}

    template<typename T2> py::array py2get_rotated_quaternions_out(double t0, double freq,
      const py::array &quat, bool rot_left, py::array &out)
      {
      auto res2 = to_vmav<T2,2>(out);
      auto quat2 = to_cmav<T,1>(quat);
      {
      py::gil_scoped_release release;
      get_rotated_quaternions(t0, freq, quat2, res2, rot_left);
      }
      return out;
      }
    py::array pyget_rotated_quaternions_out(double t0, double freq,
      const py::array &quat, bool rot_left, py::array &out)
      {
      if (isPyarr<double>(out))
        return py2get_rotated_quaternions_out<double>(t0, freq, quat, rot_left, out);
      else if (isPyarr<float>(out))
        return py2get_rotated_quaternions_out<float>(t0, freq, quat, rot_left, out);
      MR_fail("type matching failed: 'out' has neither type 'r4' nor 'r8'");
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
    the satellite orientation quaternions. Components are expected in the order
    (x, y, z, w). The quaternions need not be normalized.
nthreads : int
    number of threads to use for the interpolation operations

Returns
-------
PointingProvider : the constructed object

Notes
-----
The supplied quaternion data is assumed to be periodically repeating.

)""";

const char *get_rotated_quaternions_DS = R"""(
Produces quaternions starting at the requested time, sampled at the requested
frequency, which are rotated relative to the satellite orientation according to
a provided quaternion.

Parameters
----------
t0 : float
    the time of the first output sample
    This must use the same reference system as the time passed to the
    constructor.
    It may lie outside the interval passed to the constructor, in which case
    it will be mapped periodically to this interval.
freq : float
    the frequency at which the output orientations should be sampled
rot : numpy.ndarray((4,), dtype=numpy.float64)
    A single rotation quaternion describing the rotation from the satellite to
    the detector reference system. Components are expected in the order
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
Produces quaternions starting at the requested time, sampled at the requested
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
    the detector reference system. Components are expected in the order
    (x, y, z, w). The quaternion need not be normalized.
rot_left : bool (optional, default=True)
    if True, the rotation quaternion is multiplied from the left side,
    otherwise from the right.
out : numpy.ndarray((nval, 4), dtype=numpy.float32 or nump.float64)
    the array to put the computed quaternions into

Returns
-------
numpy.ndarray((nval, 4), same dtype as `out`) : the output quaternions
    The quaternions are normalized and in the order (x, y, z, w)
    This is identical to the provided "out" array.
)""";

void add_pointingprovider(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("pointingprovider");
  m.doc() = pointingprovider_DS;

  using pp_d = PyPointingProvider<double>;
  py::class_<pp_d>(m, "PointingProvider", py::module_local())
    .def(py::init<double, double, const py::array &, size_t>(),
         PointingProvider_init_DS, "t0"_a, "freq"_a, "quat"_a, "nthreads"_a=1)
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
