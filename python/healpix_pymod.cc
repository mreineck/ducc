/*
 *  This file is part of pyHealpix.
 *
 *  pyHealpix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  pyHealpix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with pyHealpix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.sourceforge.net
 */

/*
 *  pyHealpix is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2017-2022 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <string>

#include "ducc0/healpix/healpix_base.h"
#include "ducc0/math/constants.h"
#include "ducc0/infra/string_utils.h"
#include "ducc0/math/geom_utils.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_healpix {

using namespace std;

namespace py = pybind11;

using shape_t = fmav_info::shape_t;

template<size_t nd1, size_t nd2> shape_t repl_dim(const shape_t &s,
  const array<size_t,nd1> &si, const array<size_t,nd2> &so)
  {
  if constexpr (nd1>0)
    {
    MR_assert(s.size()>=nd1,"too few input array dimensions");
    for (size_t i=0; i<nd1; ++i)
      MR_assert(si[i]==s[s.size()-nd1+i], "input dimension mismatch");
    }
  shape_t snew(s.size()-nd1+nd2);
  for (size_t i=0; i<s.size()-nd1; ++i)
    snew[i]=s[i];
  if constexpr (nd2>0)
    for (size_t i=0; i<nd2; ++i)
      snew[i+s.size()-nd1] = so[i];
  return snew;
  }

template<typename T1, typename T2, size_t nd1, size_t nd2>
  py::array myprep(const py::array_t<T1> &ain, const array<size_t,nd1> &a1,
  const array<size_t,nd2> &a2)
  {
  auto in = to_cfmav<T1>(ain);
  auto oshp = repl_dim(in.shape(), a1, a2);
  return make_Pyarr<T2>(oshp);
  }

#define DUCC0_DISPATCH(Ti1, Ti2, To1, To2, Tni1, Tni2, arr, func, args) \
  { \
  if (isPyarr<Ti1>(arr)) return func<To1> args; \
  if (isPyarr<Ti2>(arr)) return func<To2> args; \
  MR_fail("type matching failed: '" #arr "' has neither type '" Tni1 \
          "' nor '" Tni2 "'"); \
  }

class Pyhpbase
  {
  public:
    Healpix_Base2 base;

    Pyhpbase (int64_t nside, const string &scheme)
      : base (nside, RING, SET_NSIDE)
      {
      MR_assert((scheme=="RING")||(scheme=="NEST")||(scheme=="NESTED"),
        "unknown ordering scheme");
      if ((scheme=="NEST")||(scheme=="NESTED"))
        base.SetNside(nside,NEST);
      }
    string repr() const
      {
      return "<Healpix Base: Nside=" + dataToString(base.Nside()) +
        ", Scheme=" + ((base.Scheme()==RING) ? "RING" : "NEST") +".>";
      }

    template<typename Tin> py::array pix2ang2 (const py::array &in,
      size_t nthreads) const
      {
      const auto pix = to_cfmav<Tin>(in);
      auto out = myprep<Tin, double, 0, 1>(in, {}, {2});
      auto ang = to_vfmav<double>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<0,1>([&](const auto &in, const auto &out)
        {
        pointing ptg = base.pix2ang(in());
        out(0) = ptg.theta;
        out(1) = ptg.phi;
        }, nthreads, pix, ang);
      }
      return out;
      }
    py::array pix2ang (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        pix2ang2, (in, nthreads))

    template<typename Tin> py::array ang2pix2 (const py::array &in,
      size_t nthreads) const
      {
      const auto ang = to_cfmav<Tin>(in);
      auto out = myprep<Tin, int64_t, 1, 0>(in, {2}, {});
      auto pix = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<1,0>([&](const auto &in, const auto &out)
        {
        out()=base.ang2pix(pointing(in(0),in(1)));
        }, nthreads, ang, pix);
      }
      return out;
      }
    py::array ang2pix (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(double, float, double, float, "f8", "f4", in, ang2pix2,
        (in, nthreads))
    template<typename Tin> py::array pix2vec2 (const py::array &in,
      size_t nthreads) const
      {
      const auto pix = to_cfmav<Tin>(in);
      auto out = myprep<Tin, double, 0, 1>(in, {}, {3});
      auto vec = to_vfmav<double>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<0,1>([&](const auto &in, const auto &out)
        {
        auto vec = base.pix2vec(in());
        out(0)=vec.x; out(1)=vec.y; out(2)=vec.z;
        }, nthreads, pix, vec);
      }
      return out;
      }
    py::array pix2vec (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        pix2vec2, (in, nthreads))
    template<typename Tin> py::array vec2pix2 (const py::array &in,
      size_t nthreads) const
      {
      const auto vec = to_cfmav<Tin>(in);
      auto out = myprep<Tin, int64_t, 1, 0>(in, {3}, {});
      auto pix = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<1,0>([&](const auto &in, const auto &out)
        {
        out()=base.vec2pix(vec3(in(0), in(1), in(2)));
        }, nthreads, vec, pix);
      }
      return out;
      }
    py::array vec2pix (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(double, float, double, float, "f8", "f4", in, vec2pix2,
        (in, nthreads))
    template<typename Tin> py::array pix2xyf2 (const py::array &in,
      size_t nthreads) const
      {
      const auto pix = to_cfmav<Tin>(in);
      auto out = myprep<Tin, int64_t, 0, 1>(in, {}, {3});
      auto xyf = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<0,1>([&](const auto &in, const auto &out)
        {
        int x_,y_,f_;
        base.pix2xyf(in(),x_,y_,f_);
        out(0)=x_; out(1)=y_; out(2)=f_;
        }, nthreads, pix, xyf);
      }
      return out;
      }
    py::array pix2xyf (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        pix2xyf2, (in, nthreads))
    template<typename Tin> py::array xyf2pix2 (const py::array &in,
      size_t nthreads) const
      {
      const auto xyf = to_cfmav<Tin>(in);
      auto out = myprep<Tin, int64_t, 1, 0>(in, {3}, {});
      auto pix = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<1,0>([&](const auto &in, const auto &out)
        {
        out()=base.xyf2pix(in(0), in(1), in(2));
        }, nthreads, xyf, pix);
      }
      return out;
      }
    py::array xyf2pix (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        xyf2pix2, (in, nthreads))
    template<typename Tin> py::array neighbors2 (const py::array &in,
      size_t nthreads) const
      {
      const auto pix = to_cfmav<Tin>(in);
      auto out = myprep<Tin, int64_t, 0, 1>(in, {}, {8});
      auto neigh = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<0,1>([&](const auto &in, const auto &out)
        {
        array<int64_t,8> res;
        base.neighbors(in(),res);
        for (size_t j=0; j<8; ++j) out(j)=res[j];
        }, nthreads, pix, neigh);
      }
      return out;
      }
    py::array neighbors (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        neighbors2, (in, nthreads))
    template<typename Tin> py::array ring2nest2 (const py::array &in,
      size_t nthreads) const
      {
      const auto ring = to_cfmav<Tin>(in);
      auto out = make_Pyarr<int64_t>(ring.shape());
      auto nest = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<0,0>([&](const auto &in, const auto &out)
        { out() = base.ring2nest(in()); }, nthreads, ring, nest);
      }
      return out;
      }
    py::array ring2nest (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        ring2nest2, (in, nthreads))
    template<typename Tin> py::array nest2ring2 (const py::array &in,
      size_t nthreads) const
      {
      const auto nest = to_cfmav<Tin>(in);
      auto out = make_Pyarr<int64_t>(nest.shape());
      auto ring = to_vfmav<int64_t>(out);
      {
      py::gil_scoped_release release;
      flexible_mav_apply<0,0>([&](const auto &in, const auto &out)
        { out() = base.nest2ring(in()); }, nthreads, nest,ring);
      }
      return out;
      }
    py::array nest2ring (const py::array &in, size_t nthreads) const
      DUCC0_DISPATCH(int64_t, int32_t, int64_t, int32_t, "i8", "i4", in,
        nest2ring2, (in, nthreads))
    template<typename Tin> py::array query_disc2(const py::array &ptg,
      double radius) const
      {
      MR_assert((ptg.ndim()==1)&&(ptg.shape(0)==2),
        "ptg must be a 1D array with 2 values");
      rangeset<int64_t> pixset;
      auto ptg2 = to_cmav<Tin,1>(ptg);
      {
      py::gil_scoped_release release;
      base.query_disc(pointing(ptg2(0),ptg2(1)), radius, pixset);
      }
      auto res = make_Pyarr<int64_t>(shape_t({pixset.nranges(),2}));
      auto oref=res.mutable_unchecked<2>();
      for (size_t i=0; i<pixset.nranges(); ++i)
        {
        oref(i,0)=pixset.ivbegin(i);
        oref(i,1)=pixset.ivend(i);
        }
      return res;
      }
    py::array query_disc(const py::array &ptg, double radius) const
      DUCC0_DISPATCH(double, float, double, float, "f8", "f4", ptg,
        query_disc2, (ptg, radius))
    py::dict sht_info() const
      {
      MR_assert(base.Scheme()==RING, "RING scheme required for SHTs");
      auto nside = base.Nside();
      auto nrings = size_t(4*nside-1);
      auto theta_= make_Pyarr<double>(shape_t({nrings}));
      auto theta = to_vmav<double,1>(theta_);
      auto phi0_ = make_Pyarr<double>(shape_t({nrings}));
      auto phi0 = to_vmav<double,1>(phi0_);
      auto nphi_ = make_Pyarr<size_t>(shape_t({nrings}));
      auto nphi = to_vmav<size_t,1>(nphi_);
      auto ringstart_ = make_Pyarr<size_t>(shape_t({nrings}));
      auto ringstart = to_vmav<size_t,1>(ringstart_);
      for (size_t r=0, rs=nrings-1; r<=rs; ++r, --rs)
        {
        int64_t startpix, ringpix;
        double ringtheta;
        bool shifted;
        base.get_ring_info2 (r+1, startpix, ringpix, ringtheta, shifted);
        theta(r) = ringtheta;
        theta(rs) = pi-ringtheta;
        nphi(r) = nphi(rs) = size_t(ringpix);
        phi0(r) = phi0(rs) = shifted ? (pi/ringpix) : 0.;
        ringstart(r) = size_t(startpix);
        ringstart(rs) = size_t(base.Npix() - startpix - ringpix);
        }
      py::dict res;
      res["theta"] = theta_;
      res["phi0"] = phi0_;
      res["nphi"] = nphi_;
      res["ringstart"] = ringstart_;
      return res;
      }
  };

template<typename Tin> py::array ang2vec2 (const py::array &in, size_t nthreads)
  {
  auto ang = to_cfmav<Tin>(in);
  auto out = myprep<Tin, double, 1, 1>(in, {2}, {3});
  auto vec = to_vfmav<double>(out);
  {
  py::gil_scoped_release release;
  flexible_mav_apply<1,1>([&](const auto &in, const auto &out)
    {
    vec3 v (pointing(in(0),in(1)));
    out(0)=v.x; out(1)=v.y; out(2)=v.z;
    }, nthreads, ang, vec);
  }
  return out;
  }
py::array ang2vec (const py::array &in, size_t nthreads)
  DUCC0_DISPATCH(double, float, double, float, "f8", "f4", in, ang2vec2,
    (in, nthreads))
template<typename Tin> py::array vec2ang2 (const py::array &in, size_t nthreads)
  {
  auto vec = to_cfmav<Tin>(in);
  auto out = myprep<Tin, double, 1, 1>(in, {3}, {2});
  auto ang = to_vfmav<double>(out);
  {
  py::gil_scoped_release release;
  flexible_mav_apply<1,1>([&](const auto &in, const auto &out)
    {
    pointing ptg (vec3(in(0),in(1),in(2)));
    out(0)=ptg.theta; out(1)=ptg.phi;
    }, nthreads, vec, ang);
  }
  return out;
  }
py::array vec2ang (const py::array &in, size_t nthreads)
  DUCC0_DISPATCH(double, float, double, float, "f8", "f4", in, vec2ang2,
    (in, nthreads))
template<typename Ti1, typename Ti2> py::array local_v_angle2
  (const py::array &in1, const py::array &in2, size_t nthreads)
  {
  auto vec1 = to_cfmav<Ti1>(in1);
  auto vec2 = to_cfmav<Ti2>(in2);
  auto out = myprep<Ti1, double, 1, 0>(in1, {3}, {});
  auto angle = to_vfmav<double>(out);
  {
  py::gil_scoped_release release;
  flexible_mav_apply<1,1,0>([&](const auto &in0, const auto &in1,
    const auto &out)
    {
    out()=v_angle(vec3(in0(0),in0(1),in0(2)),
                  vec3(in1(0),in1(1),in1(2)));
    }, nthreads, vec1, vec2, angle);
  }
  return out;
  }
py::array local_v_angle (const py::array &in1, const py::array &in2,
  size_t nthreads)
  {
  if (isPyarr<double>(in1) && isPyarr<double>(in2))
    return local_v_angle2<double, double> (in1, in2, nthreads);
  if (isPyarr<double>(in1) && isPyarr<float>(in2))
    return local_v_angle2<double, float> (in1, in2, nthreads);
  if (isPyarr<float>(in1) && isPyarr<float>(in2))
    return local_v_angle2<float, float> (in1, in2, nthreads);
  if (isPyarr<float>(in1) && isPyarr<double>(in2))
    return local_v_angle2<double, float> (in2, in1, nthreads);
  MR_fail("type matching failed: input arrays have neither type 'f8' nor 'f4'");
  }

#undef DUCC0_DISPATCH

constexpr const char *healpix_DS = R"""(
Python interface for some of the HEALPix C++ functionality

All angles are interpreted as radians.
The theta coordinate is measured as co-latitude, ranging from 0 (North Pole)
to pi (South Pole).

All 3-vectors returned by the functions are normalized.
However, 3-vectors provided as input to the functions need not be normalized.

Floating point input arrays can be provided as numpy.float64 or numpy.float32,
but the returned floating point arrays will always be of type numpy.float64.

Integer input arrays can be provided as numpy.int64 or numpy.int32,
but the returned integer arrays will always be of type numpy.int64.

Error conditions are reported by raising exceptions.
)""";

constexpr const char *Healpix_Base_DS = R"""(
Functionality related to the HEALPix pixelization
)""";

constexpr const char *Healpix_Base_init_DS = R"""(
Healpix_Base constructor

Parameters
----------
nside: int
    Nside parameter of the pixelization
scheme: str
    Must be "RING", "NEST", or "NESTED"
)""";

constexpr const char *order_DS = R"""(
Returns the ORDER parameter of the pixelisation.
If Nside is a power of 2, this is log_2(Nside), otherwise it is -1.
)""";

constexpr const char *nside_DS = R"""(
Returns the Nside parameter of the pixelisation.
)""";

constexpr const char *npix_DS = R"""(
Returns the total number of pixels of the pixelisation.
)""";

constexpr const char *scheme_DS = R"""(
Returns a string representation of the pixelisation's ordering scheme
("RING" or "NEST").
)""";

constexpr const char *pix_area_DS = R"""(
Returns the area (in steradian) of a single pixel.
)""";

constexpr const char *max_pixrad_DS = R"""(
Returns the maximum angular distance (in radian) between a pixel center
and its corners for this pixelisation.
)""";

constexpr const char *pix2ang_DS = R"""(
Returns a (co-latitude, longitude) tuple for each value in pix.
The result array has the same shape as pix, with an added last dimension
of size 2.
)""";

constexpr const char *ang2pix_DS = R"""(
Returns the index of the containing pixel for every (co-latitude, longitude)
tuple in ang. ang must have a last dimension of size 2; the result array
has the same shape as ang, except that ang's last dimension is removed.
)""";

constexpr const char *pix2vec_DS = R"""(
Returns a normalized 3-vector for each value in pix.
The result array has the same shape as pix, with an added last dimension
of size 3.
)""";

constexpr const char *vec2pix_DS = R"""(
Returns the index of the containing pixel for every 3-vector in vec.
vec must have a last dimension of size 3; the result array has the same shape as
vec, except that vec's last dimension is removed.
)""";

constexpr const char *ring2nest_DS = R"""(
Returns the pixel index in NEST scheme for every entry of ring.
The result array has the same shape as ring.
)""";

constexpr const char *nest2ring_DS = R"""(
Returns the pixel index in RING scheme for every entry of nest.
The result array has the same shape as nest.
)""";

constexpr const char *query_disc_DS = R"""(
Returns a range set of all pixels whose centers fall within "radius" of "ptg".
"ptg" must be a single (co-latitude, longitude) tuple. The result is a 2D array
with last dimension 2; the pixels lying inside the disc are
[res[0,0] .. res[0,1]); [res[1,0] .. res[1,1]) etc.
)""";

constexpr const char *sht_info_DS = R"""(
Returns a dictionary containing information necessary for spherical harmonic
transforms on a HEALPix grid of the given nside parameter.

The dictionary keys are chosen in such a way that the dictionary can be used
directly as kwargs in calls to `ducc0.sht.synthesis` and similar.

Returns
-------
theta: numpy.ndarray(numpy.float64, shape=(nrings,))
    the colatitudes of all rings
nphi: numpy.ndarray(numpy.uint64, shape=(nrings,))
    the number of pixels for every ring
phi0: numpy.ndarray(numpy.float64, shape=(nrings,))
    the azimuth of the first pixel in every ring
ringstart: numpy.ndarray(numpy.uint64, shape=(nrings,))
    the index of the first pixel in every ring in a typical HEALPix map array.
)""";

constexpr const char *ang2vec_DS = R"""(
Returns a normalized 3-vector for every (co-latitude, longitude)
tuple in ang. ang must have a last dimension of size 2; the result array
has the same shape as ang, except that its last dimension is 3 instead of 2.
)""";

constexpr const char *vec2ang_DS = R"""(
Returns a (co-latitude, longitude) tuple for every 3-vector in vec.
vec must have a last dimension of size 3; the result array has the same shape as
vec, except that its last dimension is 2 instead of 3.
)""";

constexpr const char *v_angle_DS = R"""(
Returns the angles between the 3-vectors in v1 and v2. The input arrays must
have identical shapes. The result array has the same shape as v1 or v2, except
that their last dimension is removed.
The employed algorithm is highly accurate, even for angles close to 0 or pi.
)""";

void add_healpix(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("healpix");
  m.doc() = healpix_DS;

  py::class_<Pyhpbase> (m, "Healpix_Base", py::module_local(), Healpix_Base_DS)
    .def(py::init<int,const string &>(), Healpix_Base_init_DS, "nside"_a,"scheme"_a)
    .def("order", [](Pyhpbase &self)
      { return self.base.Order(); }, order_DS)
    .def("nside", [](Pyhpbase &self)
      { return self.base.Nside(); }, nside_DS)
    .def("npix", [](Pyhpbase &self)
      { return self.base.Npix(); }, npix_DS)
    .def("scheme", [](Pyhpbase &self)
      { return self.base.Scheme(); }, scheme_DS)
    .def("pix_area", [](Pyhpbase &self)
      { return 4*pi/self.base.Npix(); }, pix_area_DS)
    .def("max_pixrad", [](Pyhpbase &self)
      { return self.base.max_pixrad(); }, max_pixrad_DS)
    .def("pix2ang", &Pyhpbase::pix2ang, pix2ang_DS, "pix"_a, "nthreads"_a=1)
    .def("ang2pix", &Pyhpbase::ang2pix, ang2pix_DS, "ang"_a, "nthreads"_a=1)
    .def("pix2vec", &Pyhpbase::pix2vec, pix2vec_DS, "pix"_a, "nthreads"_a=1)
    .def("vec2pix", &Pyhpbase::vec2pix, vec2pix_DS, "vec"_a, "nthreads"_a=1)
    .def("pix2xyf", &Pyhpbase::pix2xyf, "pix"_a, "nthreads"_a=1)
    .def("xyf2pix", &Pyhpbase::xyf2pix, "xyf"_a, "nthreads"_a=1)
    .def("neighbors", &Pyhpbase::neighbors,"pix"_a, "nthreads"_a=1)
    .def("ring2nest", &Pyhpbase::ring2nest, ring2nest_DS, "ring"_a, "nthreads"_a=1)
    .def("nest2ring", &Pyhpbase::nest2ring, nest2ring_DS, "nest"_a, "nthreads"_a=1)
    .def("query_disc", &Pyhpbase::query_disc, query_disc_DS, "ptg"_a, "radius"_a)
    .def("sht_info", &Pyhpbase::sht_info, sht_info_DS)
    .def("__repr__", &Pyhpbase::repr)
    ;

  m.def("ang2vec",&ang2vec, ang2vec_DS, "ang"_a, "nthreads"_a=1);
  m.def("vec2ang",&vec2ang, vec2ang_DS, "vec"_a, "nthreads"_a=1);
  m.def("v_angle",&local_v_angle, v_angle_DS, "v1"_a, "v2"_a, "nthreads"_a=1);
  }

}

using detail_pymodule_healpix::add_healpix;

}
