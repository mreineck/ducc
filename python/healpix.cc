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
 *  Copyright (C) 2017-2020 Max-Planck-Society
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
  MR_assert(s.size()+nd1,"too few input array dimensions");
  if constexpr (nd1>0)
    for (size_t i=0; i<nd1; ++i)
      MR_assert(si[i]==s[s.size()-nd1+i], "input dimension mismatch");
  shape_t snew(s.size()-nd1+nd2);
  for (size_t i=0; i<s.size()-nd1; ++i)
    snew[i]=s[i];
  if constexpr (nd2>0)
    for (size_t i=0; i<nd2; ++i)
      snew[i+s.size()-nd1] = so[i];
  return snew;
  }

template<typename T1, typename T2, size_t nd1, size_t nd2, typename Func>
  py::array doStuff(const py::array &ain, const array<size_t,nd1> &a1, const array<size_t,nd2> &a2, Func func)
  {
  auto in = to_fmav<T1>(ain);
  auto oshp = repl_dim(in.shape(), a1, a2);
  auto aout = make_Pyarr<T2>(oshp);
  auto out = to_fmav<T2>(aout,true);
  MavIter<T1,nd1+1> iin(in);
  MavIter<T2,nd2+1> iout(out);
  while (!iin.done())
    {
    func(iin, iout);
    iin.inc();iout.inc();
    }
  return move(aout);
  }

class Pyhpbase
  {
  public:
    Healpix_Base2 base;

    Pyhpbase (int64_t nside, const string &scheme)
      : base (nside, RING, SET_NSIDE)
      {
      MR_assert((scheme=="RING")||(scheme=="NEST"),
        "unknown ordering scheme");
      if (scheme=="NEST")
        base.SetNside(nside,NEST);
      }
    string repr() const
      {
      return "<Healpix Base: Nside=" + dataToString(base.Nside()) +
        ", Scheme=" + ((base.Scheme()==RING) ? "RING" : "NEST") +".>";
      }

    py::array pix2ang (const py::array &pix) const
      {
      return doStuff<int64_t, double, 0, 1>(pix, {}, {2}, [this](const MavIter<int64_t,1> &iin, MavIter<double,2> &iout)
        {
        for (size_t i=0; i<iin.shape(0); ++i)
          {
          pointing ptg=base.pix2ang(iin(i));
          iout.v(i,0) = ptg.theta; iout.v(i,1) = ptg.phi;
          }
        });
      }
    py::array ang2pix (const py::array &ang) const
      {
      return doStuff<double, int64_t, 1, 0>(ang, {2}, {}, [this](const MavIter<double,2> &iin, MavIter<int64_t,1> &iout)
        {
        for (size_t i=0; i<iout.shape(0); ++i)
          iout.v(i)=base.ang2pix(pointing(iin(i,0),iin(i,1)));
        });
      }
    py::array pix2vec (const py::array &pix) const
      {
      return doStuff<int64_t, double, 0, 1>(pix, {}, {3}, [this](const MavIter<int64_t,1> &iin, MavIter<double,2> &iout)
        {
        for (size_t i=0; i<iin.shape(0); ++i)
          {
          vec3 v=base.pix2vec(iin(i));
          iout.v(i,0)=v.x; iout.v(i,1)=v.y; iout.v(i,2)=v.z;
          }
        });
      }
    py::array vec2pix (const py::array &vec) const
      {
      return doStuff<double, int64_t, 1, 0>(vec, {3}, {}, [this](const MavIter<double,2> &iin, MavIter<int64_t,1> &iout)
        {
        for (size_t i=0; i<iout.shape(0); ++i)
          iout.v(i)=base.vec2pix(vec3(iin(i,0),iin(i,1),iin(i,2)));
        });
      }
    py::array pix2xyf (const py::array &pix) const
      {
      return doStuff<int64_t, int64_t, 0, 1>(pix, {}, {3}, [this](const MavIter<int64_t,1> &iin, MavIter<int64_t,2> &iout)
        {
        for (size_t i=0; i<iin.shape(0); ++i)
          {
          int x,y,f;
          base.pix2xyf(iin(i),x,y,f);
          iout.v(i,0)=x; iout.v(i,1)=y; iout.v(i,2)=f;
          }
        });
      }
    py::array xyf2pix (const py::array &xyf) const
      {
      return doStuff<int64_t, int64_t, 1, 0>(xyf, {3}, {}, [this](const MavIter<int64_t,2> &iin, MavIter<int64_t,1> &iout)
        {
        for (size_t i=0; i<iout.shape(0); ++i)
          iout.v(i)=base.xyf2pix(iin(i,0),iin(i,1),iin(i,2));
        });
      }
    py::array neighbors (const py::array &pix) const
      {
      return doStuff<int64_t, int64_t, 0, 1>(pix, {}, {8}, [this](const MavIter<int64_t,1> &iin, MavIter<int64_t,2> &iout)
        {
        for (size_t i=0; i<iin.shape(0); ++i)
          {
          array<int64_t,8> res;
          base.neighbors(iin(i),res);
          for (size_t j=0; j<8; ++j) iout.v(i,j)=res[j];
          }
        });
      }
    py::array ring2nest (const py::array &ring) const
      {
      return doStuff<int64_t, int64_t, 0, 0>(ring, {}, {}, [this](const MavIter<int64_t,1> &iin, MavIter<int64_t,1> &iout)
        {
        for (size_t i=0; i<iin.shape(0); ++i)
          iout.v(i)=base.ring2nest(iin(i));
        });
      }
    py::array nest2ring (const py::array &nest) const
      {
      return doStuff<int64_t, int64_t, 0, 0>(nest, {}, {}, [this](const MavIter<int64_t,1> &iin, MavIter<int64_t,1> &iout)
        {
        for (size_t i=0; i<iin.shape(0); ++i)
          iout.v(i)=base.nest2ring(iin(i));
        });
      }
    py::array query_disc(const py::array &ptg, double radius) const
      {
      MR_assert((ptg.ndim()==1)&&(ptg.shape(0)==2),
        "ptg must be a 1D array with 2 values");
      rangeset<int64_t> pixset;
      auto ptg2 = to_mav<double,1>(ptg);
      base.query_disc(pointing(ptg2(0),ptg2(1)), radius, pixset);
      auto res = make_Pyarr<int64_t>(shape_t({pixset.nranges(),2}));
      auto oref=res.mutable_unchecked<2>();
      for (size_t i=0; i<pixset.nranges(); ++i)
        {
        oref(i,0)=pixset.ivbegin(i);
        oref(i,1)=pixset.ivend(i);
        }
      return move(res);
      }
  };

py::array ang2vec (const py::array &ang)
  {
  return doStuff<double, double, 1, 1>(ang, {2}, {3}, [](const MavIter<double,2> &iin, MavIter<double,2> &iout)
    {
    for (size_t i=0; i<iin.shape(0); ++i)
      {
      vec3 v (pointing(iin(i,0),iin(i,1)));
      iout.v(i,0)=v.x; iout.v(i,1)=v.y; iout.v(i,2)=v.z;
      }
    });
  }
py::array vec2ang (const py::array &vec)
  {
  return doStuff<double, double, 1, 1>(vec, {3}, {2}, [](const MavIter<double,2> &iin, MavIter<double,2> &iout)
    {
    for (size_t i=0; i<iin.shape(0); ++i)
      {
      pointing ptg (vec3(iin(i,0),iin(i,1),iin(i,2)));
      iout.v(i,0)=ptg.theta; iout.v(i,1)=ptg.phi;
      }
    });
  }
py::array local_v_angle (const py::array &v1, const py::array &v2)
  {
  auto v12 = to_fmav<double>(v1);
  auto v22 = to_fmav<double>(v2);
  MR_assert(v12.shape()==v22.shape(), "shape mismatch");
  auto angle = make_Pyarr<double>(repl_dim<1,0>(v12.shape(),{3},{}));
  auto angle2 = to_fmav<double>(angle,true);
  MavIter<double,2> ii1(v12), ii2(v22);
  MavIter<double,1> iout(angle2);
  while (!iout.done())
    {
    for (size_t i=0; i<iout.shape(0); ++i)
      iout.v(i)=v_angle(vec3(ii1(i,0),ii1(i,1),ii1(i,2)),
                        vec3(ii2(i,0),ii2(i,1),ii2(i,2)));
    ii1.inc();ii2.inc();iout.inc();
    }
  return move(angle);
  }

const char *healpix_DS = R"""(
Python interface for some of the HEALPix C++ functionality

All angles are interpreted as radians.
The theta coordinate is measured as co-latitude, ranging from 0 (North Pole)
to pi (South Pole).

All 3-vectors returned by the functions are normalized.
However, 3-vectors provided as input to the functions need not be normalized.

Error conditions are reported by raising exceptions.
)""";

const char *order_DS = R"""(
Returns the ORDER parameter of the pixelisation.
If Nside is a power of 2, this is log_2(Nside), otherwise it is -1.
)""";

const char *nside_DS = R"""(
Returns the Nside parameter of the pixelisation.
)""";

const char *npix_DS = R"""(
Returns the total number of pixels of the pixelisation.
)""";

const char *scheme_DS = R"""(
Returns a string representation of the pixelisation's ordering scheme
("RING" or "NEST").
)""";

const char *pix_area_DS = R"""(
Returns the area (in steradian) of a single pixel.
)""";

const char *max_pixrad_DS = R"""(
Returns the maximum angular distance (in radian) between a pixel center
and its corners for this pixelisation.
)""";

const char *pix2ang_DS = R"""(
Returns a (co-latitude, longitude) tuple for each value in pix.
The result array has the same shape as pix, with an added last dimension
of size 2.
)""";

const char *ang2pix_DS = R"""(
Returns the index of the containing pixel for every (co-latitude, longitude)
tuple in ang. ang must have a last dimension of size 2; the result array
has the same shape as ang, except that ang's last dimension is removed.
)""";

const char *pix2vec_DS = R"""(
Returns a normalized 3-vector for each value in pix.
The result array has the same shape as pix, with an added last dimension
of size 3.
)""";

const char *vec2pix_DS = R"""(
Returns the index of the containing pixel for every 3-vector in vec.
vec must have a last dimension of size 3; the result array has the same shape as
vec, except that vec's last dimension is removed.
)""";

const char *ring2nest_DS = R"""(
Returns the pixel index in NEST scheme for every entry of ring.
The result array has the same shape as ring.
)""";

const char *nest2ring_DS = R"""(
Returns the pixel index in RING scheme for every entry of nest.
The result array has the same shape as nest.
)""";

const char *query_disc_DS = R"""(
Returns a range set of all pixels whose centers fall within "radius" of "ptg".
"ptg" must be a single (co-latitude, longitude) tuple. The result is a 2D array
with last dimension 2; the pixels lying inside the disc are
[res[0,0] .. res[0,1]); [res[1,0] .. res[1,1]) etc.
)""";

const char *ang2vec_DS = R"""(
Returns a normalized 3-vector for every (co-latitude, longitude)
tuple in ang. ang must have a last dimension of size 2; the result array
has the same shape as ang, except that its last dimension is 3 instead of 2.
)""";

const char *vec2ang_DS = R"""(
Returns a (co-latitude, longitude) tuple for every 3-vector in vec.
vec must have a last dimension of size 3; the result array has the same shape as
vec, except that its last dimension is 2 instead of 3.
)""";

const char *v_angle_DS = R"""(
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

  py::class_<Pyhpbase> (m, "Healpix_Base", py::module_local())
    .def(py::init<int,const string &>(),"nside"_a,"scheme"_a)
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
    .def("pix2ang", &Pyhpbase::pix2ang, pix2ang_DS, "pix"_a)
    .def("ang2pix", &Pyhpbase::ang2pix, ang2pix_DS, "ang"_a)
    .def("pix2vec", &Pyhpbase::pix2vec, pix2vec_DS, "pix"_a)
    .def("vec2pix", &Pyhpbase::vec2pix, vec2pix_DS, "vec"_a)
    .def("pix2xyf", &Pyhpbase::pix2xyf, "pix"_a)
    .def("xyf2pix", &Pyhpbase::xyf2pix, "xyf"_a)
    .def("neighbors", &Pyhpbase::neighbors,"pix"_a)
    .def("ring2nest", &Pyhpbase::ring2nest, ring2nest_DS, "ring"_a)
    .def("nest2ring", &Pyhpbase::nest2ring, nest2ring_DS, "nest"_a)
    .def("query_disc", &Pyhpbase::query_disc, query_disc_DS, "ptg"_a,"radius"_a)
    .def("__repr__", &Pyhpbase::repr)
    ;

  m.def("ang2vec",&ang2vec, ang2vec_DS, "ang"_a);
  m.def("vec2ang",&vec2ang, vec2ang_DS, "vec"_a);
  m.def("v_angle",&local_v_angle, v_angle_DS, "v1"_a, "v2"_a);
  }

}

using detail_pymodule_healpix::add_healpix;

}
