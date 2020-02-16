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

#include "Healpix_cxx/healpix_base.h"
#include "mr_util/constants.h"
#include "mr_util/string_utils.h"
#include "mr_util/geom_utils.h"

using namespace std;
using namespace mr;
using namespace healpix;

namespace py = pybind11;

namespace {
class Itnew
  {
  protected:
    using stv = vector<size_t>;
    using pdv = vector<ptrdiff_t>;

    stv shape, pos;
    pdv stride;
    bool done_;
    const char *ptr;

  public:
    Itnew (const py::array &in) : shape(max<int>(2,in.ndim())),
      pos(max<int>(2,in.ndim()),0), stride(max<int>(2,in.ndim())), done_(false),
      ptr(reinterpret_cast<const char *>(in.data()))
      {
      for (size_t i=0; i<size_t(in.ndim()); ++i)
        {
        shape[i]=in.shape(in.ndim()-1-i);
        stride[i]=(shape[i]==1) ? 0 : in.strides(in.ndim()-1-i);
        }
      for (size_t i=in.ndim(); i<shape.size(); ++i)
        {
        shape[i]=1;
        stride[i]=0;
        }
      }

    size_t dim(size_t idim) const
      { return shape[idim]; }
    bool done() const
      { return done_; }
    void inc (size_t dim)
      {
      for (size_t i=dim; i<shape.size(); ++i)
        {
        ptr+=stride[i];
        if (++pos[i]<shape[i]) return;
        pos[i]=0;
        ptr-=shape[i]*stride[i];
        }
      done_=true;
      }
  };
template<typename T> class Itnew_w: public Itnew
  {
  private:
    const T *cptr (const char *p) const { return reinterpret_cast<const T *>(p); }
    T *vptr (const char *p) { return const_cast<T *>(cptr(p)); }

  public:
    using Itnew::Itnew;

    T &operator() (size_t i)
      { return *vptr(ptr+i*stride[0]); }
    T &operator() (size_t i, size_t j)
      { return *vptr(ptr+j*stride[0]+i*stride[1]); }
  };
template<typename T> class Itnew_r: public Itnew
  {
  private:
    const T *cptr (const char *p) const { return reinterpret_cast<const T *>(p); }

  public:
    using Itnew::Itnew;

    const T &operator() (size_t i) const
      { return *cptr(ptr+i*stride[0]); }
    const T &operator() (size_t i, size_t j) const
      { return *cptr(ptr+j*stride[0]+i*stride[1]); }
  };

using a_d = py::array_t<double>;
using a_i = py::array_t<int64_t>;
using a_d_c = py::array_t<double, py::array::c_style | py::array::forcecast>;

void assert_equal_shape(const py::array &a, const py::array &b)
  {
  MR_assert(a.ndim()==b.ndim(),"array dimensions mismatch");
  for (size_t i=0; i<size_t(a.ndim()); ++i)
    MR_assert(a.shape(i)==b.shape(i), "array hape mismatch");
  }
vector<size_t> add_dim(const py::array &a, size_t dim)
  {
  vector<size_t> res(a.ndim()+1);
  for (size_t i=0; i<size_t(a.ndim()); ++i) res[i]=a.shape(i);
  res.back()=dim;
  return res;
  }
vector<size_t> subst_dim(const py::array &a, size_t d1, size_t d2)
  {
  MR_assert(a.ndim()>0,"too few array dimensions");
  MR_assert(size_t(a.shape(a.ndim()-1))==d1,
    "incorrect last array dimension");
  vector<size_t> res(a.ndim());
  for (size_t i=0; i<size_t(a.ndim()-1); ++i) res[i]=a.shape(i);
  res.back()=d2;
  return res;
  }
vector<size_t> rem_dim(const py::array &a, size_t dim)
  {
  MR_assert(a.ndim()>0,"too few array dimensions");
  MR_assert(size_t(a.shape(a.ndim()-1))==dim,
    "incorrect last array dimension");
  vector<size_t> res(a.ndim()-1);
  for (size_t i=0; i<size_t(a.ndim())-1; ++i) res[i]=a.shape(i);
  return res;
  }
vector<size_t> copy_dim(const py::array &a)
  {
  vector<size_t> res(a.ndim());
  for (size_t i=0; i<size_t(a.ndim()); ++i) res[i]=a.shape(i);
  return res;
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

    a_d pix2ang (const a_i &pix) const
      {
      a_d ang(add_dim(pix,2));
      Itnew_w<double> iout (ang);
      Itnew_r<int64_t> iin (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iin.dim(0); ++i)
          {
          pointing ptg=base.pix2ang(iin(i));
          iout(i,0)=ptg.theta; iout(i,1)=ptg.phi;
          }
        iin.inc(1);iout.inc(2);
        }
      return ang;
      }
    a_i ang2pix (const a_d &ang) const
      {
      a_i pix(rem_dim(ang,2));
      Itnew_r<double> iin (ang);
      Itnew_w<int64_t> iout (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iout.dim(0); ++i)
          iout(i)=base.ang2pix(pointing(iin(i,0),iin(i,1)));
        iin.inc(2);iout.inc(1);
        }
      return pix;
      }
    a_d pix2vec (const a_i &pix) const
      {
      a_d vec(add_dim(pix,3));
      Itnew_w<double> iout (vec);
      Itnew_r<int64_t> iin (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iin.dim(0); ++i)
          {
          vec3 v=base.pix2vec(iin(i));
          iout(i,0)=v.x; iout(i,1)=v.y; iout(i,2)=v.z;
          }
        iin.inc(1);iout.inc(2);
        }
      return vec;
      }
    a_i vec2pix (const a_d &vec) const
      {
      a_i pix(rem_dim(vec,3));
      Itnew_r<double> iin (vec);
      Itnew_w<int64_t> iout (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iout.dim(0); ++i)
          iout(i)=base.vec2pix(vec3(iin(i,0),iin(i,1),iin(i,2)));
        iin.inc(2);iout.inc(1);
        }
      return pix;
      }
    a_i pix2xyf (const a_i &pix) const
      {
      a_i xyf(add_dim(pix,3));
      Itnew_w<int64_t> iout (xyf);
      Itnew_r<int64_t> iin (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iin.dim(0); ++i)
          {
          int x,y,f;
          base.pix2xyf(iin(i),x,y,f);
          iout(i,0)=x; iout(i,1)=y; iout(i,2)=f;
          }
        iin.inc(1);iout.inc(2);
        }
      return xyf;
      }
    a_i xyf2pix (const a_i &xyf) const
      {
      a_i pix(rem_dim(xyf,3));
      Itnew_r<int64_t> iin (xyf);
      Itnew_w<int64_t> iout (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iout.dim(0); ++i)
          iout(i)=base.xyf2pix(iin(i,0),iin(i,1),iin(i,2));
        iin.inc(2);iout.inc(1);
        }
      return pix;
      }
    a_i neighbors (const a_i &pix) const
      {
      a_i nb(add_dim(pix,8));
      Itnew_w<int64_t> iout (nb);
      Itnew_r<int64_t> iin (pix);
      while (!iin.done())
        {
        for (size_t i=0; i<iin.dim(0); ++i)
          {
          array<int64_t,8> res;
          base.neighbors(iin(i),res);
          for (size_t j=0; j<8; ++j) iout(i,j)=res[j];
          }
        iin.inc(1);iout.inc(2);
        }
      return nb;
      }
    a_i ring2nest (const a_i &ring) const
      {
      a_i nest(copy_dim(ring));
      Itnew_r<int64_t> iin (ring);
      Itnew_w<int64_t> iout (nest);
      while (!iin.done())
        {
        for (size_t i=0; i<iin.dim(0); ++i)
          iout(i)=base.ring2nest(iin(i));
        iin.inc(1);iout.inc(1);
        }
      return nest;
      }
    a_i nest2ring (const a_i &nest) const
      {
      a_i ring(copy_dim(nest));
      Itnew_r<int64_t> iin (nest);
      Itnew_w<int64_t> iout (ring);
      while (!iin.done())
        {
        for (size_t i=0; i<iin.dim(0); ++i)
          iout(i)=base.nest2ring(iin(i));
        iin.inc(1);iout.inc(1);
        }
      return ring;
      }
    a_i query_disc(const a_d &ptg, double radius) const
      {
      MR_assert((ptg.ndim()==1)&&(ptg.shape(0)==2),
        "ptg must be a 1D array with 2 values");
      rangeset<int64_t> pixset;
      auto ir=ptg.unchecked<1>();
      base.query_disc(pointing(ir[0],ir[1]), radius, pixset);
      a_i res(vector<size_t>({pixset.nranges(),2}));
      auto oref=res.mutable_unchecked<2>();
      for (size_t i=0; i<pixset.nranges(); ++i)
        {
        oref(i,0)=pixset.ivbegin(i);
        oref(i,1)=pixset.ivend(i);
        }
      return res;
      }
  };

a_d ang2vec (const a_d &ang)
  {
  a_d vec(subst_dim(ang,2,3));
  Itnew_w<double> iout (vec);
  Itnew_r<double> iin (ang);
  while (!iin.done())
    {
    for (size_t i=0; i<iin.dim(1); ++i)
      {
      vec3 v (pointing(iin(i,0),iin(i,1)));
      iout(i,0)=v.x; iout(i,1)=v.y; iout(i,2)=v.z;
      }
    iin.inc(2);iout.inc(2);
    }
  return vec;
  }
a_d vec2ang (const a_d &vec)
  {
  a_d ang(subst_dim(vec,3,2));
  Itnew_w<double> iout (ang);
  Itnew_r<double> iin (vec);
  while (!iin.done())
    {
    for (size_t i=0; i<iin.dim(1); ++i)
      {
      pointing ptg (vec3(iin(i,0),iin(i,1),iin(i,2)));
      iout(i,0)=ptg.theta; iout(i,1)=ptg.phi;
      }
    iin.inc(2);iout.inc(2);
    }
  return ang;
  }
a_d local_v_angle (const a_d &v1, const a_d &v2)
  {
  assert_equal_shape(v1,v2);
  a_d angle(rem_dim(v1,3));
  Itnew_w<double> iout (angle);
  Itnew_r<double> ii1 (v1), ii2(v2);
  while (!iout.done())
    {
    for (size_t i=0; i<iout.dim(0); ++i)
      iout(i)=v_angle(vec3(ii1(i,0),ii1(i,1),ii1(i,2)),
                      vec3(ii2(i,0),ii2(i,1),ii2(i,2)));
    ii1.inc(2);ii2.inc(2);iout.inc(1);
    }
  return angle;
  }

const char *pyHealpix_DS = R"""(
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

} // unnamed namespace

PYBIND11_MODULE(pyHealpix, m)
  {
  using namespace pybind11::literals;

  m.doc() = pyHealpix_DS;

  py::class_<Pyhpbase> (m, "Healpix_Base")
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
