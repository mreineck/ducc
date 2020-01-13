/*
 *  This file is part of the MR utility library.
 *
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

/* Copyright (C) 2019 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef MRUTIL_MAV_H
#define MRUTIL_MAV_H

#include <cstdlib>
#include <array>

namespace mr {

namespace detail_mav {

using namespace std;

// "mav" stands for "multidimensional array view"
template<typename T, size_t ndim> class mav
  {
  static_assert((ndim>0) && (ndim<3), "only supports 1D and 2D arrays");

  private:
    T *d;
    array<size_t, ndim> shp;
    array<ptrdiff_t, ndim> str;

  public:
    mav(T *d_, const array<size_t,ndim> &shp_,
        const array<ptrdiff_t,ndim> &str_)
      : d(d_), shp(shp_), str(str_) {}
    mav(T *d_, const array<size_t,ndim> &shp_)
      : d(d_), shp(shp_)
      {
      str[ndim-1]=1;
      for (size_t d=2; d<=ndim; ++d)
        str[ndim-d] = str[ndim-d+1]*shp[ndim-d+1];
      }
    T &operator[](size_t i) const
      { return operator()(i); }
    T &operator()(size_t i) const
      {
      static_assert(ndim==1, "ndim must be 1");
      return d[str[0]*i];
      }
    T &operator()(size_t i, size_t j) const
      {
      static_assert(ndim==2, "ndim must be 2");
      return d[str[0]*i + str[1]*j];
      }
    size_t shape(size_t i) const { return shp[i]; }
    const array<size_t,ndim> &shape() const { return shp; }
    size_t size() const
      {
      size_t res=1;
      for (auto v: shp) res*=v;
      return res;
      }
    ptrdiff_t stride(size_t i) const { return str[i]; }
    T *data() const
      { return d; }
    bool last_contiguous() const
      { return (str[ndim-1]==1) || (str[ndim-1]==0); }
    bool contiguous() const
      {
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if (str[ndim-1-i]!=stride) return false;
        stride *= shp[ndim-1-i];
        }
      return true;
      }
    void fill(const T &val) const
      {
      // FIXME: special cases for contiguous arrays and/or zeroing?
      if (ndim==1)
        for (size_t i=0; i<shp[0]; ++i)
          d[str[0]*i]=val;
      else if (ndim==2)
        for (size_t i=0; i<shp[0]; ++i)
          for (size_t j=0; j<shp[1]; ++j)
            d[str[0]*i + str[1]*j] = val;
      }
  };

template<typename T, size_t ndim> using const_mav = mav<const T, ndim>;
template<typename T, size_t ndim> const_mav<T, ndim> cmav (const mav<T, ndim> &mav)
  { return const_mav<T, ndim>(mav.data(), mav.shape()); }
template<typename T, size_t ndim> const_mav<T, ndim> nullmav()
  {
  array<size_t,ndim> shp;
  shp.fill(0);
  return const_mav<T, ndim>(nullptr, shp);
  }

}

using detail_mav::mav;
using detail_mav::const_mav;
using detail_mav::cmav;
using detail_mav::nullmav;

}

#endif
