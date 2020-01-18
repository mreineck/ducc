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
#include <vector>
#include "mr_util/error_handling.h"

namespace mr {

namespace detail_mav {

using namespace std;

using shape_t = vector<size_t>;
using stride_t = vector<ptrdiff_t>;

class fmav_info
  {
  protected:
    shape_t shp;
    stride_t str;

    static size_t prod(const shape_t &shape)
      {
      size_t res=1;
      for (auto sz: shape)
        res*=sz;
      return res;
      }

  public:
    fmav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_)
      { MR_assert(shp.size()==str.size(), "dimensions mismatch"); }
    fmav_info(const shape_t &shape_)
      : shp(shape_), str(shape_.size())
      {
      auto ndim = shp.size();
      str[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        str[ndim-i] = str[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      }
    size_t ndim() const { return shp.size(); }
    size_t size() const { return prod(shp); }
    const shape_t &shape() const { return shp; }
    size_t shape(size_t i) const { return shp[i]; }
    const stride_t &stride() const { return str; }
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    bool last_contiguous() const
      { return (str.back()==1); }
    bool contiguous() const
      {
      auto ndim = shp.size();
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if (str[ndim-1-i]!=stride) return false;
        stride *= shp[ndim-1-i];
        }
      return true;
      }
    bool conformable(const fmav_info &other) const
      { return shp==other.shp; }
  };

// "mav" stands for "multidimensional array view"
template<typename T> class cfmav: public fmav_info
  {
  protected:
    T *d;

  public:
    cfmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : fmav_info(shp_, str_), d(const_cast<T *>(d_)) {}
    cfmav(const T *d_, const shape_t &shp_)
      : fmav_info(shp_), d(const_cast<T *>(d_)) {}
    template<typename I> const T &operator[](I i) const
      { return d[i]; }
    const T *data() const
      { return d; }
  };
template<typename T> class fmav: public cfmav<T>
  {
  protected:
    using parent = cfmav<T>;
    using parent::d;
    using parent::shp;
    using parent::str;

  public:
    fmav(T *d_, const shape_t &shp_, const stride_t &str_)
      : parent(d_, shp_, str_) {}
    fmav(T *d_, const shape_t &shp_)
      : parent(d_, shp_) {}
    template<typename I> T &operator[](I i) const
      { return d[i]; }
    using parent::shape;
    using parent::stride;
    T *data() const
      { return d; }
    using parent::last_contiguous;
    using parent::contiguous;
    using parent::conformable;
  };

template<typename T> using const_fmav = fmav<const T>;

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
      for (size_t i=2; i<=ndim; ++i)
        str[ndim-i] = str[ndim-i+1]*shp[ndim-i+1];
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
    template<typename T2> bool conformable(const mav<T2,ndim> &other) const
      { return shp==other.shp; }
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

using detail_mav::shape_t;
using detail_mav::stride_t;
using detail_mav::fmav_info;
using detail_mav::fmav;
using detail_mav::cfmav;
using detail_mav::const_fmav;
using detail_mav::mav;
using detail_mav::const_mav;
using detail_mav::cmav;
using detail_mav::nullmav;

}

#endif
