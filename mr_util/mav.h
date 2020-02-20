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

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef MRUTIL_MAV_H
#define MRUTIL_MAV_H

#include <cstdlib>
#include <array>
#include <vector>
#include <memory>
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
    size_t sz;

    static size_t prod(const shape_t &shape)
      {
      size_t res=1;
      for (auto sz: shape)
        res*=sz;
      return res;
      }

  public:
    fmav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_), sz(prod(shp))
      {
      MR_assert(shp.size()>0, "at least 1D required");
      MR_assert(shp.size()==str.size(), "dimensions mismatch");
      }
    fmav_info(const shape_t &shape_)
      : shp(shape_), str(shape_.size()), sz(prod(shp))
      {
      auto ndim = shp.size();
      MR_assert(ndim>0, "at least 1D required");
      str[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        str[ndim-i] = str[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      }
    size_t ndim() const { return shp.size(); }
    size_t size() const { return sz; }
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
        stride *= ptrdiff_t(shp[ndim-1-i]);
        }
      return true;
      }
    bool conformable(const fmav_info &other) const
      { return shp==other.shp; }
  };

template<typename T> class membuf
  {
  protected:
    using Tsp = shared_ptr<vector<T>>;
    Tsp ptr;
    const T *d;
    bool rw;

  public:
    membuf(T *d_, bool rw_=false)
      : d(d_), rw(rw_) {}
    membuf(const T *d_)
      : d(d_), rw(false) {}
    membuf(size_t sz)
      : ptr(make_unique<vector<T>>(sz)), d(ptr->data()), rw(true) {}
    membuf(const membuf &other) = default;
    membuf(membuf &&other) = default;
// Not for public use!
    membuf(T *d_, const Tsp &p, bool rw_)
      : ptr(p), d(d_), rw(rw_) {}

    template<typename I> const T &val(I i) const
      { return d[i]; }
    template<typename I> T &val(I i)
      {
      MR_assert(rw, "array is nor writable");
      return const_cast<T *>(d)[i];
      }
    template<typename I> const T &operator[](I i) const
      { return d[i]; }
    template<typename I> T &operator[](I i)
      {
      MR_assert(rw, "array is nor writable");
      return const_cast<T *>(d)[i];
      }
    const T *data() const
      { return d; }
    T *data()
      {
      MR_assert(rw, "array is nor writable");
      return const_cast<T *>(d);
      }
    bool writable() const { return rw; }
  };

// "mav" stands for "multidimensional array view"
template<typename T> class fmav: public fmav_info, public membuf<T>
  {
  protected:
    using typename membuf<T>::Tsp;

  public:
    fmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : fmav_info(shp_, str_), membuf<T>(d_) {}
    fmav(const T *d_, const shape_t &shp_)
      : fmav_info(shp_), membuf<T>(d_) {}
    fmav(T *d_, const shape_t &shp_, const stride_t &str_, bool rw_)
      : fmav_info(shp_, str_), membuf<T>(d_,rw_) {}
    fmav(T *d_, const shape_t &shp_, bool rw_)
      : fmav_info(shp_), membuf<T>(d_,rw_) {}
    fmav(const shape_t &shp_)
      : fmav_info(shp_), membuf<T>(size()) {}
    fmav(T* d_, const fmav_info &info, bool rw_=false)
      : fmav_info(info), membuf<T>(d_, rw_) {}
    fmav(const T* d_, const fmav_info &info)
      : fmav_info(info), membuf<T>(d_) {}
    fmav(const fmav &other) = delete;
    fmav(fmav &&other) = default;
// Not for public use!
    fmav(T *d_, const Tsp &p, const shape_t &shp_, const stride_t &str_, bool rw_)
      : fmav_info(shp_, str_), membuf<T>(d_, p, rw_) {}
  };

template<size_t ndim> class mav_info
  {
  protected:
    using shape_t = array<size_t, ndim>;
    using stride_t = array<ptrdiff_t, ndim>;

    shape_t shp;
    stride_t str;
    size_t sz;

    static size_t prod(const shape_t &shape)
      {
      size_t res=1;
      for (auto sz: shape)
        res*=sz;
      return res;
      }

  public:
    mav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_), sz(prod(shp))
      { MR_assert(shp.size()==str.size(), "dimensions mismatch"); }
    mav_info(const shape_t &shape_)
      : shp(shape_), sz(prod(shp))
      {
      MR_assert(ndim>0, "at least 1D required");
      str[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        str[ndim-i] = str[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      }
//    size_t ndim() const { return shp.size(); }
    size_t size() const { return sz; }
    const shape_t &shape() const { return shp; }
    size_t shape(size_t i) const { return shp[i]; }
    const stride_t &stride() const { return str; }
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    bool last_contiguous() const
      { return (str.back()==1); }
    bool contiguous() const
      {
      ptrdiff_t stride=1;
      for (size_t i=0; i<ndim; ++i)
        {
        if (str[ndim-1-i]!=stride) return false;
        stride *= ptrdiff_t(shp[ndim-1-i]);
        }
      return true;
      }
    bool conformable(const mav_info &other) const
      { return shp==other.shp; }
  };

template<typename T, size_t ndim> class mav: public mav_info<ndim>, public membuf<T>
  {
  static_assert((ndim>0) && (ndim<4), "only supports 1D, 2D, and 3D arrays");
  using typename mav_info<ndim>::shape_t;
  using typename mav_info<ndim>::stride_t;

  protected:
    using membuf<T>::Tsp;
    using mav_info<ndim>::size;
    using membuf<T>::d;
    using membuf<T>::ptr;
    using mav_info<ndim>::shp;
    using mav_info<ndim>::str;
    using membuf<T>::rw;
    using membuf<T>::val;

  public:
    mav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : mav_info<ndim>(shp_, str_), membuf<T>(d_) {}
    mav(T *d_, const shape_t &shp_, const stride_t &str_, bool rw_=false)
      : mav_info<ndim>(shp_, str_), membuf<T>(d_, rw_) {}
    mav(const T *d_, const shape_t &shp_)
      : mav_info<ndim>(shp_), membuf<T>(d_) {}
    mav(T *d_, const shape_t &shp_, bool rw_=false)
      : mav_info<ndim>(shp_), membuf<T>(d_, rw_) {}
    mav(const array<size_t,ndim> &shp_)
      : mav_info<ndim>(shp_), membuf<T>(size()) {}
    mav(const mav &other) = delete;
    mav(mav &&other) = default;
    operator fmav<T>() const
      {
      return fmav<T>(const_cast<T *>(d), ptr, {shp.begin(), shp.end()}, {str.begin(), str.end()}, rw);
      }
    const T &operator()(size_t i) const
      {
      static_assert(ndim==1, "ndim must be 1");
      return val(str[0]*i);
      }
    const T &operator()(size_t i, size_t j) const
      {
      static_assert(ndim==2, "ndim must be 2");
      return val(str[0]*i + str[1]*j);
      }
    const T &operator()(size_t i, size_t j, size_t k) const
      {
      static_assert(ndim==3, "ndim must be 3");
      return val(str[0]*i + str[1]*j + str[2]*k);
      }
    T &operator()(size_t i)
      {
      static_assert(ndim==1, "ndim must be 1");
      return val(str[0]*i);
      }
    T &operator()(size_t i, size_t j)
      {
      static_assert(ndim==2, "ndim must be 2");
      return val(str[0]*i + str[1]*j);
      }
    T &operator()(size_t i, size_t j, size_t k)
      {
      static_assert(ndim==3, "ndim must be 3");
      return val(str[0]*i + str[1]*j + str[2]*k);
      }
    void fill(const T &val)
      {
      MR_assert(rw, "array is nor writable");
      // FIXME: special cases for contiguous arrays and/or zeroing?
      if (ndim==1)
        for (size_t i=0; i<shp[0]; ++i)
          d[str[0]*i]=val;
      else if (ndim==2)
        for (size_t i=0; i<shp[0]; ++i)
          for (size_t j=0; j<shp[1]; ++j)
            d[str[0]*i + str[1]*j] = val;
      else if (ndim==3)
        for (size_t i=0; i<shp[0]; ++i)
          for (size_t j=0; j<shp[1]; ++j)
            for (size_t k=0; k<shp[2]; ++k)
              d[str[0]*i + str[1]*j + str[2]*k] = val;
      }
  };

}

using detail_mav::shape_t;
using detail_mav::stride_t;
using detail_mav::fmav;
using detail_mav::mav;

}

#endif
