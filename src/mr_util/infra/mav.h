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
#include "mr_util/infra/error_handling.h"
#define _MSC_VER
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

    membuf(const T *d_, membuf &other)
      : ptr(other.ptr), d(d_), rw(other.rw) {}
    membuf(const T *d_, const membuf &other)
      : ptr(other.ptr), d(d_), rw(false) {}

  public:
    membuf(T *d_, bool rw_=false)
      : d(d_), rw(rw_) {}
    membuf(const T *d_)
      : d(d_), rw(false) {}
    membuf(size_t sz)
      : ptr(make_unique<vector<T>>(sz)), d(ptr->data()), rw(true) {}
    membuf(const membuf &other)
      : ptr(other.ptr), d(other.d), rw(false) {}
#if defined(_MSC_VER)
    membuf(membuf &other)
      : ptr(other.ptr), d(other.d), rw(other.rw) {}
    membuf(membuf &&other)
      : ptr(move(other.ptr)), d(move(other.d)), rw(move(other.rw)) {}
#else
    membuf(membuf &other) = default;
    membuf(membuf &&other) = default;
#endif

    template<typename I> T &vraw(I i)
      {
      MR_assert(rw, "array is not writable");
      return const_cast<T *>(d)[i];
      }
    template<typename I> const T &operator[](I i) const
      { return d[i]; }
    const T *data() const
      { return d; }
    T *vdata()
      {
      MR_assert(rw, "array is not writable");
      return const_cast<T *>(d);
      }
    bool writable() const { return rw; }
  };

// "mav" stands for "multidimensional array view"
template<typename T> class fmav: public fmav_info, public membuf<T>
  {
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
    fmav(const fmav &other) = default;
#if defined(_MSC_VER)
    fmav(fmav &other) : fmav_info(other), membuf<T>(other) {}
    fmav(fmav &&other) : fmav_info(other), membuf<T>(other) {}
#else
    fmav(fmav &other) = default;
    fmav(fmav &&other) = default;
#endif
    fmav(membuf<T> &buf, const shape_t &shp_, const stride_t &str_)
      : fmav_info(shp_, str_), membuf<T>(buf) {}
    fmav(const membuf<T> &buf, const shape_t &shp_, const stride_t &str_)
      : fmav_info(shp_, str_), membuf<T>(buf) {}
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
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*n + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*n; }

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
    bool conformable(const shape_t &other) const
      { return shp==other; }
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      static_assert(ndim==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }
  };

template<typename T, size_t ndim> class mav: public mav_info<ndim>, public membuf<T>
  {
//  static_assert((ndim>0) && (ndim<4), "only supports 1D, 2D, and 3D arrays");

  protected:
    using typename mav_info<ndim>::shape_t;
    using typename mav_info<ndim>::stride_t;
    using membuf<T>::d;
    using membuf<T>::ptr;
    using mav_info<ndim>::shp;
    using mav_info<ndim>::str;
    using membuf<T>::rw;
    using membuf<T>::vraw;

    template<size_t idim, typename Func> void applyHelper(ptrdiff_t idx, Func func)
      {
      if constexpr (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<idim+1, Func>(idx+i*str[idim], func);
      else
        {
        T *d2 = vdata();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]]);
        }
      }
    template<size_t idim, typename Func> void applyHelper(ptrdiff_t idx, Func func) const
      {
      if constexpr (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<idim+1, Func>(idx+i*str[idim], func);
      else
        {
        const T *d2 = data();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]]);
        }
      }
    template<size_t idim, typename T2, typename Func>
      void applyHelper(ptrdiff_t idx, ptrdiff_t idx2,
                       const mav<T2,ndim> &other, Func func)
      {
      if constexpr (idim==0)
        MR_assert(conformable(other), "dimension mismatch");
      if constexpr (idim+1<ndim)
        for (size_t i=0; i<shp[idim]; ++i)
          applyHelper<idim+1, T2, Func>(idx+i*str[idim],
                                        idx2+i*other.str[idim], other, func);
      else
        {
        T *d2 = vdata();
        const T2 *d3 = other.data();
        for (size_t i=0; i<shp[idim]; ++i)
          func(d2[idx+i*str[idim]],d3[idx2+i*other.str[idim]]);
        }
      }

    template<size_t nd2> void subdata(const shape_t &i0, const shape_t &extent,
      array<size_t, nd2> &nshp, array<ptrdiff_t, nd2> &nstr, ptrdiff_t &nofs) const
      {
      size_t n0=0;
      for (auto x:extent) if (x==0)++n0;
      MR_assert(n0+nd2==ndim, "bad extent");
      nofs=0;
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(i0[i]<shp[i], "bad subset");
        nofs+=i0[i]*str[i];
        if (extent[i]!=0)
          {
          MR_assert(i0[i]+extent[i2]<=shp[i], "bad subset");
          nshp[i2] = extent[i]; nstr[i2]=str[i];
          ++i2;
          }
        }
      }

  public:
    using membuf<T>::operator[];
    using membuf<T>::vdata;
    using membuf<T>::data;
    using mav_info<ndim>::contiguous;
    using mav_info<ndim>::size;
    using mav_info<ndim>::idx;
    using mav_info<ndim>::conformable;

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
    mav(const mav &other) = default;
#if defined(_MSC_VER)
    mav(mav &other): mav_info<ndim>(other), membuf<T>(other) {}
    mav(mav &&other): mav_info<ndim>(other), membuf<T>(other) {}
#else
    mav(mav &other) = default;
    mav(mav &&other) = default;
#endif
    mav(const shape_t &shp_, const stride_t &str_, const T *d_, membuf<T> &mb)
      : mav_info<ndim>(shp_, str_), membuf<T>(d_, mb) {}
    mav(const shape_t &shp_, const stride_t &str_, const T *d_, const membuf<T> &mb)
      : mav_info<ndim>(shp_, str_), membuf<T>(d_, mb) {}
    operator fmav<T>() const
      {
      return fmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    operator fmav<T>()
      {
      return fmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return operator[](idx(ns...)); }
    template<typename... Ns> T &v(Ns... ns)
      { return vraw(idx(ns...)); }
    template<typename Func> void apply(Func func)
      {
      if (contiguous())
        {
        T *d2 = vdata();
        for (auto v=d2; v!=d2+size(); ++v)
          func(*v);
        return;
        }
      applyHelper<0,Func>(0,func);
      }
    template<typename T2, typename Func> void apply
      (const mav<T2, ndim> &other,Func func)
      { applyHelper<0,T2,Func>(0,0,other,func); }
    void fill(const T &val)
      { apply([val](T &v){v=val;}); }
    template<size_t nd2> mav<T,nd2> subarray(const shape_t &i0, const shape_t &extent)
      {
      array<size_t,nd2> nshp;
      array<ptrdiff_t,nd2> nstr;
      ptrdiff_t nofs;
      subdata<nd2> (i0, extent, nshp, nstr, nofs);
      return mav<T,nd2> (nshp, nstr, d+nofs, *this);
      }
    template<size_t nd2> mav<T,nd2> subarray(const shape_t &i0, const shape_t &extent) const
      {
      array<size_t,nd2> nshp;
      array<ptrdiff_t,nd2> nstr;
      ptrdiff_t nofs;
      subdata<nd2> (i0, extent, nshp, nstr, nofs);
      return mav<T,nd2> (nshp, nstr, d+nofs, *this);
      }
  };

template<typename T, size_t ndim> class MavIter
  {
  protected:
    fmav<T> mav;
    array<size_t, ndim> shp;
    array<ptrdiff_t, ndim> str;
    shape_t pos;
    ptrdiff_t idx_;
    bool done_;

    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*n + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*n; }

  public:
    MavIter(const fmav<T> &mav_)
      : mav(mav_), pos(mav.ndim()-ndim,0), idx_(0), done_(false)
      {
      for (size_t i=0; i<ndim; ++i)
        {
        shp[i] = mav.shape(mav.ndim()-ndim+i);
        str[i] = mav.stride(mav.ndim()-ndim+i);
        }
      }
    MavIter(fmav<T> &mav_)
      : mav(mav_), pos(mav.ndim()-ndim,0), idx_(0), done_(false)
      {
      for (size_t i=0; i<ndim; ++i)
        {
        shp[i] = mav.shape(mav.ndim()-ndim+i);
        str[i] = mav.stride(mav.ndim()-ndim+i);
        }
      }
    bool done() const
      { return done_; }
    void inc()
      {
      for (ptrdiff_t i=mav.ndim()-ndim-1; i>=0; --i)
        {
        idx_+=mav.stride(i);
        if (++pos[i]<mav.shape(i)) return;
        pos[i]=0;
        idx_-=mav.shape(i)*mav.stride(i);
        }
      done_=true;
      }
    size_t shape(size_t i) const { return shp[i]; }
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      { return idx_ + getIdx(0, ns...); }
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return mav[idx(ns...)]; }
    template<typename... Ns> T &v(Ns... ns)
      { return mav.vraw(idx(ns...)); }
  };

}

using detail_mav::shape_t;
using detail_mav::stride_t;
using detail_mav::fmav_info;
using detail_mav::fmav;
using detail_mav::mav;
using detail_mav::MavIter;

}

#endif
