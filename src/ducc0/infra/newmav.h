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

/*! \file ducc0/infra/mav.h
 *  Classes for dealing with multidimensional arrays
 *
 *  \copyright Copyright (C) 2019-2021 Max-Planck-Society
 *  \author Martin Reinecke
 *  */

#ifndef DUCC0_NEWMAV_H
#define DUCC0_NEWMAV_H

#include <array>
#include <vector>
#include <memory>
#include <numeric>
#include <cstddef>
#include <functional>
#include <tuple>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_mav {

using namespace std;

template<typename T> class cfmav: public fmav_info, public cmembuf<T>
  {
  protected:
    using tbuf = cmembuf<T>;
    using tinfo = fmav_info;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;


  protected:
    cfmav(const shape_t &shp_, uninitialized_dummy)
      : tinfo(shp_), tbuf(size(), UNINITIALIZED) {}
    cfmav(const shape_t &shp_, const stride_t &str_, uninitialized_dummy)
      : tinfo(shp_, str_), tbuf(size(), UNINITIALIZED)
      {
      ptrdiff_t ofs=0;
      for (size_t i=0; i<ndim(); ++i)
        ofs += (ptrdiff_t(shp[i])-1)*str[i];
      MR_assert(ofs+1==ptrdiff_t(size()), "array is not compact");
      }
    cfmav(const fmav_info &info, const T *d_, const tbuf &buf)
      : tinfo(info), tbuf(d_, buf) {}

  public:
    cfmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    cfmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
    cfmav(const T* d_, const tinfo &info)
      : tinfo(info), tbuf(d_) {}
    cfmav(const cfmav &other) = default;
    cfmav(cfmav &&other) = default;

    cfmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    cfmav(const tbuf &buf, const tinfo &info)
      : tinfo(info), tbuf(buf) {}

// FIXME temporary
    cfmav(const fmav<T> &orig)
      : tinfo(orig), tbuf(orig) {}

    void assign(const cfmav &other)
      {
      tinfo::assign(other);
      tbuf::assign(other);
      }

    /// Returns the data entry at the given set of indices.
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    /// Returns the data entry at the given set of indices.
    template<typename... Ns> const T &c(Ns... ns) const
      { return raw(idx(ns...)); }

    cfmav subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = subdata(slices);
      return cfmav(ninfo, tbuf::d+nofs, *this);
      }
  };

template<typename T> cfmav<T> subarray
  (const cfmav<T> &arr, const vector<slice> &slices)  
  { return arr.template subarray(slices); }

template<typename T> class vfmav: public cfmav<T>
  {
  protected:
    using tbuf = cmembuf<T>;
    using tinfo = fmav_info;
    using tinfo::shp, tinfo::str, tinfo::size;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;

  protected:
    vfmav(const fmav_info &info, T *d_, tbuf &buf)
      : cfmav<T>(info, d_, buf) {}

  public:
    using tbuf::raw, tbuf::data, tinfo::ndim;
    vfmav(T *d_, const fmav_info &info)
      : cfmav<T>(d_, info) {}
    vfmav(T *d_, const shape_t &shp_, const stride_t &str_)
      : cfmav<T>(d_, shp_, str_) {}
    vfmav(T *d_, const shape_t &shp_)
      : cfmav<T>(d_, shp_) {}
    vfmav(const shape_t &shp_)
      : cfmav<T>(shp_) {}
    vfmav(const shape_t &shp_, uninitialized_dummy)
      : cfmav<T>(shp_, UNINITIALIZED) {}
    vfmav(const shape_t &shp_, const stride_t &str_, uninitialized_dummy)
      : cfmav<T>(shp_, str_, UNINITIALIZED)
      {
      ptrdiff_t ofs=0;
      for (size_t i=0; i<ndim(); ++i)
        ofs += (ptrdiff_t(shp[i])-1)*str[i];
      MR_assert(ofs+1==ptrdiff_t(size()), "array is not compact");
      }
    vfmav(tbuf &buf, const tinfo &info)
      : cfmav<T>(buf, info) {}
    vfmav(tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : cfmav<T>(buf, shp_, str_) {}
    vfmav(fmav<T> &orig)
      : cfmav<T>(orig, orig) { MR_assert(orig.writable(), "fmav is not writable()"); }
    T *data()
     { return const_cast<T *>(tbuf::d); }
    // read access to element #i
    template<typename I> T &raw(I i)
      { return data()[i]; }

    vfmav(const shape_t &shp_, const stride_t &str_, T *d_, tbuf &buf)
      : cfmav<T>(shp_, str_, d_, buf) {}

    void assign(vfmav &other)
      {
      fmav_info::assign(other);
      cmembuf<T>::assign(other);
      }

    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    template<typename... Ns> T &v(Ns... ns)
      { return const_cast<T *>(raw(idx(ns...))); }

    vfmav subarray(const vector<slice> &slices)
      {
      auto [ninfo, nofs] = tinfo::subdata(slices);
      return vfmav(ninfo, data()+nofs, *this);
      }
    /** Returns a writable fmav with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  The array data is default-initialized. */
    static vfmav build_noncritical(const shape_t &shape)
      {
      auto ndim = shape.size();
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vfmav tmp(shape2);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray(slc);
      }
    /** Returns a writable fmav with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  The array data is not initialized. */
    static vfmav build_noncritical(const shape_t &shape, uninitialized_dummy)
      {
      auto ndim = shape.size();
      if (ndim<=1) return vfmav(shape, UNINITIALIZED);
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vfmav tmp(shape2, UNINITIALIZED);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray(slc);
      }
  };

template<typename T> vfmav<T> subarray
  (vfmav<T> &arr, const vector<slice> &slices)  
  { return arr.template subarray(slices); }

template<typename T, size_t ndim> class cmav: public mav_info<ndim>, public cmembuf<T>
  {
  protected:
    using tinfo = mav_info<ndim>;
    using tbuf = cmembuf<T>;
    using tinfo::shp, tinfo::str;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

  protected:
    cmav(const shape_t &shp_, uninitialized_dummy)
      : tinfo(shp_), tbuf(size(), UNINITIALIZED) {}

  public:
    cmav() {}
    cmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    cmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
#if defined(_MSC_VER)
    // MSVC is broken
    cmav(const cmav &other) : tinfo(other), tbuf(other) {}
    cmav(cmav &&other): tinfo(other), tbuf(other) {}
#else
    cmav(const cmav &other) = default;
    cmav(cmav &&other) = default;
#endif
    cmav(const tinfo &info, const T *d_, const tbuf &buf)
      : tinfo(info), tbuf(d_, buf) {}
    cmav(const shape_t &shp_)
      : tinfo(shp_), tbuf(size()) {}
    cmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    void assign(const cmav &other)
      {
      mav_info<ndim>::assign(other);
      cmembuf<T>::assign(other);
      }
    operator cfmav<T>() const
      {
      return cfmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    template<size_t nd2> cmav<T,nd2> subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = tinfo::template subdata<nd2> (slices);
      return cmav<T,nd2> (ninfo, tbuf::d+nofs, *this);
      }

    static cmav build_uniform(const shape_t &shape, const T &value)
      {
      array<size_t,1> tshp;
      tshp[0] = 1;
      cmav<T,1> tmp(tshp);
      const_cast<T &>(tmp(0)) = value;
      stride_t nstr;
      nstr.fill(0);
      return cmav(tmp, shape, nstr);
      }
  };
template<size_t nd2, typename T, size_t ndim> cmav<T,nd2> subarray
  (const cmav<T, ndim> &arr, const vector<slice> &slices)  
  { return arr.template subarray<nd2>(slices); }

template<typename T, size_t ndim> class vmav: public cmav<T, ndim>
  {
  protected:
    using parent = cmav<T, ndim>;
    using tinfo = mav_info<ndim>;
    using tbuf = cmembuf<T>;
    using tinfo::shp, tinfo::str;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

    vmav() {}
    vmav(T *d_, const shape_t &shp_, const stride_t &str_)
      : parent(d_, shp_, str_) {}
    vmav(T *d_, const shape_t &shp_)
      : parent(d_, shp_) {}
    vmav(const shape_t &shp_)
      : parent(shp_) {}
    vmav(const shape_t &shp_, uninitialized_dummy)
      : parent(shp_, UNINITIALIZED) {}
#if defined(_MSC_VER)
    // MSVC is broken
    vmav(const vmav &other) : parent(other) {}
    vmav(vmav &other): parent(other) {}
    vmav(vmav &&other): parent(other) {}
#else
    vmav(const vmav &other) = default;
    vmav(vmav &other) = default;
    vmav(vmav &&other) = default;
#endif
    vmav(const tinfo &info, T *d_, tbuf &buf)
      : parent(info, d_, buf) {}

    void assign(vmav &other)
      { parent::assign(other); }
    operator vfmav<T>()
      {
      return vfmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    template<typename... Ns> T &operator()(Ns... ns)
      { return const_cast<T &>(parent::operator()(ns...)); }
    template<size_t nd2> vmav<T,nd2> subarray(const vector<slice> &slices)
      {
      auto [ninfo, nofs] = tinfo::template subdata<nd2> (slices);
      return vmav<T,nd2> (ninfo, data()+nofs, *this);
      }

    T *data()
     { return const_cast<T *>(tbuf::d); }
    // read access to element #i
    template<typename I> T &raw(I i)
      { return data()[i]; }

    static vmav build_empty()
      {
      shape_t nshp;
      nshp.fill(0);
      return vmav(static_cast<T *>(nullptr), nshp);
      }

    static vmav build_noncritical(const shape_t &shape)
      {
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vmav tmp(shape2);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray<ndim>(slc);
      }
    static vmav build_noncritical(const shape_t &shape, uninitialized_dummy)
      {
      if (ndim<=1) return vmav(shape, UNINITIALIZED);
      auto shape2 = noncritical_shape(shape, sizeof(T));
      vmav tmp(shape2, UNINITIALIZED);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray<ndim>(slc);
      }
  };

template<size_t nd2, typename T, size_t ndim> vmav<T,nd2> subarray
  (vmav<T, ndim> &arr, const vector<slice> &slices)  
  { return arr.template subarray<nd2>(slices); }

}

using detail_mav::cfmav;
using detail_mav::vfmav;
using detail_mav::cmav;
using detail_mav::vmav;
}

#endif
