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
#if 0
template<typename T> class cmembuf
  {
  protected:
    shared_ptr<vector<T>> ptr;
    shared_ptr<aligned_array<T>> rawptr;
    const T *d;

    cmembuf(const T *d_, const cmembuf &other)
      : ptr(other.ptr), rawptr(other.rawptr), d(d_) {}

    // externally owned data pointer
    cmembuf(const T *d_)
      : d(d_) {}
    // share another memory buffer, but read-only
    cmembuf(const cmembuf &other)
      : ptr(other.ptr), rawptr(other.rawptr), d(other.d) {}
    cmembuf(size_t sz)
      : ptr(make_shared<vector<T>>(sz)), d(ptr->data()) {}
    cmembuf(size_t sz, uninitialized_dummy)
      : rawptr(make_shared<aligned_array<T>>(sz)), d(rawptr->data()) {}
    // take over another memory buffer
    cmembuf(cmembuf &&other) = default;

  public:
    cmembuf() = delete;
    void assign(const cmembuf &other)
      {
      ptr = other.ptr;
      rawptr = other.rawptr;
      d = other.d;
      }
    // read access to element #i
    template<typename I> const T &raw(I i) const
      { return d[i]; }
    // read access to data area
    const T *data() const
     { return d; }
  };
#endif

template<typename T> class cfmav: public fmav_info, public cmembuf<T>
  {
  protected:
    using tbuf = cmembuf<T>;
    using tinfo = fmav_info;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
//    cfmav() {}
    cfmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    cfmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
//    cfmav(const shape_t &shp_)
//      : tinfo(shp_), tbuf(size()) {}
    cfmav(const T* d_, const tinfo &info)
      : tinfo(info), tbuf(d_) {}
    cfmav(const cfmav &other) = default;
    cfmav(cfmav &&other) = default;

    cfmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    cfmav(const tbuf &buf, const tinfo &info)
      : tinfo(info), tbuf(buf) {}
//    cfmav(const shape_t &shp_, const stride_t &str_, const T *d_, tbuf &buf)
//      : tinfo(shp_, str_), tbuf(d_, buf) {}
    cfmav(const shape_t &shp_, const stride_t &str_, const T *d_, const tbuf &buf)
      : tinfo(shp_, str_), tbuf(d_, buf) {}
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

template<typename T> class vfmav: public cfmav<T>
  {
  protected:
    using tbuf = cmembuf<T>;
    using tinfo = fmav_info;
    using tinfo::shp, tinfo::str, tinfo::size;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;

  public:
    using tbuf::raw, tbuf::data, tinfo::ndim;
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
#if 0
    vfmav(const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(size())
      {
      ptrdiff_t ofs=0;
      for (size_t i=0; i<ndim(); ++i)
        ofs += (ptrdiff_t(shp[i])-1)*str[i];
      MR_assert(ofs+1==ptrdiff_t(size()), "array is not compact");
      }
    fmav(const T* d_, const tinfo &info)
      : tinfo(info), tbuf(d_) {}
    fmav(T* d_, const tinfo &info, bool rw_=false)
      : tinfo(info), tbuf(d_, rw_) {}
#if defined(_MSC_VER)
    // MSVC is broken
    fmav(const fmav &other) : tinfo(other), tbuf(other) {}
    fmav(fmav &other) : tinfo(other), tbuf(other) {}
    fmav(fmav &&other) : tinfo(other), tbuf(other) {}
#else
    /** Constructs a read-only fmav with the same shape and strides as \a other,
     *  pointing to the same memory. Ownership is shared. */
    fmav(const fmav &other) = default;
    /** Constructs an fmav with the same read-write status, shape and strides
     *  as \a other, pointing to the same memory. Ownership is shared. */
    fmav(fmav &other) = default;
    fmav(fmav &&other) = default;
#endif
    fmav(tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
//    fmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
//      : tinfo(shp_, str_), tbuf(buf) {}
    fmav(const shape_t &shp_, const stride_t &str_, const T *d_, tbuf &buf)
      : tinfo(shp_, str_), tbuf(d_, buf) {}
//    fmav(const shape_t &shp_, const stride_t &str_, const T *d_, const tbuf &buf)
//      : tinfo(shp_, str_), tbuf(d_, buf) {}

    void assign(vfmav &other)
      {
      fmav_info::assign(other);
      cmembuf<T>::assign(other);
      }

    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }
    template<typename... Ns> T &v(Ns... ns)
      { return const_cast<T *>(raw(idx(ns...))); }

    fmav subarray(const vector<slice> &slices)
      {
      auto [ninfo, nofs] = subdata(slices);
      return fmav(ninfo, tbuf::d+nofs, *this);
      }

    /** Returns a writable fmav with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  The array data is default-initialized. */
    static fmav build_noncritical(const shape_t &shape)
      {
      auto ndim = shape.size();
      auto shape2 = noncritical_shape(shape, sizeof(T));
      fmav tmp(shape2);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray(slc);
      }
    /** Returns a writable fmav with the specified shape.
     *  The strides are chosen in such a way that critical strides (multiples
     *  of 4096 bytes) along any dimension are avoided, by enlarging the
     *  allocated memory slightly if necessary.
     *  The array data is not initialized. */
    static fmav build_noncritical(const shape_t &shape, uninitialized_dummy)
      {
      auto ndim = shape.size();
      if (ndim<=1) return fmav(shape, UNINITIALIZED);
      auto shape2 = noncritical_shape(shape, sizeof(T));
      fmav tmp(shape2, UNINITIALIZED);
      vector<slice> slc(ndim);
      for (size_t i=0; i<ndim; ++i) slc[i] = slice(0, shape[i]);
      return tmp.subarray(slc);
      }
#endif
  };

}

using detail_mav::cfmav;
using detail_mav::vfmav;
}

#endif
