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

#ifndef DUCC0_MAV_H
#define DUCC0_MAV_H

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

namespace ducc0 {

namespace detail_mav {

using namespace std;

struct uninitialized_dummy {};
constexpr uninitialized_dummy UNINITIALIZED;

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
    cmembuf(): d(nullptr) {}
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

constexpr size_t MAXIDX=~(size_t(0));

struct slice
  {
  size_t lo, hi;
  slice() : lo(0), hi(MAXIDX) {}
  slice(size_t idx) : lo(idx), hi(idx) {}
  slice(size_t lo_, size_t hi_) : lo(lo_), hi(hi_) {}
  };

/// Helper class containing shape and stride information of an `fmav` object
class fmav_info
  {
  public:
    /// vector of nonnegative integers for storing the array shape
    using shape_t = vector<size_t>;
    /// vector of integers for storing the array strides
    using stride_t = vector<ptrdiff_t>;

  protected:
    shape_t shp;
    stride_t str;
    size_t sz;

    static stride_t shape2stride(const shape_t &shp)
      {
      auto ndim = shp.size();
      stride_t res(ndim);
      if (ndim==0) return res;
      res[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        res[ndim-i] = res[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      return res;
      }
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*ptrdiff_t(n) + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*ptrdiff_t(n); }
    ptrdiff_t getIdx(size_t /*dim*/) const
      { return 0; }

  public:
    /// Constructs a 1D object with all extents and strides set to zero.
    fmav_info() : shp(1,0), str(1,0), sz(0) {}
    /// Constructs an object with the given shape and stride.
    fmav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_),
        sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>()))
      {
      MR_assert(shp.size()==str.size(), "dimensions mismatch");
      }
    /// Constructs an object with the given shape and computes the strides
    /// automatically, assuming a C-contiguous memory layout.
    fmav_info(const shape_t &shape_)
      : fmav_info(shape_, shape2stride(shape_)) {}
    void assign(const fmav_info &other)
      {
      shp = other.shp;
      str = other.str;
      sz = other.sz;
      }
    /// Returns the dimensionality of the object.
    size_t ndim() const { return shp.size(); }
    /// Returns the total number of entries in the object.
    size_t size() const { return sz; }
    /// Returns the shape of the object.
    const shape_t &shape() const { return shp; }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return shp[i]; }
    /// Returns the strides of the object.
    const stride_t &stride() const { return str; }
    /// Returns the stride along dimension \a i.
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return ((ndim()==0) || (str.back()==1)); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
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
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const fmav_info &other) const
      { return shp==other.shp; }
    /// Returns the one-dimensional index of an entry from the given
    /// multi-dimensional index tuple, taking strides into account.
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      MR_assert(ndim()==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }
    /// Returns the common broadcast shape of *this and \a shp2
    shape_t bcast_shape(const shape_t &shp2) const
      {
      shape_t res(max(shp.size(), shp2.size()), 1);
      for (size_t i=0; i<shp.size(); ++i)
        res[i+res.size()-shp.size()] = shp[i];
      for (size_t i=0; i<shp2.size(); ++i)
        {
        size_t i2 = i+res.size()-shp2.size();
        if (res[i2]==1)
          res[i2] = shp2[i];
        else
          MR_assert((res[i2]==shp2[i])||(shp2[i]==1),
            "arrays cannot be broadcast together");
        }
      return res;
      }
    void bcast_to_shape(const shape_t &shp2)
      {
      MR_assert(shp2.size()>=shp.size(), "cannot reduce dimensionality");
      stride_t newstr(shp2.size(), 0);
      for (size_t i=0; i<shp.size(); ++i)
        {
        size_t i2 = i+shp2.size()-shp.size();
        if (shp[i]!=1)
          {
          MR_assert(shp[i]==shp2[i2], "arrays cannot be broadcast together");
          newstr[i2] = str[i];
          }
        }
      shp = shp2;
      str = newstr;
      }

    void swap_axes(size_t ax0, size_t ax1)
      {
      MR_assert(ax0<=ndim() && ax1<=ndim(), "bad axes");
      if (ax0==ax1) return;
      swap(shp[ax0], shp[ax1]);
      swap(str[ax0], str[ax1]);
      }

  protected:
    auto subdata(const vector<slice> &slices) const
      {
      auto ndim = shp.size();
      shape_t nshp(ndim);
      stride_t nstr(ndim);
      MR_assert(slices.size()==ndim, "incorrect number of slices");
      size_t n0=0;
      for (auto x:slices) if (x.lo==x.hi) ++n0;
      ptrdiff_t nofs=0;
      nshp.resize(ndim-n0);
      nstr.resize(ndim-n0);
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(slices[i].lo<shp[i], "bad subset");
        nofs+=slices[i].lo*str[i];
        if (slices[i].lo!=slices[i].hi)
          {
          auto ext = slices[i].hi-slices[i].lo;
          if (slices[i].hi==MAXIDX)
            ext = shp[i]-slices[i].lo;
          MR_assert(slices[i].lo+ext<=shp[i], "bad subset");
          nshp[i2]=ext; nstr[i2]=str[i];
          ++i2;
          }
        }
      return make_tuple(fmav_info(nshp, nstr), nofs);
      }
  };

/// Helper class containing shape and stride information of a `mav` object
template<size_t ndim> class mav_info
  {
  public:
    /// Fixed-size array of nonnegative integers for storing the array shape
    using shape_t = array<size_t, ndim>;
    /// Fixed-size array of integers for storing the array strides
    using stride_t = array<ptrdiff_t, ndim>;

  protected:
    shape_t shp;
    stride_t str;
    size_t sz;

    static stride_t shape2stride(const shape_t &shp)
      {
      stride_t res;
      if (ndim==0) return res;
      res[ndim-1]=1;
      for (size_t i=2; i<=ndim; ++i)
        res[ndim-i] = res[ndim-i+1]*ptrdiff_t(shp[ndim-i+1]);
      return res;
      }
    template<typename... Ns> ptrdiff_t getIdx(size_t dim, size_t n, Ns... ns) const
      { return str[dim]*n + getIdx(dim+1, ns...); }
    ptrdiff_t getIdx(size_t dim, size_t n) const
      { return str[dim]*n; }
    ptrdiff_t getIdx(size_t /*dim*/) const
      { return 0; }

  public:
    /// Constructs an object with all extents and strides set to zero.
    mav_info() : sz(0)
      {
      for (size_t i=0; i<ndim; ++i)
        { shp[i]=0; str[i]=0; }
      }
    /// Constructs an object with the given shape and stride.
    mav_info(const shape_t &shape_, const stride_t &stride_)
      : shp(shape_), str(stride_),
        sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>())) {}
    /// Constructs an object with the given shape and computes the strides
    /// automatically, assuming a C-contiguous memory layout.
    mav_info(const shape_t &shape_)
      : mav_info(shape_, shape2stride(shape_)) {}
    void assign(const mav_info &other)
      {
      shp = other.shp;
      str = other.str;
      sz = other.sz;
      }
    /// Returns the total number of entries in the object.
    size_t size() const { return sz; }
    /// Returns the shape of the object.
    const shape_t &shape() const { return shp; }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return shp[i]; }
    /// Returns the strides of the object.
    const stride_t &stride() const { return str; }
    /// Returns the stride along dimension \a i.
    const ptrdiff_t &stride(size_t i) const { return str[i]; }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return ((ndim==0) || (str.back()==1)); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
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
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const mav_info &other) const
      { return shp==other.shp; }
    /// Returns true iff this->shape and \a other match.
    bool conformable(const shape_t &other) const
      { return shp==other; }
    /// Returns the one-dimensional index of an entry from the given
    /// multi-dimensional index tuple, taking strides into account.
    template<typename... Ns> ptrdiff_t idx(Ns... ns) const
      {
      static_assert(ndim==sizeof...(ns), "incorrect number of indices");
      return getIdx(0, ns...);
      }

  protected:
    template<size_t nd2> auto subdata(const vector<slice> &slices) const
      {
      MR_assert(slices.size()==ndim, "bad number of slices");
      array<size_t, nd2> nshp;
      array<ptrdiff_t, nd2> nstr;

      // unnecessary, but gcc arns otherwise
      for (size_t i=0; i<nd2; ++i) nshp[i]=nstr[i]=0;

      size_t n0=0;
      for (auto x:slices) if (x.lo==x.hi) ++n0;
      MR_assert(n0+nd2==ndim, "bad extent");
      ptrdiff_t nofs=0;
      for (size_t i=0, i2=0; i<ndim; ++i)
        {
        MR_assert(slices[i].lo<shp[i], "bad subset");
        nofs+=slices[i].lo*str[i];
        if (slices[i].lo!=slices[i].hi)
          {
          auto ext = slices[i].hi-slices[i].lo;
          if (slices[i].hi==MAXIDX)
            ext = shp[i]-slices[i].lo;
          MR_assert(slices[i].lo+ext<=shp[i], "bad subset");
          nshp[i2]=ext; nstr[i2]=str[i];
          ++i2;
          }
        }
      return make_tuple(mav_info<nd2>(nshp, nstr), nofs);
      }
  };

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

    cfmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}

    void assign(const cfmav &other)
      {
      tinfo::assign(other);
      tbuf::assign(other);
      }

    /// Returns the data entry at the given set of indices.
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }

    cfmav subarray(const vector<slice> &slices) const
      {
      auto [ninfo, nofs] = subdata(slices);
      return cfmav(ninfo, tbuf::d+nofs, *this);
      }
  };

template<typename T> cfmav<T> subarray
  (const cfmav<T> &arr, const vector<slice> &slices)  
  { return arr.subarray(slices); }

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
    vfmav(tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : cfmav<T>(buf, shp_, str_) {}

    using cfmav<T>::data;
    T *data()
     { return const_cast<T *>(tbuf::d); }
    using cfmav<T>::raw;
    template<typename I> T &raw(I i)
      { return data()[i]; }

    void assign(vfmav &other)
      {
      fmav_info::assign(other);
      cmembuf<T>::assign(other);
      }

    using cfmav<T>::operator();
    template<typename... Ns> const T &operator()(Ns... ns) const
      { return raw(idx(ns...)); }

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
  { return arr.subarray(slices); }

template<typename T, size_t ndim> class cmav: public mav_info<ndim>, public cmembuf<T>
  {
  protected:
    template<typename T2, size_t nd2> friend class cmav;
    template<typename T2, size_t nd2> friend class vmav;

    using tinfo = mav_info<ndim>;
    using tbuf = cmembuf<T>;
    using tinfo::shp, tinfo::str;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

  protected:
    cmav() {}
    cmav(const shape_t &shp_, uninitialized_dummy)
      : tinfo(shp_), tbuf(size(), UNINITIALIZED) {}
    cmav(const shape_t &shp_)
      : tinfo(shp_), tbuf(size()) {}
    cmav(const tbuf &buf, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(buf) {}
    cmav(const tinfo &info, const T *d_, const tbuf &buf)
      : tinfo(info), tbuf(d_, buf) {}

  public:
    cmav(const T *d_, const shape_t &shp_, const stride_t &str_)
      : tinfo(shp_, str_), tbuf(d_) {}
    cmav(const T *d_, const shape_t &shp_)
      : tinfo(shp_), tbuf(d_) {}
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
      // Don't do this at home!
      shape_t tshp;
      tshp.fill(1);
      cmav tmp(tshp);
      const_cast<T &>(tmp.raw(0)) = value;
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
    template<typename T2, size_t nd2> friend class vmav;

    using parent = cmav<T, ndim>;
    using tinfo = mav_info<ndim>;
    using tbuf = cmembuf<T>;
    using tinfo::shp, tinfo::str;

  public:
    using typename tinfo::shape_t;
    using typename tinfo::stride_t;
    using tbuf::raw, tbuf::data;
    using tinfo::contiguous, tinfo::size, tinfo::idx, tinfo::conformable;

  protected:
    vmav(const tinfo &info, T *d_, tbuf &buf)
      : parent(info, d_, buf) {}

  public:
    vmav() {}
    vmav(T *d_, const shape_t &shp_, const stride_t &str_)
      : parent(d_, shp_, str_) {}
    vmav(T *d_, const shape_t &shp_)
      : parent(d_, shp_) {}
    vmav(const shape_t &shp_)
      : parent(shp_) {}
    vmav(const shape_t &shp_, uninitialized_dummy)
      : parent(shp_, UNINITIALIZED) {}

    void assign(vmav &other)
      { parent::assign(other); }
    operator vfmav<T>()
      {
      return vfmav<T>(*this, {shp.begin(), shp.end()}, {str.begin(), str.end()});
      }
    using parent::operator();
    template<typename... Ns> T &operator()(Ns... ns)
      { return const_cast<T &>(parent::operator()(ns...)); }

    template<size_t nd2> vmav<T,nd2> subarray(const vector<slice> &slices)
      {
      auto [ninfo, nofs] = tinfo::template subdata<nd2> (slices);
      return vmav<T,nd2> (ninfo, data()+nofs, *this);
      }

    using parent::data;
    T *data()
     { return const_cast<T *>(tbuf::d); }
    // read access to element #i
    using parent::raw;
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

// various operations involving fmav objects of the same shape -- experimental

DUCC0_NOINLINE void opt_shp_str(fmav_info::shape_t &shp, vector<fmav_info::stride_t> &str)
  {
  if (shp.size()>1)
    {
    // sort dimensions in order of descending stride, as far as possible
    vector<size_t> strcrit(shp.size(),0);
    for (const auto &curstr: str)
      for (size_t i=0; i<curstr.size(); ++i)
        strcrit[i] = (strcrit[i]==0) ?
          size_t(abs(curstr[i])) : min(strcrit[i],size_t(abs(curstr[i])));
  
    for (size_t lastdim=shp.size(); lastdim>1; --lastdim)
      {
      auto dim = size_t(min_element(strcrit.begin(),strcrit.begin()+lastdim)
                        -strcrit.begin());
      if (dim+1!=lastdim)
        {
        swap(strcrit[dim], strcrit[lastdim-1]);
        swap(shp[dim], shp[lastdim-1]);
        for (auto &curstr: str)
          swap(curstr[dim], curstr[lastdim-1]);
        }
      }
    // try merging dimensions
    size_t ndim = shp.size();
    if (ndim>1)
      for (size_t d0=ndim-2; d0+1>0; --d0)
        {
        bool can_merge = true;
        for (const auto &curstr: str)
          can_merge &= curstr[d0] == ptrdiff_t(shp[d0+1])*curstr[d0+1];
        if (can_merge)
          {
          for (auto &curstr: str)
            curstr.erase(curstr.begin()+d0);
          shp[d0+1] *= shp[d0];
          shp.erase(shp.begin()+d0);
          }
        }
    }
  }

DUCC0_NOINLINE auto multiprep(const vector<fmav_info> &info)
  {
  auto narr = info.size();
  MR_assert(narr>=1, "need at least one array");
  for (size_t i=1; i<narr; ++i)
    MR_assert(info[i].shape()==info[0].shape(), "shape mismatch");
  fmav_info::shape_t shp;
  vector<fmav_info::stride_t> str(narr);
  for (size_t i=0; i<info[0].ndim(); ++i)
    if (info[0].shape(i)!=1) // remove axes of length 1
      {
      shp.push_back(info[0].shape(i));
      for (size_t j=0; j<narr; ++j)
        str[j].push_back(info[j].stride(i));
      }
  opt_shp_str(shp, str);
  return make_tuple(shp, str);
  }

template<typename T0, typename Func>
  void applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      applyHelper(idim+1, shp, str, ptr0+i*str0, func);
  else
    for (size_t i=0; i<len; ++i)
      func(ptr0[i*str0]);
  }
template<typename T0, typename Func>
  void applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(*ptr0);
  else if (nthreads==1)
    applyHelper(0, shp, str, ptr0, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(ptr0[i*str[0][0]]);
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        applyHelper(1, shp, str, ptr0+i*str[0][0], func);
      });
  }
template<typename T0, typename T1, typename Func>
  void applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim], str1 = str[1][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      applyHelper(idim+1, shp, str, ptr0+i*str0, ptr1+i*str1, func);
  else
    for (size_t i=0; i<len; ++i)
      func(ptr0[i*str0], ptr1[i*str1]);
  }
template<typename T0, typename T1, typename Func>
  void applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1,
    Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(*ptr0, *ptr1);
  else if (nthreads==1)
    applyHelper(0, shp, str, ptr0, ptr1, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(ptr0[i*str[0][0]], ptr1[i*str[1][0]]);
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        applyHelper(1, shp, str, ptr0+i*str[0][0], ptr1+i*str[1][0], func);
      });
  }
template<typename T0, typename T1, typename T2, typename Func>
  void applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2, Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim], str1 = str[1][idim], str2 = str[2][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      applyHelper(idim+1, shp, str, ptr0+i*str0, ptr1+i*str1, ptr2+i*str2, func);
  else
    for (size_t i=0; i<len; ++i)
      func(ptr0[i*str0], ptr1[i*str1], ptr2[i*str2]);
  }
template<typename T0, typename T1, typename T2, typename Func>
  void applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2,
    Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(*ptr0, *ptr1, *ptr2);
  else if (nthreads==1)
    applyHelper(0, shp, str, ptr0, ptr1, ptr2, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(ptr0[i*str[0][0]], ptr1[i*str[1][0]], ptr2[i*str[2][0]]);
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        applyHelper(1, shp, str, ptr0+i*str[0][0], ptr1+i*str[1][0],
          ptr2+i*str[2][0], func);
      });
  }
template<typename T0, typename T1, typename T2, typename T3, typename Func>
  void applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2, T3 ptr3,
    Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim], str1 = str[1][idim], str2 = str[2][idim],
       str3 = str[3][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      applyHelper(idim+1, shp, str, ptr0+i*str0, ptr1+i*str1, ptr2+i*str2,
        ptr3+i*str3, func);
  else
    for (size_t i=0; i<len; ++i)
      func(ptr0[i*str0], ptr1[i*str1], ptr2[i*str2], ptr3[i*str3]);
  }
template<typename T0, typename T1, typename T2, typename T3, typename Func>
  void applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2, T3 ptr3,
    Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(*ptr0, *ptr1, *ptr2, *ptr3);
  else if (nthreads==1)
    applyHelper(0, shp, str, ptr0, ptr1, ptr2, ptr3, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(ptr0[i*str[0][0]], ptr1[i*str[1][0]], ptr2[i*str[2][0]],
          ptr3[i*str[3][0]]);
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        applyHelper(1, shp, str, ptr0+i*str[0][0], ptr1+i*str[1][0],
          ptr2+i*str[2][0], ptr3+i*str[3][0], func);
      });
  }

template<typename T0, typename Func>
  void mav_apply(Func func, int nthreads, T0 &&m0)
  {
  auto [shp, str] = multiprep({m0});
  applyHelper(shp, str, m0.data(), func, nthreads);
  }
template<typename T0, typename T1, typename Func>
  void mav_apply(Func func, int nthreads, T0 &&m0, T1 &&m1)
  {
  auto [shp, str] = multiprep({m0, m1});
  applyHelper(shp, str, m0.data(), m1.data(), func, nthreads);
  }
template<typename T0, typename T1, typename T2, typename Func>
  void mav_apply(Func func, int nthreads, T0 &&m0, T1 &&m1, T2 &&m2)
  {
  auto [shp, str] = multiprep({m0, m1, m2});
  applyHelper(shp, str, m0.data(), m1.data(), m2.data(), func, nthreads);
  }

template<typename T, size_t ndim> class mavref
  {
  private:
    const mav_info<ndim> &info;
    T *d;

  public:
    using shape_t = typename mav_info<ndim>::shape_t;
    using stride_t = typename mav_info<ndim>::stride_t;
    mavref(const mav_info<ndim> &info_, T *d_) : info(info_), d(d_) {}
    template<typename... Ns> T &operator()(Ns... ns) const
      { return d[info.idx(ns...)]; }
    /// Returns the total number of entries in the object.
    size_t size() const { return info.size(); }
    /// Returns the shape of the object.
    const shape_t &shape() const { return info.shape(); }
    /// Returns the length along dimension \a i.
    size_t shape(size_t i) const { return info.shape(i); }
    /// Returns the strides of the object.
    const stride_t &stride() const { return info.stride(); }
    /// Returns the stride along dimension \a i.
    const ptrdiff_t &stride(size_t i) const { return info.stride(i); }
    /// Returns true iff the last dimension has stride 1.
    /**  Typically used for optimization purposes. */
    bool last_contiguous() const
      { return info.last_contiguous(); }
    /** Returns true iff the object is C-contiguous, i.e. if the stride of the
     *  last dimension is 1, the stride for the next-to-last dimension is the
     *  shape of the last dimension etc. */
    bool contiguous() const
      { return info.contiguous(); }
    /// Returns true iff this->shape and \a other.shape match.
    bool conformable(const mavref &other) const
      { return shape()==other.shape(); }
  };
template<typename T, size_t ndim>
  mavref<T, ndim> make_mavref(const mav_info<ndim> &info_, T *d_)
  { return mavref<T, ndim>(info_, d_); }

template<size_t ndim> auto make_infos(const fmav_info &info)
  {
  MR_assert(ndim<=info.ndim(), "bad dimensionality");
  auto iterdim = info.ndim()-ndim;
  fmav_info fout({info.shape().begin(),info.shape().begin()+iterdim},
                 {info.stride().begin(),info.stride().begin()+iterdim});

  typename mav_info<ndim>::shape_t shp;
  typename mav_info<ndim>::stride_t str;
  if constexpr (ndim>0)  // just to silence compiler warnings
    for (size_t i=0; i<ndim; ++i)
      {
      shp[i] = info.shape(iterdim+i);
      str[i] = info.stride(iterdim+i);
      }
  mav_info<ndim> iout(shp, str);
  return make_tuple(fout, iout);
  }


template<typename T0, typename Ti0, typename Func>
  void flexible_mav_applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, const Ti0 &info0,
    Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      flexible_mav_applyHelper(idim+1, shp, str, ptr0+i*str0, info0, func);
  else
    for (size_t i=0; i<len; ++i)
      func(make_mavref(info0, ptr0+i*str0));
  }
template<typename T0, typename Ti0, typename Func>
  void flexible_mav_applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, const Ti0 &info0,
    Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(make_mavref(info0, ptr0));
  else if (nthreads==1)
    flexible_mav_applyHelper(0, shp, str, ptr0, info0, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(make_mavref(info0, ptr0+i*str[0][0]));
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        flexible_mav_applyHelper(1, shp, str, ptr0+i*str[0][0], info0, func);
      });
  }

template<typename T0, typename Ti0, typename T1, typename Ti1, typename Func>
  void flexible_mav_applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, const Ti0 &info0,
    T1 ptr1, const Ti1 &info1, Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim], str1 = str[1][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      flexible_mav_applyHelper(idim+1, shp, str, ptr0+i*str0, info0, ptr1+i*str1,
        info1, func);
  else
    for (size_t i=0; i<len; ++i)
      func(make_mavref(info0, ptr0+i*str0), make_mavref(info1, ptr1+i*str1));
  }
template<typename T0, typename Ti0,
         typename T1, typename Ti1, typename Func>
  void flexible_mav_applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, const Ti0 &info0,
    T1 ptr1, const Ti1 &info1, Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(mavref(info0, ptr0), mavref(info1, ptr1));
  else if (nthreads==1)
    flexible_mav_applyHelper(0, shp, str, ptr0, info0, ptr1, info1, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(make_mavref(info0, ptr0+i*str[0][0]),
             make_mavref(info1, ptr1+i*str[1][0]));
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        flexible_mav_applyHelper(1, shp, str, ptr0+i*str[0][0], info0,
          ptr1+i*str[1][0], info1, func);
      });
  }

template<typename T0, typename Ti0,
         typename T1, typename Ti1,
         typename T2, typename Ti2, typename Func>
  void flexible_mav_applyHelper(size_t idim, const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, const Ti0 &info0,
    T1 ptr1, const Ti1 &info1, T2 ptr2, const Ti2 &info2, Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim], str1 = str[1][idim], str2 = str[2][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      flexible_mav_applyHelper(idim+1, shp, str, ptr0+i*str0, info0, ptr1+i*str1, info1,
        ptr2+i*str2, info2, func);
  else
    for (size_t i=0; i<len; ++i)
      func(make_mavref(info0, ptr0+i*str0), make_mavref(info1, ptr1+i*str1),
        make_mavref(info2, ptr2+i*str2));
  }
template<typename T0, typename Ti0,
         typename T1, typename Ti1,
         typename T2, typename Ti2, typename Func>
  void flexible_mav_applyHelper(const vector<size_t> &shp,
    const vector<vector<ptrdiff_t>> &str, T0 ptr0, const Ti0 &info0,
    T1 ptr1, const Ti1 &info1, T2 ptr2, const Ti2 &info2, Func func,
    size_t nthreads)
  {
  if (shp.size()==0)
    func(mavref(info0, ptr0), mavref(info1, ptr1), mavref(info2, ptr2));
  else if (nthreads==1)
    flexible_mav_applyHelper(0, shp, str, ptr0, info0, ptr1, info1, ptr2, info2, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(make_mavref(info0, ptr0+i*str[0][0]),
             make_mavref(info1, ptr1+i*str[1][0]),
             make_mavref(info2, ptr2+i*str[2][0]));
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        flexible_mav_applyHelper(1, shp, str, ptr0+i*str[0][0], info0,
                        ptr1+i*str[1][0], info1, ptr2+i*str[2][0], info2, func);
      });
  }

template<size_t nd0, typename T0, typename Func>
  void flexible_mav_apply(Func func, size_t nthreads, T0 &&m0)
  {
  auto [f0, i0] = make_infos<nd0>(m0);
  vector<fmav_info> iterinfo{f0};
  auto [shp, str] = multiprep(iterinfo);
  flexible_mav_applyHelper(shp, str, m0.data(), i0, func, nthreads);
  }

template<size_t nd0, size_t nd1, typename T0, typename T1, typename Func>
  void flexible_mav_apply(Func func, size_t nthreads, T0 &&m0, T1 &&m1)
  {
  MR_assert(m0.ndim()-nd0 == m1.ndim()-nd1, "dimensionality mismatch");
  auto [f0, i0] = make_infos<nd0>(m0);
  auto [f1, i1] = make_infos<nd1>(m1);
  vector<fmav_info> iterinfo{f0, f1};
  auto [shp, str] = multiprep(iterinfo);
  flexible_mav_applyHelper(shp, str, m0.data(), i0, m1.data(), i1, func, nthreads);
  }

template<size_t nd0, size_t nd1, size_t nd2,
         typename T0, typename T1, typename T2, typename Func>
  void flexible_mav_apply(Func func, size_t nthreads, T0 &&m0, T1 &&m1, T2 &&m2)
  {
  MR_assert(m0.ndim()-nd0 == m1.ndim()-nd1, "dimensionality mismatch");
  MR_assert(m0.ndim()-nd0 == m2.ndim()-nd2, "dimensionality mismatch");
  auto [f0, i0] = make_infos<nd0>(m0);
  auto [f1, i1] = make_infos<nd1>(m1);
  auto [f2, i2] = make_infos<nd2>(m2);
  vector<fmav_info> iterinfo{f0, f1, f2};
  auto [shp, str] = multiprep(iterinfo);
  flexible_mav_applyHelper(shp, str, m0.data(), i0, m1.data(), i1, m2.data(), i2,
                  func, nthreads);
  }

}

using detail_mav::UNINITIALIZED;
using detail_mav::fmav_info;
using detail_mav::mav_info;
using detail_mav::slice;
using detail_mav::MAXIDX;
using detail_mav::cfmav;
using detail_mav::vfmav;
using detail_mav::cmav;
using detail_mav::vmav;
using detail_mav::subarray;
using detail_mav::mav_apply;
using detail_mav::flexible_mav_apply;
}

#endif
