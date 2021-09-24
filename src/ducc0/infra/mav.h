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
      : shp(shape_), str(stride_), sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>()))
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
      MR_assert(shp2.size()>=shp.size(), "cannot reduce dimensionallity");
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
    void prepend_dim()
      {
      shape_t shp2(shp.size()+1);
      stride_t str2(str.size()+1);
      shp2[0] = 1;
      str2[0] = 0;
      for (size_t i=0; i<shp.size(); ++i)
        {
        shp2[i+1] = shp[i];
        str2[i+1] = str[i];
        }
      shp = shp2;
      str = str2;
      }

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
      : shp(shape_), str(stride_), sz(accumulate(shp.begin(),shp.end(),size_t(1),multiplies<>())) {}
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
template<size_t nd> DUCC0_NOINLINE auto mav_multiprep(const vector<mav_info<nd>> &info)
  {
  auto narr = info.size();
  MR_assert(narr>=1, "need at least one array");
  for (size_t i=1; i<narr; ++i)
    MR_assert(info[i].shape()==info[0].shape(), "shape mismatch");
  fmav_info::shape_t shp;
  vector<fmav_info::stride_t> str(narr);
  for (size_t i=0; i<nd; ++i)
    if (info[0].shape(i)!=1) // remove axes of length 1
      {
      shp.push_back(info[0].shape(i));
      for (size_t j=0; j<narr; ++j)
        str[j].push_back(info[j].stride(i));
      }
  opt_shp_str(shp, str);
  return make_tuple(shp, str);
  }

template<typename T0, typename Func> void applyHelper(size_t idim, const vector<size_t> &shp,
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
template<typename T0, typename Func> void applyHelper(const vector<size_t> &shp,
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
template<typename T0, typename T1, typename Func> void applyHelper(size_t idim, const vector<size_t> &shp,
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
template<typename T0, typename T1, typename Func> void applyHelper(const vector<size_t> &shp,
  const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, Func func, size_t nthreads)
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
template<typename T0, typename T1, typename T2, typename Func> void applyHelper(size_t idim, const vector<size_t> &shp,
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
template<typename T0, typename T1, typename T2, typename Func> void applyHelper(const vector<size_t> &shp,
  const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2, Func func, size_t nthreads)
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
        applyHelper(1, shp, str, ptr0+i*str[0][0], ptr1+i*str[1][0], ptr2+i*str[2][0], func);
      });
  }
template<typename T0, typename T1, typename T2, typename T3, typename Func> void applyHelper(size_t idim, const vector<size_t> &shp,
  const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2, T3 ptr3, Func func)
  {
  auto len = shp[idim];
  auto str0 = str[0][idim], str1 = str[1][idim], str2 = str[2][idim], str3 = str[3][idim];
  if (idim+1<shp.size())
    for (size_t i=0; i<len; ++i)
      applyHelper(idim+1, shp, str, ptr0+i*str0, ptr1+i*str1, ptr2+i*str2, ptr3+i*str3, func);
  else
    for (size_t i=0; i<len; ++i)
      func(ptr0[i*str0], ptr1[i*str1], ptr2[i*str2], ptr3[i*str3]);
  }
template<typename T0, typename T1, typename T2, typename T3, typename Func> void applyHelper(const vector<size_t> &shp,
  const vector<vector<ptrdiff_t>> &str, T0 ptr0, T1 ptr1, T2 ptr2, T3 ptr3, Func func, size_t nthreads)
  {
  if (shp.size()==0)
    func(*ptr0, *ptr1, *ptr2, *ptr3);
  else if (nthreads==1)
    applyHelper(0, shp, str, ptr0, ptr1, ptr2, ptr3, func);
  else if (shp.size()==1)
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        func(ptr0[i*str[0][0]], ptr1[i*str[1][0]], ptr2[i*str[2][0]], ptr3[i*str[3][0]]);
      });
  else
    execParallel(shp[0], nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo; i<hi; ++i)
        applyHelper(1, shp, str, ptr0+i*str[0][0], ptr1+i*str[1][0], ptr2+i*str[2][0], ptr3+i*str[3][0], func);
      });
  }

template<typename T0, typename Func>
  void fmav_apply(Func func, int nthreads, T0 &&m0)
  {
  auto [shp, str] = multiprep({m0});
  applyHelper(shp, str, m0.data(), func, nthreads);
  }
template<typename T0, typename T1, typename Func>
  void fmav_apply(Func func, int nthreads, T0 &&m0, T1 &&m1)
  {
  auto [shp, str] = multiprep({m0, m1});
  applyHelper(shp, str, m0.data(), m1.data(), func, nthreads);
  }
template<typename T0, typename T1, typename T2, typename Func>
  void fmav_apply(Func func, int nthreads, T0 &&m0, T1 &&m1, T2 &&m2)
  {
  auto [shp, str] = multiprep({fmav_info(m0), fmav_info (m1), fmav_info (m2)});
  applyHelper(shp, str, m0.data(), m1.data(), m2.data(), func, nthreads);
  }
template<typename T0, typename T1, typename T2, typename T3, typename Func>  void fmav_apply(Func func, int nthreads, T0 &&m0, T1 &&m1, T2 &&m2, T3 &&m3)
  {
  auto [shp, str] = multiprep({m0, m1, m2, m3});
  applyHelper(shp, str, m0.data(), m1.data(), m2.data(), m3.data(), func, nthreads);
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

}

using detail_mav::UNINITIALIZED;
using detail_mav::fmav_info;
using detail_mav::mav_info;
using detail_mav::slice;
using detail_mav::MAXIDX;
using detail_mav::fmav_apply;
using detail_mav::mav_apply;
}

#endif
