/*
This file is part of the ducc FFT library.

Copyright (C) 2010-2025 Max-Planck-Society
Copyright (C) 2019 Peter Bell

Authors: Martin Reinecke, Peter Bell
*/

/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
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

#ifndef DUCC0_FFTND_IMPL_H
#define DUCC0_FFTND_IMPL_H

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <vector>
#include <complex>
#include <algorithm>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/math/unity_roots.h"
#include "ducc0/fft/fft1d_impl.h"

/** \file fftnd_impl.h
 *  Implementation of multi-dimensional Fast Fourier and related transforms
 *  \copyright Copyright (C) 2010-2021 Max-Planck-Society
 *  \copyright Copyright (C) 2019 Peter Bell
 *  \copyright
 *  \copyright For the odd-sized DCT-IV transforms:
 *  \copyright   Copyright (C) 2003, 2007-14 Matteo Frigo
 *  \copyright   Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * \authors Martin Reinecke, Peter Bell
 */

namespace DUCC0_NAMESPACE {

namespace detail_fft {

using namespace std;

namespace {

struct util // hack to avoid duplicate symbols
  {
  static void sanity_check_axes(size_t ndim, const shape_t &axes)
    {
    if (ndim==1)
      {
      if ((axes.size()!=1) || (axes[0]!=0))
        throw invalid_argument("bad axes");
      return;
      }
    shape_t tmp(ndim,0);
    if (axes.empty()) throw invalid_argument("no axes specified");
    for (auto ax : axes)
      {
      if (ax>=ndim) throw invalid_argument("bad axis number");
      if (++tmp[ax]>1) throw invalid_argument("axis specified repeatedly");
      }
    }

  DUCC0_NOINLINE static void sanity_check_onetype(const fmav_info &a1,
    const fmav_info &a2, bool inplace, const shape_t &axes)
    {
    MR_assert(a1.ndim()>0, "arrays must be at least 1D");
    sanity_check_axes(a1.ndim(), axes);
    for (auto ax : axes)
      MR_assert(a1.shape(ax)>0, "no zero-sized FFTs");
    MR_assert(a1.conformable(a2), "array sizes are not conformable");
    if (inplace) MR_assert(a1.stride()==a2.stride(), "stride mismatch");
    }
  DUCC0_NOINLINE static void sanity_check_cr(const fmav_info &ac,
    const fmav_info &ar, bool inplace, const size_t axis)
    {
    MR_assert(ac.ndim()>0, "arrays must be at least 1D");
    if (axis>=ac.ndim()) throw invalid_argument("bad axis number");
    MR_assert(ac.ndim()==ar.ndim(), "dimension mismatch");
    MR_assert(ar.shape(axis)>0, "no zero-sized FFTs");
    for (size_t i=0; i<ac.ndim(); ++i)
      {
      if (i==axis)
        {
        MR_assert(ac.shape(i) == ar.shape(i)/2+1, "axis length mismatch");
        if (inplace)
          {
          MR_assert((ac.shape(i)==1) || (ac.stride(i)==1),
            "stride on halfcomplex axis must be 1");
          MR_assert((ar.shape(i)==1) || (ar.stride(i)==1),
            "stride on halfcomplex axis must be 1");
          }
        }
      else
        {
        MR_assert(ac.shape(i) == ar.shape(i), "axis length mismatch");
        if (inplace)
          MR_assert((ac.shape(i)==1) || (2*ac.stride(i)==ar.stride(i)),
            "stride mismatch");
        }
      }
    }
  DUCC0_NOINLINE static void sanity_check_cr(const fmav_info &ac,
    const fmav_info &ar, bool inplace, const shape_t &axes)
    {
    sanity_check_axes(ac.ndim(), axes);
    sanity_check_cr(ac, ar, inplace, axes.back());
    for (auto ax : axes)
      MR_assert(ar.shape(ax)>0, "no zero-sized FFTs");
    }

  static size_t thread_count (size_t nthreads, const fmav_info &info,
    size_t axis, size_t vlen)
    {
    if (nthreads==1) return 1;
    size_t size = info.size();
    if (size<=32768) return 1;  // not worth opening a parallel region
    size_t max_parallel = size / (info.shape(axis)*vlen);
    size_t max_threads = adjust_nthreads(nthreads);
    return std::max(size_t(1), std::min(max_parallel, max_threads));
    }
  };

//
// multi-D infrastructure
//

template<typename T> shared_ptr<T> get_plan(size_t length, bool vectorize=false)
  {
#ifdef DUCC0_NO_FFT_CACHE
  return make_shared<T>(length, vectorize);
#else
  constexpr size_t nmax=10;
  struct entry { size_t n; bool vectorize; shared_ptr<T> ptr; };
  static array<entry, nmax> cache{{{0,0,nullptr}}};
  static array<size_t, nmax> last_access{{0}};
  static size_t access_counter = 0;
  static Mutex mut;

  auto find_in_cache = [&]() -> shared_ptr<T>
    {
    for (size_t i=0; i<nmax; ++i)
      if (cache[i].ptr && (cache[i].n==length) && (cache[i].vectorize==vectorize))
        {
        // no need to update if this is already the most recent entry
        if (last_access[i]!=access_counter)
          {
          last_access[i] = ++access_counter;
          // Guard against overflow
          if (access_counter == 0)
            last_access.fill(0);
          }
        return cache[i].ptr;
        }

    return nullptr;
    };

  {
  LockGuard lock(mut);

  auto p = find_in_cache();
  if (p) return p;
  }
  auto plan = make_shared<T>(length, vectorize);
  // building a plan costs up to half a transform at every length (SIMD
  // driver), so cache plans up to a total size
  constexpr size_t budget = size_t(1)<<28;
  if (plan->footprint() > budget)
    return plan;
  {
  LockGuard lock(mut);

  auto p = find_in_cache();
  if (p) return p;

  // evict least recently used plans until the new one fits
  size_t total = plan->footprint(), nused = 0;
  for (const auto &e: cache)
    if (e.ptr) { total += e.ptr->footprint(); ++nused; }
  while ((total>budget) || (nused==nmax))
    {
    size_t lru = nmax;
    for (size_t i=0; i<nmax; ++i)
      if (cache[i].ptr && ((lru==nmax) || (last_access[i]<last_access[lru])))
        lru = i;
    total -= cache[lru].ptr->footprint(); --nused;
    cache[lru] = {0, 0, nullptr};
    }
  size_t slot = 0;
  while (cache[slot].ptr) ++slot;
  cache[slot] = {length, vectorize, plan};
  last_access[slot] = ++access_counter;
  }
  return plan;
#endif
  }

template<size_t N> class multi_iter
  {
  private:
    // the dimensions other than the active one, all in one allocation
    struct dim_t { size_t shp, pos; ptrdiff_t str_i, str_o; };
    vector<dim_t> d;
    size_t cshp_i, cshp_o, rem;
    ptrdiff_t cstr_i, cstr_o, sstr_i, sstr_o, p_ii, p_i[N], p_oi, p_o[N];
    bool uni_i, uni_o;

    void advance_i()
      {
      for (auto &x: d)
        {
        p_ii += x.str_i;
        p_oi += x.str_o;
        if (++x.pos < x.shp)
          return;
        x.pos = 0;
        p_ii -= ptrdiff_t(x.shp)*x.str_i;
        p_oi -= ptrdiff_t(x.shp)*x.str_o;
        }
      }

  public:
    multi_iter(const fmav_info &iarr, const fmav_info &oarr, size_t idim,
      size_t nshares, size_t myshare)
      : rem(iarr.size()/iarr.shape(idim)), sstr_i(0), sstr_o(0), p_ii(0), p_oi(0)
      {
      MR_assert(oarr.ndim()==iarr.ndim(), "dimension mismatch");
      MR_assert(iarr.ndim()>=1, "not enough dimensions");
      MR_assert(idim<iarr.ndim(), "bad active dimension");
      d.reserve(iarr.ndim());
      for (size_t i=0; i<iarr.ndim(); ++i)
        if (i!=idim)
          {
          MR_assert(iarr.shape(i)==oarr.shape(i), "shape mismatch");
          d.push_back({iarr.shape(i), 0, iarr.stride(i), oarr.stride(i)});
          }
      // Sort the extraneous dimensions in order of ascending output stride;
      // this should improve overall cache re-use and avoid clashes between
      // threads as much as possible.
      sort(d.begin(), d.end(), [](const dim_t &a, const dim_t &b) { return a.str_o<b.str_o; });
      cstr_i = iarr.stride(idim);
      cstr_o = oarr.stride(idim);
      cshp_i = iarr.shape(idim);
      cshp_o = oarr.shape(idim);

      // collapse unneeded dimensions
      for (size_t i=1; i<d.size();)
        if ((d[i].str_i == d[i-1].str_i*ptrdiff_t(d[i-1].shp))
         && (d[i].str_o == d[i-1].str_o*ptrdiff_t(d[i-1].shp)))
          {
          d[i-1].shp *= d[i].shp;
          d.erase(d.begin()+ptrdiff_t(i));
          }
        else
          ++i;
      if (!d.empty())
        {
        sstr_i = d[0].str_i;
        sstr_o = d[0].str_o;
        }

      if (nshares==1) return;
      if (nshares==0) throw runtime_error("can't run with zero threads");
      if (myshare>=nshares) throw runtime_error("impossible share requested");
      auto [lo, hi] = calcShare(nshares, myshare, rem);
      size_t todo = hi-lo;

      // this shouldn't normally happen, but make sure we handle it properly
      if (todo==0)
        { rem=0; return; }

      size_t chunk = rem;
      for (size_t i2=0, i=d.size()-1; i2<d.size(); ++i2,--i)
        {
        chunk /= d[i].shp;
        size_t n_advance = lo/chunk;
        d[i].pos += n_advance;
        p_ii += ptrdiff_t(n_advance)*d[i].str_i;
        p_oi += ptrdiff_t(n_advance)*d[i].str_o;
        lo -= n_advance*chunk;
        }
      MR_assert(lo==0, "must not happen");
      rem = todo;
      }
    void advance(size_t n)
      {
      if (rem<n) throw runtime_error("underrun");
      for (size_t i=0; i<n; ++i)
        {
        p_i[i] = p_ii;
        p_o[i] = p_oi;
        advance_i();
        }
      uni_i = uni_o = true;
      for (size_t i=1; i<n; ++i)
        {
        uni_i = uni_i && (p_i[i]-p_i[i-1] == sstr_i);
        uni_o = uni_o && (p_o[i]-p_o[i-1] == sstr_o);
        }
      rem -= n;
      }
    ptrdiff_t iofs(size_t i) const { return p_i[0] + ptrdiff_t(i)*cstr_i; }
    ptrdiff_t iofs(size_t j, size_t i) const { return p_i[j] + ptrdiff_t(i)*cstr_i; }
    ptrdiff_t iofs_uni(size_t j, size_t i) const { return p_i[0] + ptrdiff_t(j)*sstr_i + ptrdiff_t(i)*cstr_i; }
    ptrdiff_t oofs(size_t i) const { return p_o[0] + ptrdiff_t(i)*cstr_o; }
    ptrdiff_t oofs(size_t j, size_t i) const { return p_o[j] + ptrdiff_t(i)*cstr_o; }
    ptrdiff_t oofs_uni(size_t j, size_t i) const { return p_o[0] + ptrdiff_t(j)*sstr_o + ptrdiff_t(i)*cstr_o; }
    bool uniform_i() const { return uni_i; }
    ptrdiff_t unistride_i() const { return sstr_i; }
    bool uniform_o() const { return uni_o; }
    ptrdiff_t unistride_o() const { return sstr_o; }
    size_t length_in() const { return cshp_i; }
    size_t length_out() const { return cshp_o; }
    ptrdiff_t stride_in() const { return cstr_i; }
    ptrdiff_t stride_out() const { return cstr_o; }
    size_t remaining() const { return rem; }
    bool critical_stride_trans(size_t tsz) const
      {
      return ((abs<ptrdiff_t>(stride_in() *tsz)&4095)==0)
          || ((abs<ptrdiff_t>(stride_out()*tsz)&4095)==0);
      }
    bool critical_stride_other(size_t tsz) const
      {
      if (unistride_i()==0) return false;  // it's just one transform
      return ((abs<ptrdiff_t>(unistride_i()*tsz)&4095)==0)
          || ((abs<ptrdiff_t>(unistride_o()*tsz)&4095)==0);
      }
  };

template<typename T, typename T0> class TmpStorage
  {
  private:
    aligned_array<T> d;
    size_t dofs, dstride;

  public:
    TmpStorage(size_t n_trafo, size_t bufsize_data, size_t bufsize_trafo,
               size_t n_simultaneous, bool inplace)
      {
      if (inplace)
        {
        d.realloc(bufsize_trafo);
        return;
        }
      constexpr auto vlen = native_simd<T0>::size();
      size_t buffct = std::min(vlen, n_trafo);
      size_t datafct = std::min(vlen, n_trafo);
      if (n_trafo>=n_simultaneous*vlen) datafct = n_simultaneous*vlen;
      dstride = bufsize_data;
      dofs = bufsize_trafo;
      // critical stride avoidance
      if ((dstride!=0)&&((dstride&256)==0)) dstride+=16;
      if ((dofs!=0)&&((dofs&256)==0)) dofs += 16;
      d.realloc(buffct*dofs + datafct*dstride);
      }

    template<typename T2> T2 *transformBuf()
      { return reinterpret_cast<T2 *>(d.data()); }
    template<typename T2> T2 *dataBuf()
      { return reinterpret_cast<T2 *>(d.data()) + dofs; }
    size_t data_stride() const
      { return dstride; }
  };

template<typename T2, typename T, typename T0> class TmpStorage2
  {
  private:
    TmpStorage<T, T0> &stg;

  public:
    using datatype = T2;
    TmpStorage2(TmpStorage<T,T0> &stg_): stg(stg_) {}

    T2 *transformBuf() { return stg.template transformBuf<T2>(); }
    T2 *dataBuf() { return stg.template dataBuf<T2>(); }
    size_t data_stride() const { return stg.data_stride(); }
  };

/* Gather nvec groups of vlen lines of the input into SIMD buffers (group g at
   dst+g*vstr, lane l of element i = element i of line g*vlen+l) and scatter
   back. Three cases: lines adjacent in memory (plain vector loads/stores),
   contiguous lines (vlen x vlen register transposes), general strides
   (lane by lane). */
template <simd_like Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<Cmplx<typename Tsimd::value_type>> &src, Cmplx<Tsimd> * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  const Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = src.data();
  const size_t len = it.length_in();
  if (it.uniform_i() && (it.unistride_i()==1))
    {
    for (size_t i=0; i<len; ++i)
      for (size_t j0=0; j0<nvec; ++j0)
        dst[j0*vstr+i] = vload<Tsimd>(ptr+it.iofs_uni(j0*vlen,i));
    return;
    }
  size_t i=0;
  if (it.stride_in()==1)
    for (; i+vlen<=len; i+=vlen)
      for (size_t j0=0; j0<nvec; ++j0)
        {
        array<Cmplx<Tsimd>,vlen> v;
        for (size_t j1=0; j1<vlen; ++j1) v[j1] = vload<Tsimd>(ptr+it.iofs(j0*vlen+j1,i));
        transpose(v);
        for (size_t k=0; k<vlen; ++k) dst[j0*vstr+i+k] = v[k];
        }
  for (; i<len; ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      for (size_t j1=0; j1<vlen; ++j1)
        {
        dst[j0*vstr+i].r[j1] = ptr[it.iofs(j0*vlen+j1,i)].r;
        dst[j0*vstr+i].i[j1] = ptr[it.iofs(j0*vlen+j1,i)].i;
        }
  }
template <simd_like Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<typename Tsimd::value_type> &src, Tsimd * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  const typename Tsimd::value_type * DUCC0_RESTRICT ptr = src.data();
  const size_t len = it.length_in();
  if (it.uniform_i() && (it.unistride_i()==1))
    {
    for (size_t i=0; i<len; ++i)
      for (size_t j0=0; j0<nvec; ++j0)
        dst[j0*vstr+i] = Tsimd(ptr+it.iofs_uni(j0*vlen,i), element_aligned_tag());
    return;
    }
  size_t i=0;
  if (it.stride_in()==1)
    for (; i+vlen<=len; i+=vlen)
      for (size_t j0=0; j0<nvec; ++j0)
        {
        array<Tsimd,vlen> v;
        for (size_t j1=0; j1<vlen; ++j1) v[j1] = Tsimd(ptr+it.iofs(j0*vlen+j1,i), element_aligned_tag());
        transpose(v);
        for (size_t k=0; k<vlen; ++k) dst[j0*vstr+i+k] = v[k];
        }
  for (; i<len; ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      for (size_t j1=0; j1<vlen; ++j1)
        dst[j0*vstr+i][j1] = ptr[it.iofs(j0*vlen+j1,i)];
  }
template <typename T, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<T> &src, T * DUCC0_RESTRICT dst, size_t nvec, size_t vstr)
  {
  const T * DUCC0_RESTRICT ptr = src.data();
  if ((nvec==1) && (dst == &src.raw(it.iofs(0)))) return;  // in-place
  const size_t len = it.length_in();
  if (it.stride_in()==1)  // line by line, so that the copies vectorize
    for (size_t j0=0; j0<nvec; ++j0)
      std::copy_n(ptr+it.iofs(j0,0), len, dst+j0*vstr);
  else
    for (size_t i=0; i<len; ++i)
      for (size_t j0=0; j0<nvec; ++j0)
        dst[j0*vstr+i] = ptr[it.iofs(j0,i)];
  }
template <typename Tbuf, typename Titer, typename T> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<T> &src, Tbuf * DUCC0_RESTRICT dst)
  { copy_input(it, src, dst, 1, 0); }

template<simd_like Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Cmplx<Tsimd> * DUCC0_RESTRICT src, const vfmav<Cmplx<typename Tsimd::value_type>> &dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = dst.data();
  const size_t len = it.length_out();
  if (it.uniform_o() && (it.unistride_o()==1))
    {
    for (size_t i=0; i<len; ++i)
      for (size_t j0=0; j0<nvec; ++j0)
        vstore(src[j0*vstr+i], ptr+it.oofs_uni(j0*vlen,i));
    return;
    }
  size_t i=0;
  if (it.stride_out()==1)
    for (; i+vlen<=len; i+=vlen)
      for (size_t j0=0; j0<nvec; ++j0)
        {
        array<Cmplx<Tsimd>,vlen> v;
        for (size_t k=0; k<vlen; ++k) v[k] = src[j0*vstr+i+k];
        transpose(v);
        for (size_t j1=0; j1<vlen; ++j1) vstore(v[j1], ptr+it.oofs(j0*vlen+j1,i));
        }
  for (; i<len; ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      for (size_t j1=0; j1<vlen; ++j1)
        ptr[it.oofs(j0*vlen+j1,i)].Set(src[j0*vstr+i].r[j1],src[j0*vstr+i].i[j1]);
  }
template<simd_like Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Tsimd * DUCC0_RESTRICT src, const vfmav<typename Tsimd::value_type> &dst, size_t nvec, size_t vstr)
  {
  constexpr auto vlen=Tsimd::size();
  typename Tsimd::value_type * DUCC0_RESTRICT ptr = dst.data();
  const size_t len = it.length_out();
  if (it.uniform_o() && (it.unistride_o()==1))
    {
    for (size_t i=0; i<len; ++i)
      for (size_t j0=0; j0<nvec; ++j0)
        src[j0*vstr+i].copy_to(ptr+it.oofs_uni(j0*vlen,i), element_aligned_tag());
    return;
    }
  size_t i=0;
  if (it.stride_out()==1)
    for (; i+vlen<=len; i+=vlen)
      for (size_t j0=0; j0<nvec; ++j0)
        {
        array<Tsimd,vlen> v;
        for (size_t k=0; k<vlen; ++k) v[k] = src[j0*vstr+i+k];
        transpose(v);
        for (size_t j1=0; j1<vlen; ++j1) v[j1].copy_to(ptr+it.oofs(j0*vlen+j1,i), element_aligned_tag());
        }
  for (; i<len; ++i)
    for (size_t j0=0; j0<nvec; ++j0)
      for (size_t j1=0; j1<vlen; ++j1)
        ptr[it.oofs(j0*vlen+j1,i)] = src[j0*vstr+i][j1];
  }
template<typename T, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const T * DUCC0_RESTRICT src, const vfmav<T> &dst, size_t nvec, size_t vstr)
  {
  T * DUCC0_RESTRICT ptr = dst.data();
  if ((nvec==1) && (src == &dst.raw(it.oofs(0)))) return;  // in-place
  const size_t len = it.length_out();
  if (it.stride_out()==1)  // line by line, so that the copies vectorize
    for (size_t j0=0; j0<nvec; ++j0)
      std::copy_n(src+j0*vstr, len, ptr+it.oofs(j0,0));
  else
    for (size_t i=0; i<len; ++i)
      for (size_t j0=0; j0<nvec; ++j0)
        ptr[it.oofs(j0,i)] = src[j0*vstr+i];
  }
template<typename Tbuf, typename Titer, typename T> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Tbuf * DUCC0_RESTRICT src, const vfmav<T> &dst)
  { copy_output(it, src, dst, 1, 0); }

template <typename T, size_t vlen> struct add_vec
  { using type = typename simd_select<T, vlen>::type; };
template <typename T, size_t vlen> struct add_vec<Cmplx<T>, vlen>
  { using type = Cmplx<typename simd_select<T, vlen>::type>; };
template <typename T, size_t vlen> using add_vec_t = typename add_vec<T, vlen>::type;

template<typename Tplan, typename T, typename T0, typename Exec>
DUCC0_NOINLINE void general_nd(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, T0 fct, size_t nthreads, const Exec &exec,
  const bool /*allow_inplace*/=true)
  {
  if ((in.ndim()==1)&&(in.stride(0)==1)&&(out.stride(0)==1))
    {
    auto plan = get_plan<Tplan>(in.shape(0), true);
    exec.exec_simple(in.data(), out.data(), *plan, fct, nthreads);
    return;
    }
  shared_ptr<Tplan> plan, vplan;
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;

  for (size_t iax=0; iax<axes.size(); ++iax)
    {
    size_t len=in.shape(axes[iax]);
    if ((!plan) || (len!=plan->length()))
      {
      plan = get_plan<Tplan>(len, in.ndim()==1);
      vplan = (in.ndim()==1) ? plan : get_plan<Tplan>(len, true);
      }

    auto work = [&](Scheduler &sched)
      {
      constexpr auto vlen = native_simd<T0>::size();
      constexpr size_t nmax = 64;
      const auto &tin(iax==0? in : out);
      multi_iter<nmax> it(tin, out, axes[iax], sched.num_threads(), sched.thread_num());

      // n_simul: vector size
      // n_bunch: total size of bunch (multiple of n_simul)
      size_t n_simul=1, n_bunch=1;
      bool critstride = (((in.stride(axes[iax])*sizeof(T))&4095)==0)
                     || (((out.stride(axes[iax])*sizeof(T))&4095)==0);
      bool nostride = (in.stride(axes[iax])==1) && (out.stride(axes[iax])==1);

      constexpr size_t l2cache=size_t(1)<<20;
      constexpr size_t cacheline=64;

      // working set size
      auto wss = [&](size_t vl) { return sizeof(T)*(2*len*vl + plan->bufsize()); };
      // is the FFT small enough to fit into L2 vectorized?
      if (wss(1)>l2cache) // "long" FFT, don't execute more than one at the same time
        {
        n_simul=1;
        if (critstride)  // make bunch large to reduce overall copy cost
          {
          n_bunch=n_simul;
          while ((n_bunch<nmax) && (sizeof(T)*n_bunch<2*cacheline)) n_bunch*=2;
          }
        else if (nostride)  // simple scalar "in-place" transform
          n_bunch=n_simul;
        else  // we have some strides, use a medium-sized bunch
          {
          n_bunch=n_simul;
          while ((n_bunch<nmax) && (sizeof(T)*n_bunch<cacheline)) n_bunch*=2;
          }
        }
      else  // fairly small individual FFT, vectorizing probably beneficial
        {
        // if no stride, only vectorize up to a total size that still fits into L2
        // if strided, always vectorize (TBC)
        if (nostride)
          {
          n_simul = vlen;
          while((n_simul>1) && (wss(n_simul)>l2cache)) n_simul /= 2;
          }
        else
          n_simul = vlen;
        // lines at a critical stride (a multiple of 4 KB apart): gather as many
        // as fill a tile of the L2 size (at least 16), so that the input is
        // read in long contiguous pieces; with several threads and an array
        // that fits into L3, a quarter of that, so that the tiles do not push
        // the array out. At other strides larger bunches were slower.
        constexpr size_t l3cache=size_t(16)<<20;
        size_t arraybytes = sizeof(T)*in.size()*((tin.data()==out.data()) ? 1 : 2);
        size_t tile = ((sched.num_threads()>1) && (arraybytes<=l3cache)) ? l2cache/4 : l2cache;
        n_bunch = critstride ? max<size_t>(16, min(nmax, tile/(sizeof(T)*len)))/n_simul*n_simul
                             : n_simul;
        }

      // Contiguous complex lines that have a SIMD single-transform plan (length a
      // multiple of the SIMD width, >=128) are transformed in place one at a time:
      // faster than SIMD over lines, which needs register transposes here.
      constexpr size_t vl1 = fft1d_simdlen<T0>;
      if (nostride && is_same_v<T,Cmplx<T0>> && (len%vl1==0) && (len>=max<size_t>(128,vl1*vl1)))
        n_simul = n_bunch = 1;
      bool inplace = (in.stride(axes[iax])==1) && (out.stride(axes[iax])==1) && (n_bunch==1);
      MR_assert(n_bunch<=nmax, "must not happen");
      TmpStorage<T,T0> storage(in.size()/len, len, max(plan->bufsize(),vplan->bufsize()), (n_bunch+vlen-1)/vlen, inplace);

      // first, do all possible steps of size n_bunch, then n_simul
      if (n_bunch>1)
        {
#ifndef DUCC0_NO_SIMD
        if constexpr (vlen>1)
          {
          constexpr size_t lvlen = vlen;
          if (n_simul>=lvlen)
            {
            if ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=n_bunch)
                {
                it.advance(n_bunch);
                exec.exec_n(it, tin, out, storage2, *plan, fct, n_bunch/lvlen, nth1d);
                }
              }
            }
          if (n_simul==lvlen)
            {
            if (it.remaining()>=lvlen)
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=lvlen)
                {
                it.advance(lvlen);
                exec(it, tin, out, storage2, *plan, fct, nth1d);
                }
              }
            }
          }
        if constexpr ((vlen>2) && (simd_exists<T0,vlen/2>))
          {
          constexpr size_t lvlen = vlen/2;
          if (n_simul>=lvlen)
            {
            if ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=n_bunch)
                {
                it.advance(n_bunch);
                exec.exec_n(it, tin, out, storage2, *plan, fct, n_bunch/lvlen, nth1d);
                }
              }
            }
          if (n_simul==lvlen)
            {
            if (it.remaining()>=lvlen)
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=lvlen)
                {
                it.advance(lvlen);
                exec(it, tin, out, storage2, *plan, fct, nth1d);
                }
              }
            }
          }
        if constexpr ((vlen>4) && (simd_exists<T0,vlen/4>))
          {
          constexpr size_t lvlen = vlen/4;
          if (n_simul>=lvlen)
            {
            if ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=n_bunch)
                {
                it.advance(n_bunch);
                exec.exec_n(it, tin, out, storage2, *plan, fct, n_bunch/lvlen, nth1d);
                }
              }
            }
          if (n_simul==lvlen)
            {
            if (it.remaining()>=lvlen)
              {
              TmpStorage2<add_vec_t<T, lvlen>,T,T0> storage2(storage);
              while (it.remaining()>=lvlen)
                {
                it.advance(lvlen);
                exec(it, tin, out, storage2, *plan, fct, nth1d);
                }
              }
            }
          }
#endif
        {
        TmpStorage2<T,T,T0> storage2(storage);
        while ((n_bunch>n_simul) && (it.remaining()>=n_bunch))
          {
          it.advance(n_bunch);
          exec.exec_n(it, tin, out, storage2, *vplan, fct, n_bunch, nth1d);
          }
        }
        }
        {
        TmpStorage2<T,T,T0> storage2(storage);
        while (it.remaining()>0)
          {
          it.advance(1);
          exec(it, tin, out, storage2, *vplan, fct, nth1d, inplace);
          }
        }
      };
    // passed by reference: the closure is too big for std::function's inline storage
    execParallel(util::thread_count(nthreads, in, axes[iax], native_simd<T0>::size()), std::ref(work));
    fct = T0(1); // factor has been applied, use 1 for remaining axes
    }
  }

struct ExecC2C
  {
  bool forward;

  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    const vfmav<Cmplx<T0>> &out, Tstorage &storage, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads, bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<Cmplx<T0>, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, forward, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    const vfmav<Cmplx<T0>> &out, Tstorage &storage, const pocketfft_c<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, forward, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const Cmplx<T0> *in, Cmplx<T0> *out, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, forward, nthreads);
    }
  };

/* ExecC2C, with output k of every line multiplied by tw[k] afterwards (the
   twiddles of the first step of c2c_fourstep) */
struct ExecC2CTw
  {
  bool forward;
  const void *tw;  // const Cmplx<T0> *

  template <typename T0, typename T> DUCC0_ALWAYS_INLINE void twiddle(T *p, size_t n) const
    {
    auto w = static_cast<const Cmplx<T0> *>(tw);
    if (forward)
      for (size_t k=0; k<n; ++k) p[k] = p[k].template special_mul<true>(w[k]);
    else
      for (size_t k=0; k<n; ++k) p[k] = p[k].template special_mul<false>(w[k]);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    const vfmav<Cmplx<T0>> &out, Tstorage &storage, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads, bool /*inplace*/=false) const
    {
    using T = typename Tstorage::datatype;
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    twiddle<T0>(res, it.length_out());
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    const vfmav<Cmplx<T0>> &out, Tstorage &storage, const pocketfft_c<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      {
      plan.exec_copyback(buf2+i*dstr, buf1, fct, forward, nthreads);
      twiddle<T0>(buf2+i*dstr, it.length_out());
      }
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const Cmplx<T0> *in, Cmplx<T0> *out, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, forward, nthreads);
    twiddle<T0>(out, plan.length());
    }
  };

struct [[maybe_unused]] ExecHartley
  {
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_hartley<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_hartley<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_hartley<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, nthreads);
    }
  };

struct [[maybe_unused]] ExecFHT
  {
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_fht<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_fht<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_fht<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, nthreads);
    }
  };

struct [[maybe_unused]] ExecFFTW
  {
  bool forward;

  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_fftw<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, forward, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_fftw<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, forward, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_fftw<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, forward, nthreads);
    }
  };

struct ExecDcst
  {
  bool ortho;
  int type;
  bool cosine;

  template <typename T0, typename Tstorage, typename Tplan, typename Titer>
  DUCC0_NOINLINE void operator() (const Titer &it, const cfmav<T0> &in,
    const vfmav <T0> &out, Tstorage &storage, const Tplan &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data()+it.oofs(0));
        plan.exec_copyback(out.data()+it.oofs(0), storage.transformBuf(), fct, ortho, type, cosine, nthreads);
        return;
        }
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, ortho, type, cosine, nthreads);
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Tplan, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const Tplan &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, ortho, type, cosine, nthreads);
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0, typename Tplan> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const Tplan &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    plan.exec(out, fct, ortho, type, cosine, nthreads);
    }
  };

/* r2c and c2r along one axis. Lines are gathered into buffers as in
   general_nd (SIMD over lv lines, at a critical stride in L2-sized tiles),
   and the halfcomplex <-> complex repacking happens in the buffers, so that
   the copies use the fast paths of copy_input/copy_output. The data buffer of
   a group holds the real line (len) followed by the complex line (lenc). */
template<bool r2c, typename T, typename Tin, typename Tout> DUCC0_NOINLINE void general_r2c_c2r(
  const cfmav<Tin> &in, const vfmav<Tout> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;
  size_t len = r2c ? in.shape(axis) : out.shape(axis), lenc=len/2+1;
  auto plan = get_plan<pocketfft_r<T>>(len, in.ndim()==1);
  auto work = [&](Scheduler &sched) {
    constexpr auto vlen = native_simd<T>::size();
    constexpr size_t nmax = 64, l2cache=size_t(1)<<20, l3cache=size_t(16)<<20;
    size_t blen = len+(len&1)+2*lenc;  // even, so that the complex part has an integral stride
    bool critstride = (((in.stride(axis)*sizeof(Tin))&4095)==0)
                   || (((out.stride(axis)*sizeof(Tout))&4095)==0);
    size_t arraybytes = sizeof(Tin)*in.size() + sizeof(Tout)*out.size();
    size_t tile = ((sched.num_threads()>1) && (arraybytes<=l3cache)) ? l2cache/4 : l2cache;
    size_t n_bunch = critstride ? max<size_t>(16, min(nmax, tile/(sizeof(T)*len)))/vlen*vlen : vlen;
    TmpStorage<T,T> storage(in.size()/in.shape(axis), blen, plan->bufsize(), (n_bunch+vlen-1)/vlen, false);
    multi_iter<nmax> it(in, out, axis, sched.num_threads(), sched.thread_num());
    // one bunch of n_bunch lines, or one group of lv lines (lv=1: a single line)
    auto lines = [&](auto lvc) -> bool
      {
      constexpr size_t lv = decltype(lvc)::value;
      if constexpr ((lv==0) || ((lv>1) && !simd_exists<T,lv>)) return false;
      else
        {
        using T2 = typename conditional_t<(lv>1), add_vec<T,lv>, type_identity<T>>::type;
        if ((lv>1) && (it.remaining()<lv)) return false;
        TmpStorage2<T2,T,T> st(storage);
        T2 *dbuf=st.dataBuf(), *tbuf=st.transformBuf();
        size_t dstr=st.data_stride(), n = ((lv==vlen) && (it.remaining()>=n_bunch)) ? n_bunch : lv, nvec=n/lv;
        auto cbuf = reinterpret_cast<Cmplx<T2> *>(dbuf+len+(len&1));
        it.advance(n);
        if constexpr (r2c)
          {
          if constexpr (lv==1)  // a single contiguous line: repack straight into the array
            if (it.stride_out()==1) cbuf = &out.raw(it.oofs(0));
          copy_input(it, in, dbuf, nvec, dstr);
          for (size_t k=0; k<nvec; ++k)
            {
            auto res = plan->exec(dbuf+k*dstr, tbuf, fct, true, nth1d);
            auto cb = cbuf+k*(dstr/2);
            cb[0].Set(res[0], T2(0));
            size_t i=1, ii=1;
            if (forward)
              for (; i<len-1; i+=2, ++ii) cb[ii].Set(res[i], res[i+1]);
            else
              for (; i<len-1; i+=2, ++ii) cb[ii].Set(res[i], -res[i+1]);
            if (i<len) cb[ii].Set(res[i], T2(0));
            }
          copy_output(it, cbuf, out, nvec, dstr/2);
          }
        else
          {
          const Cmplx<T2> *cin = cbuf;  // a single contiguous line: repack straight from the array
          if constexpr (lv==1)
            if (it.stride_in()==1) cin = &in.raw(it.iofs(0));
          if (cin==cbuf) copy_input(it, in, cbuf, nvec, dstr/2);
          for (size_t k=0; k<nvec; ++k)
            {
            auto cb = cin+k*(dstr/2);
            auto db = dbuf+k*dstr;
            db[0] = cb[0].r;
            size_t i=1, ii=1;
            if (forward)
              for (; i<len-1; i+=2, ++ii) { db[i]=cb[ii].r; db[i+1]=-cb[ii].i; }
            else
              for (; i<len-1; i+=2, ++ii) { db[i]=cb[ii].r; db[i+1]=cb[ii].i; }
            if (i<len) db[i] = cb[ii].r;
            if (nvec==1)  // copy the result out from wherever the plan left it
              dbuf = plan->exec(db, tbuf, fct, false, nth1d);
            else
              plan->exec_copyback(db, tbuf, fct, false, nth1d);
            }
          copy_output(it, dbuf, out, nvec, dstr);
          }
        return true;
        }
      };
    while (it.remaining()>0)
      {
#ifndef DUCC0_NO_SIMD
      if (lines(integral_constant<size_t,vlen>())) continue;
      if (lines(integral_constant<size_t,vlen/2>())) continue;
      if (lines(integral_constant<size_t,vlen/4>())) continue;
#endif
      lines(integral_constant<size_t,1>());
      }
    };
  execParallel(util::thread_count(nthreads, in, axis, native_simd<T>::size()), std::ref(work));
  }
template<typename T> DUCC0_NOINLINE void general_r2c(
  const cfmav<T> &in, const vfmav<Cmplx<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  { general_r2c_c2r<true,T>(in, out, axis, forward, fct, nthreads); }
template<typename T> DUCC0_NOINLINE void general_c2r(
  const cfmav<Cmplx<T>> &in, const vfmav<T> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  { general_r2c_c2r<false,T>(in, out, axis, forward, fct, nthreads); }

struct ExecR2R
  {
  bool r2c, forward;

  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out, Tstorage &storage,
    const pocketfft_r<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    using T = typename Tstorage::datatype;
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        T *buf1=storage.transformBuf(), *buf2=out.data()+it.oofs(0);
        if (in.data()!=buf2)
          copy_input(it, in, buf2);
        if ((!r2c) && forward)
          for (size_t i=2; i<it.length_out(); i+=2)
            buf2[i] = -buf2[i];
        plan.exec_copyback(buf2, buf1, fct, r2c, nthreads);
        if (r2c && (!forward))
          for (size_t i=2; i<it.length_out(); i+=2)
            buf2[i] = -buf2[i];
        return;
        }

    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    if ((!r2c) && forward)
      for (size_t i=2; i<it.length_out(); i+=2)
        buf2[i] = -buf2[i];
    auto res = plan.exec(buf2, buf1, fct, r2c, nthreads);
    if (r2c && (!forward))
      for (size_t i=2; i<it.length_out(); i+=2)
        res[i] = -res[i];
    copy_output(it, res, out);
    }
  template <typename T0, typename Tstorage, typename Titer> DUCC0_NOINLINE void exec_n (
    const Titer &it, const cfmav<T0> &in,
    const vfmav<T0> &out, Tstorage &storage, const pocketfft_r<T0> &plan, T0 fct, size_t nvec,
    size_t nthreads) const
    {
    using T = typename Tstorage::datatype;
    size_t dstr = storage.data_stride();
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2, nvec, dstr);
    if ((!r2c) && forward)
      for (size_t k=0; k<nvec; ++k)
        for (size_t i=2; i<it.length_out(); i+=2)
          buf2[i+k*dstr] = -buf2[i+k*dstr];
    for (size_t i=0; i<nvec; ++i)
      plan.exec_copyback(buf2+i*dstr, buf1, fct, r2c, nthreads);
    if (r2c && (!forward))
      for (size_t k=0; k<nvec; ++k)
        for (size_t i=2; i<it.length_out(); i+=2)
          buf2[i+k*dstr] = -buf2[i+k*dstr];
    copy_output(it, buf2, out, nvec, dstr);
    }
  template <typename T0> DUCC0_NOINLINE void exec_simple (
    const T0 *in, T0 *out, const pocketfft_r<T0> &plan, T0 fct,
    size_t nthreads) const
    {
    if (in!=out) copy_n(in, plan.length(), out);
    if ((!r2c) && forward)
      for (size_t i=2; i<plan.length(); i+=2)
        out[i] = -out[i];
    plan.exec(out, fct, r2c, nthreads);
    if (r2c && (!forward))
      for (size_t i=2; i<plan.length(); i+=2)
        out[i] = -out[i];
    }
  };

} // unnamed namespace

template<typename T> class Long1dPlan: public UnityRoots<T,complex<T>>
  {
  public:
    Long1dPlan(size_t length, bool)
      : UnityRoots<T,complex<T>>(length) {}
  };

/* c2c along a strided axis as a four-step transform. Along an axis whose
   elements are a multiple of 4 KB apart, each line is read in pieces of a few
   cache lines, which makes the transform memory-bound far above the cost of
   the same transform along a contiguous axis. With n = n1*n2 the transform is
   done in two steps of short lengths, which fit many more lines into the L2
   tiles of general_nd, so the rows are read in long contiguous pieces:
     1. for each j2: transforms of length n1 over the elements j2, j2+n2, ...;
        output k1 is multiplied by w_n^(k1*j2)
     2. transforms of length n2 over consecutive elements
     3. element k1*n2+k2 now holds output k2*n1+k1: the hyperplanes along the
        axis are permuted in place, cycle by cycle
   Measured on a Ryzen 7 7840U with 8 threads: worth it from n=4096 on, and
   only with several threads (one thread is compute-bound and loses). */
template<typename T> size_t fourstep_n1(const fmav_info &in, const fmav_info &out,
  size_t axis, size_t nthreads)
  {
  size_t n = in.shape(axis);
  if ((in.ndim()<2) || (n<4096)) return 0;
  if (util::thread_count(nthreads, in, axis, native_simd<T>::size())<2) return 0;
  auto crit = [](ptrdiff_t str) { return ((str*ptrdiff_t(sizeof(Cmplx<T>)))&4095)==0; };
  if (!(crit(in.stride(axis)) || crit(out.stride(axis)))) return 0;
  for (size_t d=size_t(std::sqrt(double(n))); d>=16; --d)  // the most balanced split
    if (n%d==0) return (n/d>=16) ? d : 0;
  return 0;
  }
template<typename T> void c2c_fourstep(const cfmav<Cmplx<T>> &in,
  const vfmav<Cmplx<T>> &out, size_t axis, size_t n1, bool forward, T fct,
  size_t nthreads)
  {
  const size_t n=in.shape(axis), n2=n/n1, nd=in.ndim();
  const ptrdiff_t si=in.stride(axis), so=out.stride(axis);
  // step 1
  UnityRoots<T,Cmplx<T>> roots(n);
  shape_t shp1(in.shape());
  shp1[axis] = n1;
  auto stri(in.stride()), stro(out.stride());
  stri[axis] = si*ptrdiff_t(n2);
  stro[axis] = so*ptrdiff_t(n2);
  vector<Cmplx<T>> tw(n1);
  for (size_t j2=0; j2<n2; ++j2)
    {
    for (size_t k1=0; k1<n1; ++k1) tw[k1] = roots[(k1*j2)%n];
    cfmav<Cmplx<T>> sub_in(in.data()+ptrdiff_t(j2)*si, shp1, stri);
    vfmav<Cmplx<T>> sub_out(out.data()+ptrdiff_t(j2)*so, shp1, stro);
    general_nd<pocketfft_c<T>>(sub_in, sub_out, {axis}, fct, nthreads, ExecC2CTw{forward, tw.data()});
    }
  // step 2: the axis split into (n1, n2), transforms along the second part
  shape_t shp2;
  auto str2 = out.stride();
  str2.clear();
  for (size_t i=0; i<nd; ++i)
    {
    if (i==axis)
      { shp2.push_back(n1); str2.push_back(so*ptrdiff_t(n2)); shp2.push_back(n2); str2.push_back(so); }
    else
      { shp2.push_back(out.shape(i)); str2.push_back(out.stride(i)); }
    }
  vfmav<Cmplx<T>> v2(out.data(), shp2, str2);
  general_nd<pocketfft_c<T>>(cfmav<Cmplx<T>>(v2), v2, {axis+1}, T(1), nthreads, ExecC2C{forward});
  // step 3: hyperplane j must receive the content of plane src(j)
  auto src = [n1,n2](size_t j) { return (j%n1)*n2 + j/n1; };
  vector<size_t> leaders;  // smallest index of every cycle of length > 1
  for (size_t i=0; i<n; ++i)
    {
    size_t j=src(i);
    while (j>i) j=src(j);
    if ((j==i) && (src(i)!=i)) leaders.push_back(i);
    }
  shape_t shpp;
  auto strp = out.stride();
  strp.clear();
  for (size_t i=0; i<nd; ++i)
    if (i!=axis) { shpp.push_back(out.shape(i)); strp.push_back(out.stride(i)); }
  auto plane = [&](size_t j) { return vfmav<Cmplx<T>>(out.data()+ptrdiff_t(j)*so, shpp, strp); };
  auto assign = [](Cmplx<T> &a, const Cmplx<T> &b) { a=b; };
  execDynamic(leaders.size(), nthreads, 1, [&](Scheduler &sched)
    {
    vfmav<Cmplx<T>> tmp(shpp);
    while (auto rng=sched.getNext())
      for (auto c=rng.lo; c<rng.hi; ++c)
        {
        size_t i0=leaders[c], j=i0;
        mav_apply(assign, 1, tmp, plane(i0));
        for (size_t s2=src(j); s2!=i0; j=s2, s2=src(j))
          mav_apply(assign, 1, plane(j), plane(s2));
        mav_apply(assign, 1, plane(j), tmp);
        }
    });
  }

template<typename T> DUCC0_NOINLINE void c2c(const cfmav<complex<T>> &in,
  const vfmav<complex<T>> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;

  // strided axes of length >= 4096 at a critical stride: four-step (see
  // c2c_fourstep); then all axes one by one, in the given order
  for (auto ax: axes)
    if (fourstep_n1<T>(in, out, ax, nthreads))
      {
      const auto &in2(reinterpret_cast<const cfmav<Cmplx<T> >&>(in));
      const auto &out2(reinterpret_cast<const vfmav<Cmplx<T> >&>(out));
      bool first = true;
      for (auto a: axes)
        {
        const auto &src(first ? in2 : cfmav<Cmplx<T>>(out2));
        if (size_t n1=fourstep_n1<T>(src, out2, a, nthreads))
          c2c_fourstep<T>(src, out2, a, n1, forward, first ? fct : T(1), nthreads);
        else
          general_nd<pocketfft_c<T>>(src, out2, {a}, first ? fct : T(1), nthreads, ExecC2C{forward});
        first = false;
        }
      return;
      }

  // special treatment for long 1D transforms (Bailey's algorithm)
  // TODO:
  //  - if not in-place and out has no critical stride, the "tmp" array can be avoided
  // (the SIMD single-transform plan is faster for lengths divisible by 16,
  //  but it runs on one thread, so the 2-D route keeps multithreading)
  if ((in.ndim()==1) && (in.shape(0)>=65536*2) && (((in.shape(0)&15)!=0) || (nthreads>1)))
    {
    size_t ip = in.shape(0);
    auto factors = util1d::prime_factors(ip);
    sort(factors.begin(), factors.end(), std::greater<size_t>());
    size_t f1=1, f2=1;
    for (auto fct: factors)
      (f2>f1) ? f1*=fct : f2*=fct;
    if (f1>f2) swap(f1,f2);
    if (f1>=16) // 2D algorithm makes sense
      {
      auto istr=in.stride(0);
      auto ostr=out.stride(0);
      cmav<complex<T>,2> in2 (in.data(), {f1,f2}, {ptrdiff_t(f2)*istr, istr});
      auto tmp (vmav<complex<T>,2>::build_noncritical({f1,f2}));
      vmav<complex<T>,2> out2 (out.data(), {f1,f2}, {ostr, ptrdiff_t(f1)*ostr});
      auto fin2(in2.to_fmav());
      auto ftmp(tmp.to_fmav());
      auto fout2(out2.to_fmav());
      c2c(fin2, ftmp, {0}, forward, T(1), nthreads);
      auto roots_p = get_plan<Long1dPlan<T>>(ip);
      const auto &roots(*roots_p);
      if (forward)
        execStatic(f1, nthreads, 0, [&](Scheduler &sched) {
          while (auto rng=sched.getNext())
            for (auto i=rng.lo; i<rng.hi; ++i)
              for (size_t j=0; j<f2; ++j)
                tmp(i,j) *= conj(roots[i*j]);
          });
      else
        execStatic(f1, nthreads, 0, [&](Scheduler &sched) {
          while (auto rng=sched.getNext())
            for (auto i=rng.lo; i<rng.hi; ++i)
              for (size_t j=0; j<f2; ++j)
                tmp(i,j) *= roots[i*j];
          });
      c2c(ftmp, fout2, {1}, forward, fct, nthreads);
      return;
      }
    }

  const auto &in2(reinterpret_cast<const cfmav<Cmplx<T> >&>(in));
  const auto &out2(reinterpret_cast<const vfmav<Cmplx<T> >&>(out));
  if ((axes.size()>1) && (in.data()!=out.data())) // optimize axis order
    {
    if ((in.stride(axes[0])!=1)&&(out.stride(axes[0])==1))
      {
      shape_t axes2(axes);
      swap(axes2[0],axes2.back());
      general_nd<pocketfft_c<T>>(in2, out2, axes2, fct, nthreads, ExecC2C{forward});
      return;
      }
    for (size_t i=1; i<axes.size(); ++i)
      if (in.stride(axes[i])==1)
        {
        shape_t axes2(axes);
        swap(axes2[0],axes2[i]);
        general_nd<pocketfft_c<T>>(in2, out2, axes2, fct, nthreads, ExecC2C{forward});
        return;
        }
    }
  general_nd<pocketfft_c<T>>(in2, out2, axes, fct, nthreads, ExecC2C{forward});
  }

template<typename T> DUCC0_NOINLINE void dct(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads)
  {
  if ((type<1) || (type>4)) throw invalid_argument("invalid DCT type");
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  const ExecDcst exec{ortho, type, true};
  if (type==1)
    general_nd<T_dct1<T>>(in, out, axes, fct, nthreads, exec);
  else if (type==4)
    general_nd<T_dcst4<T>>(in, out, axes, fct, nthreads, exec);
  else
    general_nd<T_dcst23<T>>(in, out, axes, fct, nthreads, exec);
  }

template<typename T> DUCC0_NOINLINE void dst(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads)
  {
  if ((type<1) || (type>4)) throw invalid_argument("invalid DST type");
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  const ExecDcst exec{ortho, type, false};
  if (type==1)
    general_nd<T_dst1<T>>(in, out, axes, fct, nthreads, exec);
  else if (type==4)
    general_nd<T_dcst4<T>>(in, out, axes, fct, nthreads, exec);
  else
    general_nd<T_dcst23<T>>(in, out, axes, fct, nthreads, exec);
  }

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  const vfmav<complex<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  bool inplace = in.data() == reinterpret_cast<const T*>(out.data());
  util::sanity_check_cr(out, in, inplace, axis);
  if (in.size()==0) return;
  const auto &out2(reinterpret_cast<const vfmav<Cmplx<T>>&>(out));
  general_r2c(in, out2, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  const vfmav<complex<T>> &out, const shape_t &axes,
  bool forward, T fct, size_t nthreads)
  {
  bool inplace = in.data() == reinterpret_cast<const T*>(out.data());
  util::sanity_check_cr(out, in, inplace, axes);
  if (in.size()==0) return;
  r2c(in, out, axes.back(), forward, fct, nthreads);
  if (axes.size()==1) return;

  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(out, out, newaxes, forward, T(1), nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<complex<T>> &in,
  const vfmav<T> &out,  size_t axis, bool forward, T fct, size_t nthreads)
  {
  bool inplace = reinterpret_cast<const T*>(in.data()) == out.data();
  util::sanity_check_cr(in, out, inplace, axis);
  if (in.size()==0) return;
  const auto &in2(reinterpret_cast<const cfmav<Cmplx<T>>&>(in));
  general_c2r(in2, out, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r_mut(const vfmav<complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads)
  {
  if (axes.size()==1)
    return c2r(in, out, axes[0], forward, fct, nthreads);
  bool inplace = reinterpret_cast<const T*>(in.data()) == out.data();
  util::sanity_check_cr(in, out, inplace, axes);
  if (in.size()==0) return;
  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(in, in, newaxes, forward, T(1), nthreads);
  c2r(in, out, axes.back(), forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads)
  {
  if (axes.size()==1)
    return c2r(in, out, axes[0], forward, fct, nthreads);
  bool inplace = reinterpret_cast<const T*>(in.data()) == out.data();
  util::sanity_check_cr(in, out, inplace, axes);
  if (in.size()==0) return;
  if (inplace)  // OK, we can overwrite the complex array after all
    {
    vfmav<complex<T>> in2(const_cast<complex<T> *>(in.data()), in.shape(), in.stride());
    return c2r_mut(in2, out, axes, forward, fct, nthreads);
    }
  auto atmp(vfmav<complex<T>>::build_noncritical(in.shape(), PAGE_IN(nthreads)));
  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(in, atmp, newaxes, forward, T(1), nthreads);
  c2r(atmp, out, axes.back(), forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2r_fftpack(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool real2hermitian, bool forward,
  T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_r<T>>(in, out, axes, fct, nthreads,
    ExecR2R{real2hermitian, forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_fftw(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_fftw<T>>(in, out, axes, fct, nthreads,
    ExecFFTW{forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_separable_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_hartley<T>>(in, out, axes, fct, nthreads,
    ExecHartley{}, false);
  }

template<typename T> DUCC0_NOINLINE void r2r_separable_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_fht<T>>(in, out, axes, fct, nthreads,
    ExecFHT{}, false);
  }

namespace {

template<typename T> [[maybe_unused]] void oscarize(const vfmav<T> &data, size_t ax0, size_t ax1,
  size_t nthreads)
  {
  auto nu=data.shape(ax0), nv=data.shape(ax1);
  if ((nu<3)||(nv<3)) return;
  vector<slice> slc(data.ndim());
  slc[ax0] = slice(1,(nu+1)/2);
  slc[ax1] = slice(1,(nv+1)/2);
  auto all = data.subarray(slc);
  slc[ax0] = slice(nu-1,nu/2,-1);
  auto ahl = data.subarray(slc);
  slc[ax1] = slice(nv-1,nv/2,-1);
  auto ahh = data.subarray(slc);
  slc[ax0] = slice(1,(nu+1)/2);
  auto alh = data.subarray(slc);
  mav_apply([](T &ll, T &hl, T &hh, T &lh)
    {
    T tll=ll, thl=hl, tlh=lh, thh=hh;
    T v = T(0.5)*(tll+tlh+thl+thh);
    ll = v-thh;
    hl = v-tlh;
    lh = v-thl;
    hh = v-tll;
    }, nthreads, all, ahl, ahh, alh);
  }

} // unnamed namespace

template<typename T> void r2r_genuine_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  if (axes.size()==1)
    return r2r_separable_hartley(in, out, axes, fct, nthreads);
  if (axes.size()==2)
    {
    r2r_separable_hartley(in, out, axes, fct, nthreads);
    oscarize(out, axes[0], axes[1], nthreads);
    return;
    }
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  shape_t tshp(in.shape());
  tshp[axes.back()] = tshp[axes.back()]/2+1;
  auto atmp(vfmav<complex<T>>::build_noncritical(tshp, PAGE_IN(nthreads)));
  r2c(in, atmp, axes, true, fct, nthreads);
  hermiteHelper(0, 0, 0, 0, atmp, out, axes, [](const complex<T> &c, T &r0, T &r1)
    {
    auto ccopy = c;
    r0 = ccopy.real()+ccopy.imag();
    r1 = ccopy.real()-ccopy.imag();
    }, nthreads);
  }

template<typename T> void r2r_genuine_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads)
  {
  if (axes.size()==1)
    return r2r_separable_fht(in, out, axes, fct, nthreads);
  if (axes.size()==2)
    {
    r2r_separable_fht(in, out, axes, fct, nthreads);
    oscarize(out, axes[0], axes[1], nthreads);
    return;
    }
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  shape_t tshp(in.shape());
  tshp[axes.back()] = tshp[axes.back()]/2+1;
  auto atmp(vfmav<complex<T>>::build_noncritical(tshp, PAGE_IN(nthreads)));
  r2c(in, atmp, axes, true, fct, nthreads);
  hermiteHelper(0, 0, 0, 0, atmp, out, axes, [](const complex<T> &c, T &r0, T &r1)
    {
    auto ccopy = c;
    r0 = ccopy.real()-ccopy.imag();
    r1 = ccopy.real()+ccopy.imag();
    }, nthreads);
  }

namespace {

template<typename Tplan, typename T0, typename T, typename Exec>
 [[maybe_unused]] DUCC0_NOINLINE void general_convolve_axis(const cfmav<T> &in, const vfmav<T> &out,
  const size_t axis, const cmav<T,1> &kernel, size_t nthreads,
  const Exec &exec)
  {
  size_t l_in=in.shape(axis), l_out=out.shape(axis);
  MR_assert(kernel.size()==l_in, "bad kernel size");
  auto plan1 = get_plan<Tplan>(l_in, in.ndim()==1);
  auto plan2 = get_plan<Tplan>(l_out, in.ndim()==1);
  size_t bufsz = max(plan1->bufsize(), plan2->bufsize());

  vmav<T,1> fkernel({kernel.shape(0)}, UNINITIALIZED);
  for (size_t i=0; i<kernel.shape(0); ++i)
    fkernel(i) = kernel(i);
  plan1->exec(fkernel.data(), T0(1)/T0(l_in), true, nthreads);

  auto work = [&](Scheduler &sched) {
      constexpr auto vlen = native_simd<T0>::size();
      TmpStorage<T,T0> storage(in.size()/l_in, l_in+l_out, bufsz, 1, false);
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        {
        TmpStorage2<add_vec_t<T, vlen>,T,T0> storage2(storage);
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          exec(it, in, out, storage2, *plan1, *plan2, fkernel);
          }
        }
      if constexpr (vlen>2)
        if constexpr (simd_exists<T,vlen/2>)
          if (it.remaining()>=vlen/2)
            {
            TmpStorage2<add_vec_t<T, vlen/2>,T,T0> storage2(storage);
            it.advance(vlen/2);
            exec(it, in, out, storage2, *plan1, *plan2, fkernel);
            }
      if constexpr (vlen>4)
        if constexpr (simd_exists<T,vlen/4>)
          if (it.remaining()>=vlen/4)
            {
            TmpStorage2<add_vec_t<T, vlen/4>,T,T0> storage2(storage);
            it.advance(vlen/4);
            exec(it, in, out, storage2, *plan1, *plan2, fkernel);
            }
#endif
      {
      TmpStorage2<T,T,T0> storage2(storage);
      while (it.remaining()>0)
        {
        it.advance(1);
        exec(it, in, out, storage2, *plan1, *plan2, fkernel);
        }
      }
    };
  execParallel(util::thread_count(nthreads, in, axis, native_simd<T0>::size()), std::ref(work));
  }

struct [[maybe_unused]] ExecConv1R
  {
  template <typename T0, typename Tstorage, typename Titer> void operator() (
    const Titer &it, const cfmav<T0> &in, const vfmav<T0> &out,
    Tstorage &storage, const pocketfft_r<T0> &plan1, const pocketfft_r<T0> &plan2,
    const cmav<T0,1> &fkernel) const
    {
    using T = typename Tstorage::datatype;
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out);
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    plan1.exec_copyback(buf2, buf1, T0(1), true);
    auto res = buf2;
    {
    res[0] *= fkernel(0);
    size_t i;
    for (i=1; 2*i<l_min; ++i)
      {
      Cmplx<T> t1(res[2*i-1], res[2*i]);
      Cmplx<T0> t2(fkernel(2*i-1), fkernel(2*i));
      auto t3 = t1*t2;
      res[2*i-1] = t3.r;
      res[2*i] = t3.i;
      }
    if (2*i==l_min)
      {
      if (l_min<l_out) // padding
        res[2*i-1] *= fkernel(2*i-1)*T0(0.5);
      else if (l_min<l_in) // truncation
        {
        Cmplx<T> t1(res[2*i-1], res[2*i]);
        Cmplx<T0> t2(fkernel(2*i-1), fkernel(2*i));
        res[2*i-1] = (t1*t2).r*T0(2);
        }
      else
        res[2*i-1] *= fkernel(2*i-1);
      }
    }
    for (size_t i=l_in; i<l_out; ++i) res[i] = T(0);
    res = plan2.exec(res, buf1, T0(1), false);
    copy_output(it, res, out);
    }
  };
struct [[maybe_unused]] ExecConv1C
  {
  template <typename T0, typename Tstorage, typename Titer> void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in, const vfmav<Cmplx<T0>> &out,
    Tstorage &storage, const pocketfft_c<T0> &plan1, const pocketfft_c<T0> &plan2,
    const cmav<Cmplx<T0>,1> &fkernel) const
    {
    using T = typename Tstorage::datatype;
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out);
    T *buf1=storage.transformBuf(), *buf2=storage.dataBuf();
    copy_input(it, in, buf2);
    auto res = plan1.exec(buf2, buf1, T0(1), true);
    auto res2 = buf2+l_in;
    {
    res2[0] = res[0]*fkernel(0);
    size_t i;
    for (i=1; 2*i<l_min; ++i)
      {
      res2[i] = res[i]*fkernel(i);
      res2[l_out-i] = res[l_in-i]*fkernel(l_in-i);
      }
    if (2*i==l_min)
      {
      if (l_min<l_out) // padding
        res2[l_out-i] = res2[i] = res[i]*fkernel(i)*T0(.5);
      else if (l_min<l_in) // truncation
        res2[i] = res[i]*fkernel(i) + res[l_in-i]*fkernel(l_in-i);
      else
        res2[i] = res[i]*fkernel(i);
      ++i;
      }
    for (; 2*i<=l_out; ++i)
      res2[i] = res2[l_out-i] = T(0,0);
    }
    res = plan2.exec(res2, buf1, T0(1), false);
    copy_output(it, res, out);
    }
  };

} // unnamed namespace

template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<T> &in,
  const vfmav<T> &out, size_t axis, const cmav<T,1> &kernel, size_t nthreads)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0 || out.size()==0) return;
  general_convolve_axis<pocketfft_r<T>, T>(in, out, axis, kernel, nthreads,
    ExecConv1R());
  }
template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<complex<T>> &in,
  const vfmav<complex<T>> &out, size_t axis, const cmav<complex<T>,1> &kernel,
  size_t nthreads)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0 || out.size()==0) return;
  const auto &in2(reinterpret_cast<const cfmav<Cmplx<T>>&>(in));
  const auto &out2(reinterpret_cast<const vfmav<Cmplx<T>>&>(out));
  const auto &kernel2(reinterpret_cast<const cmav<Cmplx<T>,1>&>(kernel));
  general_convolve_axis<pocketfft_c<T>, T>(in2, out2, axis, kernel2, nthreads,
    ExecConv1C());
  }

} // namespace detail_fft

} // namespace DUCC0_NAMESPACE

#endif // POCKETFFT_HDRONLY_H
