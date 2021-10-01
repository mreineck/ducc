/*
This file is part of pocketfft.

Copyright (C) 2010-2021 Max-Planck-Society
Copyright (C) 2019 Peter Bell

For the odd-sized DCT-IV transforms:
  Copyright (C) 2003, 2007-14 Matteo Frigo
  Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

Authors: Martin Reinecke, Peter Bell

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

#ifndef DUCC0_FFT_H
#define DUCC0_FFT_H

#include <cmath>
#include <cstddef>
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
#include "ducc0/fft/fft1d.h"

/** \file fft.h
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

namespace ducc0 {

namespace detail_fft {

template<typename T> constexpr inline size_t fft_simdlen
  = min<size_t>(8, native_simd<T>::size());
template<> constexpr inline size_t fft_simdlen<double>
  = min<size_t>(2, native_simd<double>::size());
template<> constexpr inline size_t fft_simdlen<float>
  = min<size_t>(4, native_simd<float>::size());
template<typename T> using fft_simd = typename simd_select<T,fft_simdlen<T>>::type;
template<typename T> constexpr inline bool fft_simd_exists = (fft_simdlen<T> > 1);

using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

constexpr bool FORWARD  = true,
               BACKWARD = false;

struct util // hack to avoid duplicate symbols
  {
  static void sanity_check_axes(size_t ndim, const shape_t &axes)
    {
    shape_t tmp(ndim,0);
    if (axes.empty()) throw std::invalid_argument("no axes specified");
    for (auto ax : axes)
      {
      if (ax>=ndim) throw std::invalid_argument("bad axis number");
      if (++tmp[ax]>1) throw std::invalid_argument("axis specified repeatedly");
      }
    }

  DUCC0_NOINLINE static void sanity_check_onetype(const fmav_info &a1,
    const fmav_info &a2, bool inplace, const shape_t &axes)
    {
    sanity_check_axes(a1.ndim(), axes);
    MR_assert(a1.conformable(a2), "array sizes are not conformable");
    if (inplace) MR_assert(a1.stride()==a2.stride(), "stride mismatch");
    }
  DUCC0_NOINLINE static void sanity_check_cr(const fmav_info &ac,
    const fmav_info &ar, const shape_t &axes)
    {
    sanity_check_axes(ac.ndim(), axes);
    MR_assert(ac.ndim()==ar.ndim(), "dimension mismatch");
    for (size_t i=0; i<ac.ndim(); ++i)
      MR_assert(ac.shape(i)== (i==axes.back()) ? (ar.shape(i)/2+1) : ar.shape(i),
        "axis length mismatch");
    }
  DUCC0_NOINLINE static void sanity_check_cr(const fmav_info &ac,
    const fmav_info &ar, const size_t axis)
    {
    if (axis>=ac.ndim()) throw std::invalid_argument("bad axis number");
    MR_assert(ac.ndim()==ar.ndim(), "dimension mismatch");
    for (size_t i=0; i<ac.ndim(); ++i)
      MR_assert(ac.shape(i)== (i==axis) ? (ar.shape(i)/2+1) : ar.shape(i),
        "axis length mismatch");
    }

#ifdef DUCC0_NO_THREADING
  static size_t thread_count (size_t /*nthreads*/, const fmav_info &/*info*/,
    size_t /*axis*/, size_t /*vlen*/)
    { return 1; }
#else
  static size_t thread_count (size_t nthreads, const fmav_info &info,
    size_t axis, size_t vlen)
    {
    if (nthreads==1) return 1;
    size_t size = info.size();
    size_t parallel = size / (info.shape(axis) * vlen);
    if (info.shape(axis) < 1000)
      parallel /= 4;
    size_t max_threads = (nthreads==0) ? ducc0::get_default_nthreads() : nthreads;
    return std::max(size_t(1), std::min(parallel, max_threads));
    }
#endif
  };


//
// sine/cosine transforms
//

template<typename T0> class T_dct1
  {
  private:
    pocketfft_r<T0> fftplan;

  public:
    DUCC0_NOINLINE T_dct1(size_t length, bool /*vectorize*/=false)
      : fftplan(2*(length-1)) {}

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
      size_t N=fftplan.length(), n=N/2+1;
      if (ortho)
        { c[0]*=sqrt2; c[n-1]*=sqrt2; }
      auto tmp=&buf[0];
      tmp[0] = c[0];
      for (size_t i=1; i<n; ++i)
        tmp[i] = tmp[N-i] = c[i];
      auto res = fftplan.exec(tmp, &buf[N], fct, true, nthreads);
      c[0] = res[0];
      for (size_t i=1; i<n; ++i)
        c[i] = res[2*i-1];
      if (ortho)
        { c[0]*=sqrt2*T0(0.5); c[n-1]*=sqrt2*T0(0.5); }
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      exec(c, buf, fct, ortho, 1, true, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      aligned_array<T> buf(bufsize());
      exec_copyback(c, buf.data(), fct, ortho, 1, true, nthreads);
      }

    size_t length() const { return fftplan.length()/2+1; }
    size_t bufsize() const { return fftplan.length()+fftplan.bufsize(); }
  };

template<typename T0> class T_dst1
  {
  private:
    pocketfft_r<T0> fftplan;

  public:
    DUCC0_NOINLINE T_dst1(size_t length, bool /*vectorize*/=false)
      : fftplan(2*(length+1)) {}

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      size_t N=fftplan.length(), n=N/2-1;
      auto tmp = &buf[0];
      tmp[0] = tmp[n+1] = c[0]*0;
      for (size_t i=0; i<n; ++i)
        { tmp[i+1]=c[i]; tmp[N-1-i]=-c[i]; }
      auto res = fftplan.exec(tmp, buf+N, fct, true, nthreads);
      for (size_t i=0; i<n; ++i)
        c[i] = -res[2*i+2];
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/, size_t nthreads=1) const
      {
      exec(c, buf, fct, true, 1, false, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/, size_t nthreads) const
      {
      aligned_array<T> buf(bufsize());
      exec_copyback(c, buf.data(), fct, true, 1, false, nthreads);
      }

    size_t length() const { return fftplan.length()/2-1; }
    size_t bufsize() const { return fftplan.length()+fftplan.bufsize(); }
  };

template<typename T0> class T_dcst23
  {
  private:
    pocketfft_r<T0> fftplan;
    std::vector<T0> twiddle;

  public:
    DUCC0_NOINLINE T_dcst23(size_t length, bool /*vectorize*/=false)
      : fftplan(length), twiddle(length)
      {
      UnityRoots<T0,Cmplx<T0>> tw(4*length);
      for (size_t i=0; i<length; ++i)
        twiddle[i] = tw[i+1].r;
      }

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool ortho,
      int type, bool cosine, size_t nthreads=1) const
      {
      constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
      size_t N=length();
      size_t NS2 = (N+1)/2;
      if (type==2)
        {
        if (!cosine)
          for (size_t k=1; k<N; k+=2)
            c[k] = -c[k];
        c[0] *= 2;
        if ((N&1)==0) c[N-1]*=2;
        for (size_t k=1; k<N-1; k+=2)
          MPINPLACE(c[k+1], c[k]);
        auto res = fftplan.exec(c, buf, fct, false, nthreads);
        c[0] = res[0];
        for (size_t k=1, kc=N-1; k<NS2; ++k, --kc)
          {
          T t1 = twiddle[k-1]*res[kc]+twiddle[kc-1]*res[k];
          T t2 = twiddle[k-1]*res[k]-twiddle[kc-1]*res[kc];
          c[k] = T0(0.5)*(t1+t2); c[kc]=T0(0.5)*(t1-t2);
          }
        if ((N&1)==0)
          c[NS2] = res[NS2]*twiddle[NS2-1];
        if (!cosine)
          for (size_t k=0, kc=N-1; k<kc; ++k, --kc)
            std::swap(c[k], c[kc]);
        if (ortho) c[0]*=sqrt2*T0(0.5);
        }
      else
        {
        if (ortho) c[0]*=sqrt2;
        if (!cosine)
          for (size_t k=0, kc=N-1; k<NS2; ++k, --kc)
            std::swap(c[k], c[kc]);
        for (size_t k=1, kc=N-1; k<NS2; ++k, --kc)
          {
          T t1=c[k]+c[kc], t2=c[k]-c[kc];
          c[k] = twiddle[k-1]*t2+twiddle[kc-1]*t1;
          c[kc]= twiddle[k-1]*t1-twiddle[kc-1]*t2;
          }
        if ((N&1)==0)
          c[NS2] *= 2*twiddle[NS2-1];
        auto res = fftplan.exec(c, buf, fct, true, nthreads);
        if (res != c) // FIXME: not yet optimal
          copy_n(res, N, c);
        for (size_t k=1; k<N-1; k+=2)
          MPINPLACE(c[k], c[k+1]);
        if (!cosine)
          for (size_t k=1; k<N; k+=2)
            c[k] = -c[k];
        }
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct,
      bool ortho, int type, bool cosine, size_t nthreads=1) const
      {
      exec(c, buf, fct, ortho, type, cosine, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool ortho,
      int type, bool cosine, size_t nthreads=1) const
      {
      aligned_array<T> buf(bufsize());
      exec(c, &buf[0], fct, ortho, type, cosine, nthreads);
      }

    size_t length() const { return fftplan.length(); }
    size_t bufsize() const { return fftplan.bufsize(); }
  };

template<typename T0> class T_dcst4
  {
  private:
    size_t N;
    std::unique_ptr<pocketfft_c<T0>> fft;
    std::unique_ptr<pocketfft_r<T0>> rfft;
    aligned_array<Cmplx<T0>> C2;

  public:
    DUCC0_NOINLINE T_dcst4(size_t length, bool /*vectorize*/=false)
      : N(length),
        fft((N&1) ? nullptr : make_unique<pocketfft_c<T0>>(N/2)),
        rfft((N&1)? make_unique<pocketfft_r<T0>>(N) : nullptr),
        C2((N&1) ? 0 : N/2)
      {
      if ((N&1)==0)
        {
        UnityRoots<T0,Cmplx<T0>> tw(16*N);
        for (size_t i=0; i<N/2; ++i)
          C2[i] = tw[8*i+1].conj();
        }
      }

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T /*buf*/[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine, size_t nthreads) const
      {
      size_t n2 = N/2;
      if (!cosine)
        for (size_t k=0, kc=N-1; k<n2; ++k, --kc)
          std::swap(c[k], c[kc]);
      if (N&1)
        {
        // The following code is derived from the FFTW3 function apply_re11()
        // and is released under the 3-clause BSD license with friendly
        // permission of Matteo Frigo and Steven G. Johnson.

        aligned_array<T> y(N);
        {
        size_t i=0, m=n2;
        for (; m<N; ++i, m+=4)
          y[i] = c[m];
        for (; m<2*N; ++i, m+=4)
          y[i] = -c[2*N-m-1];
        for (; m<3*N; ++i, m+=4)
          y[i] = -c[m-2*N];
        for (; m<4*N; ++i, m+=4)
          y[i] = c[4*N-m-1];
        for (; i<N; ++i, m+=4)
          y[i] = c[m-4*N];
        }
        rfft->exec(y.data(), fct, true, nthreads);
        {
        auto SGN = [](size_t i)
           {
           constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
           return (i&2) ? -sqrt2 : sqrt2;
           };
        c[n2] = y[0]*SGN(n2+1);
        size_t i=0, i1=1, k=1;
        for (; k<n2; ++i, ++i1, k+=2)
          {
          c[i    ] = y[2*k-1]*SGN(i1)     + y[2*k  ]*SGN(i);
          c[N -i1] = y[2*k-1]*SGN(N -i)   - y[2*k  ]*SGN(N -i1);
          c[n2-i1] = y[2*k+1]*SGN(n2-i)   - y[2*k+2]*SGN(n2-i1);
          c[n2+i1] = y[2*k+1]*SGN(n2+i+2) + y[2*k+2]*SGN(n2+i1);
          }
        if (k == n2)
          {
          c[i   ] = y[2*k-1]*SGN(i+1) + y[2*k]*SGN(i);
          c[N-i1] = y[2*k-1]*SGN(i+2) + y[2*k]*SGN(i1);
          }
        }

        // FFTW-derived code ends here
        }
      else
        {
        // even length algorithm from
        // https://www.appletonaudio.com/blog/2013/derivation-of-fast-dct-4-algorithm-based-on-dft/
        aligned_array<Cmplx<T>> y(n2);
        for(size_t i=0; i<n2; ++i)
          {
          y[i].Set(c[2*i],c[N-1-2*i]);
          y[i] *= C2[i];
          }
        fft->exec(y.data(), fct, true, nthreads);
        for(size_t i=0, ic=n2-1; i<n2; ++i, --ic)
          {
          c[2*i  ] = T0( 2)*(y[i ].r*C2[i ].r-y[i ].i*C2[i ].i);
          c[2*i+1] = T0(-2)*(y[ic].i*C2[ic].r+y[ic].r*C2[ic].i);
          }
        }
      if (!cosine)
        for (size_t k=1; k<N; k+=2)
          c[k] = -c[k];
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec_copyback(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine, size_t nthreads=1) const
      {
      exec(c, buf, fct, true, 4, cosine, nthreads);
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine, size_t nthreads=1) const
      {
      exec(c, nullptr, fct, true, 4, cosine, nthreads);
      }

    size_t length() const { return N; }
    size_t bufsize() const { return 0; }
  };


//
// multi-D infrastructure
//

template<size_t N> class multi_iter
  {
  private:
    shape_t shp, pos;
    stride_t str_i, str_o;
    size_t cshp_i, cshp_o, rem;
    ptrdiff_t cstr_i, cstr_o, sstr_i, sstr_o, p_ii, p_i[N], p_oi, p_o[N];
    bool uni_i, uni_o;

    void advance_i()
      {
      for (size_t i=0; i<pos.size(); ++i)
        {
        p_ii += str_i[i];
        p_oi += str_o[i];
        if (++pos[i] < shp[i])
          return;
        pos[i] = 0;
        p_ii -= ptrdiff_t(shp[i])*str_i[i];
        p_oi -= ptrdiff_t(shp[i])*str_o[i];
        }
      }

  public:
    multi_iter(const fmav_info &iarr, const fmav_info &oarr, size_t idim,
      size_t nshares, size_t myshare)
      : rem(iarr.size()/iarr.shape(idim)), sstr_i(0), sstr_o(0), p_ii(0), p_oi(0)
      {
      MR_assert(oarr.ndim()==iarr.ndim(), "dimension mismatch");
      MR_assert(iarr.ndim()>=1, "not enough dimensions");
      // Sort the extraneous dimensions in order of ascending output stride;
      // this should improve overall cache re-use and avoid clashes between
      // threads as much as possible.
      shape_t idx(iarr.ndim());
      std::iota(idx.begin(), idx.end(), 0);
      sort(idx.begin(), idx.end(),
        [&oarr](size_t i1, size_t i2) {return oarr.stride(i1) < oarr.stride(i2);});
      for (auto i: idx)
        if (i!=idim)
          {
          pos.push_back(0);
          MR_assert(iarr.shape(i)==oarr.shape(i), "shape mismatch");
          shp.push_back(iarr.shape(i));
          str_i.push_back(iarr.stride(i));
          str_o.push_back(oarr.stride(i));
          }
      MR_assert(idim<iarr.ndim(), "bad active dimension");
      cstr_i = iarr.stride(idim);
      cstr_o = oarr.stride(idim);
      cshp_i = iarr.shape(idim);
      cshp_o = oarr.shape(idim);

// collapse unneeded dimensions
      bool done = false;
      while(!done)
        {
        done=true;
        for (size_t i=1; i<shp.size(); ++i)
          if ((str_i[i] == str_i[i-1]*ptrdiff_t(shp[i-1]))
           && (str_o[i] == str_o[i-1]*ptrdiff_t(shp[i-1])))
            {
            shp[i-1] *= shp[i];
            str_i.erase(str_i.begin()+ptrdiff_t(i));
            str_o.erase(str_o.begin()+ptrdiff_t(i));
            shp.erase(shp.begin()+ptrdiff_t(i));
            pos.pop_back();
            done=false;
            }
        }
      if (pos.size()>0)
        {
        sstr_i = str_i[0];
        sstr_o = str_o[0];
        }

      if (nshares==1) return;
      if (nshares==0) throw std::runtime_error("can't run with zero threads");
      if (myshare>=nshares) throw std::runtime_error("impossible share requested");
      auto [lo, hi] = calcShare(nshares, myshare, rem);
      size_t todo = hi-lo;

      size_t chunk = rem;
      for (size_t i2=0, i=pos.size()-1; i2<pos.size(); ++i2,--i)
        {
        chunk /= shp[i];
        size_t n_advance = lo/chunk;
        pos[i] += n_advance;
        p_ii += ptrdiff_t(n_advance)*str_i[i];
        p_oi += ptrdiff_t(n_advance)*str_o[i];
        lo -= n_advance*chunk;
        }
      MR_assert(lo==0, "must not happen");
      rem = todo;
      }
    void advance(size_t n)
      {
      if (rem<n) throw std::runtime_error("underrun");
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
  };

template<typename T, typename T0> DUCC0_NOINLINE aligned_array<T> alloc_tmp
  (const fmav_info &info, size_t axsize, size_t bufsize, bool inplace=false)
  {
  if (inplace)
    return aligned_array<T>(bufsize);
  auto othersize = info.size()/axsize;
  constexpr auto vlen = fft_simdlen<T0>;
  // FIXME: when switching to C++20, use bit_floor(othersize)
  return aligned_array<T>((axsize+bufsize)*std::min(vlen, othersize));
  }

// NOTE: gcc 10 seems to pessimize this code by rearranging loops when tree
// vectorization is on. So I'm currently using an explicit "-fno-tree-vectorize"
// in setup.py.
template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<Cmplx<typename Tsimd::value_type>> &src, Cmplx<Tsimd> *DUCC0_RESTRICT dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_i())
    {
    auto ptr = &src.raw(it.iofs_uni(0,0));
    auto jstr = it.unistride_i();
    auto istr = it.stride_in();
    if (istr==1)
      for (size_t i=0; i<it.length_in(); ++i)
        for (size_t j=0; j<vlen; ++j)
          {
          auto tmp = ptr[ptrdiff_t(j)*jstr+ptrdiff_t(i)];
          dst[i].r[j] = tmp.r;
          dst[i].i[j] = tmp.i;
          }
    else if (jstr==1)
      for (size_t i=0; i<it.length_in(); ++i)
        for (size_t j=0; j<vlen; ++j)
          {
          auto tmp = ptr[ptrdiff_t(j)+ptrdiff_t(i)*istr];
          dst[i].r[j] = tmp.r;
          dst[i].i[j] = tmp.i;
          }
    else
      for (size_t i=0; i<it.length_in(); ++i)
        for (size_t j=0; j<vlen; ++j)
          {
          auto tmp = src.raw(it.iofs_uni(j,i));
          dst[i].r[j] = tmp.r;
          dst[i].i[j] = tmp.i;
          }
    }
  else
    for (size_t i=0; i<it.length_in(); ++i)
      for (size_t j=0; j<vlen; ++j)
        {
        dst[i].r[j] = src.raw(it.iofs(j,i)).r;
        dst[i].i[j] = src.raw(it.iofs(j,i)).i;
        }
  }

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const cfmav<typename Tsimd::value_type> &src, Tsimd *DUCC0_RESTRICT dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_i())
    {
    auto ptr = &src.raw(it.iofs_uni(0,0));
    auto jstr = it.unistride_i();
    auto istr = it.stride_in();
// FIXME: flip loops to avoid critical strides?
    if (istr==1)
      for (size_t i=0; i<it.length_in(); ++i)
        for (size_t j=0; j<vlen; ++j)
          dst[i][j] = ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i)];
    else if (jstr==1)
      for (size_t i=0; i<it.length_in(); ++i)
        for (size_t j=0; j<vlen; ++j)
          dst[i][j] = ptr[ptrdiff_t(j) + ptrdiff_t(i)*istr];
    else
      for (size_t i=0; i<it.length_in(); ++i)
        for (size_t j=0; j<vlen; ++j)
          dst[i][j] = src.raw(it.iofs_uni(j,i));
    }
  else
    for (size_t i=0; i<it.length_in(); ++i)
      for (size_t j=0; j<vlen; ++j)
        dst[i][j] = src.raw(it.iofs(j,i));
  }

template <typename T, size_t vlen> DUCC0_NOINLINE void copy_input(const multi_iter<vlen> &it,
  const cfmav<T> &src, T *DUCC0_RESTRICT dst)
  {
  if (dst == &src.raw(it.iofs(0))) return;  // in-place
  for (size_t i=0; i<it.length_in(); ++i)
    dst[i] = src.raw(it.iofs(i));
  }

// NOTE: gcc 10 seems to pessimize this code by rearranging loops when tree
// vectorization is on. So I'm currently using an explicit "-fno-tree-vectorize"
// in setup.py.
template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Cmplx<Tsimd> *DUCC0_RESTRICT src, vfmav<Cmplx<typename Tsimd::value_type>> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_o())
    {
    Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = &dst.raw(it.oofs_uni(0,0));
    auto jstr = it.unistride_o();
    auto istr = it.stride_out();
    if (istr==1)
      for (size_t i=0; i<it.length_out(); ++i)
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i)].Set(src[i].r[j],src[i].i[j]);
    else if (jstr==1)
      for (size_t i=0; i<it.length_out(); ++i)
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j) + ptrdiff_t(i)*istr].Set(src[i].r[j],src[i].i[j]);
    else
      for (size_t i=0; i<it.length_out(); ++i)
        for (size_t j=0; j<vlen; ++j)
          dst.raw(it.oofs_uni(j,i)).Set(src[i].r[j],src[i].i[j]);
    }
  else
    {
    Cmplx<typename Tsimd::value_type> * DUCC0_RESTRICT ptr = dst.data();
    for (size_t i=0; i<it.length_out(); ++i)
      for (size_t j=0; j<vlen; ++j)
        ptr[it.oofs(j,i)].Set(src[i].r[j],src[i].i[j]);
    }
  }

template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Tsimd *DUCC0_RESTRICT src, vfmav<typename Tsimd::value_type> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_o())
    {
    auto ptr = &dst.raw(it.oofs_uni(0,0));
    auto jstr = it.unistride_o();
    auto istr = it.stride_out();
    if (istr==1)
      for (size_t i=0; i<it.length_out(); ++i)
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i)] = src[i][j];
    else if (jstr==1)
      for (size_t i=0; i<it.length_out(); ++i)
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j) + ptrdiff_t(i)*istr] = src[i][j];
    else
      for (size_t i=0; i<it.length_out(); ++i)
        for (size_t j=0; j<vlen; ++j)
          dst.raw(it.oofs_uni(j,i)) = src[i][j];
    }
  else
    {
    auto ptr=dst.data();
    for (size_t i=0; i<it.length_out(); ++i)
      for (size_t j=0; j<vlen; ++j)
        ptr[it.oofs(j,i)] = src[i][j];
    }
  }

template<typename T, size_t vlen> DUCC0_NOINLINE void copy_output(const multi_iter<vlen> &it,
  const T *DUCC0_RESTRICT src, vfmav<T> &dst)
  {
  auto ptr=dst.data();
  if (src == &dst.raw(it.oofs(0))) return;  // in-place
  for (size_t i=0; i<it.length_out(); ++i)
    ptr[it.oofs(i)] = src[i];
  }

template <typename T, size_t vlen> struct add_vec
  { using type = typename simd_select<T, vlen>::type; };
template <typename T, size_t vlen> struct add_vec<Cmplx<T>, vlen>
  { using type = Cmplx<typename simd_select<T, vlen>::type>; };
template <typename T, size_t vlen> using add_vec_t = typename add_vec<T, vlen>::type;

template<typename Tplan, typename T, typename T0, typename Exec>
DUCC0_NOINLINE void general_nd(const cfmav<T> &in, vfmav<T> &out,
  const shape_t &axes, T0 fct, size_t nthreads, const Exec &exec,
  const bool /*allow_inplace*/=true)
  {
  std::unique_ptr<Tplan> plan;
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;
  bool inplace = (out.ndim()==1)&&(out.stride(0)==1);

  for (size_t iax=0; iax<axes.size(); ++iax)
    {
    size_t len=in.shape(axes[iax]);
    if ((!plan) || (len!=plan->length()))
      plan = std::make_unique<Tplan>(len, in.ndim()==1);

    execParallel(
      util::thread_count(nthreads, in, axes[iax], fft_simdlen<T0>),
      [&](Scheduler &sched) {
        constexpr auto vlen = fft_simdlen<T0>;
        auto storage = alloc_tmp<T,T0>(in, len, plan->bufsize(), inplace);
        const auto &tin(iax==0? in : out);
        multi_iter<vlen> it(tin, out, axes[iax], sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
        if constexpr (vlen>1)
          while (it.remaining()>=vlen)
            {
            it.advance(vlen);
            auto tdatav = reinterpret_cast<add_vec_t<T, vlen> *>(storage.data());
            exec(it, tin, out, tdatav, *plan, fct, nth1d);
            }
        if constexpr (vlen>2)
          if constexpr (simd_exists<T0,vlen/2>)
            if (it.remaining()>=vlen/2)
              {
              it.advance(vlen/2);
              auto tdatav = reinterpret_cast<add_vec_t<T, vlen/2> *>(storage.data());
              exec(it, tin, out, tdatav, *plan, fct, nth1d);
              }
        if constexpr (vlen>4)
          if constexpr (simd_exists<T0,vlen/4>)
            if (it.remaining()>=vlen/4)
              {
              it.advance(vlen/4);
              auto tdatav = reinterpret_cast<add_vec_t<T, vlen/4> *>(storage.data());
              exec(it, tin, out, tdatav, *plan, fct, nth1d);
              }
#endif
        while (it.remaining()>0)
          {
          it.advance(1);
          exec(it, tin, out, storage.data(), *plan, fct, nth1d, inplace);
          }
      });  // end of parallel region
    fct = T0(1); // factor has been applied, use 1 for remaining axes
    }
  }

struct ExecC2C
  {
  bool forward;

  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in,
    vfmav<Cmplx<T0>> &out, T *buf, const pocketfft_c<T0> &plan, T0 fct,
    size_t nthreads, bool inplace=false) const
    {
    if constexpr(is_same<Cmplx<T0>, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data());
        plan.exec_copyback(out.data(), buf, fct, forward, nthreads);
        return;
        }
    T *buf1=buf, *buf2=buf+plan.bufsize();
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    copy_output(it, res, out);
    }
  };

struct ExecHartley
  {
  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator () (
    const Titer &it, const cfmav<T0> &in, vfmav<T0> &out,
    T *buf, const pocketfft_hartley<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data());
        plan.exec_copyback(out.data(), buf, fct, nthreads);
        return;
        }
    T *buf1=buf, *buf2=buf+plan.bufsize(); 
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, nthreads);
    copy_output(it, res, out);
    }
  };

struct ExecFFTW
  {
  bool forward;

  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator () (
    const Titer &it, const cfmav<T0> &in, vfmav<T0> &out,
    T *buf, const pocketfft_fftw<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data());
        plan.exec_copyback(out.data(), buf, fct, forward, nthreads);
        return;
        }
    T *buf1=buf, *buf2=buf+plan.bufsize(); 
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward, nthreads);
    copy_output(it, res, out);
    }
  };

struct ExecDcst
  {
  bool ortho;
  int type;
  bool cosine;

  template <typename T0, typename T, typename Tplan, typename Titer>
  DUCC0_NOINLINE void operator () (const Titer &it, const cfmav<T0> &in,
    vfmav <T0> &out, T * buf, const Tplan &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        if (in.data()!=out.data())
          copy_input(it, in, out.data());
        plan.exec_copyback(out.data(), buf, fct, ortho, type, cosine, nthreads);
        return;
        }
    T *buf1=buf, *buf2=buf+plan.bufsize(); 
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, ortho, type, cosine, nthreads);
    copy_output(it, res, out);
    }
  };

template<typename T> DUCC0_NOINLINE void general_r2c(
  const cfmav<T> &in, vfmav<Cmplx<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;
  auto plan = std::make_unique<pocketfft_r<T>>(in.shape(axis));
  size_t len=in.shape(axis);
  execParallel(
    util::thread_count(nthreads, in, axis, fft_simdlen<T>),
    [&](Scheduler &sched) {
    constexpr auto vlen = fft_simdlen<T>;
    auto storage = alloc_tmp<T,T>(in, len, plan->bufsize());
    multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
    if constexpr (vlen>1)
      while (it.remaining()>=vlen)
        {
        it.advance(vlen);
        auto tdatav = reinterpret_cast<fft_simd<T> *>(storage.data());
        copy_input(it, in, tdatav);
        plan->exec(tdatav, fct, true, nth1d);
        auto vout = out.data();
        for (size_t j=0; j<vlen; ++j)
          vout[it.oofs(j,0)].Set(tdatav[0][j]);
        size_t i=1, ii=1;
        if (forward)
          for (; i<len-1; i+=2, ++ii)
            for (size_t j=0; j<vlen; ++j)
              vout[it.oofs(j,ii)].Set(tdatav[i][j], tdatav[i+1][j]);
        else
          for (; i<len-1; i+=2, ++ii)
            for (size_t j=0; j<vlen; ++j)
              vout[it.oofs(j,ii)].Set(tdatav[i][j], -tdatav[i+1][j]);
        if (i<len)
          for (size_t j=0; j<vlen; ++j)
            vout[it.oofs(j,ii)].Set(tdatav[i][j]);
        }
    if constexpr (vlen>2)
      if constexpr (simd_exists<T,vlen/2>)
        if (it.remaining()>=vlen/2)
          {
          it.advance(vlen/2);
          auto tdatav = reinterpret_cast<typename simd_select<T,vlen/2>::type *>(storage.data());
          copy_input(it, in, tdatav);
          plan->exec(tdatav, fct, true, nth1d);
          auto vout = out.data();
          for (size_t j=0; j<vlen/2; ++j)
            vout[it.oofs(j,0)].Set(tdatav[0][j]);
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/2; ++j)
                vout[it.oofs(j,ii)].Set(tdatav[i][j], tdatav[i+1][j]);
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/2; ++j)
                vout[it.oofs(j,ii)].Set(tdatav[i][j], -tdatav[i+1][j]);
          if (i<len)
            for (size_t j=0; j<vlen/2; ++j)
              vout[it.oofs(j,ii)].Set(tdatav[i][j]);
          }
    if constexpr (vlen>4)
      if constexpr( simd_exists<T,vlen/4>)
        if (it.remaining()>=vlen/4)
          {
          it.advance(vlen/4);
          auto tdatav = reinterpret_cast<typename simd_select<T,vlen/4>::type *>(storage.data());
          copy_input(it, in, tdatav);
          plan->exec(tdatav, fct, true, nth1d);
          auto vout = out.data();
          for (size_t j=0; j<vlen/4; ++j)
            vout[it.oofs(j,0)].Set(tdatav[0][j]);
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/4; ++j)
                vout[it.oofs(j,ii)].Set(tdatav[i][j], tdatav[i+1][j]);
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/4; ++j)
                vout[it.oofs(j,ii)].Set(tdatav[i][j], -tdatav[i+1][j]);
          if (i<len)
            for (size_t j=0; j<vlen/4; ++j)
              vout[it.oofs(j,ii)].Set(tdatav[i][j]);
          }
#endif
    while (it.remaining()>0)
      {
      it.advance(1);
      auto tdata = reinterpret_cast<T *>(storage.data());
      copy_input(it, in, tdata);
      plan->exec(tdata, fct, true, nth1d);
      auto vout = out.data();
      vout[it.oofs(0)].Set(tdata[0]);
      size_t i=1, ii=1;
      if (forward)
        for (; i<len-1; i+=2, ++ii)
          vout[it.oofs(ii)].Set(tdata[i], tdata[i+1]);
      else
        for (; i<len-1; i+=2, ++ii)
          vout[it.oofs(ii)].Set(tdata[i], -tdata[i+1]);
      if (i<len)
        vout[it.oofs(ii)].Set(tdata[i]);
      }
    });  // end of parallel region
  }
template<typename T> DUCC0_NOINLINE void general_c2r(
  const cfmav<Cmplx<T>> &in, vfmav<T> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  size_t nth1d = (in.ndim()==1) ? nthreads : 1;
  auto plan = std::make_unique<pocketfft_r<T>>(out.shape(axis));
  size_t len=out.shape(axis);
  execParallel(
    util::thread_count(nthreads, in, axis, fft_simdlen<T>),
    [&](Scheduler &sched) {
      constexpr auto vlen = fft_simdlen<T>;
      auto storage = alloc_tmp<T,T>(out, len, plan->bufsize());
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          auto tdatav = reinterpret_cast<fft_simd<T> *>(storage.data());
          for (size_t j=0; j<vlen; ++j)
            tdatav[0][j]=in.raw(it.iofs(j,0)).r;
          {
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen; ++j)
                {
                tdatav[i  ][j] =  in.raw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = -in.raw(it.iofs(j,ii)).i;
                }
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen; ++j)
                {
                tdatav[i  ][j] = in.raw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = in.raw(it.iofs(j,ii)).i;
                }
          if (i<len)
            for (size_t j=0; j<vlen; ++j)
              tdatav[i][j] = in.raw(it.iofs(j,ii)).r;
          }
          plan->exec(tdatav, fct, false, nth1d);
          copy_output(it, tdatav, out);
          }
      if constexpr (vlen>2)
        if constexpr (simd_exists<T,vlen/2>)
          if (it.remaining()>=vlen/2)
            {
            it.advance(vlen/2);
            auto tdatav = reinterpret_cast<typename simd_select<T,vlen/2>::type *>(storage.data());
            for (size_t j=0; j<vlen/2; ++j)
              tdatav[0][j]=in.raw(it.iofs(j,0)).r;
            {
            size_t i=1, ii=1;
            if (forward)
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/2; ++j)
                  {
                  tdatav[i  ][j] =  in.raw(it.iofs(j,ii)).r;
                  tdatav[i+1][j] = -in.raw(it.iofs(j,ii)).i;
                  }
            else
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/2; ++j)
                  {
                  tdatav[i  ][j] = in.raw(it.iofs(j,ii)).r;
                  tdatav[i+1][j] = in.raw(it.iofs(j,ii)).i;
                  }
            if (i<len)
              for (size_t j=0; j<vlen/2; ++j)
                tdatav[i][j] = in.raw(it.iofs(j,ii)).r;
            }
            plan->exec(tdatav, fct, false, nth1d);
            copy_output(it, tdatav, out);
            }
      if constexpr (vlen>4)
        if constexpr(simd_exists<T,vlen/4>)
          if (it.remaining()>=vlen/4)
            {
            it.advance(vlen/4);
            auto tdatav = reinterpret_cast<typename simd_select<T,vlen/4>::type *>(storage.data());
            for (size_t j=0; j<vlen/4; ++j)
              tdatav[0][j]=in.raw(it.iofs(j,0)).r;
            {
            size_t i=1, ii=1;
            if (forward)
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/4; ++j)
                  {
                  tdatav[i  ][j] =  in.raw(it.iofs(j,ii)).r;
                  tdatav[i+1][j] = -in.raw(it.iofs(j,ii)).i;
                  }
            else
              for (; i<len-1; i+=2, ++ii)
                for (size_t j=0; j<vlen/4; ++j)
                  {
                  tdatav[i  ][j] = in.raw(it.iofs(j,ii)).r;
                  tdatav[i+1][j] = in.raw(it.iofs(j,ii)).i;
                  }
            if (i<len)
              for (size_t j=0; j<vlen/4; ++j)
                tdatav[i][j] = in.raw(it.iofs(j,ii)).r;
            }
            plan->exec(tdatav, fct, false, nth1d);
            copy_output(it, tdatav, out);
            }
#endif
      while (it.remaining()>0)
        {
        it.advance(1);
        auto tdata = reinterpret_cast<T *>(storage.data());
        tdata[0]=in.raw(it.iofs(0)).r;
        {
        size_t i=1, ii=1;
        if (forward)
          for (; i<len-1; i+=2, ++ii)
            {
            tdata[i  ] =  in.raw(it.iofs(ii)).r;
            tdata[i+1] = -in.raw(it.iofs(ii)).i;
            }
        else
          for (; i<len-1; i+=2, ++ii)
            {
            tdata[i  ] = in.raw(it.iofs(ii)).r;
            tdata[i+1] = in.raw(it.iofs(ii)).i;
            }
        if (i<len)
          tdata[i] = in.raw(it.iofs(ii)).r;
        }
        plan->exec(tdata, fct, false, nth1d);
        copy_output(it, tdata, out);
        }
    });  // end of parallel region
  }

struct ExecR2R
  {
  bool r2c, forward;

  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator () (
    const Titer &it, const cfmav<T0> &in, vfmav<T0> &out, T *buf,
    const pocketfft_r<T0> &plan, T0 fct, size_t nthreads,
    bool inplace=false) const
    {
    if constexpr(is_same<T0, T>::value)
      if (inplace)
        {
        T *buf1=buf, *buf2=out.data();
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

    T *buf1=buf, *buf2=buf+plan.bufsize();
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
  };

/// Complex-to-complex Fast Fourier Transform
/** This executes a Fast Fourier Transform on \a in and stores the result in
 *  \a out.
 *
 *  \a in and \a out must have identical shapes; they may point to the same
 *  memory; in this case their strides must also be identical.
 *
 *  \a axes specifies the axes over which the transform is carried out.
 * 
 *  If \a forward is true, a minus sign will be used in the exponent.
 * 
 *  No normalization factors will be applied by default; if multiplication by
 *  a constant is desired, it can be supplied in \a fct.
 * 
 *  If the underlying array has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void c2c(const cfmav<std::complex<T>> &in,
  vfmav<std::complex<T>> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  cfmav<Cmplx<T>> in2(reinterpret_cast<const Cmplx<T> *>(in.data()), in);
  vfmav<Cmplx<T>> out2(reinterpret_cast<Cmplx<T> *>(out.data()), out);
  general_nd<pocketfft_c<T>>(in2, out2, axes, fct, nthreads, ExecC2C{forward});
  }

/// Fast Discrete Cosine Transform
/** This executes a DCT on \a in and stores the result in \a out.
 *
 *  \a in and \a out must have identical shapes; they may point to the same
 *  memory; in this case their strides must also be identical.
 *
 *  \a axes specifies the axes over which the transform is carried out.
 * 
 *  If \a forward is true, a DCT is computed, otherwise an inverse DCT.
 *
 *  \a type specifies the desired type (1-4) of the transform.
 * 
 *  No normalization factors will be applied by default; if multiplication by
 *  a constant is desired, it can be supplied in \a fct.
 *
 *  If \a ortho is true, the first and last array entries are corrected (if
 *  necessary) to allow an orthonormalized transform.
 * 
 *  If the underlying array has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void dct(const cfmav<T> &in, vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads=1)
  {
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DCT type");
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

/// Fast Discrete Sine Transform
/** This executes a DST on \a in and stores the result in \a out.
 *
 *  \a in and \a out must have identical shapes; they may point to the same
 *  memory; in this case their strides must also be identical.
 *
 *  \a axes specifies the axes over which the transform is carried out.
 * 
 *  If \a forward is true, a DST is computed, otherwise an inverse DST.
 *
 *  \a type specifies the desired type (1-4) of the transform.
 * 
 *  No normalization factors will be applied by default; if multiplication by
 *  a constant is desired, it can be supplied in \a fct.
 *
 *  If \a ortho is true, the first and last array entries are corrected (if
 *  necessary) to allow an orthonormalized transform.
 * 
 *  If the underlying array has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void dst(const cfmav<T> &in, vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads=1)
  {
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DST type");
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
  vfmav<std::complex<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads=1)
  {
  util::sanity_check_cr(out, in, axis);
  if (in.size()==0) return;
  vfmav<Cmplx<T>> out2(reinterpret_cast<Cmplx<T> *>(out.data()), out);
  general_r2c(in, out2, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2c(const cfmav<T> &in,
  vfmav<std::complex<T>> &out, const shape_t &axes,
  bool forward, T fct, size_t nthreads=1)
  {
  util::sanity_check_cr(out, in, axes);
  if (in.size()==0) return;
  r2c(in, out, axes.back(), forward, fct, nthreads);
  if (axes.size()==1) return;

  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(out, out, newaxes, forward, T(1), nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<std::complex<T>> &in,
  vfmav<T> &out,  size_t axis, bool forward, T fct, size_t nthreads=1)
  {
  util::sanity_check_cr(in, out, axis);
  if (in.size()==0) return;
  cfmav<Cmplx<T>> in2(reinterpret_cast<const Cmplx<T> *>(in.data()), in);
  general_c2r(in2, out, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const cfmav<std::complex<T>> &in,
  vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads=1)
  {
  if (axes.size()==1)
    return c2r(in, out, axes[0], forward, fct, nthreads);
  util::sanity_check_cr(in, out, axes);
  if (in.size()==0) return;
  auto atmp(vfmav<std::complex<T>>::build_noncritical(in.shape(), UNINITIALIZED));
  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(in, atmp, newaxes, forward, T(1), nthreads);
  c2r(atmp, out, axes.back(), forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2r_fftpack(const cfmav<T> &in,
  vfmav<T> &out, const shape_t &axes, bool real2hermitian, bool forward,
  T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_r<T>>(in, out, axes, fct, nthreads,
    ExecR2R{real2hermitian, forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_fftw(const cfmav<T> &in,
  vfmav<T> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_fftw<T>>(in, out, axes, fct, nthreads,
    ExecFFTW{forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_separable_hartley(const cfmav<T> &in,
  vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_hartley<T>>(in, out, axes, fct, nthreads,
    ExecHartley{}, false);
  }

template<typename T0, typename T1, typename Func> void hermiteHelper(size_t idim, ptrdiff_t iin,
  ptrdiff_t iout0, ptrdiff_t iout1, const cfmav<T0> &c,
  vfmav<T1> &r, const shape_t &axes, Func func, size_t nthreads)
  {
  auto cstr=c.stride(idim), str=r.stride(idim);
  auto len=r.shape(idim);

  if (idim+1==c.ndim())  // last dimension, not much gain in parallelizing
    {
    if (idim==axes.back())  // halfcomplex axis
      for (size_t i=0; i<len/2+1; ++i)
        {
        size_t j = (i==0) ? 0 : len-i;
        size_t io0=iout0+i*str, io1=iout1+j*str;
        func (c.raw(iin+i*cstr), r.raw(io0), r.raw(io1));
        }
    else if (find(axes.begin(), axes.end(), idim) != axes.end())  // FFT axis
      for (size_t i=0; i<len; ++i)
        {
        size_t j = (i==0) ? 0 : len-i;
        size_t io0=iout0+i*str, io1=iout1+j*str;
        func (c.raw(iin+i*cstr), r.raw(io0), r.raw(io1));
        }
    else  // non-FFT axis
      for (size_t i=0; i<len; ++i)
        func (c.raw(iin+i*cstr), r.raw(iout0+i*str), r.raw(iout1+i*str));
    }
  else
    {
    if (idim==axes.back())
      {
      if (nthreads==1)
        for (size_t i=0; i<len/2+1; ++i)
          {
          size_t j = (i==0) ? 0 : len-i;
          size_t io0=iout0+i*str, io1=iout1+j*str;
          hermiteHelper(idim+1, iin+i*cstr, io0, io1, c, r, axes, func, 1);
          }
      else
        execParallel(0, len/2+1, nthreads, [&](size_t lo, size_t hi)
          {
          for (size_t i=lo; i<hi; ++i)
            {
            size_t j = (i==0) ? 0 : len-i;
            size_t io0=iout0+i*str, io1=iout1+j*str;
            hermiteHelper(idim+1, iin+i*cstr, io0, io1, c, r, axes, func, 1);
            }
          });
      }
    else if (find(axes.begin(), axes.end(), idim) != axes.end())
      {
      if (nthreads==1)
        {
        for (size_t i=0; i<len; ++i)
          {
          size_t j = (i==0) ? 0 : len-i;
          size_t io0=iout0+i*str, io1=iout1+j*str;
          hermiteHelper(idim+1, iin+i*cstr, io0, io1, c, r, axes, func, 1);
          }
        }
      else
        execParallel(0, len/2+1, nthreads, [&](size_t lo, size_t hi)
          {
          for (size_t i=lo; i<hi; ++i)
            {
            size_t j = (i==0) ? 0 : len-i;
            size_t io0=iout0+i*str, io1=iout1+j*str;
            hermiteHelper(idim+1, iin+i*cstr, io0, io1, c, r, axes, func, 1);
            if (i!=j)
              hermiteHelper(idim+1, iin+j*cstr, io1, io0, c, r, axes, func, 1);
            }
          });
      }
    else
      {
      if (nthreads==1)
        for (size_t i=0; i<len; ++i)
          hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+i*str, c, r, axes, func, 1);
      else
         execParallel(0, len, nthreads, [&](size_t lo, size_t hi)
          {
          for (size_t i=lo; i<hi; ++i)
            hermiteHelper(idim+1, iin+i*cstr, iout0+i*str, iout1+i*str, c, r, axes, func, 1);
          });
      }
    }
  }

template<typename T> void oscarize(vfmav<T> &data, size_t ax0, size_t ax1,
  size_t nthreads)
  {
  vfmav d(data);
  // sort axes to have decreasing strides from ax0 to ax1
  if (d.stride(ax0)<d.stride(ax1)) swap(ax0, ax1);
  d.swap_axes(ax0, d.ndim()-2);
  d.swap_axes(ax1, d.ndim()-1);
  flexible_mav_apply<2>([nthreads](const auto &plane)
    {
    auto nu=plane.shape(0), nv=plane.shape(1);
    execParallel((nu+1)/2-1, nthreads, [&](size_t lo, size_t hi)
      {
      for(auto i=lo+1; i<hi+1; ++i)
        for(size_t j=1; j<(nv+1)/2; ++j)
          {
          T ll = plane(i   ,j   );
          T hl = plane(nu-i,j   );
          T lh = plane(i   ,nv-j);
          T hh = plane(nu-i,nv-j);
          T v = T(0.5)*(ll+lh+hl+hh);
          plane(i   ,j   ) = v-hh;
          plane(nu-i,j   ) = v-lh;
          plane(i   ,nv-j) = v-hl;
          plane(nu-i,nv-j) = v-ll;
          }
      });
    }, 1, d);
  }

// Bortfeld & Dinter, IEEE Transactions on Signal Processing 43, 1995, 1306 
template<typename T> void oscarize3(vfmav<T> &data, size_t ax0, size_t ax1, size_t ax2,
  size_t nthreads)
  {
  vfmav d(data);
  // sort axes to have decreasing strides from ax0 to ax2
  if (d.stride(ax0)<d.stride(ax1)) swap(ax0, ax1);
  if (d.stride(ax0)<d.stride(ax2)) swap(ax0, ax2);
  if (d.stride(ax1)<d.stride(ax2)) swap(ax1, ax2);
  d.swap_axes(ax0, d.ndim()-3);
  d.swap_axes(ax1, d.ndim()-2);
  d.swap_axes(ax2, d.ndim()-1);
  flexible_mav_apply<3>([nthreads](const auto &plane)
    {
    auto nu=plane.shape(0), nv=plane.shape(1), nw=plane.shape(2);
    execParallel(nu/2+1, nthreads, [&](size_t lo, size_t hi)
      {
      for(auto i=lo, xi=(i==0)?0:nu-i; i<hi; ++i, xi=nu-i)
        for(size_t j=0, xj=0; j<=xj; ++j, xj=nv-j)
          for(size_t k=0, xk=0; k<=xk; ++k, xk=nw-k)
            {
            T lll = plane(i ,j ,k );
            T hll = plane(xi,j ,k );
            T lhl = plane(i ,xj,k );
            T hhl = plane(xi,xj,k );
            T llh = plane(i ,j ,xk);
            T hlh = plane(xi,j ,xk);
            T lhh = plane(i ,xj,xk);
            T hhh = plane(xi,xj,xk);
            plane(i ,j ,k ) = T(0.5)*(llh+lhl+hll-hhh);
            plane(xi,j ,k ) = T(0.5)*(hlh+hhl+lll-lhh);
            plane(i ,xj,k ) = T(0.5)*(lhh+lll+hhl-hlh);
            plane(xi,xj,k ) = T(0.5)*(hhh+hll+lhl-llh);
            plane(i ,j ,xk) = T(0.5)*(lll+lhh+hlh-hhl);
            plane(xi,j ,xk) = T(0.5)*(hll+hhh+llh-lhl);
            plane(i ,xj,xk) = T(0.5)*(lhl+llh+hhh-hll);
            plane(xi,xj,xk) = T(0.5)*(hhl+hlh+lhh-lll);
            }
      });
    }, 1, d);
  }

template<typename T> void r2r_genuine_hartley(const cfmav<T> &in,
  vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1)
  {
  if (axes.size()==1)
    return r2r_separable_hartley(in, out, axes, fct, nthreads);
  if (axes.size()==2)
    {
    r2r_separable_hartley(in, out, axes, fct, nthreads);
    oscarize(out, axes[0], axes[1], nthreads);
    return;
    }
  if (axes.size()==3)
    {
    r2r_separable_hartley(in, out, axes, fct, nthreads);
    oscarize3(out, axes[0], axes[1], axes[2], nthreads);
    return;
    }
  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  if (in.size()==0) return;
  shape_t tshp(in.shape());
  tshp[axes.back()] = tshp[axes.back()]/2+1;
  auto atmp(vfmav<std::complex<T>>::build_noncritical(tshp, UNINITIALIZED));
  r2c(in, atmp, axes, true, fct, nthreads);
  hermiteHelper(0, 0, 0, 0, atmp, out, axes, [](const std::complex<T> &c, T &r0, T &r1)
    {
#ifdef DUCC0_USE_PROPER_HARTLEY_CONVENTION
    r0 = c.real()-c.imag();
    r1 = c.real()+c.imag();
#else
    r0 = c.real()+c.imag();
    r1 = c.real()-c.imag();
#endif
    }, nthreads);
  }

template<typename T, typename T0> aligned_array<T> alloc_tmp_conv_axis
  (const fmav_info &info, size_t axis, size_t len, size_t bufsize)
  {
  auto othersize = info.size()/info.shape(axis);
  constexpr auto vlen = fft_simdlen<T0>;
  return aligned_array<T>((len+bufsize)*std::min(vlen, othersize));
  }

template<typename Tplan, typename T0, typename T, typename Exec>
DUCC0_NOINLINE void general_convolve_axis(const cfmav<T> &in, vfmav<T> &out,
  const size_t axis, const cmav<T,1> &kernel, size_t nthreads,
  const Exec &exec)
  {
  std::unique_ptr<Tplan> plan1, plan2;

  size_t l_in=in.shape(axis), l_out=out.shape(axis);
  size_t l_max=std::max(l_in, l_out);
  MR_assert(kernel.size()==l_in, "bad kernel size");
  plan1 = std::make_unique<Tplan>(l_in);
  plan2 = std::make_unique<Tplan>(l_out);
  size_t bufsz = max(plan1->bufsize(), plan2->bufsize());

  vmav<T,1> fkernel({kernel.shape(0)});
  for (size_t i=0; i<kernel.shape(0); ++i)
    fkernel(i) = kernel(i);
  plan1->exec(fkernel.data(), T0(1)/T0(l_in), true, nthreads);

  execParallel(
    util::thread_count(nthreads, in, axis, fft_simdlen<T0>),
    [&](Scheduler &sched) {
      constexpr auto vlen = fft_simdlen<T0>;
      auto storage = alloc_tmp_conv_axis<T,T0>(in, axis, l_max, bufsz);
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          auto tdatav = reinterpret_cast<add_vec_t<T, vlen> *>(storage.data());
          exec(it, in, out, tdatav, *plan1, *plan2, fkernel);
          }
      if constexpr (vlen>2)
        if constexpr (simd_exists<T,vlen/2>)
          if (it.remaining()>=vlen/2)
            {
            it.advance(vlen/2);
            auto tdatav = reinterpret_cast<add_vec_t<T, vlen/2> *>(storage.data());
            exec(it, in, out, tdatav, *plan1, *plan2, fkernel);
            }
      if constexpr (vlen>4)
        if constexpr (simd_exists<T,vlen/4>)
          if (it.remaining()>=vlen/4)
            {
            it.advance(vlen/4);
            auto tdatav = reinterpret_cast<add_vec_t<T, vlen/4> *>(storage.data());
            exec(it, in, out, tdatav, *plan1, *plan2, fkernel);
            }
#endif
      while (it.remaining()>0)
        {
        it.advance(1);
        auto buf = reinterpret_cast<T *>(storage.data());
        exec(it, in, out, buf, *plan1, *plan2, fkernel);
        }
    });  // end of parallel region
  }

struct ExecConv1R
  {
  template <typename T0, typename T, typename Titer> void operator() (
    const Titer &it, const cfmav<T0> &in, vfmav<T0> &out,
    T * buf, const pocketfft_r<T0> &plan1, const pocketfft_r<T0> &plan2,
    const cmav<T0,1> &fkernel) const
    {
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out),
           bufsz = max(plan1.bufsize(), plan2.bufsize());
    T *buf1=buf, *buf2=buf+bufsz; 
    copy_input(it, in, buf2);
    auto res = plan1.exec(buf2, buf1, T0(1), true);
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
    res = plan2.exec(res, res==buf2 ? buf1 : buf2, T0(1), false);
    copy_output(it, res, out);
    }
  };
struct ExecConv1C
  {
  template <typename T0, typename T, typename Titer> void operator() (
    const Titer &it, const cfmav<Cmplx<T0>> &in, vfmav<Cmplx<T0>> &out,
    T *buf, const pocketfft_c<T0> &plan1, const pocketfft_c<T0> &plan2,
    const cmav<Cmplx<T0>,1> &fkernel) const
    {
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out),
           bufsz = max(plan1.bufsize(), plan2.bufsize());
    T *buf1=buf, *buf2=buf+bufsz;
    copy_input(it, in, buf2);
    auto res = plan1.exec(buf2, buf1, T0(1), true);
    auto res2 = (res==buf2) ? buf1 : buf2;
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
    res = plan2.exec(res2, res, T0(1), false);
    copy_output(it, res, out);
    }
  };

/// Convolution and zero-padding/truncation along one axis
/** This performs a circular convolution with the kernel \a kernel on axis
 *  \a axis of \a in, applies the necessary zero-padding/truncation on this
 *  axis to give it the length \a out.shape(axis),and returns the result
 *  in \a out.
 *
 *  The main purpose of this routine is efficiency: the combination of the above
 *  operations can be carried out more quickly than running the individual
 *  operations in succession.
 * 
 *  \a in and \a out must have identical shapes, with the possible exception
 *  of the axis \a axis; they may point to the same memory; in this case all
 *  of their strides must be identical.
 *
 *  \a axis specifies the axis over which the operation is carried out.
 *
 *  \a kernel must have the same length as \a in.shape(axis); it must be
 *  provided in the same domain as \a in (i.e. not pre-transformed).
 * 
 *  If \a in has more than one dimension, the computation will
 *  be distributed over \a nthreads threads.
 */
template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<T> &in,
  vfmav<T> &out, size_t axis, const cmav<T,1> &kernel, size_t nthreads=1)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0) return;
  general_convolve_axis<pocketfft_r<T>, T>(in, out, axis, kernel, nthreads,
    ExecConv1R());
  }
template<typename T> DUCC0_NOINLINE void convolve_axis(const cfmav<complex<T>> &in,
  vfmav<complex<T>> &out, size_t axis, const cmav<complex<T>,1> &kernel,
  size_t nthreads=1)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0) return;
  cfmav<Cmplx<T>> in2(reinterpret_cast<const Cmplx<T> *>(in.data()), in);
  vfmav<Cmplx<T>> out2(reinterpret_cast<Cmplx<T> *>(out.data()), out);
  cmav<Cmplx<T>,1> kernel2(reinterpret_cast<const Cmplx<T> *>(kernel.data()), kernel.shape());
  general_convolve_axis<pocketfft_c<T>, T>(in2, out2, axis, kernel2, nthreads,
    ExecConv1C());
  }

} // namespace detail_fft

using detail_fft::FORWARD;
using detail_fft::BACKWARD;
using detail_fft::c2c;
using detail_fft::c2r;
using detail_fft::r2c;
using detail_fft::r2r_fftpack;
using detail_fft::r2r_fftw;
using detail_fft::r2r_separable_hartley;
using detail_fft::r2r_genuine_hartley;
using detail_fft::dct;
using detail_fft::dst;
using detail_fft::convolve_axis;

} // namespace ducc0

#endif // POCKETFFT_HDRONLY_H
