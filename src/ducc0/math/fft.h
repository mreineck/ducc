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

#include "ducc0/math/fft1d.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <vector>
#include <complex>
#include <algorithm>
#include "ducc0/infra/threading.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/mav.h"
#ifndef DUCC0_NO_THREADING
#include <mutex>
#endif

namespace ducc0 {

namespace detail_fft {

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
    DUCC0_NOINLINE T_dct1(size_t length)
      : fftplan(2*(length-1)) {}

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/) const
      {
      constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);
      size_t N=fftplan.length(), n=N/2+1;
      if (ortho)
        { c[0]*=sqrt2; c[n-1]*=sqrt2; }
      auto tmp=&buf[0];
      tmp[0] = c[0];
      for (size_t i=1; i<n; ++i)
        tmp[i] = tmp[N-i] = c[i];
      auto res = fftplan.exec(tmp, &buf[N], fct, true);
      c[0] = res[0];
      for (size_t i=1; i<n; ++i)
        c[i] = res[2*i-1];
      if (ortho)
        { c[0]*=sqrt2*T0(0.5); c[n-1]*=sqrt2*T0(0.5); }
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool ortho,
      int /*type*/, bool /*cosine*/) const
      {
      aligned_array<T> buf(bufsize());
      exec(c, buf.data(), fct, ortho, 1, true);
      }

    size_t length() const { return fftplan.length()/2+1; }
    size_t bufsize() const { return fftplan.length()+fftplan.bufsize(); }
  };

template<typename T0> class T_dst1
  {
  private:
    pocketfft_r<T0> fftplan;

  public:
    DUCC0_NOINLINE T_dst1(size_t length)
      : fftplan(2*(length+1)) {}

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/) const
      {
      size_t N=fftplan.length(), n=N/2-1;
      auto tmp = &buf[0];
      tmp[0] = tmp[n+1] = c[0]*0;
      for (size_t i=0; i<n; ++i)
        { tmp[i+1]=c[i]; tmp[N-1-i]=-c[i]; }
      auto res = fftplan.exec(tmp, buf+N, fct, true);
      for (size_t i=0; i<n; ++i)
        c[i] = -res[2*i+2];
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct,
      bool /*ortho*/, int /*type*/, bool /*cosine*/) const
      {
      aligned_array<T> buf(bufsize());
      exec(c, buf.data(), fct, true, 1, false);
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
    DUCC0_NOINLINE T_dcst23(size_t length)
      : fftplan(length), twiddle(length)
      {
      UnityRoots<T0,Cmplx<T0>> tw(4*length);
      for (size_t i=0; i<length; ++i)
        twiddle[i] = tw[i+1].r;
      }

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool ortho,
      int type, bool cosine) const
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
        auto res = fftplan.exec(c, buf, fct, false);
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
        auto res = fftplan.exec(c, buf, fct, true);
        if (res != c) // FIXME: not yet optimal
          memcpy(c, res, N*sizeof(T));
        for (size_t k=1; k<N-1; k+=2)
          MPINPLACE(c[k], c[k+1]);
        if (!cosine)
          for (size_t k=1; k<N; k+=2)
            c[k] = -c[k];
        }
      return c;
      }
    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool ortho,
      int type, bool cosine) const
      {
      aligned_array<T> buf(bufsize());
      exec(c, &buf[0], fct, ortho, type, cosine);
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
    DUCC0_NOINLINE T_dcst4(size_t length)
      : N(length),
        fft((N&1) ? nullptr : new pocketfft_c<T0>(N/2)),
        rfft((N&1)? new pocketfft_r<T0>(N) : nullptr),
        C2((N&1) ? 0 : N/2)
      {
      if ((N&1)==0)
        {
        UnityRoots<T0,Cmplx<T0>> tw(16*N);
        for (size_t i=0; i<N/2; ++i)
          C2[i] = conj(tw[8*i+1]);
        }
      }

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T /*buf*/[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine) const
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
        rfft->exec(y.data(), fct, true);
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
        fft->exec(y.data(), fct, true);
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

    template<typename T> DUCC0_NOINLINE T *exec(T c[], T0 fct,
      bool /*ortho*/, int /*type*/, bool cosine) const
      {
      exec(c, nullptr, fct, true, 4, cosine);
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

class rev_iter
  {
  private:
    shape_t pos;
    fmav_info arr;
    std::vector<char> rev_axis;
    std::vector<char> rev_jump;
    size_t last_axis, last_size;
    shape_t shp;
    ptrdiff_t p, rp;
    size_t rem;

  public:
    rev_iter(const fmav_info &arr_, const shape_t &axes)
      : pos(arr_.ndim(), 0), arr(arr_), rev_axis(arr_.ndim(), 0),
        rev_jump(arr_.ndim(), 1), p(0), rp(0)
      {
      for (auto ax: axes)
        rev_axis[ax]=1;
      last_axis = axes.back();
      last_size = arr.shape(last_axis)/2 + 1;
      shp = arr.shape();
      shp[last_axis] = last_size;
      rem=1;
      for (auto i: shp)
        rem *= i;
      }
    void advance()
      {
      --rem;
      for (int i_=int(pos.size())-1; i_>=0; --i_)
        {
        auto i = size_t(i_);
        p += arr.stride(i);
        if (!rev_axis[i])
          rp += arr.stride(i);
        else
          {
          rp -= arr.stride(i);
          if (rev_jump[i])
            {
            rp += ptrdiff_t(arr.shape(i))*arr.stride(i);
            rev_jump[i] = 0;
            }
          }
        if (++pos[i] < shp[i])
          return;
        pos[i] = 0;
        p -= ptrdiff_t(shp[i])*arr.stride(i);
        if (rev_axis[i])
          {
          rp -= ptrdiff_t(arr.shape(i)-shp[i])*arr.stride(i);
          rev_jump[i] = 1;
          }
        else
          rp -= ptrdiff_t(shp[i])*arr.stride(i);
        }
      }
    ptrdiff_t ofs() const { return p; }
    ptrdiff_t rev_ofs() const { return rp; }
    size_t remaining() const { return rem; }
  };

template<typename T, typename T0> DUCC0_NOINLINE aligned_array<T> alloc_tmp
  (const fmav_info &info, size_t axsize)
  {
  auto othersize = info.size()/axsize;
  constexpr auto vlen = native_simd<T0>::size();
  // FIXME: when switching to C++20, use bit_floor(othersize)
  return aligned_array<T>(axsize*std::min(vlen, othersize));
  }

template<typename T, typename T0> DUCC0_NOINLINE aligned_array<T> alloc_tmp
  (const fmav_info &info, size_t axsize, size_t bufsize)
  {
  auto othersize = info.size()/axsize;
  constexpr auto vlen = native_simd<T0>::size();
  // FIXME: when switching to C++20, use bit_floor(othersize)
  return aligned_array<T>((axsize+bufsize)*std::min(vlen, othersize));
  }

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const fmav<Cmplx<typename Tsimd::Ts>> &src, Cmplx<Tsimd> *DUCC0_RESTRICT dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_i())
    {
    auto ptr = &src.craw(it.iofs_uni(0,0));
    auto jstr = it.unistride_i();
    auto istr = it.stride_in();
    if (istr==1)
      for (size_t i=0; i<it.length_in(); ++i)
        {
        Cmplx<Tsimd> stmp;
        for (size_t j=0; j<vlen; ++j)
          {
          auto tmp = ptr[ptrdiff_t(j)*jstr+ptrdiff_t(i)];
          stmp.r[j] = tmp.r;
          stmp.i[j] = tmp.i;
          }
        dst[i] = stmp;
        }
    else if (jstr==1)
      for (size_t i=0; i<it.length_in(); ++i)
        {
        Cmplx<Tsimd> stmp;
        for (size_t j=0; j<vlen; ++j)
          {
          auto tmp = ptr[ptrdiff_t(j)+ptrdiff_t(i)*istr];
          stmp.r[j] = tmp.r;
          stmp.i[j] = tmp.i;
          }
        dst[i] = stmp;
        }
    else
      for (size_t i=0; i<it.length_in(); ++i)
        {
        Cmplx<Tsimd> stmp;
        for (size_t j=0; j<vlen; ++j)
          {
          auto tmp = src.craw(it.iofs_uni(j,i));
          stmp.r[j] = tmp.r;
          stmp.i[j] = tmp.i;
          }
        dst[i] = stmp;
        }
    }
  else
    for (size_t i=0; i<it.length_in(); ++i)
      {
      Cmplx<Tsimd> stmp;
      for (size_t j=0; j<vlen; ++j)
        {
        auto tmp = src.craw(it.iofs(j,i));
        stmp.r[j] = tmp.r;
        stmp.i[j] = tmp.i;
        }
      dst[i] = stmp;
      }
  }

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_input(const Titer &it,
  const fmav<typename Tsimd::Ts> &src, Tsimd *DUCC0_RESTRICT dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_i())
    {
    auto ptr = &src.craw(it.iofs_uni(0,0));
    auto jstr = it.unistride_i();
    auto istr = it.stride_in();
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
          dst[i][j] = src.craw(it.iofs_uni(j,i));
    }
  else
    for (size_t i=0; i<it.length_in(); ++i)
      for (size_t j=0; j<vlen; ++j)
        dst[i][j] = src.craw(it.iofs(j,i));
  }

template <typename T, size_t vlen> DUCC0_NOINLINE void copy_input(const multi_iter<vlen> &it,
  const fmav<T> &src, T *DUCC0_RESTRICT dst)
  {
  if (dst == &src.craw(it.iofs(0))) return;  // in-place
  for (size_t i=0; i<it.length_in(); ++i)
    dst[i] = src.craw(it.iofs(i));
  }

template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Cmplx<Tsimd> *DUCC0_RESTRICT src, fmav<Cmplx<typename Tsimd::Ts>> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_o())
    {
    auto ptr = &dst.vraw(it.oofs_uni(0,0));
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
          dst.vraw(it.oofs_uni(j,i)).Set(src[i].r[j],src[i].i[j]);
    }
  else
    {
    auto ptr = dst.vdata();
    for (size_t i=0; i<it.length_out(); ++i)
      for (size_t j=0; j<vlen; ++j)
        ptr[it.oofs(j,i)].Set(src[i].r[j],src[i].i[j]);
    }
  }

template<typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_output(const Titer &it,
  const Tsimd *DUCC0_RESTRICT src, fmav<typename Tsimd::Ts> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_o())
    {
    auto ptr = &dst.vraw(it.oofs_uni(0,0));
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
          dst.vraw(it.oofs_uni(j,i)) = src[i][j];
    }
  else
    {
    auto ptr=dst.vdata();
    for (size_t i=0; i<it.length_out(); ++i)
      for (size_t j=0; j<vlen; ++j)
        ptr[it.oofs(j,i)] = src[i][j];
    }
  }

template<typename T, size_t vlen> DUCC0_NOINLINE void copy_output(const multi_iter<vlen> &it,
  const T *DUCC0_RESTRICT src, fmav<T> &dst)
  {
  auto ptr=dst.vdata();
  if (src == &dst.craw(it.oofs(0))) return;  // in-place
  for (size_t i=0; i<it.length_out(); ++i)
    ptr[it.oofs(i)] = src[i];
  }

template <typename T, size_t vlen> struct add_vec { using type = simd<T, vlen>; };
template <typename T, size_t vlen> struct add_vec<Cmplx<T>, vlen>
  { using type = Cmplx<simd<T, vlen>>; };
template <typename T, size_t vlen> using add_vec_t = typename add_vec<T, vlen>::type;

template<typename Tplan, typename T, typename T0, typename Exec>
DUCC0_NOINLINE void general_nd(const fmav<T> &in, fmav<T> &out,
  const shape_t &axes, T0 fct, size_t nthreads, const Exec &exec,
  const bool /*allow_inplace*/=true)
  {
  std::unique_ptr<Tplan> plan;

  for (size_t iax=0; iax<axes.size(); ++iax)
    {
    size_t len=in.shape(axes[iax]);
    if ((!plan) || (len!=plan->length()))
      plan = std::make_unique<Tplan>(len);

    execParallel(
      util::thread_count(nthreads, in, axes[iax], native_simd<T0>::size()),
      [&](Scheduler &sched) {
        constexpr auto vlen = native_simd<T0>::size();
        auto storage = alloc_tmp<T,T0>(in, len, plan->bufsize());
        const auto &tin(iax==0? in : out);
        multi_iter<vlen> it(tin, out, axes[iax], sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
        if constexpr (vlen>1)
          while (it.remaining()>=vlen)
            {
            it.advance(vlen);
            auto tdatav = reinterpret_cast<add_vec_t<T, vlen> *>(storage.data());
            exec(it, tin, out, tdatav, *plan, fct);
            }
        if constexpr (simd_exists<T0,vlen/2>)
          if (it.remaining()>=vlen/2)
            {
            it.advance(vlen/2);
            auto tdatav = reinterpret_cast<add_vec_t<T, vlen/2> *>(storage.data());
            exec(it, tin, out, tdatav, *plan, fct);
            }
        if constexpr (simd_exists<T0,vlen/4>)
          if (it.remaining()>=vlen/4)
            {
            it.advance(vlen/4);
            auto tdatav = reinterpret_cast<add_vec_t<T, vlen/4> *>(storage.data());
            exec(it, tin, out, tdatav, *plan, fct);
            }
#endif
        while (it.remaining()>0)
          {
          it.advance(1);
          exec(it, tin, out, storage.data(), *plan, fct);
          }
      });  // end of parallel region
    fct = T0(1); // factor has been applied, use 1 for remaining axes
    }
  }

struct ExecC2C
  {
  bool forward;

  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator() (
    const Titer &it, const fmav<Cmplx<T0>> &in,
    fmav<Cmplx<T0>> &out, T *buf, const pocketfft_c<T0> &plan, T0 fct) const
    {
    T *buf1=buf, *buf2=buf+plan.bufsize(); 
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, forward);
    copy_output(it, res, out);
    }
  };

template <typename Tsimd, typename Titer> DUCC0_NOINLINE void copy_hartley(const Titer &it,
  const Tsimd *DUCC0_RESTRICT src, fmav<typename Tsimd::Ts> &dst)
  {
  constexpr auto vlen=Tsimd::size();
  if (it.uniform_o())
    {
    auto ptr = &dst.vraw(it.oofs_uni(0,0));
    auto jstr = it.unistride_o();
    auto istr = it.stride_out();
    if (istr==1)
      {
      for (size_t j=0; j<vlen; ++j)
        ptr[ptrdiff_t(j)*jstr] = src[0][j];
      size_t i=1, i1=1, i2=it.length_out()-1;
      for (i=1; i<it.length_out()-1; i+=2, ++i1, --i2)
        for (size_t j=0; j<vlen; ++j)
          {
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i1)] = src[i][j]+src[i+1][j];
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i2)] = src[i][j]-src[i+1][j];
          }
      if (i<it.length_out())
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i1)] = src[i][j];
      }
    else if (jstr==1)
      {
      for (size_t j=0; j<vlen; ++j)
        ptr[ptrdiff_t(j)] = src[0][j];
      size_t i=1, i1=1, i2=it.length_out()-1;
      for (i=1; i<it.length_out()-1; i+=2, ++i1, --i2)
        for (size_t j=0; j<vlen; ++j)
          {
          ptr[ptrdiff_t(j) + ptrdiff_t(i1)*istr] = src[i][j]+src[i+1][j];
          ptr[ptrdiff_t(j) + ptrdiff_t(i2)*istr] = src[i][j]-src[i+1][j];
          }
      if (i<it.length_out())
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j) + ptrdiff_t(i1)*istr] = src[i][j];
      }
    else
      {
      for (size_t j=0; j<vlen; ++j)
        ptr[ptrdiff_t(j)*jstr] = src[0][j];
      size_t i=1, i1=1, i2=it.length_out()-1;
      for (i=1; i<it.length_out()-1; i+=2, ++i1, --i2)
        for (size_t j=0; j<vlen; ++j)
          {
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i1)*istr] = src[i][j]+src[i+1][j];
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i2)*istr] = src[i][j]-src[i+1][j];
          }
      if (i<it.length_out())
        for (size_t j=0; j<vlen; ++j)
          ptr[ptrdiff_t(j)*jstr + ptrdiff_t(i1)*istr] = src[i][j];
      }
    }
  else
    {
    auto ptr = dst.vdata();
    for (size_t j=0; j<vlen; ++j)
      ptr[it.oofs(j,0)] = src[0][j];
    size_t i=1, i1=1, i2=it.length_out()-1;
    for (i=1; i<it.length_out()-1; i+=2, ++i1, --i2)
      for (size_t j=0; j<vlen; ++j)
        {
        ptr[it.oofs(j,i1)] = src[i][j]+src[i+1][j];
        ptr[it.oofs(j,i2)] = src[i][j]-src[i+1][j];
        }
    if (i<it.length_out())
      for (size_t j=0; j<vlen; ++j)
        ptr[it.oofs(j,i1)] = src[i][j];
    }
  }

template <typename T, size_t vlen> DUCC0_NOINLINE void copy_hartley(const multi_iter<vlen> &it,
  const T *DUCC0_RESTRICT src, fmav<T> &dst)
  {
  auto ptr = dst.vdata();
  ptr[it.oofs(0)] = src[0];
  size_t i=1, i1=1, i2=it.length_out()-1;
  for (i=1; i<it.length_out()-1; i+=2, ++i1, --i2)
    {
    ptr[it.oofs(i1)] = src[i]+src[i+1];
    ptr[it.oofs(i2)] = src[i]-src[i+1];
    }
  if (i<it.length_out())
    ptr[it.oofs(i1)] = src[i];
  }

struct ExecHartley
  {
  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator () (
    const Titer &it, const fmav<T0> &in, fmav<T0> &out,
    T *buf, const pocketfft_r<T0> &plan, T0 fct) const
    {
    T *buf1=buf, *buf2=buf+plan.bufsize(); 
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, true);
    copy_hartley(it, res, out);
    }
  };

struct ExecDcst
  {
  bool ortho;
  int type;
  bool cosine;

  template <typename T0, typename T, typename Tplan, typename Titer>
  DUCC0_NOINLINE void operator () (const Titer &it, const fmav<T0> &in,
    fmav <T0> &out, T * buf, const Tplan &plan, T0 fct) const
    {
    T *buf1=buf, *buf2=buf+plan.bufsize(); 
    copy_input(it, in, buf2);
    auto res = plan.exec(buf2, buf1, fct, ortho, type, cosine);
    copy_output(it, res, out);
    }
  };

template<typename T> DUCC0_NOINLINE void general_r2c(
  const fmav<T> &in, fmav<Cmplx<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  auto plan = std::make_unique<pocketfft_r<T>>(in.shape(axis));
  size_t len=in.shape(axis);
  execParallel(
    util::thread_count(nthreads, in, axis, native_simd<T>::size()),
    [&](Scheduler &sched) {
    constexpr auto vlen = native_simd<T>::size();
    auto storage = alloc_tmp<T,T>(in, len, plan->bufsize());
    multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
    if constexpr (vlen>1)
      while (it.remaining()>=vlen)
        {
        it.advance(vlen);
        auto tdatav = reinterpret_cast<simd<T,vlen> *>(storage.data());
        copy_input(it, in, tdatav);
        plan->exec(tdatav, fct, true);
        auto vout = out.vdata();
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
    if constexpr (simd_exists<T,vlen/2>)
      if (it.remaining()>=vlen/2)
        {
        it.advance(vlen/2);
        auto tdatav = reinterpret_cast<simd<T,vlen/2> *>(storage.data());
        copy_input(it, in, tdatav);
        plan->exec(tdatav, fct, true);
        auto vout = out.vdata();
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
    if constexpr (simd_exists<T,vlen/4>)
      if (it.remaining()>=vlen/4)
        {
        it.advance(vlen/4);
        auto tdatav = reinterpret_cast<simd<T,vlen/4> *>(storage.data());
        copy_input(it, in, tdatav);
        plan->exec(tdatav, fct, true);
        auto vout = out.vdata();
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
      plan->exec(tdata, fct, true);
      auto vout = out.vdata();
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
  const fmav<Cmplx<T>> &in, fmav<T> &out, size_t axis, bool forward, T fct,
  size_t nthreads)
  {
  auto plan = std::make_unique<pocketfft_r<T>>(out.shape(axis));
  size_t len=out.shape(axis);
  execParallel(
    util::thread_count(nthreads, in, axis, native_simd<T>::size()),
    [&](Scheduler &sched) {
      constexpr auto vlen = native_simd<T>::size();
      auto storage = alloc_tmp<T,T>(out, len, plan->bufsize());
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          auto tdatav = reinterpret_cast<native_simd<T> *>(storage.data());
          for (size_t j=0; j<vlen; ++j)
            tdatav[0][j]=in.craw(it.iofs(j,0)).r;
          {
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen; ++j)
                {
                tdatav[i  ][j] =  in.craw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = -in.craw(it.iofs(j,ii)).i;
                }
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen; ++j)
                {
                tdatav[i  ][j] = in.craw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = in.craw(it.iofs(j,ii)).i;
                }
          if (i<len)
            for (size_t j=0; j<vlen; ++j)
              tdatav[i][j] = in.craw(it.iofs(j,ii)).r;
          }
          plan->exec(tdatav, fct, false);
          copy_output(it, tdatav, out);
          }
      if constexpr (simd_exists<T,vlen/2>)
        if (it.remaining()>=vlen/2)
          {
          it.advance(vlen/2);
          auto tdatav = reinterpret_cast<simd<T,vlen/2> *>(storage.data());
          for (size_t j=0; j<vlen/2; ++j)
            tdatav[0][j]=in.craw(it.iofs(j,0)).r;
          {
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/2; ++j)
                {
                tdatav[i  ][j] =  in.craw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = -in.craw(it.iofs(j,ii)).i;
                }
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/2; ++j)
                {
                tdatav[i  ][j] = in.craw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = in.craw(it.iofs(j,ii)).i;
                }
          if (i<len)
            for (size_t j=0; j<vlen/2; ++j)
              tdatav[i][j] = in.craw(it.iofs(j,ii)).r;
          }
          plan->exec(tdatav, fct, false);
          copy_output(it, tdatav, out);
          }
      if constexpr (simd_exists<T,vlen/4>)
        if (it.remaining()>=vlen/4)
          {
          it.advance(vlen/4);
          auto tdatav = reinterpret_cast<simd<T,vlen/4> *>(storage.data());
          for (size_t j=0; j<vlen/4; ++j)
            tdatav[0][j]=in.craw(it.iofs(j,0)).r;
          {
          size_t i=1, ii=1;
          if (forward)
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/4; ++j)
                {
                tdatav[i  ][j] =  in.craw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = -in.craw(it.iofs(j,ii)).i;
                }
          else
            for (; i<len-1; i+=2, ++ii)
              for (size_t j=0; j<vlen/4; ++j)
                {
                tdatav[i  ][j] = in.craw(it.iofs(j,ii)).r;
                tdatav[i+1][j] = in.craw(it.iofs(j,ii)).i;
                }
          if (i<len)
            for (size_t j=0; j<vlen/4; ++j)
              tdatav[i][j] = in.craw(it.iofs(j,ii)).r;
          }
          plan->exec(tdatav, fct, false);
          copy_output(it, tdatav, out);
          }
#endif
      while (it.remaining()>0)
        {
        it.advance(1);
        auto tdata = reinterpret_cast<T *>(storage.data());
        tdata[0]=in.craw(it.iofs(0)).r;
        {
        size_t i=1, ii=1;
        if (forward)
          for (; i<len-1; i+=2, ++ii)
            {
            tdata[i  ] =  in.craw(it.iofs(ii)).r;
            tdata[i+1] = -in.craw(it.iofs(ii)).i;
            }
        else
          for (; i<len-1; i+=2, ++ii)
            {
            tdata[i  ] = in.craw(it.iofs(ii)).r;
            tdata[i+1] = in.craw(it.iofs(ii)).i;
            }
        if (i<len)
          tdata[i] = in.craw(it.iofs(ii)).r;
        }
        plan->exec(tdata, fct, false);
        copy_output(it, tdata, out);
        }
    });  // end of parallel region
  }

struct ExecR2R
  {
  bool r2c, forward;

  template <typename T0, typename T, typename Titer> DUCC0_NOINLINE void operator () (
    const Titer &it, const fmav<T0> &in, fmav<T0> &out, T *buf,
    const pocketfft_r<T0> &plan, T0 fct) const
    {
    T *buf1=buf, *buf2=buf+plan.bufsize();
    copy_input(it, in, buf2);
    if ((!r2c) && forward)
      for (size_t i=2; i<it.length_out(); i+=2)
        buf2[i] = -buf2[i];
    auto res = plan.exec(buf2, buf1, fct, r2c);
    if (r2c && (!forward))
      for (size_t i=2; i<it.length_out(); i+=2)
        res[i] = -res[i];
    copy_output(it, res, out);
    }
  };

template<typename T> DUCC0_NOINLINE void c2c(const fmav<std::complex<T>> &in,
  fmav<std::complex<T>> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.cdata()==out.cdata(), axes);
  if (in.size()==0) return;
  fmav<Cmplx<T>> in2(reinterpret_cast<const Cmplx<T> *>(in.cdata()), in);
  fmav<Cmplx<T>> out2(reinterpret_cast<Cmplx<T> *>(out.vdata()), out, out.writable());
  general_nd<pocketfft_c<T>>(in2, out2, axes, fct, nthreads, ExecC2C{forward});
  }

template<typename T> DUCC0_NOINLINE void dct(const fmav<T> &in, fmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads=1)
  {
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DCT type");
  util::sanity_check_onetype(in, out, in.cdata()==out.cdata(), axes);
  if (in.size()==0) return;
  const ExecDcst exec{ortho, type, true};
  if (type==1)
    general_nd<T_dct1<T>>(in, out, axes, fct, nthreads, exec);
  else if (type==4)
    general_nd<T_dcst4<T>>(in, out, axes, fct, nthreads, exec);
  else
    general_nd<T_dcst23<T>>(in, out, axes, fct, nthreads, exec);
  }

template<typename T> DUCC0_NOINLINE void dst(const fmav<T> &in, fmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads=1)
  {
  if ((type<1) || (type>4)) throw std::invalid_argument("invalid DST type");
  util::sanity_check_onetype(in, out, in.cdata()==out.cdata(), axes);
  const ExecDcst exec{ortho, type, false};
  if (type==1)
    general_nd<T_dst1<T>>(in, out, axes, fct, nthreads, exec);
  else if (type==4)
    general_nd<T_dcst4<T>>(in, out, axes, fct, nthreads, exec);
  else
    general_nd<T_dcst23<T>>(in, out, axes, fct, nthreads, exec);
  }

template<typename T> DUCC0_NOINLINE void r2c(const fmav<T> &in,
  fmav<std::complex<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads=1)
  {
  util::sanity_check_cr(out, in, axis);
  if (in.size()==0) return;
  fmav<Cmplx<T>> out2(reinterpret_cast<Cmplx<T> *>(out.vdata()), out, out.writable());
  general_r2c(in, out2, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2c(const fmav<T> &in,
  fmav<std::complex<T>> &out, const shape_t &axes,
  bool forward, T fct, size_t nthreads=1)
  {
  util::sanity_check_cr(out, in, axes);
  if (in.size()==0) return;
  r2c(in, out, axes.back(), forward, fct, nthreads);
  if (axes.size()==1) return;

  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(out, out, newaxes, forward, T(1), nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const fmav<std::complex<T>> &in,
  fmav<T> &out,  size_t axis, bool forward, T fct, size_t nthreads=1)
  {
  util::sanity_check_cr(in, out, axis);
  if (in.size()==0) return;
  fmav<Cmplx<T>> in2(reinterpret_cast<const Cmplx<T> *>(in.cdata()), in);
  general_c2r(in2, out, axis, forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void c2r(const fmav<std::complex<T>> &in,
  fmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads=1)
  {
  if (axes.size()==1)
    return c2r(in, out, axes[0], forward, fct, nthreads);
  util::sanity_check_cr(in, out, axes);
  if (in.size()==0) return;
  fmav<std::complex<T>> atmp(in.shape());
  auto newaxes = shape_t{axes.begin(), --axes.end()};
  c2c(in, atmp, newaxes, forward, T(1), nthreads);
  c2r(atmp, out, axes.back(), forward, fct, nthreads);
  }

template<typename T> DUCC0_NOINLINE void r2r_fftpack(const fmav<T> &in,
  fmav<T> &out, const shape_t &axes, bool real2hermitian, bool forward,
  T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.cdata()==out.cdata(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_r<T>>(in, out, axes, fct, nthreads,
    ExecR2R{real2hermitian, forward});
  }

template<typename T> DUCC0_NOINLINE void r2r_separable_hartley(const fmav<T> &in,
  fmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1)
  {
  util::sanity_check_onetype(in, out, in.cdata()==out.cdata(), axes);
  if (in.size()==0) return;
  general_nd<pocketfft_r<T>>(in, out, axes, fct, nthreads, ExecHartley{},
    false);
  }

template<typename T> void r2r_genuine_hartley(const fmav<T> &in,
  fmav<T> &out, const shape_t &axes, T fct, size_t nthreads=1)
  {
  if (axes.size()==1)
    return r2r_separable_hartley(in, out, axes, fct, nthreads);
  util::sanity_check_onetype(in, out, in.cdata()==out.cdata(), axes);
  if (in.size()==0) return;
  shape_t tshp(in.shape());
  tshp[axes.back()] = tshp[axes.back()]/2+1;
  fmav<std::complex<T>> atmp(tshp);
  r2c(in, atmp, axes, true, fct, nthreads);
  FmavIter iin(atmp);
  rev_iter iout(out, axes);
  auto vout = out.vdata();
  while(iin.remaining()>0)
    {
    auto v = atmp.craw(iin.ofs());
    vout[iout.ofs()] = v.real()+v.imag();
    vout[iout.rev_ofs()] = v.real()-v.imag();
    iin.advance(); iout.advance();
    }
  }

} // namespace detail_fft

using detail_fft::FORWARD;
using detail_fft::BACKWARD;
using detail_fft::c2c;
using detail_fft::c2r;
using detail_fft::r2c;
using detail_fft::r2r_fftpack;
using detail_fft::r2r_separable_hartley;
using detail_fft::r2r_genuine_hartley;
using detail_fft::dct;
using detail_fft::dst;

} // namespace ducc0

#endif // POCKETFFT_HDRONLY_H
