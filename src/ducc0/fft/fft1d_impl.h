/*
This file is part of the ducc FFT library

Copyright (C) 2010-2026 Max-Planck-Society
Copyright (C) 2019 Peter Bell
Copyright (C) 2026 Philipp Arras

For the odd-sized DCT-IV transforms:
  Copyright (C) 2003, 2007-14 Matteo Frigo
  Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

Authors: Martin Reinecke, Philipp Arras, Peter Bell
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

#ifndef DUCC0_FFT1D_IMPL_H
#define DUCC0_FFT1D_IMPL_H

#include <memory>
#include <array>
#include <bit>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>
#include <typeinfo>
#include <typeindex>
#include <numbers>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/threading.h"
#include "ducc0/math/unity_roots.h"
#include "ducc0/fft/fft.h"

namespace DUCC0_NAMESPACE {

namespace detail_fft {

using namespace std;

namespace {

// Check for gcc versions affected by
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=124138
// which breaks double precision real-valued transforms with AVX512.
// If detected, switch maximum double precision FFT vector length to 4.
#if defined(__GNUC__) && !defined(__clang__) && \
  ((__GNUC__<13) || \
   ((__GNUC__==13) && (__GNUC_MINOR__<5)) || \
   ((__GNUC__==14) && (__GNUC_MINOR__<4)) || \
   ((__GNUC__==15) && (__GNUC_MINOR__<3)))
template<typename T> constexpr inline size_t fft1d_max_simdlen
  = (sizeof(T)<8) ? native_simd<T>::size() : ((native_simd<T>::size()<=4) ? native_simd<T>::size() : 4);
#else
template<typename T> constexpr inline size_t fft1d_max_simdlen
  = native_simd<T>::size();
#endif

template <typename Tfs> class cfftp1: public cfftpass<Tfs>
  {
  public:
    cfftp1() {}
    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return false; }
    virtual size_t footprint() const { return 0; }

    virtual void *exec(const type_index & /*ti*/, void * in, void * /*copy*/,
      void * /*buf*/, bool /*fwd*/, size_t /*nthreads*/) const
      { return in; }
  };

// ---------------------------------------------------------------------------
// Compile-time generated radix codelets and the SoA single-transform driver
// ---------------------------------------------------------------------------

/* static_for<N>(f) calls f(integral_constant<size_t,I>()) for I=0..N-1. The
   lambdas must be inlined unconditionally: in large functions GCC left them
   out of line, which forced the codelets' temporaries onto the stack and made
   the scalar radix-7 pass 7x slower. */
#if defined(__GNUC__)
#define DUCC0_LAMBDA_INLINE __attribute__((always_inline))
#else
#define DUCC0_LAMBDA_INLINE
#endif
/* GCC 14.2 (manylinux_2_28 image, compiler of the official wheels) crashes
   on some static_for lambdas with -Wall -Wextra ("internal compiler error:
   in cxx_eval_indirect_ref"); for it, the code avoids them. */
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__==14) && (__GNUC_MINOR__==2)
#define DUCC0_GCC142_ICE 1
#else
#define DUCC0_GCC142_ICE 0
#endif
// if the second version works reliably, I'd like to switch to that one,
// but it's purely aesthetic ...
#if 0
template<typename F, size_t... I> DUCC0_ALWAYS_INLINE void static_for_impl(F &f, index_sequence<I...>)
  { (f(integral_constant<size_t,I>()), ...); }
template<size_t N, typename F> DUCC0_ALWAYS_INLINE void static_for(F &&f)
  { static_for_impl(f, make_index_sequence<N>()); }
#else
template<size_t N, typename F> DUCC0_ALWAYS_INLINE void static_for(F &&f)
  {
  [&]<size_t... Is>(index_sequence<Is...>) DUCC0_LAMBDA_INLINE {(f(integral_constant<size_t,Is>()), ...);}
    (make_index_sequence<N>());
  }
#endif
// f(j) for j<n: unrolled with compile-time j if N>0 (then N==n), else a loop
template<size_t N, typename F> DUCC0_ALWAYS_INLINE void xfor(size_t n, F &&f)
  { if constexpr (N>0) static_for<N>(f); else for (size_t j=0; j<n; ++j) f(j); }

/* The virtual exec() of a pass class D (CRTP): calls D::exec_<fwd>() with the
   pointers cast to the data type that ti names, the scalar type or the SIMD
   type of one of the widths fft1d_max_simdlen<Tfs>>>s (W wraps them: Cmplx for
   the complex passes). D declares this class a friend. */
template<typename D, typename Tfs, typename Base, template<typename> class W> class pass_impl: public Base
  {
  private:
    template<typename T> void *call(void *in, void *copy, void *buf, bool fwd, size_t nthreads) const
      {
      const D &self = static_cast<const D &>(*this);
      auto in1 = static_cast<T *>(in), copy1 = static_cast<T *>(copy), buf1 = static_cast<T *>(buf);
      return fwd ? self.template exec_<true>(in1, copy1, buf1, nthreads)
                 : self.template exec_<false>(in1, copy1, buf1, nthreads);
      }
#if DUCC0_GCC142_ICE  // a static_for lambda in exec() crashes GCC 14.2
    template<size_t s> void *exec_simd(const type_index &ti, void *in, void *copy, void *buf,
      bool fwd, size_t nthreads) const
      {
      constexpr size_t v = fft1d_max_simdlen<Tfs> >> s;
      if constexpr (v>1)
        {
        if constexpr (simd_exists<Tfs,v>)
          {
          using T = W<typename simd_select<Tfs,v>::type>;
          static const auto tiv = tidx<T *>();
          if (ti==tiv) return call<T>(in, copy, buf, fwd, nthreads);
          }
        return exec_simd<s+1>(ti, in, copy, buf, fwd, nthreads);
        }
      else
        MR_fail("impossible vector length requested");
      }
#endif

  public:
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf,
      bool fwd, size_t nthreads) const
      {
      static const auto tis = tidx<W<Tfs> *>();
      if (ti==tis) return call<W<Tfs>>(in, copy, buf, fwd, nthreads);
#if DUCC0_GCC142_ICE
      return exec_simd<0>(ti, in, copy, buf, fwd, nthreads);
#else
      void *res = nullptr;
      static_for<4>([&](auto s) DUCC0_LAMBDA_INLINE
        {
        constexpr size_t v = fft1d_max_simdlen<Tfs> >> decltype(s)::value;
        if constexpr ((v>1) && simd_exists<Tfs,v>)
          {
          using T = W<typename simd_select<Tfs,v>::type>;
          static const auto tiv = tidx<T *>();
          if ((!res) && (ti==tiv)) res = call<T>(in, copy, buf, fwd, nthreads);
          }
        });
      MR_assert(res, "impossible vector length requested");
      return res;
#endif
      }
  };
template<typename D, typename Tfs> using cfftpass_impl = pass_impl<D, Tfs, cfftpass<Tfs>, Cmplx>;
template<typename D, typename Tfs> using rfftpass_impl = pass_impl<D, Tfs, rfftpass<Tfs>, type_identity_t>;

struct ct_cs { long double c, s; };
/* cos and sin of 2*pi*num/den, evaluated at compile time. long double may be
   just double (e.g. on arm64 macOS), so the angle is reduced exactly, in
   integers, to the nearest quarter turn q, and the Taylor series of the rest
   (|x|<=pi/4) is summed from the smallest term: at most 1.6 ulp in double. */
consteval ct_cs ct_sincos(size_t num, size_t den)
  {
  num %= den;
  size_t q = (8*num+den)/(2*den);
  constexpr long double pi = 3.141592653589793238462643383279502884197L; //std::numbers::pi_v<long double>
  long double x = pi/2*(ptrdiff_t(4*num)-ptrdiff_t(q*den))/den,
              x2 = x*x, c = 0, s = 0;
  for (size_t n=15; n>0; --n)
    { c = -x2/((2*n-1)*(2*n))*(1+c); s = -x2/((2*n)*(2*n+1))*(1+s); }
  c += 1; s = x*(1+s);
  const ct_cs res[4] = {{c,s}, {-s,c}, {-c,-s}, {s,-c}};
  return res[q&3];
  }
consteval bool ct_isprime(size_t n)
  {
  for (size_t d=2; d*d<=n; ++d) if (n%d==0) return false;
  return n>1;
  }
// first factor R1 of the split R=R1*R2 used by the codelet
consteval size_t ct_split(size_t n)
  {
  size_t res=1;  // largest divisor d>1 with d*d<=n
  for (size_t d=2; d*d<=n; ++d) if (n%d==0) res=d;
  return res;
  }
template<size_t R, typename Tfs> struct SingletonTab
  { Tfs c[(R-1)/2][(R-1)/2], s[(R-1)/2][(R-1)/2]; };
template<size_t R, typename Tfs> consteval SingletonTab<R,Tfs> ct_singleton()
  {
  SingletonTab<R,Tfs> res{};
  for (size_t j=0; j<(R-1)/2; ++j)
    for (size_t k=0; k<(R-1)/2; ++k)
      {
      auto v = ct_sincos((j+1)*(k+1), R);
      res.c[j][k] = Tfs(v.c); res.s[j][k] = Tfs(v.s);
      }
  return res;
  }

// multiply x by exp(-+2*pi*i*e/R)
template<size_t e, size_t R, bool fwd, typename Tfs, typename V>
  DUCC0_ALWAYS_INLINE Cmplx<V> twmul(const Cmplx<V> &x)
  {
  using C = Cmplx<V>;
  constexpr size_t m = e%R;
  if constexpr (m==0) return x;
  else if constexpr (2*m==R) return C(-x.r,-x.i);
  else if constexpr (4*m==R) return fwd ? C(x.i,-x.r) : C(-x.i,x.r);
  else if constexpr (4*m==3*R) return fwd ? C(-x.i,x.r) : C(x.i,-x.r);
  else if constexpr ((8*m)%R==0)  // odd multiple of 45 degrees: |cos|==|sin|
    {
    constexpr auto cs = ct_sincos(m,R);
    constexpr bool pc = cs.c>0, ps = (fwd ? -cs.s : cs.s)>0;
    constexpr Tfs h = Tfs(1.414213562373095048801688724209698L/2); //Tfs(std::numbers::sqrt2_v<long double>/2);
    V t1=x.r+x.i, t2=x.r-x.i;
    if constexpr (pc==ps) return C(t2*(pc ? h : -h), t1*(pc ? h : -h));
    else return C(t1*(pc ? h : -h), t2*(ps ? h : -h));
    }
  else
    {
    constexpr auto cs = ct_sincos(m,R);
    constexpr Tfs c = Tfs(cs.c), s = Tfs(fwd ? -cs.s : cs.s);
    return C(x.r*c-x.i*s, x.r*s+x.i*c);
    }
  }

// in-place DFT of length R, natural order in and out, fully unrolled
template<size_t R, bool fwd, typename Tfs, typename V>
  DUCC0_ALWAYS_INLINE void dft(array<Cmplx<V>,R> &x)
  {
  using C = Cmplx<V>;
  if constexpr (R==1) {}
  else if constexpr (R==2)
    { C t=x[0]; x[0]=t+x[1]; x[1]=t-x[1]; }
  else if constexpr (ct_isprime(R))  // Singleton's algorithm
    {
    constexpr size_t h=(R-1)/2;
    constexpr auto cs = ct_singleton<R,Tfs>();
    array<C,h> t, d;
    static_for<h>([&](auto j) DUCC0_LAMBDA_INLINE{ t[j]=x[j+1]+x[R-1-j]; d[j]=x[j+1]-x[R-1-j]; });
    C x0=x[0];
    static_for<h>([&](auto j) DUCC0_LAMBDA_INLINE{ x[0]+=t[j]; });
    static_for<h>([&](auto k) DUCC0_LAMBDA_INLINE
      {
      C a=x0+t[0]*cs.c[0][k], b=d[0]*cs.s[0][k];
      static_for<h-1>([&](auto j) DUCC0_LAMBDA_INLINE{ a+=t[j+1]*cs.c[j+1][k]; b+=d[j+1]*cs.s[j+1][k]; });
      // with ib=i*b, the same operation in real and imaginary part: GCC
      // cannot vectorize loops over scalar data otherwise
      C ib(-b.i, b.r);
      x[k+1] = fwd ? a-ib : a+ib;
      x[R-1-k] = fwd ? a+ib : a-ib;
      });
    }
  else  // Cooley-Tukey, R = R1*R2
    {
    constexpr size_t R1=ct_split(R), R2=R/R1;
    array<C,R> t;
    static_for<R2>([&](auto n2) DUCC0_LAMBDA_INLINE
      {
      array<C,R1> y;
      static_for<R1>([&](auto n1) DUCC0_LAMBDA_INLINE{ y[n1]=x[n1*R2+n2]; });
      dft<R1,fwd,Tfs>(y);
      static_for<R1>([&](auto k1) DUCC0_LAMBDA_INLINE
        { t[n2*R1+k1]=twmul<decltype(n2)::value*decltype(k1)::value,R,fwd,Tfs>(y[k1]); });
      });
    static_for<R1>([&](auto k1) DUCC0_LAMBDA_INLINE
      {
      array<C,R2> z;
      static_for<R2>([&](auto n2) DUCC0_LAMBDA_INLINE{ z[n2]=t[n2*R1+k1]; });
      dft<R2,fwd,Tfs>(z);
      static_for<R2>([&](auto k2) DUCC0_LAMBDA_INLINE{ x[k1+R1*k2]=z[k2]; });
      });
    }
  }
/* Singleton's algorithm for an odd length n known at run time, O(n^2): from
   t[j]=x[j+1]+x[n-1-j] and t[h+j]=x[j+1]-x[n-1-j] (j<h=(n-1)/2) it passes to
   f(k,a,b), k=1..h, a=x0+sum_j t[j]*cos(2pi(j+1)k/n) and b=sum_j t[h+j]*sin(..);
   the outputs are a-+ib (k) and a+-ib (n-k). cs[m] = exp(2*pi*i*m/n). */
template<typename Tfs, typename C, size_t M, typename F>
  DUCC0_ALWAYS_INLINE void singleton_rt(const array<C,M> &t, const C &x0, size_t n, const Cmplx<Tfs> *cs, F &&f)
  {
  const size_t h=(n-1)/2;
  for (size_t k=1; k<=h; ++k)
    {
    C a=x0+t[0]*cs[k].r, b=t[h]*cs[k].i;
    for (size_t j=1, m=k; j<h; ++j)
      { m+=k; m-=(m>=n) ? n : 0; a+=t[j]*cs[m].r; b+=t[h+j]*cs[m].i; }
    f(k, a, b);
    }
  }
// in-place DFT of an odd length n<=M known at run time
template<bool fwd, typename Tfs, typename C, size_t M>
  DUCC0_ALWAYS_INLINE void dft_rt(array<C,M> &x, size_t n, const Cmplx<Tfs> *cs)
  {
  const size_t h=(n-1)/2;
  array<C,M> t;
  for (size_t j=0; j<h; ++j) { t[j]=x[j+1]+x[n-1-j]; t[h+j]=x[j+1]-x[n-1-j]; }
  const C x0=x[0];
  for (size_t j=0; j<h; ++j) x[0]+=t[j];
  singleton_rt(t, x0, n, cs, [&](size_t k, const C &a, const C &b) DUCC0_LAMBDA_INLINE
    {
    C ib(-b.i, b.r);
    x[k] = fwd ? a-ib : a+ib;
    x[n-k] = fwd ? a+ib : a-ib;
    });
  }

/* Lane shuffles on SIMD types: lane P of the result is lane F(P) of the
   concatenation of a and b. The index patterns are constexpr functions, not
   lambdas, because clang does not accept constants of an enclosing lambda
   inside a lambda used as template argument. */
template<auto F, typename Tfv> DUCC0_ALWAYS_INLINE Tfv shuf(const Tfv &a, const Tfv &b)
  {
#if defined(__GNUC__) || defined(__clang__)
  typedef typename Tfv::value_type G __attribute__((vector_size(sizeof(Tfv))));
  return [&]<size_t... P>(index_sequence<P...>)
    { return bit_cast<Tfv>(__builtin_shufflevector(bit_cast<G>(a), bit_cast<G>(b), F(P)...)); }
    (make_index_sequence<Tfv::size()>());
#else
  constexpr size_t n = Tfv::size();
  Tfv res;
  for (size_t p=0; p<n; ++p) res[p] = (F(p)<n) ? a[F(p)] : b[F(p)-n];
  return res;
#endif
  }
constexpr size_t sh_even(size_t i) { return 2*i; }
constexpr size_t sh_odd(size_t i) { return 2*i+1; }
template<size_t n> constexpr size_t sh_zip_lo(size_t i) { return (i&1) ? n+i/2 : i/2; }
template<size_t n> constexpr size_t sh_zip_hi(size_t i) { return (i&1) ? n+n/2+i/2 : n/2+i/2; }
template<size_t n> constexpr size_t sh_rev(size_t i) { return n-1-i; }
template<size_t b, size_t n> constexpr size_t sh_tr_lo(size_t p) { return (p&b) ? p-b+n : p; }
template<size_t b, size_t n> constexpr size_t sh_tr_hi(size_t p) { return (p&b) ? p+n : p+b; }
template<size_t J, size_t M, size_t R, size_t n> constexpr size_t sh_gather(size_t t)
  { return ((J+R*t)/n==M) ? n+(J+R*t)%n : t; }
// vlen consecutive complex numbers in memory <-> one SIMD complex number
template<typename Tfv, typename Tfs> DUCC0_ALWAYS_INLINE Cmplx<Tfv> vload(const Cmplx<Tfs> *p)
  {
  constexpr size_t vlen = Tfv::size();
  auto q = reinterpret_cast<const Tfs *>(p);
  Tfv a=loadu<Tfv>(q), b=loadu<Tfv>(q+vlen);
  return {shuf<sh_even>(a,b), shuf<sh_odd>(a,b)};
  }
template<typename Tfv, typename Tfs> DUCC0_ALWAYS_INLINE void vstore(const Cmplx<Tfv> &x, Cmplx<Tfs> *p)
  {
  constexpr size_t vlen = Tfv::size();
  auto q = reinterpret_cast<Tfs *>(p);
  storeu(shuf<sh_zip_lo<vlen>>(x.r, x.i), q);
  storeu(shuf<sh_zip_hi<vlen>>(x.r, x.i), q+vlen);
  }
/* the R x vl block of reals at p, transposed: x[j] lane t = p[j+R*t]; for
   R==0 the runtime size n, through a buffer */
template<size_t R, typename Tfv, typename Tfs, size_t L> DUCC0_ALWAYS_INLINE void vgather(const Tfs *p, array<Tfv,L> &x, size_t n=R)
  {
  constexpr size_t vl = Tfv::size();
  if constexpr (R==0)
    {
    array<Tfs,L*vl> b;
    for (size_t t=0; t<vl; ++t) for (size_t j=0; j<n; ++j) b[j*vl+t] = p[j+n*t];
    for (size_t j=0; j<n; ++j) x[j] = loadu<Tfv>(&b[j*vl]);
    return;
    }
  array<Tfv,R> v;
  static_for<R>([&](auto m) DUCC0_LAMBDA_INLINE{ v[m] = loadu<Tfv>(p+m*vl); });
  static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE
    {
    constexpr size_t J = decltype(j)::value;
    static_for<R>([&](auto m) DUCC0_LAMBDA_INLINE  // collect the lanes lying in v[m]
      {
      constexpr size_t M = decltype(m)::value;
      x[j] = shuf<sh_gather<J,M,R,vl>>(M==0 ? v[0] : x[j], v[m]);
      });
    });
  }
// the inverse: p[j+R*t] = x[j] lane t
template<size_t J, size_t M, size_t R, size_t n> constexpr size_t sh_scatter(size_t p)
  { return ((M*n+p)%R==J) ? n+(M*n+p)/R : p; }
template<size_t R, typename Tfv, typename Tfs, size_t L> DUCC0_ALWAYS_INLINE void vscatter(const array<Tfv,L> &x, Tfs *p, size_t n=R)
  {
  constexpr size_t vl = Tfv::size();
  if constexpr (R==0)
    {
    array<Tfs,L*vl> b;
    for (size_t j=0; j<n; ++j) storeu(x[j], &b[j*vl]);
    for (size_t t=0; t<vl; ++t) for (size_t j=0; j<n; ++j) p[j+n*t] = b[j*vl+t];
    return;
    }
  static_for<R>([&](auto m) DUCC0_LAMBDA_INLINE
    {
    constexpr size_t M = decltype(m)::value;
    Tfv v = x[0];
    static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE  // collect the lanes going to memory vector m
      { v = shuf<sh_scatter<decltype(j)::value,M,R,vl>>(v, x[j]); });
    storeu(v, p+M*vl);
    });
  }
// transpose a vlen x vlen block of numbers held in vlen SIMD values (real or complex)
template<size_t vlen, typename Tfv> DUCC0_ALWAYS_INLINE void transpose(array<Tfv,vlen> &v)
  {
  static_for<countr_zero(vlen)>([&](auto sc) DUCC0_LAMBDA_INLINE
    {
    constexpr size_t b = size_t(1)<<decltype(sc)::value;
    static_for<vlen>([&](auto ic) DUCC0_LAMBDA_INLINE
      {
      constexpr size_t i = decltype(ic)::value;
      if constexpr ((i&b)==0)
        {
        auto a=v[i], c=v[i+b];
        v[i]   = shuf<sh_tr_lo<b,vlen>>(a,c);
        v[i+b] = shuf<sh_tr_hi<b,vlen>>(a,c);
        }
      });
    });
  }
template<size_t vlen, typename Tfv> DUCC0_ALWAYS_INLINE void transpose(array<Cmplx<Tfv>,vlen> &v)
  {
  array<Tfv,vlen> re, im;
  static_for<vlen>([&](auto k) DUCC0_LAMBDA_INLINE{ re[k]=v[k].r; im[k]=v[k].i; });
  transpose(re); transpose(im);
  static_for<vlen>([&](auto k) DUCC0_LAMBDA_INLINE{ v[k]={re[k],im[k]}; });
  }


/* Radix-R pass, with the DFT of length R generated at compile time, or, for
   R0==0, of an odd length R known at run time (primes 17..109 inside
   composite lengths: the O(R^2) dft_rt). Indexing as in FFTPACK: input
   cc[i+ido*(j+R*k)], output ch[i+ido*(k+l1*j)], where output j>0 is
   multiplied by the twiddle w^(l1*i*j). exec works on interleaved complex
   data of any type, exec_planar on real and imaginary planes of scalar data,
   SIMD over vl consecutive butterflies (consecutive in memory, so without
   shuffles; the last pass, ido==1, gathers). */
template <size_t R0, typename Tfs> class cfftpR: public cfftpass_impl<cfftpR<R0,Tfs>,Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;
    static constexpr size_t vl = fft1d_max_simdlen<Tfs>, L = R0 ? R0 : 109;  // L: capacity of the butterfly
    // compile-time loop count n, or 0 (a runtime loop) if the radix is not known at compile time
    static constexpr size_t ct(size_t n) { return R0 ? n : 0; }
    size_t l1, ido, ip;  // ip: the runtime radix (R0==0)
    /* twiddles w^(l1*i*(j+1)), i<ido, j<R-1, in blocks of vl values of i:
       real parts at w[wi(i,j)], imaginary parts vl entries later. The 2(R-1)
       twiddles of one i then lie close together, and the vl consecutive ones
       of one j load as a vector. */
    aligned_array<Tfs> w;
    aligned_array<Tcs> cs;  // only for R0==0: cs[m] = exp(2*pi*i*m/R)
    size_t wi(size_t i, size_t j) const { return ((i/vl)*(rdx()-1)+j)*2*vl + i%vl; }
    DUCC0_ALWAYS_INLINE size_t rdx() const { if constexpr (R0>0) return R0; else return ip; }
    template<bool fwd, typename C> DUCC0_ALWAYS_INLINE void dftR(array<C,L> &x) const
      { if constexpr (R0>0) dft<R0,fwd,Tfs>(x); else dft_rt<fwd,Tfs>(x, ip, cs.data()); }

    /* not inlined since GCC seem to drop the restrict qualifiers. Scalar data
       does not rely on the compiler's vectorization of the loop over i (it
       failed for radix 5 and for the blocked twiddles), but takes the
       explicit SIMD path of the planar passes with interleaved data. */
    template<bool fwd, typename Tcd> DUCC0_NOINLINE Tcd *exec_ (Tcd *cc, Tcd *ch,
      Tcd * /*buf*/, size_t /*nthreads*/) const
      {
      const size_t R = rdx();
      if constexpr (is_same_v<Tcd,Tcs>)  // scalar data: explicit SIMD over i, as for planes
        {
        exec_planar_<fwd,true,true>(reinterpret_cast<const Tfs *>(cc), reinterpret_cast<Tfs *>(ch));
        return ch;
        }
      else
        {
        if (cc==ch)  // SIMD data, requested by the caller: in place, only possible for l1==1
          {
          MR_assert(l1==1, "in-place pass needs l1==1");
          for (size_t i=0; i<ido; ++i)  // every butterfly reads and writes the same places
            {
            array<Tcd,L> x;
            xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=cc[i+ido*j]; });
            dftR<fwd>(x);
            cc[i]=x[0];
            const Tfs *wp = w.data()+wi(i,0);
            xfor<ct(R0-1)>(R-1, [&](auto j) DUCC0_LAMBDA_INLINE
              { cc[i+ido*(j+1)]=x[j+1].template special_mul<fwd>(Tcs(wp[2*vl*j], wp[2*vl*j+vl])); });
            }
          return cc;
          }
        const size_t so = ido*l1;
        for (size_t k=0; k<l1; ++k)
          {
          // few base pointers: otherwise the compiler runs out of registers for them
          const Tcd * DUCC0_RESTRICT cck = cc+ido*R*k;
          Tcd * DUCC0_RESTRICT chk = ch+ido*k;
            {
            array<Tcd,L> x;
            xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=cck[ido*j]; });
            dftR<fwd>(x);
            xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ chk[so*j]=x[j]; });
            }
          for (size_t i=1; i<ido; ++i)
            {
            array<Tcd,L> x;
            xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=cck[i+ido*j]; });
            dftR<fwd>(x);
            chk[i]=x[0];
            const Tfs *wp = w.data()+wi(i,0);
            xfor<ct(R0-1)>(R-1, [&](auto j) DUCC0_LAMBDA_INLINE
              { chk[i+so*(j+1)]=x[j+1].template special_mul<fwd>(Tcs(wp[2*vl*j], wp[2*vl*j+vl])); });
            }
          }
        return ch;
        }
      }

    // one (V=Tfs) or vl consecutive (V=SIMD) complex values at idx, from/to
    // planes of N values (imaginary at p+N) or interleaved complex numbers (il)
    template<typename V, bool il> DUCC0_ALWAYS_INLINE static Cmplx<V> ld(const Tfs *p, size_t N, size_t idx)
      {
      if constexpr (il)
        {
        auto c = reinterpret_cast<const Tcs *>(p);
        if constexpr (is_same_v<V,Tfs>) return c[idx]; else return vload<V>(c+idx);
        }
      else if constexpr (is_same_v<V,Tfs>) return Cmplx<V>(p[idx], p[N+idx]);
      else return Cmplx<V>(loadu<V>(p+idx), loadu<V>(p+N+idx));
      }
    template<typename V, bool il> DUCC0_ALWAYS_INLINE static void st(const Cmplx<V> &x, Tfs *p, size_t N, size_t idx)
      {
      if constexpr (il)
        {
        auto c = reinterpret_cast<Tcs *>(p);
        if constexpr (is_same_v<V,Tfs>) c[idx]=x; else vstore(x, c+idx);
        }
      else if constexpr (is_same_v<V,Tfs>) { p[idx]=x.r; p[N+idx]=x.i; }
      else { storeu(x.r, p+idx); storeu(x.i, p+N+idx); }
      }
    // one butterfly (V=Tfs) or vl consecutive ones (V=SIMD) at index i of transform k
    template<bool fwd, typename V, bool iin, bool iout> DUCC0_ALWAYS_INLINE void pstep(size_t i, size_t k,
      const Tfs *in, Tfs *out) const
      {
      const size_t R=rdx(), N=R*l1*ido;
      array<Cmplx<V>,L> x;
      xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=ld<V,iin>(in, N, i+ido*(j+R*k)); });
      dftR<fwd>(x);
      st<V,iout>(x[0], out, N, i+ido*k);
      xfor<ct(R0-1)>(R-1, [&](auto j) DUCC0_LAMBDA_INLINE
        { st<V,iout>(x[j+1].template special_mul<fwd>(ld<V,false>(w.data(), vl, wi(i,j))), out, N, i+ido*(k+l1*(j+1))); });
      }
    template<bool fwd, bool iin, bool iout> void exec_planar_(const Tfs *in, Tfs *out) const
      {
      const size_t R=rdx(), N=R*l1*ido;
      // full SIMD width first, then narrower vectors for the rest, then scalars
      if constexpr (!iin)
        if (ido==1)  // inputs consecutive in j: transpose blocks of v transforms
          {
          size_t k=0;
          static_for<4>([&](auto s) DUCC0_LAMBDA_INLINE
            {
            constexpr size_t v = vl>>decltype(s)::value;
            if constexpr ((v>=2) && simd_exists<Tfs,v>)
              {
              using Tfv = typename simd_select<Tfs,v>::type;
              for (; k+v<=l1; k+=v)
                {
                array<Tfv,L> xr, xi;
                vgather<R0>(in+R*k, xr, R);
                vgather<R0>(in+N+R*k, xi, R);
                array<Cmplx<Tfv>,L> x;
                xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=Cmplx<Tfv>(xr[j], xi[j]); });
                dftR<fwd>(x);
                xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ st<Tfv,iout>(x[j], out, N, k+l1*j); });
                }
              }
            });
          for (; k<l1; ++k) pstep<fwd,Tfs,iin,iout>(0, k, in, out);
          return;
          }
      for (size_t k=0; k<l1; ++k)
        {
        size_t i=0;
        static_for<4>([&](auto s) DUCC0_LAMBDA_INLINE
          {
          constexpr size_t v = vl>>decltype(s)::value;
          if constexpr ((v>=2) && simd_exists<Tfs,v>)
            for (; i+v<=ido; i+=v) pstep<fwd,typename simd_select<Tfs,v>::type,iin,iout>(i, k, in, out);
          });
        for (; i<ido; ++i) pstep<fwd,Tfs,iin,iout>(i, k, in, out);
        }
      }

  public:
    cfftpR(size_t l1_, size_t ido_, const Troots<Tfs> &roots, size_t ip_=R0)
      : l1(l1_), ido(ido_), ip(ip_), w(((ido+vl-1)/vl)*(ip-1)*2*vl), cs(R0 ? 0 : ip)
      {
      const size_t R=rdx();
      MR_assert((R0>0) || ((R&1) && (R>=3) && (R<=L)), "bad runtime radix");
      size_t N=R*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=0; i<ido; ++i)
        for (size_t j=0; j<R-1; ++j)
          {
          auto v = (*roots)[rfct*(j+1)*l1*i];
          w[wi(i,j)] = v.r; w[wi(i,j)+vl] = v.i;
          }
      for (size_t m=0; m<cs.size(); ++m) cs[m] = (*roots)[rfct*l1*ido*m];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const { return w.size()*sizeof(Tfs)+cs.size()*sizeof(Tcs); }
    virtual bool planar() const { return true; }
    virtual void exec_planar(const Tfs *in, Tfs *out, bool fwd, bool iin, bool iout) const
      {
      MR_assert(!(iin&&iout), "cannot convert input and output in one pass");
      if (iin) fwd ? exec_planar_<true,true,false>(in,out) : exec_planar_<false,true,false>(in,out);
      else if (iout) fwd ? exec_planar_<true,false,true>(in,out) : exec_planar_<false,false,true>(in,out);
      else fwd ? exec_planar_<true,false,false>(in,out) : exec_planar_<false,false,false>(in,out);
      }

    friend cfftpass_impl<cfftpR<R0,Tfs>,Tfs>;
  };

/* Single 1-D transform of length N=vlen*M, SIMD over the whole transform:
   x[l+vlen*i] is put into lane l of SIMD element i, the M-point transforms of
   all lanes run with the ordinary passes, then for blocks of vlen outputs
   the lanes are twiddled, transposed and combined by a vlen-point DFT. */
template <size_t vlen, typename Tfs> class cfftp_soa: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;
    using Tfv = typename simd_select<Tfs,vlen>::type;
    using Tcv = Cmplx<Tfv>;
    size_t N, M;
    Tcpass<Tfs> mpass;
    aligned_array<Tcv> tw;  // lane l of tw[k] holds w_N^(l*k)
    size_t bufsz;

    template<bool fwd> Tcs *exec_(Tcs *cc, Tcs * /*ch*/, Tcs *sbuf, size_t nthreads) const
      {
      auto xbuf = reinterpret_cast<uintptr_t>(sbuf);
      xbuf = (xbuf+alignof(Tcv)-1) & ~uintptr_t(alignof(Tcv)-1);
      Tcv *cc2 = reinterpret_cast<Tcv *>(xbuf);
      cc2 += skew(cc, cc2);
      Tcv *ch2 = cc2+M;
      ch2 += skew(cc2, ch2);
      Tcv *buf2 = ch2+M;
      for (size_t i=0; i<M; ++i)
        cc2[i] = vload<Tfv>(cc+vlen*i);
      static const auto ticv = tidx<Tcv *>();
      auto res = static_cast<Tcv *>(mpass->exec(ticv, cc2, ch2, buf2, fwd, nthreads));
      auto block = [&](size_t b)
        {
        array<Tcv,vlen> x;
        static_for<vlen>([&](auto t) DUCC0_LAMBDA_INLINE{ x[t] = res[b+t].template special_mul<fwd>(tw[b+t]); });
        transpose(x);
        dft<vlen,fwd,Tfs>(x);
        static_for<vlen>([&](auto k2) DUCC0_LAMBDA_INLINE{ vstore(x[k2], cc+b+M*k2); });
        };
      size_t b=0;
      for (; b+vlen<=M; b+=vlen) block(b);
      if (b<M) block(M-vlen);  // recompute an overlapping last block
      return cc;
      }

  public:
    cfftp_soa(size_t n, const Troots<Tfs> &roots)
      : N(n), M(n/vlen), mpass(cfftpass<Tfs>::make_pass(1,1,M,roots)), tw(M)
      {
      MR_assert((N==M*vlen) && (M>=vlen), "bad length for cfftp_soa");
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t k=0; k<M; ++k)
        for (size_t l=0; l<vlen; ++l)
          {
          auto w = (*roots)[(rfct*k)*l];  // l*k < N
          tw[k].r[l] = w.r; tw[k].i[l] = w.i;
          }
      bufsz = (2*M+mpass->bufsize())*vlen + 2*vlen + 2*4096/sizeof(Tcs);  // in units of Tcs
      }
    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return false; }
    virtual size_t footprint() const
      { return tw.size()*sizeof(Tcv)+mpass->footprint(); }
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf,
      bool fwd, size_t nthreads=1) const
      {
      static const auto tics = tidx<Tcs *>();
      MR_assert(ti==tics, "bad input type");
      auto in1 = static_cast<Tcs *>(in);
      auto copy1 = static_cast<Tcs *>(copy);
      auto buf1 = static_cast<Tcs *>(buf);
      return fwd ? exec_<true>(in1, copy1, buf1, nthreads)
                 : exec_<false>(in1, copy1, buf1, nthreads);
      }
  };

template <typename Tfs> class cfftpblue: public cfftpass_impl<cfftpblue<Tfs>,Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    const size_t l1, ido, ip;
    const size_t ip2;
    const Tcpass<Tfs> subplan;
    aligned_array<Tcs> wa, bk, bkf;
    size_t bufsz;
    bool need_cpy;

    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch,
       Tcd * DUCC0_RESTRICT buf, size_t nthreads) const
      {
      static const auto ti=tidx<Tcd *>();
      Tcd *akf = buf;
      Tcd *akf2 = subplan->needs_copy() ? (buf+ip2) : akf;
      Tcd *subbuf = akf2+ip2;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tcd&
        { return cc[a+ido*(b+ip*c)]; };

//FIXME: parallelize here?
      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          {
          /* initialize a_k and FFT it */
          for (size_t m=0; m<ip; ++m)
            special_mul<fwd>(CC(i,m,k),bk[m],akf[m]);
          auto zero = akf[0]*Tfs(0);
          for (size_t m=ip; m<ip2; ++m)
            akf[m]=zero;

          auto res = static_cast<Tcd *>(subplan->exec(ti,akf,akf2,
            subbuf, true, nthreads));

          /* do the convolution */
          res[0] = res[0].template special_mul<!fwd>(bkf[0]);
          for (size_t m=1; m<(ip2+1)/2; ++m)
            {
            res[m] = res[m].template special_mul<!fwd>(bkf[m]);
            res[ip2-m] = res[ip2-m].template special_mul<!fwd>(bkf[m]);
            }
          if ((ip2&1)==0)
            res[ip2/2] = res[ip2/2].template special_mul<!fwd>(bkf[ip2/2]);

          /* inverse FFT */
          res = static_cast<Tcd *>(subplan->exec(ti, res,
            (res==akf) ? akf2 : akf, subbuf, false, nthreads));

          /* multiply by b_k and write to output buffer */
          if (l1>1)
            {
            if (i==0)
              for (size_t m=0; m<ip; ++m)
                CH(0,k,m) = res[m].template special_mul<fwd>(bk[m]);
            else
              {
              CH(i,k,0) = res[0].template special_mul<fwd>(bk[0]);
              for (size_t m=1; m<ip; ++m)
                CH(i,k,m) = res[m].template special_mul<fwd>(bk[m]*WA(m-1,i));
              }
            }
          else
            {
            if (i==0)
              for (size_t m=0; m<ip; ++m)
                CC(0,m,0) = res[m].template special_mul<fwd>(bk[m]);
            else
              {
              CC(i,0,0) = res[0].template special_mul<fwd>(bk[0]);
              for (size_t m=1; m<ip; ++m)
                CC(i,m,0) = res[m].template special_mul<fwd>(bk[m]*WA(m-1,i));
              }
            }
          }

      return (l1>1) ? ch : cc;
      }

  public:
    cfftpblue(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots,
      bool vectorize=false)
      : l1(l1_), ido(ido_), ip(ip_),
        // a vectorized sub-transform wants a length divisible by 16 (cfftp_soa)
        ip2(vectorize ? util1d::good_size_cmplx(ip*2-1, 16) : util1d::good_size_cmplx(ip*2-1)),
        subplan(cfftpass<Tfs>::make_pass(ip2, vectorize)), wa((ip-1)*(ido-1)),
        bk(ip), bkf(ip2/2+1)
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];

      /* initialize b_k */
      bk[0].Set(1, 0);
      size_t coeff=0;
      auto roots2 = ((roots->size()/(2*ip))*2*ip==roots->size()) ?
                    roots : make_shared<const UnityRoots<Tfs,Tcs>>(2*ip);
      size_t rfct2 = roots2->size()/(2*ip);
      for (size_t m=1; m<ip; ++m)
        {
        coeff+=2*m-1;
        if (coeff>=2*ip) coeff-=2*ip;
        bk[m] = (*roots2)[coeff*rfct2];
        }

      /* initialize the zero-padded, Fourier transformed b_k. Add normalisation. */
      aligned_array<Tcs> tbkf(ip2), tbkf2(ip2);
      Tfs xn2 = Tfs(1)/Tfs(ip2);
      tbkf[0] = bk[0]*xn2;
      for (size_t m=1; m<ip; ++m)
        tbkf[m] = tbkf[ip2-m] = bk[m]*xn2;
      for (size_t m=ip;m<=(ip2-ip);++m)
        tbkf[m].Set(0.,0.);
      aligned_array<Tcs> buf(subplan->bufsize());
      static const auto tics=tidx<Tcs *>();
      auto res = static_cast<Tcs *>(subplan->exec(tics, tbkf.data(),
        tbkf2.data(), buf.data(), true));
      for (size_t i=0; i<ip2/2+1; ++i)
        bkf[i] = res[i];

      need_cpy = l1>1;
      bufsz = ip2*(1+subplan->needs_copy()) + subplan->bufsize();
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }
    virtual size_t footprint() const
      { return subplan->footprint() + (wa.size()+bk.size()+bkf.size())*sizeof(Tcs); }

    friend cfftpass_impl<cfftpblue<Tfs>,Tfs>;
  };

/* Rader's algorithm: a DFT of prime length p is a cyclic convolution of
   length p-1 (input and output indices permuted by the powers of a primitive
   root g modulo p), which is computed with FFTs of length p-1. */
template <typename Tfs> class cfftprader: public cfftpass_impl<cfftprader<Tfs>,Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    const size_t l1, ido, ip;
    const Tcpass<Tfs> subplan;
    aligned_array<Tcs> wa, cf;  // twiddles; FFT of the convolution kernel, scaled by 1/(ip-1)
// FIXME: gout is just a reversed gin with exception of index 0. Perhaps something can be optimized here.
    aligned_array<size_t> gin, gout;  // gin[r]=g^r mod ip, gout[q]=g^-q mod ip
    size_t bufsz;

    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch,
       Tcd * DUCC0_RESTRICT buf, size_t nthreads) const
      {
      static const auto ti=tidx<Tcd *>();
      const size_t n = ip-1;
      Tcd *akf = buf;
      Tcd *akf2 = subplan->needs_copy() ? (buf+n) : akf;
      Tcd *subbuf = akf2+n;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tcd&
        { return cc[a+ido*(b+ip*c)]; };

      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          {
          auto x0 = CC(i,0,k);
          for (size_t r=0; r<n; ++r)
            akf[r] = CC(i,gin[r],k);
          auto res = static_cast<Tcd *>(subplan->exec(ti, akf, akf2, subbuf, fwd, nthreads));
          auto X0 = x0+res[0];  // X[0] = x[0] + sum of the others
          for (size_t m=0; m<n; ++m)
            res[m] = res[m].template special_mul<!fwd>(cf[m]);
          res = static_cast<Tcd *>(subplan->exec(ti, res, (res==akf) ? akf2 : akf, subbuf, !fwd, nthreads));
          auto out = [&](size_t m) -> Tcd& { return (l1>1) ? CH(i,k,m) : CC(i,m,0); };
          out(0) = X0;
          if (i==0)
            for (size_t q=0; q<n; ++q)
              out(gout[q]) = x0+res[q];
          else
            for (size_t q=0; q<n; ++q)
              out(gout[q]) = (x0+res[q]).template special_mul<fwd>(WA(gout[q]-1,i));
          }
      return (l1>1) ? ch : cc;
      }

  public:
    cfftprader(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots,
      bool vectorize=false)
      : l1(l1_), ido(ido_), ip(ip_),
        subplan(cfftpass<Tfs>::make_pass(ip-1, vectorize)), wa((ip-1)*(ido-1)),
        cf(ip-1), gin(ip-1), gout(ip-1)
      {
      const size_t N=ip*l1*ido, n=ip-1;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];

      size_t g=2;  // primitive root: smallest g whose powers run through all of 1..ip-1
// MR FIXME: this looks slow! Perhaps there is a more efficient way to achieve this.
// See, e.g., https://cp-algorithms.com/algebra/primitive-root.html
      for (;; ++g)
        {
        size_t x=g, cnt=1;
        while (x!=1) { x=(x*g)%ip; ++cnt; }
        if (cnt==n) break;
        }
      for (size_t r=0, rx=0, pos=1; r<n; ++r, rx=n-r, pos=(pos*g)%ip)
        gin[r] = gout[rx] = pos;

      /* kernel c[m] = exp(-2 pi i g^-m/ip); store its forward FFT divided by n */
      auto rootsp = ((roots->size()%ip)==0) ?
                    roots : make_shared<const UnityRoots<Tfs,Tcs>>(ip);
      size_t rfctp = rootsp->size()/ip;
      aligned_array<Tcs> c(n), c2(n), sbuf(subplan->bufsize());
      for (size_t m=0; m<n; ++m)
        c[m] = (*rootsp)[gout[m]*rfctp].conj()*(Tfs(1)/Tfs(n));
      static const auto tics=tidx<Tcs *>();
      auto res = static_cast<Tcs *>(subplan->exec(tics, c.data(), c2.data(), sbuf.data(), true));
      for (size_t m=0; m<n; ++m) cf[m] = res[m];

      bufsz = n*(1+subplan->needs_copy()) + subplan->bufsize();
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return l1>1; }
    virtual size_t footprint() const
      { return subplan->footprint() + (wa.size()+cf.size())*sizeof(Tcs) + 2*gin.size()*sizeof(size_t); }

    friend cfftpass_impl<cfftprader<Tfs>,Tfs>;
  };

template <typename Tfs> class cfft_multipass: public cfftpass_impl<cfft_multipass<Tfs>,Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;
    static constexpr size_t bunchsize=8;

    const size_t l1, ido;
    size_t ip;
    vector<Tcpass<Tfs>> passes;
    size_t bufsz;
    bool need_cpy, planar;  // planar: all passes can work on re/im planes
    size_t rfct;
    Troots<Tfs> myroots;

// FIXME split into sub-functions. This is too long!
    template<bool fwd, typename T> Cmplx<T> *exec_(Cmplx<T> *cc, Cmplx<T> *ch,
      Cmplx<T> *buf, size_t nthreads) const
      {
      using Tc = Cmplx<T>;
      if ((l1==1) && (ido==1)) // a single transform
        {
        if constexpr (is_same_v<T,Tfs>)
          if (planar)  // scalar data: run the passes on real and imaginary planes
            {
            // (the first pass reads interleaved input, the last writes interleaved output)
            Tfs *pin=reinterpret_cast<Tfs *>(cc), *pout=reinterpret_cast<Tfs *>(ch);
            for (size_t n=0; n<passes.size(); ++n)
              {
              passes[n]->exec_planar(pin, pout, fwd, n==0, n+1==passes.size());
              swap(pin, pout);
              }
            return reinterpret_cast<Tc *>(pin);
            }
        static const auto tic=tidx<Tc *>();
        Tc *p1=cc, *p2=ch;
        // With an odd number of codelet passes the result would end up in ch;
        // the first pass can run in place on SIMD data (requested by cc==ch),
        // which is faster and leaves the result in cc.
        bool first_inplace = planar && (passes.size()&1) && (passes.size()>1) && (!is_same_v<T,Tfs>);
        for(const auto &pass: passes)
          {
          auto res = static_cast<Tc *>(pass->exec(tic, p1, first_inplace ? p1 : p2, buf,
            fwd, nthreads));
          if (res==p2) swap (p1,p2);
          first_inplace = false;
          }
        return p1;
        }
      else
        {
        if constexpr(is_same<T,Tfs>::value && simd_exists<Tfs, fft1d_max_simdlen<Tfs>>) // we can vectorize!
          {
          using Tfv = typename simd_select<Tfs, fft1d_max_simdlen<Tfs>>::type;
          using Tcv = Cmplx<Tfv>;
          constexpr size_t vlen = Tfv::size();
          size_t nvtrans = (l1*ido + vlen-1)/vlen;
          // NOTE: removed "static" here, because it leads to trouble with gcc 7
          // static const type_index ticv = tidx<Tcv *>();
          const type_index ticv = tidx<Tcv *>();

          if (ido==1)
            {
            auto CH = [ch,this](size_t b, size_t c) -> Tc&
              { return ch[b+l1*c]; };
            auto CC = [cc,this](size_t b, size_t c) -> Tc&
              { return cc[b+ip*c]; };

            execStatic(nvtrans, nthreads, 0, [&](auto &sched)
              {
              aligned_array<Tcv> tbuf(2*ip+32+bufsize());
              auto cc2 = tbuf.data();
              auto ch2 = tbuf.data()+ip+16;
              auto buf2 = tbuf.data()+2*ip+32;

              while (auto rng=sched.getNext())
                for(auto itrans=rng.lo; itrans<rng.hi; ++itrans)
                  {
                  for (size_t n=0; n<vlen; ++n)
                    for (size_t m=0; m<ip; ++m)
                      {
                      size_t k = min(l1-1, itrans*vlen+n);
                      cc2[m].r[n] = CC(m,k).r;
                      cc2[m].i[n] = CC(m,k).i;
                      }

                  Tcv *p1=cc2, *p2=ch2;
                  for(const auto &pass: passes)
                    {
                    auto res = static_cast<Tcv *>(pass->exec(ticv,
                      p1, p2, buf2, fwd));
                    if (res==p2) swap (p1,p2);
                    }

                  for (size_t m=0; m<ip; ++m)
                    for (size_t n=0; n<vlen; ++n)
                      {
                      auto k = min(l1-1, itrans*vlen+n);
                      CH(k,m) = { p1[m].r[n], p1[m].i[n] };
                      }
                  }
              });
            return ch;
            }

          if (l1==1)
            {
            auto CC = [cc,this](size_t a, size_t b) -> Tc&
              { return cc[a+ido*b]; };

            execStatic(nvtrans, nthreads, 0, [&](auto &sched)
              {
              aligned_array<Tcv> tbuf(2*ip+32+bufsize());
              auto cc2 = tbuf.data();
              auto ch2 = tbuf.data()+ip+16;
              auto buf2 = tbuf.data()+2*ip+32;

              while (auto rng=sched.getNext())
                for(auto itrans=rng.lo; itrans<rng.hi; ++itrans)
                  {
                  for (size_t m=0; m<ip; ++m)
                    for (size_t n=0; n<vlen; ++n)
                      {
                      size_t i = min(ido-1, itrans*vlen+n);
                      cc2[m].r[n] = CC(i,m).r;
                      cc2[m].i[n] = CC(i,m).i;
                      }

                  Tcv *p1=cc2, *p2=ch2;
                  for(const auto &pass: passes)
                    {
                    auto res = static_cast<Tcv *>(pass->exec(ticv,
                      p1, p2, buf2, fwd));
                    if (res==p2) swap (p1,p2);
                    }

                  for (size_t m=0; m<ip; ++m)
                    for (size_t n=0; n<vlen; ++n)
                      {
                      auto i = itrans*vlen+n;
                      if (i >= ido) break;
                      if (i==0)
                        CC(0,m) = { p1[m].r[n], p1[m].i[n] };
                      else
                        {
                        if (m==0)
                          CC(i,0) = { p1[0].r[n], p1[0].i[n] } ;
                        else
                          CC(i,m) = Tcs(p1[m].r[n],p1[m].i[n]).template special_mul<fwd>((*myroots)[rfct*m*i]);
                        }
                      }
                  }
              });
            return cc;
            }

MR_fail("must not get here");
#if 0
//FIXME this code path is currently unused
          aligned_array<Tcv> tbuf(2*ip+bufsize());
          auto cc2 = tbuf;
          auto ch2 = tbuf+ip;
          auto buf2 = tbuf+2*ip;

          auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
            { return ch[a+ido*(b+l1*c)]; };
          auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tc&
            { return cc[a+ido*(b+ip*c)]; };

//FIXME parallelize?
          for (size_t itrans=0; itrans<nvtrans; ++itrans)
            {
            array<size_t, vlen> ix, kx;
            size_t ixcur = (itrans*vlen)%ido;
            size_t kxcur = (itrans*vlen)/ido;
            for (size_t n=0; n<vlen; ++n)
              {
              ix[n] = ixcur;
              kx[n] = min(l1-1,kxcur);
              if (++ixcur==ido)
                {
                ixcur=0;
                ++kxcur;
                }
              }

            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<vlen; ++n)
                {
                cc2[m].r[n] = CC(ix[n],m,kx[n]).r;
                cc2[m].i[n] = CC(ix[n],m,kx[n]).i;
                }

            Tcv *p1=cc2, *p2=ch2;
            for(const auto &pass: passes)
              {
              auto res = static_cast<Tcv *>(pass->exec(ticv,
                p1, p2, buf2, fwd));
              if (res==p2) swap (p1,p2);
              }

            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<vlen; ++n)
                {
                auto i = ix[n];
                auto k = kx[n];
                if (itrans*vlen+n >= l1*ido) break;
                if (i==0)
                  CH(0,k,m) = { p1[m].r[n], p1[m].i[n] };
                else
                  {
                  if (m==0)
                    CH(i,k,0) = { p1[0].r[n], p1[0].i[n] } ;
                  else
                    CH(i,k,m) = Tcs(p1[m].r[n],p1[m].i[n]).template special_mul<fwd>((*myroots)[rfct*l1*m*i]);
                  }
                }
            }
          return ch;
#endif
          }
        else
          {
          static const auto tic = tidx<Cmplx<T> *>();
          if (ido==1)
            {
// parallelize here!
            for (size_t n=0; n<l1; ++n)
              {
              Cmplx<T> *p1=&cc[n*ip], *p2=ch;
              Cmplx<T> *res = nullptr;
              for(const auto &pass: passes)
                {
                res = static_cast<Cmplx<T> *>(pass->exec(tic,
                  p1, p2, buf, fwd));
                if (res==p2) swap (p1,p2);
                }
              if (res != &cc[n*ip])
                copy(res, res+ip, cc+n*ip);
              }
            // transpose
            size_t nbunch = (l1*ido + bunchsize-1)/bunchsize;
// parallelize here!
            for (size_t ibunch=0; ibunch<nbunch; ++ibunch)
              {
              size_t ntrans = min(bunchsize, l1-ibunch*bunchsize);
              for (size_t m=0; m<ip; ++m)
                for (size_t n=0; n<ntrans; ++n)
                  {
                  size_t itrans = ibunch*bunchsize + n;
                  ch[itrans+m*l1] = cc[m+itrans*ip];
                  }
              }
            return ch;
            }
          if (l1==1)
            {
            auto cc2 = buf;
            auto ch2 = buf+bunchsize*ip;
            auto buf2 = buf+(bunchsize+1)*ip;
            size_t nbunch = (ido + bunchsize-1)/bunchsize;

            auto CC = [cc,this](size_t a, size_t b) -> Tc&
              { return cc[a+ido*b]; };

// parallelize here!
             for (size_t ibunch=0; ibunch<nbunch; ++ibunch)
              {
              size_t ntrans = min(bunchsize, ido-ibunch*bunchsize);

              for (size_t m=0; m<ip; ++m)
                for (size_t n=0; n<ntrans; ++n)
                  cc2[m+n*ip] = CC(n+ibunch*bunchsize,m);

              for (size_t n=0; n<ntrans; ++n)
                {
                auto i = n+ibunch*bunchsize;
                Cmplx<T> *p1=&cc2[n*ip], *p2=ch2;
                Cmplx<T> *res = nullptr;
                for(const auto &pass: passes)
                  {
                  res = static_cast<Cmplx<T> *>(pass->exec(tic,
                    p1, p2, buf2, fwd));
                  if (res==p2) swap (p1,p2);
                  }
                if (res==&cc2[n*ip]) // no copying necessary
                  {
                  if (i!=0)
                    {
                    for (size_t m=1; m<ip; ++m)
                      cc2[n*ip+m] = cc2[n*ip+m].template special_mul<fwd>((*myroots)[rfct*m*i]);
                    }
                  }
                else
                  {
                  if (i==0)
                    for (size_t m=0; m<ip; ++m)
                      cc2[n*ip+m] = res[m];
                  else
                    {
                    cc2[n*ip] = res[0];
                    for (size_t m=1; m<ip; ++m)
                      cc2[n*ip+m] = res[m].template special_mul<fwd>((*myroots)[rfct*m*i]);
                    }
                  }
                }
              for (size_t m=0; m<ip; ++m)
                for (size_t n=0; n<ntrans; ++n)
                  CC(n+ibunch*bunchsize, m) = cc2[m+n*ip];
              }
            return cc;
            }

MR_fail("must not get here");
#if 0
//FIXME this code path is currently unused
          auto cc2 = buf;
          auto ch2 = buf+bunchsize*ip;
          auto buf2 = buf+(bunchsize+1)*ip;
          size_t nbunch = (l1*ido + bunchsize-1)/bunchsize;

          auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
            { return ch[a+ido*(b+l1*c)]; };
          auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tc&
            { return cc[a+ido*(b+ip*c)]; };

// parallelize here!
          for (size_t ibunch=0; ibunch<nbunch; ++ibunch)
            {
            size_t ntrans = min(bunchsize, l1*ido-ibunch*bunchsize);
            array<size_t, bunchsize> ix, kx;
            size_t ixcur = (ibunch*bunchsize)%ido;
            size_t kxcur = (ibunch*bunchsize)/ido;
            for (size_t n=0; n<bunchsize; ++n)
              {
              ix[n] = ixcur;
              kx[n] = min(l1-1,kxcur);
              if (++ixcur==ido)
                {
                ixcur=0;
                ++kxcur;
                }
              }
            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<ntrans; ++n)
                cc2[m+n*ip] = CC(ix[n],m,kx[n]);

            for (size_t n=0; n<ntrans; ++n)
              {
              auto i = ix[n];
              Cmplx<T> *p1=&cc2[n*ip], *p2=ch2;
              Cmplx<T> *res = nullptr;
              for(const auto &pass: passes)
                {
                res = static_cast<Cmplx<T> *>(pass->exec(tic,
                  p1, p2, buf2, fwd));
                if (res==p2) swap (p1,p2);
                }
              if (res==&cc2[n*ip]) // no copying necessary
                {
                if (i!=0)
                  {
                  for (size_t m=1; m<ip; ++m)
                    cc2[n*ip+m] = cc2[n*ip+m].template special_mul<fwd>((*myroots)[rfct*l1*m*i]);
                  }
                }
              else
                {
                if (i==0)
                  for (size_t m=0; m<ip; ++m)
                    cc2[n*ip+m] = res[m];
                else
                  {
                  cc2[n*ip] = res[0];
                  for (size_t m=1; m<ip; ++m)
                    cc2[n*ip+m] = res[m].template special_mul<fwd>((*myroots)[rfct*l1*m*i]);
                  }
                }
              }
            for (size_t m=0; m<ip; ++m)
              for (size_t n=0; n<ntrans; ++n)
                CH(ix[n], kx[n], m) = cc2[m+n*ip];
            }
          return ch;
#endif
          }
        }
      }

  public:
    cfft_multipass(size_t l1_, size_t ido_, size_t ip_,
      const Troots<Tfs> &roots, bool /*vectorize*/=false)
      : l1(l1_), ido(ido_), ip(ip_), bufsz(0), need_cpy(false),
        myroots(roots)
      {
      size_t N=ip*l1*ido;
      rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");

      // up to 8 MB of data all passes run over the whole array; beyond that
      // the transform is split into two nested ones that fit the caches
      // better (measured on a 16 MB L3: splitting smaller arrays was up to 3x
      // slower, not splitting 13 MB and more up to 1.7x slower)
      size_t lim = (size_t(1)<<23)/sizeof(Cmplx<Tfs>);
      if (ip<=lim)
        {
        auto factors = cfftpass<Tfs>::factorize(ip);
        size_t l1l=1;
        for (auto fct: factors)
          {
          passes.push_back(cfftpass<Tfs>::make_pass(l1l, ip/(fct*l1l), fct, roots, false));
          l1l*=fct;
          }
        }
      else
        {
        vector<size_t> packets(2,1);
        auto factors = util1d::prime_factors(ip);
        sort(factors.begin(), factors.end(), std::greater<size_t>());
        for (auto fct: factors)
          (packets[0]>packets[1]) ? packets[1]*=fct : packets[0]*=fct;
        size_t l1l=1;
        for (auto pkt: packets)
          {
          passes.push_back(cfftpass<Tfs>::make_pass(l1l, ip/(pkt*l1l), pkt, roots, false));
          l1l*=pkt;
          }
        }
      planar = true;
      for (const auto &pass: passes)
        {
        bufsz = max(bufsz, pass->bufsize());
        need_cpy |= pass->needs_copy();
        planar &= pass->planar();
        }
      if ((l1!=1)||(ido!=1))
        {
        need_cpy=true;
        bufsz += (bunchsize+1)*ip;
        }
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }
    virtual size_t footprint() const
      {
      size_t res = myroots->footprint();
      for (const auto &p: passes) res += p->footprint();
      return res;
      }

    friend cfftpass_impl<cfft_multipass<Tfs>,Tfs>;
  };


} // unnamed namespace

#if DUCC0_GCC142_ICE
// SIMD single-transform plan with vl lanes, if it exists and pays for length ip
template<size_t vl, typename Tfs> Tcpass<Tfs> make_soa(size_t ip, const Troots<Tfs> &roots)
  {
  if constexpr ((vl>=2) && simd_exists<Tfs,vl>)
    if ((ip%vl==0) && (ip>=max<size_t>(vl*vl,16)))  // below 16 the plain codelet is faster
      return make_shared<cfftp_soa<vl,Tfs>>(ip, roots);
  return {};
  }
#endif

template<typename Tfs> Tcpass<Tfs> cfftpass<Tfs>::make_pass(size_t l1,
  size_t ido, size_t ip, const Troots<Tfs> &roots, bool vectorize)
  {
  MR_assert(ip>=1, "no zero-sized FFTs");
  // do we have an 1D vectorizable FFT?
  if (vectorize && (l1==1) && (ido==1))
    {
#if DUCC0_GCC142_ICE  // the lambda below crashes GCC 14.2
    constexpr size_t vl = fft1d_max_simdlen<Tfs>;  // try the full SIMD width first
    if (auto res = make_soa<vl,Tfs>(ip, roots)) return res;
    if (auto res = make_soa<vl/2,Tfs>(ip, roots)) return res;
    if (auto res = make_soa<vl/4,Tfs>(ip, roots)) return res;
#else
    Tcpass<Tfs> res;
    static_for<3>([&](auto s) DUCC0_LAMBDA_INLINE  // try the full SIMD width first
      {
      constexpr size_t vl = fft1d_max_simdlen<Tfs> >> decltype(s)::value;
      if constexpr ((vl>=2) && simd_exists<Tfs,vl>)
        if ((!res) && (ip%vl==0) && (ip>=max<size_t>(vl*vl,16)))  // below 16 the plain codelet is faster
          res = make_shared<cfftp_soa<vl,Tfs>>(ip, roots);
      });
    if (res) return res;
#endif
    }

  if (ip==1) return make_shared<cfftp1<Tfs>>();
  auto factors=cfftpass<Tfs>::factorize(ip);
  if (factors.size()==1)
    {
    switch(ip)
      {
      case  2: return make_shared<cfftpR< 2,Tfs>>(l1, ido, roots);
      case  3: return make_shared<cfftpR< 3,Tfs>>(l1, ido, roots);
      case  4: return make_shared<cfftpR< 4,Tfs>>(l1, ido, roots);
      case  5: return make_shared<cfftpR< 5,Tfs>>(l1, ido, roots);
      case  7: return make_shared<cfftpR< 7,Tfs>>(l1, ido, roots);
      case  8: return make_shared<cfftpR< 8,Tfs>>(l1, ido, roots);
      case  9: return make_shared<cfftpR< 9,Tfs>>(l1, ido, roots);
      case 11: return make_shared<cfftpR<11,Tfs>>(l1, ido, roots);
      case 13: return make_shared<cfftpR<13,Tfs>>(l1, ido, roots);
      default:
        /* Primes up to about 100 inside a composite length: the generic
           O(ip^2) pass is fastest (it vectorizes along ido). A prime as the
           whole transform, or a larger prime factor: convolution algorithms.
           Rader needs FFTs of length ip-1; it wins when these need only
           codelet passes and can use SIMD lanes (ip-1 divisible by 4),
           otherwise Bluestein's padded power-of-two-like length is faster. */
        if ((ip<110) && (l1*ido>1))
          return make_shared<cfftpR<0,Tfs>>(l1, ido, roots, ip);
        if (((ip-1)%4==0) && (util1d::prime_factors(ip-1).back()<=13))
          return make_shared<cfftprader<Tfs>>(l1, ido, ip, roots, vectorize);
        return make_shared<cfftpblue<Tfs>>(l1, ido, ip, roots, vectorize);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<cfft_multipass<Tfs>>(l1, ido, ip, roots, vectorize);
  }

namespace {

/* (a+ib) = conj(c+id) * (e+if) */
template<typename T1, typename T2, typename T3> inline void MULPM
  (T1 &a, T1 &b, T2 c, T2 d, T3 e, T3 f)
  {  a=c*e+d*f; b=c*f-d*e; }

template <typename Tfs> class rfftp1: public rfftpass<Tfs>
  {
  public:
    rfftp1() {}
    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return false; }
    virtual size_t footprint() const { return 0; }

    virtual void *exec(const type_index & /*ti*/, void * in, void * /*copy*/,
      void * /*buf*/, bool /*fwd*/, size_t /*nthreads*/) const
      { return in; }
  };

/* Real radix-R pass (FFTPACK halfcomplex layout) on top of the complex DFT
   of length R generated at compile time, or, for R0==0, of an odd length R
   known at run time (dft_rt, as in cfftpR). Forward: cc[a+ido*(k+l1*j)] ->
   ch[a+ido*(j+R*k)], backward the other way. Index i=0 holds real values,
   (i-1,i) for even i>0 a complex value whose conjugate partner is stored
   mirrored at ido-i; for even R and even ido, i=ido-1 is real again. */
template <size_t R0, typename Tfs> class rfftpR: public rfftpass_impl<rfftpR<R0,Tfs>,Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    static constexpr size_t vl = fft1d_max_simdlen<Tfs>, L = R0 ? R0 : 134;  // L: capacity of the butterfly
    static constexpr bool even = (R0>0) && ((R0&1)==0);
    // compile-time loop count n, or 0 (a runtime loop) if the radix is not known at compile time
    static constexpr size_t ct(size_t n) { return R0 ? n : 0; }
    size_t l1, ido, ip;  // ip: the runtime radix (R0==0)
    /* twiddles w^(l1*j*i/2), i=2,4,..<ido, j=1..R-1, in blocks of vl values of
       i (as in cfftpR): real parts at wa[wi(i/2-1,j-1)], imaginary parts vl later */
    aligned_array<Tfs> wa;
    aligned_array<Tcs> cs;  // only for R0==0: cs[m] = exp(2*pi*i*m/R)
    size_t wi(size_t q, size_t j) const { return ((q/vl)*(rdx()-1)+j)*2*vl + q%vl; }
    DUCC0_ALWAYS_INLINE size_t rdx() const { if constexpr (R0>0) return R0; else return ip; }
    template<bool fwd, typename C> DUCC0_ALWAYS_INLINE void dftR(array<C,L> &x) const
      { if constexpr (R0>0) dft<R0,fwd,Tfs>(x); else dft_rt<fwd,Tfs>(x, ip, cs.data()); }

    // one complex value at p (V==Tfd), or V::size() consecutive ones of scalar data, optionally reversed
    template<typename V, bool rev, typename Tfd> DUCC0_ALWAYS_INLINE static Cmplx<V> ld(const Tfd *p)
      {
      if constexpr (is_same_v<V,Tfd>) return Cmplx<V>(p[0], p[1]);
      else
        {
        auto x = vload<V>(reinterpret_cast<const Tcs *>(p));
        if constexpr (rev) return Cmplx<V>(shuf<sh_rev<V::size()>>(x.r,x.r), shuf<sh_rev<V::size()>>(x.i,x.i));
        else return x;
        }
      }
    template<typename V, bool rev, typename Tfd> DUCC0_ALWAYS_INLINE static void st(const Cmplx<V> &x, Tfd *p)
      {
      if constexpr (is_same_v<V,Tfd>) { p[0]=x.r; p[1]=x.i; }
      else if constexpr (rev)
        vstore(Cmplx<V>(shuf<sh_rev<V::size()>>(x.r,x.r), shuf<sh_rev<V::size()>>(x.i,x.i)), reinterpret_cast<Tcs *>(p));
      else vstore(x, reinterpret_cast<Tcs *>(p));
      }
    template<typename V, typename Tfd> DUCC0_ALWAYS_INLINE auto tw(size_t q, size_t j) const
      {
      const Tfs *p = wa.data()+wi(q,j);
      if constexpr (is_same_v<V,Tfd>) return Tcs(p[0], p[vl]);
      else return Cmplx<V>(loadu<V>(p), loadu<V>(p+vl));
      }
    /* the butterfly at i (V==Tfd), or the V::size() ones starting there (scalar
       data); complex values at (i-1,i), partners mirrored at (ido-i-1,ido-i);
       c, o: block k of input, output */
    template<bool fwd, typename V, typename Tfd> DUCC0_ALWAYS_INLINE void bfly(const Tfd *c, Tfd *o, size_t i) const
      {
      constexpr size_t n = is_same_v<V,Tfd> ? 1 : sizeof(V)/sizeof(Tfs);
      const size_t R=rdx(), so=ido*l1, q=i/2-1, im=ido-i-1-2*(n-1);  // im: lowest index of the mirrored values
      array<Cmplx<V>,L> x;
      if constexpr (fwd)
        {
        x[0]=ld<V,false>(c+i-1);
        xfor<ct(R0-1)>(R-1, [&](auto j) DUCC0_LAMBDA_INLINE
          { x[j+1]=ld<V,false>(c+i-1+so*(j+1)).template special_mul<true>(tw<V,Tfd>(q,j)); });
        dftR<true>(x);
        xfor<ct((R0+1)/2)>((R+1)/2, [&](auto m) DUCC0_LAMBDA_INLINE{ st<V,false>(x[m], o+i-1+ido*2*m); });
        xfor<ct(R0/2)>(R/2, [&](auto m) DUCC0_LAMBDA_INLINE
          { st<V,true>(Cmplx<V>(x[R-1-m].r, -x[R-1-m].i), o+im+ido*(2*m+1)); });
        }
      else
        {
        xfor<ct((R0+1)/2)>((R+1)/2, [&](auto m) DUCC0_LAMBDA_INLINE{ x[m]=ld<V,false>(c+i-1+ido*2*m); });
        xfor<ct(R0/2)>(R/2, [&](auto m) DUCC0_LAMBDA_INLINE
          { auto y=ld<V,true>(c+im+ido*(2*m+1)); x[R-1-m]=Cmplx<V>(y.r, -y.i); });
        dftR<false>(x);
        st<V,false>(x[0], o+i-1);
        xfor<ct(R0-1)>(R-1, [&](auto j) DUCC0_LAMBDA_INLINE
          { st<V,false>(x[j+1].template special_mul<false>(tw<V,Tfd>(q,j)), o+i-1+so*(j+1)); });
        }
      }

    /* DFT of R reals y[j] in place, result in halfcomplex order: X0.r, X1.r,
       X1.i, ..., (X(R/2).r for even R); backward the inverse (unnormalized).
       For runtime R Singleton's algorithm on real data: half the work of the
       complex DFT, whose zero imaginary parts the compiler can only fold
       for compile-time R. */
    template<bool fwd, typename V> DUCC0_ALWAYS_INLINE void redge(array<V,L> &y) const
      {
      const size_t R=rdx(), h=(R-1)/2;
      const V zero(Tfs(0));
      if constexpr (R0>0)
        {
        array<Cmplx<V>,L> x;
        if constexpr (fwd)
          {
          static_for<R0>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=Cmplx<V>(y[j], zero); });
          dftR<true>(x);
          y[0]=x[0].r;
          static_for<(R0-1)/2>([&](auto m) DUCC0_LAMBDA_INLINE{ y[2*m+1]=x[m+1].r; y[2*m+2]=x[m+1].i; });
          if constexpr (even) y[R0-1]=x[R0/2].r;
          }
        else
          {
          x[0]=Cmplx<V>(y[0], zero);
          static_for<(R0-1)/2>([&](auto m) DUCC0_LAMBDA_INLINE
            { x[m+1]=Cmplx<V>(y[2*m+1], y[2*m+2]); x[R0-1-m]=Cmplx<V>(y[2*m+1], -y[2*m+2]); });
          if constexpr (even) x[R0/2]=Cmplx<V>(y[R0-1], zero);
          dftR<false>(x);
          static_for<R0>([&](auto j) DUCC0_LAMBDA_INLINE{ y[j]=x[j].r; });
          }
        }
      else
        {
        array<V,L> t;  // as in dft_rt; backward: 2*Re and 2*Im of the inputs
        for (size_t j=0; j<h; ++j)
          if constexpr (fwd) { t[j]=y[j+1]+y[R-1-j]; t[h+j]=y[j+1]-y[R-1-j]; }
          else { t[j]=Tfs(2)*y[2*j+1]; t[h+j]=Tfs(2)*y[2*j+2]; }
        const V x0=y[0];
        for (size_t j=0; j<h; ++j) y[0]+=t[j];
        singleton_rt(t, x0, R, cs.data(), [&](size_t k, const V &a, const V &b) DUCC0_LAMBDA_INLINE
          {
          if constexpr (fwd) { y[2*k-1]=a; y[2*k]=-b; }
          else { y[k]=a-b; y[R-k]=a+b; }
          });
        }
      }

    // not inlined, for the same reason as cfftpR::exec_
    template<bool fwd, typename Tfd> DUCC0_NOINLINE Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      const size_t R=rdx(), so=ido*l1;
      // i=0, real values: own loops, SIMD over consecutive k for ido==1
      size_t k0=0;
      if constexpr (is_same_v<Tfd,Tfs>)
        if (ido==1)  // scalar data, only the real butterflies at i=0
          static_for<4>([&](auto s) DUCC0_LAMBDA_INLINE  // full SIMD width, then narrower vectors
            {
            constexpr size_t v = vl>>decltype(s)::value;
            if constexpr ((v>=2) && simd_exists<Tfs,v>)
              for (; k0+v<=l1; k0+=v)
                {
                using V = typename simd_select<Tfs,v>::type;
                array<V,L> y;
                if constexpr (fwd)
                  xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ y[j]=loadu<V>(cc+k0+l1*j); });
                else vgather<R0>(cc+R*k0, y, R);
                redge<fwd>(y);
                if constexpr (fwd) vscatter<R0>(y, ch+R*k0, R);
                else xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ storeu(y[j], ch+k0+l1*j); });
                }
            });
      // offset of halfcomplex value p of a transform: 0, ido-1, ido, 3*ido-1, 3*ido, ...
      auto hc = [this](size_t p) DUCC0_LAMBDA_INLINE { return (p&1) ? ido*(p+1)-1 : ido*p; };
      for (size_t k=k0; k<l1; ++k)
        {
        const Tfd *c = cc+ido*k*(fwd ? 1 : R);
        Tfd *o = ch+ido*k*(fwd ? R : 1);
        array<Tfd,L> y;
        xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ y[j]=c[fwd ? so*j : hc(j)]; });
        redge<fwd>(y);
        xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE{ o[fwd ? hc(j) : so*j]=y[j]; });
        }
      if (ido==1) return ch;
      for (size_t k=0; k<l1; ++k)
        {
        // forward: cc[a+ido*(k+l1*j)] -> ch[a+ido*(j+R*k)], backward the other way
        const Tfd * DUCC0_RESTRICT c = cc+ido*k*(fwd ? 1 : R);
        Tfd * DUCC0_RESTRICT o = ch+ido*k*(fwd ? R : 1);
        size_t i=2;
        if constexpr (is_same_v<Tfd,Tfs>)
          static_for<4>([&](auto s) DUCC0_LAMBDA_INLINE  // full SIMD width, then narrower vectors
            {
            constexpr size_t v = vl>>decltype(s)::value;
            if constexpr ((v>=2) && simd_exists<Tfs,v>)
              for (; i+2*v-2<ido; i+=2*v) bfly<fwd,typename simd_select<Tfs,v>::type,Tfd>(c, o, i);
            });
        for (; i<ido; i+=2) bfly<fwd,Tfd,Tfd>(c, o, i);
        // even R, even ido: real values at i=ido-1, with half-step twiddles
        if constexpr (even)
          if ((ido&1)==0)
            {
            using C = Cmplx<Tfd>;
            const Tfd zero(0);
            array<C,L> x;
            if constexpr (fwd)
              {
              xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE
                { x[j]=twmul<decltype(j)::value,2*R0,true,Tfs>(C(c[ido-1+so*j], zero)); });
              dftR<true>(x);
              xfor<ct(R0/2)>(R/2, [&](auto m) DUCC0_LAMBDA_INLINE{ o[ido-1+ido*2*m]=x[m].r; o[ido*(2*m+1)]=x[m].i; });
              }
            else
              {
              xfor<ct(R0/2)>(R/2, [&](auto m) DUCC0_LAMBDA_INLINE
                { x[m]=C(c[ido-1+ido*2*m], c[ido*(2*m+1)]); x[R-1-m]=C(x[m].r, -x[m].i); });
              dftR<false>(x);
              xfor<R0>(R, [&](auto j) DUCC0_LAMBDA_INLINE
                { o[ido-1+so*j]=twmul<decltype(j)::value,2*R0,false,Tfs>(x[j]).r; });
              }
            }
        }
      return ch;
      }

  public:
    rfftpR(size_t l1_, size_t ido_, const Troots<Tfs> &roots, size_t ip_=R0)
      : l1(l1_), ido(ido_), ip(ip_), wa((((ido-1)/2+vl-1)/vl)*(ip-1)*2*vl), cs(R0 ? 0 : ip)
      {
      const size_t R=rdx();
      MR_assert((R0>0) || ((R&1) && (R>=3) && (R<=L)), "bad runtime radix");
      MR_assert((ido&1) || ((R&1)==0), "ido must be odd for odd radices");
      size_t N=R*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t q=0; q<(ido-1)/2; ++q)
        for (size_t j=0; j<R-1; ++j)
          {
          auto val = (*roots)[rfct*(j+1)*l1*(q+1)];
          wa[wi(q,j)] = val.r; wa[wi(q,j)+vl] = val.i;
          }
      for (size_t m=0; m<cs.size(); ++m) cs[m] = (*roots)[rfct*l1*ido*m];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const { return wa.size()*sizeof(Tfs)+cs.size()*sizeof(Tcs); }

    friend rfftpass_impl<rfftpR<R0,Tfs>,Tfs>;
  };

template <typename Tfs> class rfftpblue: public rfftpass_impl<rfftpblue<Tfs>,Tfs>
  {
  private:
    const size_t l1, ido, ip;
    aligned_array<Tfs> wa;
    const Tcpass<Tfs> cplan;
    size_t bufsz;
    bool need_cpy;

    auto WA(size_t x, size_t i) const
      { return wa[i+x*(ido-1)]; }

    template<bool fwd, typename Tfd> Tfd *exec_
      (Tfd * DUCC0_RESTRICT cc, Tfd * DUCC0_RESTRICT ch,
       Tfd * DUCC0_RESTRICT buf_, size_t nthreads) const
      {
      using Tcd = Cmplx<Tfd>;
      auto buf = reinterpret_cast<Tcd *>(buf_);
      Tcd *cc2 = buf;
      Tcd *ch2 = buf+ip;
      Tcd *subbuf = buf+2*ip;
      static const auto ticd = tidx<Tcd *>();

      if constexpr(fwd)
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+ip*c)]; };

        for (size_t k=0; k<l1; ++k)
          {
          // copy in
          for (size_t m=0; m<ip; ++m)
            cc2[m] = {CC(0,k,m),Tfd(0)};
          auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
            subbuf, fwd, nthreads));
          // copy out
          CH(0,0,k) = res[0].r;
          for (size_t m=1; m<=ip/2; ++m)
            {
            CH(ido-1,2*m-1,k)=res[m].r;
            CH(0,2*m,k)=res[m].i;
            }
          }
        if (ido==1) return ch;
        size_t ipph = (ip+1)/2;
        for (size_t k=0; k<l1; ++k)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            // copy in
            cc2[0] = {CC(i-1,k,0),CC(i,k,0)};
            for (size_t m=1; m<ipph; ++m)
              {
              MULPM (cc2[m].r,cc2[m].i,WA(m-1,i-2),WA(m-1,i-1),CC(i-1,k,m),CC(i,k,m));
              MULPM (cc2[ip-m].r,cc2[ip-m].i,WA(ip-m-1,i-2),WA(ip-m-1,i-1),CC(i-1,k,ip-m),CC(i,k,ip-m));
              }
            auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
              subbuf, fwd, nthreads));
            CH(i-1,0,k) = res[0].r;
            CH(i,0,k) = res[0].i;
            for (size_t m=1; m<ipph; ++m)
              {
              CH(i-1,2*m,k) = res[m].r;
              CH(ic-1,2*m-1,k) = res[ip-m].r;
              CH(i  ,2*m,k) = res[m].i;
              CH(ic  ,2*m-1,k) = -res[ip-m].i;
              }
            }
        }
      else
        {
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };

        for (size_t k=0; k<l1; k++)
          {
          cc2[0] = {CC(0,0,k), Tfd(0)};
          for (size_t m=1; m<=ip/2; ++m)
            {
            cc2[m] = {CC(ido-1,2*m-1,k),CC(0,2*m,k)};
            cc2[ip-m] = {CC(ido-1,2*m-1,k),-CC(0,2*m,k)};
            }
          auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
            subbuf, fwd, nthreads));
          for (size_t m=0; m<ip; ++m)
            CH(0,k,m) = res[m].r;
          }
        if (ido==1) return ch;
        for (size_t k=0; k<l1; ++k)
          for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
            {
            // copy in
            cc2[0] = {CC(i-1,0,k),CC(i,0,k)};
            for (size_t m=1; m<=ip/2; ++m)
              {
              cc2[m] = {CC(i-1,2*m,k),CC(i,2*m,k)};
              cc2[ip-m] = {CC(ic-1,2*m-1,k),-CC(ic,2*m-1,k)};
              }
            auto res = static_cast<Tcd *>(cplan->exec(ticd, cc2, ch2,
              subbuf, fwd, nthreads));
            CH(i-1,k,0) = res[0].r;
            CH(i,k,0) = res[0].i;
            for (size_t m=1; m<ip; ++m)
              {
              MULPM(CH(i-1,k,m),CH(i,k,m),WA(m-1,i-2),-WA(m-1,i-1),res[m].r,res[m].i);
              MULPM(CH(i-1,k,ip-m),CH(i,k,ip-m),WA(ip-m-1,i-2),-WA(ip-m-1,i-1),res[ip-m].r,res[ip-m].i);
              }
            }
        }
      return ch;
      }

  public:
    rfftpblue(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots, bool vectorize=false)
      : l1(l1_), ido(ido_), ip(ip_), wa((ip-1)*(ido-1)),
        cplan(cfftpass<Tfs>::make_pass(1,1,ip,roots,vectorize))
      {
      MR_assert(ip&1, "Bluestein length must be odd");
      MR_assert(ido&1, "ido must be odd");
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      }

    virtual size_t bufsize() const { return 4*ip + 2*cplan->bufsize(); }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const
      { return wa.size()*sizeof(Tfs) + cplan->footprint(); }

    friend rfftpass_impl<rfftpblue<Tfs>,Tfs>;
  };

template <typename Tfs> class rfft_multipass: public rfftpass_impl<rfft_multipass<Tfs>,Tfs>
  {
  private:
    const size_t l1, ido;
    size_t ip;
    vector<Trpass<Tfs>> passes;
    size_t bufsz;
    bool need_cpy;
    aligned_array<Tfs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[(i-1)*(ip-1)+x]; }

    template<bool fwd, typename Tfd> Tfd *exec_(Tfd *cc, Tfd *ch, Tfd *buf,
      size_t nthreads) const
      {
      static const auto tifd = tidx<Tfd *>();
      if ((l1==1) && (ido==1))
        {
        Tfd *p1=cc, *p2=ch;
        if constexpr (fwd)
          for (auto it=passes.rbegin(); it!=passes.rend(); ++it)
            {
            auto res = static_cast<Tfd *>((*it)->exec(tifd,
              p1, p2, buf, fwd, nthreads));
            if (res==p2) swap(p1,p2);
            }
        else
          for (const auto &pass: passes)
            {
            auto res = static_cast<Tfd *>(pass->exec(tifd,
              p1, p2, buf, fwd, nthreads));
            if (res==p2) swap(p1,p2);
            }
        return p1;
        }
      else
        MR_fail("not yet supported");
      }

  public:
    rfft_multipass(size_t l1_, size_t ido_, size_t ip_,
      const Troots<Tfs> &roots, bool /*vectorize*/=false)
      : l1(l1_), ido(ido_), ip(ip_), bufsz(0), need_cpy(false),
        wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }

      auto factors = rfftpass<Tfs>::factorize(ip);

      size_t l1l=1;
      for (auto fct: factors)
        {
        passes.push_back(rfftpass<Tfs>::make_pass(l1l, ip/(fct*l1l), fct, roots));
        l1l*=fct;
        }
      for (const auto &pass: passes)
        {
        bufsz = max(bufsz, pass->bufsize());
        need_cpy |= pass->needs_copy();
        }
      if ((l1!=1)||(ido!=1))
        {
        need_cpy=true;
        bufsz += 2*ip;
        }
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }
    virtual size_t footprint() const
      {
      size_t res = wa.size()*sizeof(Tfs);
      for (const auto &p: passes) res += p->footprint();
      return res;
      }

    friend rfftpass_impl<rfft_multipass<Tfs>,Tfs>;
  };

/* Real transform of even length N via a complex transform of length N/2
   and a pass that combines the outputs k and N/2-k with the twiddles
   tw[k]=exp(-2*pi*i*k/N), k<=N/4. */
template <typename Tfs> class rfftp_complexify: public rfftpass_impl<rfftp_complexify<Tfs>,Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    static constexpr size_t vl = fft1d_max_simdlen<Tfs>;
    static constexpr bool havesimd = (vl>1) && simd_exists<Tfs,vl>;
    size_t N;
    Tcpass<Tfs> pass;
    aligned_array<Tcs> tw;

    // vl complex values at p, optionally in reversed order
    template<typename V> static Cmplx<V> ldv(const Tcs *p, bool rev)
      {
      auto x = vload<V>(p);
      return rev ? Cmplx<V>(shuf<sh_rev<vl>>(x.r,x.r), shuf<sh_rev<vl>>(x.i,x.i)) : x;
      }
    template<typename V> static void stv(const Cmplx<V> &x, Tcs *p, bool rev)
      { vstore(rev ? Cmplx<V>(shuf<sh_rev<vl>>(x.r,x.r), shuf<sh_rev<vl>>(x.i,x.i)) : x, p); }

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * buf, size_t nthreads) const
      {
      using Tcd = Cmplx<Tfd>;
      auto ccc = reinterpret_cast<Tcd *>(cc);
      auto cch = reinterpret_cast<Tcd *>(ch);
      auto cbuf = reinterpret_cast<Tcd *>(buf);
      static const auto ticd = tidx<Tcd *>();
      if constexpr(fwd)
        {
        auto res = static_cast<Tcd *>(pass->exec(ticd,
          ccc, cch, cbuf, true, nthreads));
        auto rres = (res==ccc) ? ch : cc;
        rres[0] = res[0].r+res[0].i;
        size_t i=1, xi=N/2-1;
        if constexpr (is_same_v<Tfd,Tfs> && havesimd)  // vl pairs (i, N/2-i) at a time
          {
          using V = typename simd_select<Tfs,vl>::type;
          for (; i+2*vl<=xi+2; i+=vl, xi-=vl)
            {
            auto a=ldv<V>(res+i, false), b=ldv<V>(res+xi+1-vl, true), t=ldv<V>(tw.data()+i, false);
            V er=a.r+b.r, ei=a.i-b.i, dr=a.i+b.i, di=b.r-a.r;
            V xr=dr*t.r-di*t.i, xi_=dr*t.i+di*t.r;
            stv(Cmplx<V>(Tfs(0.5)*(er+xr), Tfs(0.5)*(ei+xi_)), reinterpret_cast<Tcs *>(rres+2*i-1), false);
            stv(Cmplx<V>(Tfs(0.5)*(er-xr), Tfs(0.5)*(xi_-ei)), reinterpret_cast<Tcs *>(rres+2*(xi+1-vl)-1), true);
            }
          }
        for (; i<=xi; ++i, --xi)
          {
          auto a=res[i], b=res[xi];
          Tfd er=a.r+b.r, ei=a.i-b.i, dr=a.i+b.i, di=b.r-a.r;
          Tfd xr=dr*tw[i].r-di*tw[i].i, xi_=dr*tw[i].i+di*tw[i].r;
          rres[2*i-1] = Tfs(0.5)*(er+xr);
          rres[2*i] = Tfs(0.5)*(ei+xi_);
          rres[2*xi-1] = Tfs(0.5)*(er-xr);
          rres[2*xi] = Tfs(0.5)*(xi_-ei);
          }
        rres[N-1] = res[0].r-res[0].i;
        return rres;
        }
      else
        {
        cch[0] = Tcd(cc[0]+cc[N-1], cc[0]-cc[N-1]);
        size_t i=1, xi=N/2-1;
        if constexpr (is_same_v<Tfd,Tfs> && havesimd)
          {
          using V = typename simd_select<Tfs,vl>::type;
          for (; i+2*vl<=xi+2; i+=vl, xi-=vl)
            {
            auto t1=ldv<V>(reinterpret_cast<const Tcs *>(cc+2*i-1), false),
                 t2=ldv<V>(reinterpret_cast<const Tcs *>(cc+2*(xi+1-vl)-1), true), t=ldv<V>(tw.data()+i, false);
            V er=t1.r+t2.r, ei=t1.i-t2.i, dr=t1.r-t2.r, di=t1.i+t2.i;
            V xr=dr*t.r+di*t.i, xi_=di*t.r-dr*t.i;
            stv(Cmplx<V>(er-xi_, ei+xr), cch+i, false);
            stv(Cmplx<V>(er+xi_, xr-ei), cch+xi+1-vl, true);
            }
          }
        for (; i<=xi; ++i, --xi)
          {
          Tfd t1r=cc[2*i-1], t1i=cc[2*i], t2r=cc[2*xi-1], t2i=-cc[2*xi];
          Tfd er=t1r+t2r, ei=t1i+t2i, dr=t1r-t2r, di=t1i-t2i;
          Tfd xr=dr*tw[i].r+di*tw[i].i, xi_=di*tw[i].r-dr*tw[i].i;  // (t1-t2)*conj(tw)
          cch[i] = Tcd(er-xi_, ei+xr);
          cch[xi] = Tcd(er+xi_, xr-ei);
          }
        auto res = static_cast<Tcd *>(pass->exec(ticd,
          cch, ccc, cbuf, false, nthreads));
        return (res==ccc) ? cc : ch;
        }
      }

  public:
    rfftp_complexify(size_t N_, const Troots<Tfs> &roots, bool vectorize=false)
      : N(N_), pass(cfftpass<Tfs>::make_pass(N/2, vectorize)), tw(N/4+1)
      {
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      MR_assert((N&1)==0, "N must be even");
      for (size_t i=0; i<tw.size(); ++i)
        tw[i] = (*roots)[rfct*i].conj();
      }

    virtual size_t bufsize() const { return 2*pass->bufsize(); }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const
      { return pass->footprint() + tw.size()*sizeof(Tcs); }

    friend rfftpass_impl<rfftp_complexify<Tfs>,Tfs>;
  };

} // unnamed namespace

template<typename Tfs> Trpass<Tfs> rfftpass<Tfs>::make_pass(size_t l1,
  size_t ido, size_t ip, const Troots<Tfs> &roots, bool vectorize)
  {
  MR_assert(ip>=1, "no zero-sized FFTs");
  if (ip==1) return make_shared<rfftp1<Tfs>>();
  // a single transform of even length is fastest via the SIMD complex plan of
  // half the length (measured: from 4 SIMD widths on; slower below)
  if (vectorize && (l1==1) && (ido==1) && (ip>=4*fft1d_max_simdlen<Tfs>) && ((ip&1)==0))
    return make_shared<rfftp_complexify<Tfs>>(ip, roots, vectorize);
  if ((ip>1000) && ((ip&1)==0))  // use complex transform
    {
    bool doit = vectorize&&((ip&7)==0);  // vecpass might be beneficial
    doit |= ip>10000;  // complex multipass might be beneficial
    if (!doit)
      {
      auto factors = rfftpass<Tfs>::factorize(ip);
      for (auto factor: factors)
        // complex Bluestein or larger prime factor functions might be beneficial
        if (factor>5) { doit=true; break; }
      }
    if (doit)
      return make_shared<rfftp_complexify<Tfs>>(ip, roots, vectorize);
    }
  auto factors=rfftpass<Tfs>::factorize(ip);
  if (factors.size()==1)
    {
    switch(ip)
      {
      case  2: return make_shared<rfftpR< 2,Tfs>>(l1, ido, roots);
      case  3: return make_shared<rfftpR< 3,Tfs>>(l1, ido, roots);
      case  4: return make_shared<rfftpR< 4,Tfs>>(l1, ido, roots);
      case  5: return make_shared<rfftpR< 5,Tfs>>(l1, ido, roots);
      case  7: return make_shared<rfftpR< 7,Tfs>>(l1, ido, roots);
      case  9: return make_shared<rfftpR< 9,Tfs>>(l1, ido, roots);
      case 11: return make_shared<rfftpR<11,Tfs>>(l1, ido, roots);
      case 13: return make_shared<rfftpR<13,Tfs>>(l1, ido, roots);
      default:
        if (ip<135)
          return make_shared<rfftpR<0,Tfs>>(l1, ido, roots, ip);
        else
          return make_shared<rfftpblue<Tfs>>(l1, ido, ip, roots, vectorize);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<rfft_multipass<Tfs>>(l1, ido, ip, roots, vectorize);
  }

}}

#endif
