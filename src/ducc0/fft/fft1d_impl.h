/*
This file is part of the ducc FFT library

Copyright (C) 2010-2023 Max-Planck-Society
Copyright (C) 2019 Peter Bell

For the odd-sized DCT-IV transforms:
  Copyright (C) 2003, 2007-14 Matteo Frigo
  Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

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

template<typename T> constexpr inline size_t fft1d_simdlen
  = native_simd<T>::size();

#define POCKETFFT_EXEC_DISPATCH \
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf, \
      bool fwd, size_t nthreads=1) const \
      { \
      static const auto tics = tidx<Tcs *>(); \
      if (ti==tics) \
        { \
        auto in1 = static_cast<Tcs *>(in); \
        auto copy1 = static_cast<Tcs *>(copy); \
        auto buf1 = static_cast<Tcs *>(buf); \
        return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                   : exec_<false>(in1, copy1, buf1, nthreads); \
        } \
      if constexpr (fft1d_simdlen<Tfs> > 1) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 2) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/2>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/2>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 4) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/4>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/4>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 8) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/8>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/8>::type; \
          using Tcv = Cmplx<Tfv>; \
          static const auto ticv = tidx<Tcv *>(); \
          if (ti==ticv) \
            {  \
            auto in1 = static_cast<Tcv *>(in); \
            auto copy1 = static_cast<Tcv *>(copy); \
            auto buf1 = static_cast<Tcv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      MR_fail("impossible vector length requested"); \
      }

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
template<typename F, size_t... I> DUCC0_ALWAYS_INLINE void static_for_impl(F &f, index_sequence<I...>)
  { (f(integral_constant<size_t,I>()), ...); }
template<size_t N, typename F> DUCC0_ALWAYS_INLINE void static_for(F &&f)
  { static_for_impl(f, make_index_sequence<N>()); }

struct ct_cs { long double c, s; };
// cos and sin of 2*pi*num/den, evaluated at compile time
consteval ct_cs ct_sincos(size_t num, size_t den)
  {
  constexpr long double pi = 3.141592653589793238462643383279502884L,
                        r2 = 0.7071067811865475244008443621048490393L;
  num %= den;
  long double a = 8.0L*num/den;   // angle in eighths of a turn
  size_t k = size_t(a+0.5L);
  long double x = (a-k)*(pi/4), x2 = x*x, c = 1, s = x, tc = 1, ts = x;
  for (size_t n=1; n<16; ++n)
    { tc *= -x2/((2*n-1)*(2*n)); c += tc; ts *= -x2/((2*n)*(2*n+1)); s += ts; }
  constexpr long double kc[8] = {1, r2, 0, -r2, -1, -r2, 0, r2},
                        ks[8] = {0, r2, 1, r2, 0, -r2, -1, -r2};
  k &= 7;
  return {c*kc[k]-s*ks[k], c*ks[k]+s*kc[k]};
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
    constexpr Tfs h = Tfs(0.7071067811865475244008443621048490393L);
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
  Tfv a(q, element_aligned_tag()), b(q+vlen, element_aligned_tag());
  return {shuf<sh_even>(a,b), shuf<sh_odd>(a,b)};
  }
template<typename Tfv, typename Tfs> DUCC0_ALWAYS_INLINE void vstore(const Cmplx<Tfv> &x, Cmplx<Tfs> *p)
  {
  constexpr size_t vlen = Tfv::size();
  auto q = reinterpret_cast<Tfs *>(p);
  shuf<sh_zip_lo<vlen>>(x.r, x.i).copy_to(q, element_aligned_tag());
  shuf<sh_zip_hi<vlen>>(x.r, x.i).copy_to(q+vlen, element_aligned_tag());
  }
// the R x vl block of reals at p, transposed: x[j] lane t = p[j+R*t]
template<size_t R, typename Tfv, typename Tfs> DUCC0_ALWAYS_INLINE void vgather(const Tfs *p, array<Tfv,R> &x)
  {
  constexpr size_t vl = Tfv::size();
  array<Tfv,R> v;
  static_for<R>([&](auto m) DUCC0_LAMBDA_INLINE{ v[m] = Tfv(p+m*vl, element_aligned_tag()); });
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
template<size_t R, typename Tfv, typename Tfs> DUCC0_ALWAYS_INLINE void vscatter(const array<Tfv,R> &x, Tfs *p)
  {
  constexpr size_t vl = Tfv::size();
  static_for<R>([&](auto m) DUCC0_LAMBDA_INLINE
    {
    constexpr size_t M = decltype(m)::value;
    Tfv v = x[0];
    static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE  // collect the lanes going to memory vector m
      { v = shuf<sh_scatter<decltype(j)::value,M,R,vl>>(v, x[j]); });
    v.copy_to(p+M*vl, element_aligned_tag());
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


/* Radix-R pass, with the DFT of length R generated at compile time. Indexing
   as in FFTPACK: input cc[i+ido*(j+R*k)], output ch[i+ido*(k+l1*j)], where
   output j>0 is multiplied by the twiddle w^(l1*i*j). exec works on
   interleaved complex data of any type, exec_planar on real and imaginary
   planes of scalar data, SIMD over vl consecutive butterflies (consecutive
   in memory, so without shuffles; the last pass, ido==1, gathers). */
template <size_t R, typename Tfs> class cfftpR: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;
    static constexpr size_t vl = fft1d_simdlen<Tfs>;
    size_t l1, ido;
    /* twiddles w^(l1*i*(j+1)), i<ido, j<R-1, in blocks of vl values of i:
       real parts at w[wi(i,j)], imaginary parts vl entries later. The 2(R-1)
       twiddles of one i then lie close together, and the vl consecutive ones
       of one j load as a vector. */
    aligned_array<Tfs> w;
    size_t wi(size_t i, size_t j) const { return ((i/vl)*(R-1)+j)*2*vl + i%vl; }

    /* not inlined since GCC seem to drop the restrict qualifiers. Scalar data
       does not rely on the compiler's vectorization of the loop over i (it
       failed for radix 5 and for the blocked twiddles), but takes the
       explicit SIMD path of the planar passes with interleaved data. */
    template<bool fwd, typename Tcd> DUCC0_NOINLINE Tcd *exec_ (Tcd *cc, Tcd *ch,
      Tcd * /*buf*/, size_t /*nthreads*/) const
      {
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
            array<Tcd,R> x;
            static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=cc[i+ido*j]; });
            dft<R,fwd,Tfs>(x);
            cc[i]=x[0];
            const Tfs *wp = w.data()+wi(i,0);
            static_for<R-1>([&](auto j) DUCC0_LAMBDA_INLINE
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
            array<Tcd,R> x;
            static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=cck[ido*j]; });
            dft<R,fwd,Tfs>(x);
            static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ chk[so*j]=x[j]; });
            }
          for (size_t i=1; i<ido; ++i)
            {
            array<Tcd,R> x;
            static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=cck[i+ido*j]; });
            dft<R,fwd,Tfs>(x);
            chk[i]=x[0];
            const Tfs *wp = w.data()+wi(i,0);
            static_for<R-1>([&](auto j) DUCC0_LAMBDA_INLINE
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
      else return Cmplx<V>(V(p+idx, element_aligned_tag()), V(p+N+idx, element_aligned_tag()));
      }
    template<typename V, bool il> DUCC0_ALWAYS_INLINE static void st(const Cmplx<V> &x, Tfs *p, size_t N, size_t idx)
      {
      if constexpr (il)
        {
        auto c = reinterpret_cast<Tcs *>(p);
        if constexpr (is_same_v<V,Tfs>) c[idx]=x; else vstore(x, c+idx);
        }
      else if constexpr (is_same_v<V,Tfs>) { p[idx]=x.r; p[N+idx]=x.i; }
      else { x.r.copy_to(p+idx, element_aligned_tag()); x.i.copy_to(p+N+idx, element_aligned_tag()); }
      }
    // one butterfly (V=Tfs) or vl consecutive ones (V=SIMD) at index i of transform k
    template<bool fwd, typename V, bool iin, bool iout> DUCC0_ALWAYS_INLINE void pstep(size_t i, size_t k,
      const Tfs *in, Tfs *out) const
      {
      const size_t N=R*l1*ido;
      array<Cmplx<V>,R> x;
      static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=ld<V,iin>(in, N, i+ido*(j+R*k)); });
      dft<R,fwd,Tfs>(x);
      st<V,iout>(x[0], out, N, i+ido*k);
      static_for<R-1>([&](auto j) DUCC0_LAMBDA_INLINE
        { st<V,iout>(x[j+1].template special_mul<fwd>(ld<V,false>(w.data(), vl, wi(i,j))), out, N, i+ido*(k+l1*(j+1))); });
      }
    template<bool fwd, bool iin, bool iout> void exec_planar_(const Tfs *in, Tfs *out) const
      {
      const size_t N=R*l1*ido;
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
                array<Tfv,R> xr, xi;
                vgather<R>(in+R*k, xr);
                vgather<R>(in+N+R*k, xi);
                array<Cmplx<Tfv>,R> x;
                static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=Cmplx<Tfv>(xr[j], xi[j]); });
                dft<R,fwd,Tfs>(x);
                static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ st<Tfv,iout>(x[j], out, N, k+l1*j); });
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
    cfftpR(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), w(((ido+vl-1)/vl)*(R-1)*2*vl)
      {
      size_t N=R*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t i=0; i<ido; ++i)
        for (size_t j=0; j<R-1; ++j)
          {
          auto v = (*roots)[rfct*(j+1)*l1*i];
          w[wi(i,j)] = v.r; w[wi(i,j)+vl] = v.i;
          }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const { return w.size()*sizeof(Tfs); }
    virtual bool planar() const { return true; }
    virtual void exec_planar(const Tfs *in, Tfs *out, bool fwd, bool iin, bool iout) const
      {
      MR_assert(!(iin&&iout), "cannot convert input and output in one pass");
      if (iin) fwd ? exec_planar_<true,true,false>(in,out) : exec_planar_<false,true,false>(in,out);
      else if (iout) fwd ? exec_planar_<true,false,true>(in,out) : exec_planar_<false,false,true>(in,out);
      else fwd ? exec_planar_<true,false,false>(in,out) : exec_planar_<false,false,false>(in,out);
      }

    POCKETFFT_EXEC_DISPATCH
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
          auto w = (*roots)[rfct*((l*k)%N)];
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

template <typename Tfs> class cfftpg: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    size_t l1, ido;
    size_t ip;
    aligned_array<Tcs> wa;
    aligned_array<Tcs> csarr;

    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename Tcd> Tcd *exec_
      (Tcd * DUCC0_RESTRICT cc, Tcd * DUCC0_RESTRICT ch, Tcd * /*buf*/, size_t /*nthreads*/) const
      {
      size_t ipph = (ip+1)/2;
      size_t idl1 = ido*l1;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tcd&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tcd&
        { return cc[a+ido*(b+ip*c)]; };
      auto CX = [cc,this](size_t a, size_t b, size_t c) -> Tcd&
        { return cc[a+ido*(b+l1*c)]; };
      auto CX2 = [cc, idl1](size_t a, size_t b) -> Tcd&
        { return cc[a+idl1*b]; };
      auto CH2 = [ch, idl1](size_t a, size_t b) -> const Tcd&
        { return ch[a+idl1*b]; };

      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          CH(i,k,0) = CC(i,0,k);
      for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)
        for (size_t k=0; k<l1; ++k)
          for (size_t i=0; i<ido; ++i)
            PM(CH(i,k,j),CH(i,k,jc),CC(i,j,k),CC(i,jc,k));
      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          {
          Tcd tmp = CH(i,k,0);
          for (size_t j=1; j<ipph; ++j)
            tmp+=CH(i,k,j);
          CX(i,k,0) = tmp;
          }
      for (size_t l=1, lc=ip-1; l<ipph; ++l, --lc)
        {
        // j=0,1,2
        {
        auto wal  = fwd ? csarr[  l].conj() : csarr[  l];
        auto wal2 = fwd ? csarr[2*l].conj() : csarr[2*l];
        for (size_t ik=0; ik<idl1; ++ik)
          {
          CX2(ik,l ).r = CH2(ik,0).r+wal.r*CH2(ik,1).r+wal2.r*CH2(ik,2).r;
          CX2(ik,l ).i = CH2(ik,0).i+wal.r*CH2(ik,1).i+wal2.r*CH2(ik,2).i;
          CX2(ik,lc).r =-wal.i*CH2(ik,ip-1).i-wal2.i*CH2(ik,ip-2).i;
          CX2(ik,lc).i = wal.i*CH2(ik,ip-1).r+wal2.i*CH2(ik,ip-2).r;
          }
        }

        size_t iwal=2*l;
        size_t j=3, jc=ip-3;
        for (; j<ipph-1; j+=2, jc-=2)
          {
          iwal+=l; if (iwal>ip) iwal-=ip;
          Tcs xwal=fwd ? csarr[iwal].conj() : csarr[iwal];
          iwal+=l; if (iwal>ip) iwal-=ip;
          Tcs xwal2=fwd ? csarr[iwal].conj() : csarr[iwal];
          for (size_t ik=0; ik<idl1; ++ik)
            {
            CX2(ik,l).r += CH2(ik,j).r*xwal.r+CH2(ik,j+1).r*xwal2.r;
            CX2(ik,l).i += CH2(ik,j).i*xwal.r+CH2(ik,j+1).i*xwal2.r;
            CX2(ik,lc).r -= CH2(ik,jc).i*xwal.i+CH2(ik,jc-1).i*xwal2.i;
            CX2(ik,lc).i += CH2(ik,jc).r*xwal.i+CH2(ik,jc-1).r*xwal2.i;
            }
          }
        for (; j<ipph; ++j, --jc)
          {
          iwal+=l; if (iwal>ip) iwal-=ip;
          Tcs xwal=fwd ? csarr[iwal].conj() : csarr[iwal];
          for (size_t ik=0; ik<idl1; ++ik)
            {
            CX2(ik,l).r += CH2(ik,j).r*xwal.r;
            CX2(ik,l).i += CH2(ik,j).i*xwal.r;
            CX2(ik,lc).r -= CH2(ik,jc).i*xwal.i;
            CX2(ik,lc).i += CH2(ik,jc).r*xwal.i;
            }
          }
        }

      // shuffling and twiddling
      if (ido==1)
        for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)
          for (size_t ik=0; ik<idl1; ++ik)
            {
            Tcd t1=CX2(ik,j), t2=CX2(ik,jc);
            PM(CX2(ik,j),CX2(ik,jc),t1,t2);
            }
      else
        {
        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)
          for (size_t k=0; k<l1; ++k)
            {
            Tcd t1=CX(0,k,j), t2=CX(0,k,jc);
            PM(CX(0,k,j),CX(0,k,jc),t1,t2);
            for (size_t i=1; i<ido; ++i)
              {
              Tcd x1, x2;
              PM(x1,x2,CX(i,k,j),CX(i,k,jc));
              size_t idij=(j-1)*(ido-1)+i-1;
              special_mul<fwd>(x1,wa[idij],CX(i,k,j));
              idij=(jc-1)*(ido-1)+i-1;
              special_mul<fwd>(x2,wa[idij],CX(i,k,jc));
              }
            }
        }
      return cc;
      }

  public:
    cfftpg(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), ip(ip_), wa((ip-1)*(ido-1)), csarr(ip)
      {
      MR_assert((ip&1)&&(ip>=5), "need an odd number >=5");
      size_t N=ip*l1*ido;
      auto rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      for (size_t i=0; i<ip; ++i)
        csarr[i] = (*roots)[rfct*ido*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const
      { return (wa.size()+csarr.size())*sizeof(Tcs); }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfftpblue: public cfftpass<Tfs>
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

    POCKETFFT_EXEC_DISPATCH
  };

/* Rader's algorithm: a DFT of prime length p is a cyclic convolution of
   length p-1 (input and output indices permuted by the powers of a primitive
   root g modulo p), which is computed with FFTs of length p-1. */
template <typename Tfs> class cfftprader: public cfftpass<Tfs>
  {
  private:
    using typename cfftpass<Tfs>::Tcs;

    const size_t l1, ido, ip;
    const Tcpass<Tfs> subplan;
    aligned_array<Tcs> wa, cf;  // twiddles; FFT of the convolution kernel, scaled by 1/(ip-1)
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
      for (;; ++g)
        {
        size_t x=g, cnt=1;
        while (x!=1) { x=x*g%ip; ++cnt; }
        if (cnt==n) break;
        }
      gin[0]=1;
      for (size_t r=1; r<n; ++r) gin[r] = gin[r-1]*g%ip;
      for (size_t r=0; r<n; ++r) gout[(n-r)%n] = gin[r];

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

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class cfft_multipass: public cfftpass<Tfs>
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
        if constexpr(is_same<T,Tfs>::value && simd_exists<Tfs, native_simd<Tfs>::size()>) // we can vectorize!
          {
          using Tfv = native_simd<Tfs>;
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

    POCKETFFT_EXEC_DISPATCH
  };

#undef POCKETFFT_EXEC_DISPATCH

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
    constexpr size_t vl = fft1d_simdlen<Tfs>;  // try the full SIMD width first
    if (auto res = make_soa<vl,Tfs>(ip, roots)) return res;
    if (auto res = make_soa<vl/2,Tfs>(ip, roots)) return res;
    if (auto res = make_soa<vl/4,Tfs>(ip, roots)) return res;
#else
    Tcpass<Tfs> res;
    static_for<3>([&](auto s) DUCC0_LAMBDA_INLINE  // try the full SIMD width first
      {
      constexpr size_t vl = fft1d_simdlen<Tfs> >> decltype(s)::value;
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
      case 2:
        return make_shared<cfftpR<2,Tfs>>(l1, ido, roots);
      case 3:
        return make_shared<cfftpR<3,Tfs>>(l1, ido, roots);
      case 4:
        return make_shared<cfftpR<4,Tfs>>(l1, ido, roots);
      case 5:
        return make_shared<cfftpR<5,Tfs>>(l1, ido, roots);
      case 7:
        return make_shared<cfftpR<7,Tfs>>(l1, ido, roots);
      case 8:
        return make_shared<cfftpR<8,Tfs>>(l1, ido, roots);
      case 11:
        return make_shared<cfftpR<11,Tfs>>(l1, ido, roots);
      default:
        /* Primes up to about 100 inside a composite length: the generic
           O(ip^2) pass is fastest (it vectorizes along ido). A prime as the
           whole transform, or a larger prime factor: convolution algorithms.
           Rader needs FFTs of length ip-1; it wins when these need only
           codelet passes and can use SIMD lanes (ip-1 divisible by 4),
           otherwise Bluestein's padded power-of-two-like length is faster. */
        if ((ip<110) && (l1*ido>1))
          return make_shared<cfftpg<Tfs>>(l1, ido, ip, roots);
        if (((ip-1)%4==0) && (util1d::prime_factors(ip-1).back()<=13))
          return make_shared<cfftprader<Tfs>>(l1, ido, ip, roots, vectorize);
        return make_shared<cfftpblue<Tfs>>(l1, ido, ip, roots, vectorize);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<cfft_multipass<Tfs>>(l1, ido, ip, roots, vectorize);
  }

namespace {

#define POCKETFFT_EXEC_DISPATCH \
    virtual void *exec(const type_index &ti, void *in, void *copy, void *buf, \
      bool fwd, size_t nthreads) const \
      { \
      static const auto tifs=tidx<Tfs *>(); \
      if (ti==tifs) \
        { \
        auto in1 = static_cast<Tfs *>(in); \
        auto copy1 = static_cast<Tfs *>(copy); \
        auto buf1 = static_cast<Tfs *>(buf); \
        return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                   : exec_<false>(in1, copy1, buf1, nthreads); \
        } \
      if constexpr (fft1d_simdlen<Tfs> > 1) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 2) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/2>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/2>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 4) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/4>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/4>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      if constexpr (fft1d_simdlen<Tfs> > 8) \
        if constexpr (simd_exists<Tfs, fft1d_simdlen<Tfs>/8>) \
          { \
          using Tfv = typename simd_select<Tfs, fft1d_simdlen<Tfs>/8>::type; \
          static const auto tifv=tidx<Tfv *>(); \
          if (ti==tifv) \
            {  \
            auto in1 = static_cast<Tfv *>(in); \
            auto copy1 = static_cast<Tfv *>(copy); \
            auto buf1 = static_cast<Tfv *>(buf); \
            return fwd ? exec_<true>(in1, copy1, buf1, nthreads) \
                       : exec_<false>(in1, copy1, buf1, nthreads); \
            } \
          } \
      MR_fail("impossible vector length requested"); \
      }

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
   of length R generated at compile time. Forward: cc[a+ido*(k+l1*j)] ->
   ch[a+ido*(j+R*k)], backward the other way. Index i=0 holds real values,
   (i-1,i) for even i>0 a complex value whose conjugate partner is stored
   mirrored at ido-i; for even R and even ido, i=ido-1 is real again. */
template <size_t R, typename Tfs> class rfftpR: public rfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    static constexpr size_t vl = fft1d_simdlen<Tfs>;
    size_t l1, ido;
    /* twiddles w^(l1*j*i/2), i=2,4,..<ido, j=1..R-1, in blocks of vl values of
       i (as in cfftpR): real parts at wa[wi(i/2-1,j-1)], imaginary parts vl later */
    aligned_array<Tfs> wa;
    size_t wi(size_t q, size_t j) const { return ((q/vl)*(R-1)+j)*2*vl + q%vl; }

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
      else return Cmplx<V>(V(p, element_aligned_tag()), V(p+vl, element_aligned_tag()));
      }
    /* the butterfly at i (V==Tfd), or the V::size() ones starting there (scalar
       data); complex values at (i-1,i), partners mirrored at (ido-i-1,ido-i);
       c, o: block k of input, output */
    template<bool fwd, typename V, typename Tfd> DUCC0_ALWAYS_INLINE void bfly(const Tfd *c, Tfd *o, size_t i) const
      {
      constexpr size_t n = is_same_v<V,Tfd> ? 1 : sizeof(V)/sizeof(Tfs);
      const size_t so=ido*l1, q=i/2-1, im=ido-i-1-2*(n-1);  // im: lowest index of the mirrored values
      array<Cmplx<V>,R> x;
      if constexpr (fwd)
        {
        x[0]=ld<V,false>(c+i-1);
        static_for<R-1>([&](auto j) DUCC0_LAMBDA_INLINE
          { x[j+1]=ld<V,false>(c+i-1+so*(j+1)).template special_mul<true>(tw<V,Tfd>(q,j)); });
        dft<R,true,Tfs>(x);
        static_for<(R+1)/2>([&](auto m) DUCC0_LAMBDA_INLINE{ st<V,false>(x[m], o+i-1+ido*2*m); });
        static_for<R/2>([&](auto m) DUCC0_LAMBDA_INLINE
          { st<V,true>(Cmplx<V>(x[R-1-m].r, -x[R-1-m].i), o+im+ido*(2*m+1)); });
        }
      else
        {
        static_for<(R+1)/2>([&](auto m) DUCC0_LAMBDA_INLINE{ x[m]=ld<V,false>(c+i-1+ido*2*m); });
        static_for<R/2>([&](auto m) DUCC0_LAMBDA_INLINE
          { auto y=ld<V,true>(c+im+ido*(2*m+1)); x[R-1-m]=Cmplx<V>(y.r, -y.i); });
        dft<R,false,Tfs>(x);
        st<V,false>(x[0], o+i-1);
        static_for<R-1>([&](auto j) DUCC0_LAMBDA_INLINE
          { st<V,false>(x[j+1].template special_mul<false>(tw<V,Tfd>(q,j)), o+i-1+so*(j+1)); });
        }
      }

    // not inlined, for the same reason as cfftpR::exec_
    template<bool fwd, typename Tfd> DUCC0_NOINLINE Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      using C = Cmplx<Tfd>;
      constexpr size_t h=(R-1)/2;
      const size_t so=ido*l1;
      const Tfd zero(0);
      size_t k0=0;
      if constexpr (is_same_v<Tfd,Tfs>)
        if (ido==1)  // scalar data, only the real butterflies at i=0: SIMD over consecutive k
          static_for<4>([&](auto s) DUCC0_LAMBDA_INLINE  // full SIMD width, then narrower vectors
          {
          constexpr size_t v = vl>>decltype(s)::value;
          if constexpr ((v>=2) && simd_exists<Tfs,v>)
          {
          using V = typename simd_select<Tfs,v>::type;
          const V vzero(Tfs(0));
          for (; k0+v<=l1; k0+=v)
            {
            array<Cmplx<V>,R> x;
            array<V,R> y;
            if constexpr (fwd)
              {
              static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=Cmplx<V>(V(cc+k0+l1*j, element_aligned_tag()), vzero); });
              dft<R,true,Tfs>(x);
              y[0]=x[0].r;
              static_for<h>([&](auto m) DUCC0_LAMBDA_INLINE{ y[2*m+1]=x[m+1].r; y[2*m+2]=x[m+1].i; });
              if constexpr ((R&1)==0) y[R-1]=x[R/2].r;
              vscatter<R>(y, ch+R*k0);
              }
            else
              {
              vgather<R>(cc+R*k0, y);
              x[0]=Cmplx<V>(y[0], vzero);
              static_for<h>([&](auto m) DUCC0_LAMBDA_INLINE
                { x[m+1]=Cmplx<V>(y[2*m+1], y[2*m+2]); x[R-1-m]=Cmplx<V>(y[2*m+1], -y[2*m+2]); });
              if constexpr ((R&1)==0) x[R/2]=Cmplx<V>(y[R-1], vzero);
              dft<R,false,Tfs>(x);
              static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j].r.copy_to(ch+k0+l1*j, element_aligned_tag()); });
              }
            }
          }
          });
      // i=0: real values split into separate loop for vectorization over k
      for (size_t k=k0; k<l1; ++k)
        {
        const size_t ck = ido*k*(fwd ? 1 : R), ok = ido*k*(fwd ? R : 1);
        if constexpr (fwd)
          {
          array<C,R> x;
          static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ x[j]=C(cc[ck+so*j], zero); });
          dft<R,true,Tfs>(x);
          ch[ok+0]=x[0].r;
          static_for<h>([&](auto m) DUCC0_LAMBDA_INLINE{ ch[ok+ido*(2*m+2)-1]=x[m+1].r; ch[ok+ido*(2*m+2)]=x[m+1].i; });
          if constexpr ((R&1)==0) ch[ok+ido*R-1]=x[R/2].r;
          }
        else
          {
          array<C,R> x;
          x[0]=C(cc[ck+0], zero);
          static_for<h>([&](auto m) DUCC0_LAMBDA_INLINE
            { x[m+1]=C(cc[ck+ido*(2*m+2)-1], cc[ck+ido*(2*m+2)]); x[R-1-m]=C(x[m+1].r, -x[m+1].i); });
          if constexpr ((R&1)==0) x[R/2]=C(cc[ck+ido*R-1], zero);
          dft<R,false,Tfs>(x);
          static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE{ ch[ok+so*j]=x[j].r; });
          }
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
        if constexpr ((R&1)==0)
          if ((ido&1)==0)
            {
            array<C,R> x;
            if constexpr (fwd)
              {
              static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE
                { x[j]=twmul<decltype(j)::value,2*R,true,Tfs>(C(c[ido-1+so*j], zero)); });
              dft<R,true,Tfs>(x);
              static_for<R/2>([&](auto m) DUCC0_LAMBDA_INLINE{ o[ido-1+ido*2*m]=x[m].r; o[ido*(2*m+1)]=x[m].i; });
              }
            else
              {
              static_for<R/2>([&](auto m) DUCC0_LAMBDA_INLINE
                { x[m]=C(c[ido-1+ido*2*m], c[ido*(2*m+1)]); x[R-1-m]=C(x[m].r, -x[m].i); });
              dft<R,false,Tfs>(x);
              static_for<R>([&](auto j) DUCC0_LAMBDA_INLINE
                { o[ido-1+so*j]=twmul<decltype(j)::value,2*R,false,Tfs>(x[j]).r; });
              }
            }
        }
      return ch;
      }

  public:
    rfftpR(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((((ido-1)/2+vl-1)/vl)*(R-1)*2*vl)
      {
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
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const { return wa.size()*sizeof(Tfs); }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftpg: public rfftpass<Tfs>
  {
  private:
    size_t l1, ido;
    size_t ip;
    aligned_array<Tfs> wa, csarr;

    template<bool fwd, typename Tfd> Tfd *exec_ (Tfd * DUCC0_RESTRICT cc,
      Tfd * DUCC0_RESTRICT ch, Tfd * /*buf*/, size_t /*nthreads*/) const
      {
      if constexpr(fwd)
        {
        size_t ipph=(ip+1)/2;
        size_t idl1 = ido*l1;

        auto CC = [cc,this](size_t a, size_t b, size_t c) -> Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return ch[a+ido*(b+l1*c)]; };
        auto C1 = [cc,this] (size_t a, size_t b, size_t c) -> Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto C2 = [cc,idl1] (size_t a, size_t b) -> Tfd&
          { return cc[a+idl1*b]; };
        auto CH2 = [ch,idl1] (size_t a, size_t b) -> Tfd&
          { return ch[a+idl1*b]; };

        if (ido>1)
          {
          for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)              // 114
            {
            size_t is=(j-1)*(ido-1),
                   is2=(jc-1)*(ido-1);
            for (size_t k=0; k<l1; ++k)                            // 113
              {
              size_t idij=is;
              size_t idij2=is2;
              for (size_t i=1; i<=ido-2; i+=2)                      // 112
                {
                Tfd t1=C1(i,k,j ), t2=C1(i+1,k,j ),
                    t3=C1(i,k,jc), t4=C1(i+1,k,jc);
                Tfd x1=wa[idij]*t1 + wa[idij+1]*t2,
                    x2=wa[idij]*t2 - wa[idij+1]*t1,
                    x3=wa[idij2]*t3 + wa[idij2+1]*t4,
                    x4=wa[idij2]*t4 - wa[idij2+1]*t3;
                PM(C1(i,k,j),C1(i+1,k,jc),x3,x1);
                PM(C1(i+1,k,j),C1(i,k,jc),x2,x4);
                idij+=2;
                idij2+=2;
                }
              }
            }
          }

        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 123
          for (size_t k=0; k<l1; ++k)                              // 122
            MPINPLACE(C1(0,k,jc), C1(0,k,j));

      //everything in C
      //memset(ch,0,ip*l1*ido*sizeof(double));

        for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)                 // 127
          {
          for (size_t ik=0; ik<idl1; ++ik)                         // 124
            {
            CH2(ik,l ) = C2(ik,0)+csarr[2*l]*C2(ik,1)+csarr[4*l]*C2(ik,2);
            CH2(ik,lc) = csarr[2*l+1]*C2(ik,ip-1)+csarr[4*l+1]*C2(ik,ip-2);
            }
          size_t iang = 2*l;
          size_t j=3, jc=ip-3;
          for (; j<ipph-3; j+=4,jc-=4)              // 126
            {
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar3=csarr[2*iang], ai3=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar4=csarr[2*iang], ai4=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)                       // 125
              {
              CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1)
                           +ar3*C2(ik,j +2)+ar4*C2(ik,j +3);
              CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1)
                           +ai3*C2(ik,jc-2)+ai4*C2(ik,jc-3);
              }
            }
          for (; j<ipph-1; j+=2,jc-=2)              // 126
            {
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)                       // 125
              {
              CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1);
              CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1);
              }
            }
          for (; j<ipph; ++j,--jc)              // 126
            {
            iang+=l; if (iang>=ip) iang-=ip;
            Tfs ar=csarr[2*iang], ai=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)                       // 125
              {
              CH2(ik,l ) += ar*C2(ik,j );
              CH2(ik,lc) += ai*C2(ik,jc);
              }
            }
          }
        for (size_t ik=0; ik<idl1; ++ik)                         // 101
          CH2(ik,0) = C2(ik,0);
        for (size_t j=1; j<ipph; ++j)                              // 129
          for (size_t ik=0; ik<idl1; ++ik)                         // 128
            CH2(ik,0) += C2(ik,j);

      // everything in CH at this point!
      //memset(cc,0,ip*l1*ido*sizeof(double));

        for (size_t k=0; k<l1; ++k)                                // 131
          for (size_t i=0; i<ido; ++i)                             // 130
            CC(i,0,k) = CH(i,k,0);

        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 137
          {
          size_t j2=2*j-1;
          for (size_t k=0; k<l1; ++k)                              // 136
            {
            CC(ido-1,j2,k) = CH(0,k,j);
            CC(0,j2+1,k) = CH(0,k,jc);
            }
          }

        if (ido==1) return cc;

        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 140
          {
          size_t j2=2*j-1;
          for(size_t k=0; k<l1; ++k)                               // 139
            for(size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 138
              {
              CC(i   ,j2+1,k) = CH(i  ,k,j )+CH(i  ,k,jc);
              CC(ic  ,j2  ,k) = CH(i  ,k,j )-CH(i  ,k,jc);
              CC(i+1 ,j2+1,k) = CH(i+1,k,j )+CH(i+1,k,jc);
              CC(ic+1,j2  ,k) = CH(i+1,k,jc)-CH(i+1,k,j );
              }
          }
        return cc;
        }
      else
        {
        size_t ipph=(ip+1)/ 2;
        size_t idl1 = ido*l1;

        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+ip*c)]; };
        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tfd&
          { return ch[a+ido*(b+l1*c)]; };
        auto C1 = [cc,this](size_t a, size_t b, size_t c) -> const Tfd&
          { return cc[a+ido*(b+l1*c)]; };
        auto C2 = [cc,idl1](size_t a, size_t b) -> Tfd&
          { return cc[a+idl1*b]; };
        auto CH2 = [ch,idl1](size_t a, size_t b) -> Tfd&
          { return ch[a+idl1*b]; };

        for (size_t k=0; k<l1; ++k)        // 102
          for (size_t i=0; i<ido; ++i)     // 101
            CH(i,k,0) = CC(i,0,k);
        for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)   // 108
          {
          size_t j2=2*j-1;
          for (size_t k=0; k<l1; ++k)
            {
            CH(0,k,j ) = Tfs(2)*CC(ido-1,j2,k);
            CH(0,k,jc) = Tfs(2)*CC(0,j2+1,k);
            }
          }

        if (ido!=1)
          {
          for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 111
            {
            size_t j2=2*j-1;
            for (size_t k=0; k<l1; ++k)
              for (size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 109
                {
                CH(i  ,k,j ) = CC(i  ,j2+1,k)+CC(ic  ,j2,k);
                CH(i  ,k,jc) = CC(i  ,j2+1,k)-CC(ic  ,j2,k);
                CH(i+1,k,j ) = CC(i+1,j2+1,k)-CC(ic+1,j2,k);
                CH(i+1,k,jc) = CC(i+1,j2+1,k)+CC(ic+1,j2,k);
                }
            }
          }
        for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)
          {
          for (size_t ik=0; ik<idl1; ++ik)
            {
            C2(ik,l ) = CH2(ik,0)+csarr[2*l]*CH2(ik,1)+csarr[4*l]*CH2(ik,2);
            C2(ik,lc) = csarr[2*l+1]*CH2(ik,ip-1)+csarr[4*l+1]*CH2(ik,ip-2);
            }
          size_t iang=2*l;
          size_t j=3,jc=ip-3;
          for(; j<ipph-3; j+=4,jc-=4)
            {
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar3=csarr[2*iang], ai3=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar4=csarr[2*iang], ai4=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)
              {
              C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1)
                          +ar3*CH2(ik,j +2)+ar4*CH2(ik,j +3);
              C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1)
                          +ai3*CH2(ik,jc-2)+ai4*CH2(ik,jc-3);
              }
            }
          for(; j<ipph-1; j+=2,jc-=2)
            {
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar1=csarr[2*iang], ai1=csarr[2*iang+1];
            iang+=l; if(iang>ip) iang-=ip;
            Tfs ar2=csarr[2*iang], ai2=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)
              {
              C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1);
              C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1);
              }
            }
          for(; j<ipph; ++j,--jc)
            {
            iang+=l; if(iang>ip) iang-=ip;
            Tfs war=csarr[2*iang], wai=csarr[2*iang+1];
            for (size_t ik=0; ik<idl1; ++ik)
              {
              C2(ik,l ) += war*CH2(ik,j );
              C2(ik,lc) += wai*CH2(ik,jc);
              }
            }
          }
        for (size_t j=1; j<ipph; ++j)
          for (size_t ik=0; ik<idl1; ++ik)
            CH2(ik,0) += CH2(ik,j);
        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 124
          for (size_t k=0; k<l1; ++k)
            PM(CH(0,k,jc),CH(0,k,j),C1(0,k,j),C1(0,k,jc));

        if (ido==1) return ch;

        for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)  // 127
          for (size_t k=0; k<l1; ++k)
            for (size_t i=1; i<=ido-2; i+=2)
              {
              CH(i  ,k,j ) = C1(i  ,k,j)-C1(i+1,k,jc);
              CH(i  ,k,jc) = C1(i  ,k,j)+C1(i+1,k,jc);
              CH(i+1,k,j ) = C1(i+1,k,j)+C1(i  ,k,jc);
              CH(i+1,k,jc) = C1(i+1,k,j)-C1(i  ,k,jc);
              }

      // All in CH

        for (size_t j=1; j<ip; ++j)
          {
          size_t is = (j-1)*(ido-1);
          for (size_t k=0; k<l1; ++k)
            {
            size_t idij = is;
            for (size_t i=1; i<=ido-2; i+=2)
              {
              Tfd t1=CH(i,k,j), t2=CH(i+1,k,j);
              CH(i  ,k,j) = wa[idij]*t1-wa[idij+1]*t2;
              CH(i+1,k,j) = wa[idij]*t2+wa[idij+1]*t1;
              idij+=2;
              }
            }
          }
        return ch;
        }
      }

  public:
    rfftpg(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), ip(ip_), wa((ip-1)*(ido-1)), csarr(2*ip)
      {
      MR_assert(ido&1, "ido must be odd");
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          auto val = (*roots)[rfct*j*l1*i];
          wa[(j-1)*(ido-1)+2*i-2] = val.r;
          wa[(j-1)*(ido-1)+2*i-1] = val.i;
          }
      csarr[0] = Tfs(1);
      csarr[1] = Tfs(0);
      for (size_t i=2, ic=2*ip-2; i<=ic; i+=2, ic-=2)
        {
        auto val = (*roots)[i/2*rfct*(N/ip)];
        csarr[i   ] = val.r;
        csarr[i +1] = val.i;
        csarr[ic  ] = val.r;
        csarr[ic+1] = -val.i;
        }
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }
    virtual size_t footprint() const
      { return (wa.size()+csarr.size())*sizeof(Tfs); }

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfftpblue: public rfftpass<Tfs>
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

    POCKETFFT_EXEC_DISPATCH
  };

template <typename Tfs> class rfft_multipass: public rfftpass<Tfs>
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

    POCKETFFT_EXEC_DISPATCH
  };

/* Real transform of even length N via a complex transform of length N/2
   and a pass that combines the outputs k and N/2-k with the twiddles
   tw[k]=exp(-2*pi*i*k/N), k<=N/4. */
template <typename Tfs> class rfftp_complexify: public rfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    static constexpr size_t vl = fft1d_simdlen<Tfs>;
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

    POCKETFFT_EXEC_DISPATCH
  };
#undef POCKETFFT_EXEC_DISPATCH

} // unnamed namespace

template<typename Tfs> Trpass<Tfs> rfftpass<Tfs>::make_pass(size_t l1,
  size_t ido, size_t ip, const Troots<Tfs> &roots, bool vectorize)
  {
  MR_assert(ip>=1, "no zero-sized FFTs");
  if (ip==1) return make_shared<rfftp1<Tfs>>();
  // a single transform of even length is fastest via the SIMD complex plan of
  // half the length (measured: from 4 SIMD widths on; slower below)
  if (vectorize && (l1==1) && (ido==1) && (ip>=4*fft1d_simdlen<Tfs>) && ((ip&1)==0))
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
      case 2:
        return make_shared<rfftpR<2,Tfs>>(l1, ido, roots);
      case 3:
        return make_shared<rfftpR<3,Tfs>>(l1, ido, roots);
      case 4:
        return make_shared<rfftpR<4,Tfs>>(l1, ido, roots);
      case 5:
        return make_shared<rfftpR<5,Tfs>>(l1, ido, roots);
      case 7:
        return make_shared<rfftpR<7,Tfs>>(l1, ido, roots);
      case 9:
        return make_shared<rfftpR<9,Tfs>>(l1, ido, roots);
      case 11:
        return make_shared<rfftpR<11,Tfs>>(l1, ido, roots);
      case 13:
        return make_shared<rfftpR<13,Tfs>>(l1, ido, roots);
      default:
        if (ip<135)
          return make_shared<rfftpg<Tfs>>(l1, ido, ip, roots);
        else
          return make_shared<rfftpblue<Tfs>>(l1, ido, ip, roots, vectorize);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<rfft_multipass<Tfs>>(l1, ido, ip, roots, vectorize);
  }

}}

#endif
