/*
This file is part of pocketfft.

Copyright (C) 2010-2021 Max-Planck-Society
Copyright (C) 2019 Peter Bell

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

#ifndef DUCC0_FFT1D_H
#define DUCC0_FFT1D_H

#include <cstring>
#include <stdexcept>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/math/unity_roots.h"

namespace ducc0 {

namespace detail_fft {

using std::size_t;
using std::ptrdiff_t;

// Always use std:: for <cmath> functions
template <typename T> T cos(T) = delete;
template <typename T> T sin(T) = delete;
template <typename T> T sqrt(T) = delete;

template<typename T> inline void PM(T &a, T &b, T c, T d)
  { a=c+d; b=c-d; }
template<typename T> inline void PMINPLACE(T &a, T &b)
  { T t = a; a+=b; b=t-b; }
template<typename T> inline void MPINPLACE(T &a, T &b)
  { T t = a; a-=b; b=t+b; }
template<typename T> Cmplx<T> conj(const Cmplx<T> &a)
  { return {a.r, -a.i}; }
template<bool fwd, typename T, typename T2> void special_mul (const Cmplx<T> &v1, const Cmplx<T2> &v2, Cmplx<T> &res)
  {
  res = fwd ? Cmplx<T>(v1.r*v2.r+v1.i*v2.i, v1.i*v2.r-v1.r*v2.i)
            : Cmplx<T>(v1.r*v2.r-v1.i*v2.i, v1.r*v2.i+v1.i*v2.r);
  }

template<typename T> void ROT90(Cmplx<T> &a)
  { auto tmp_=a.r; a.r=-a.i; a.i=tmp_; }
template<bool fwd, typename T> void ROTX90(Cmplx<T> &a)
  { auto tmp_= fwd ? -a.r : a.r; a.r = fwd ? a.i : -a.i; a.i=tmp_; }

struct util1d // hack to avoid duplicate symbols
  {
  DUCC0_NOINLINE static size_t largest_prime_factor (size_t n)
    {
    size_t res=1;
    while ((n&1)==0)
      { res=2; n>>=1; }
    for (size_t x=3; x*x<=n; x+=2)
      while ((n%x)==0)
        { res=x; n/=x; }
    if (n>1) res=n;
    return res;
    }

  DUCC0_NOINLINE static double cost_guess (size_t n)
    {
    constexpr double lfp=1.1; // penalty for non-hardcoded larger factors
    size_t ni=n;
    double result=0.;
    while ((n&1)==0)
      { result+=2; n>>=1; }
    for (size_t x=3; x*x<=n; x+=2)
      while ((n%x)==0)
        {
        result+= (x<=5) ? double(x) : lfp*double(x); // penalize larger prime factors
        n/=x;
        }
    if (n>1) result+=(n<=5) ? double(n) : lfp*double(n);
    return result*double(ni);
    }

  /* returns the smallest composite of 2, 3, 5, 7 and 11 which is >= n */
  DUCC0_NOINLINE static size_t good_size_cmplx(size_t n)
    {
    if (n<=12) return n;

    size_t bestfac=2*n;
    for (size_t f11=1; f11<bestfac; f11*=11)
      for (size_t f117=f11; f117<bestfac; f117*=7)
        for (size_t f1175=f117; f1175<bestfac; f1175*=5)
          {
          size_t x=f1175;
          while (x<n) x*=2;
          for (;;)
            {
            if (x<n)
              x*=3;
            else if (x>n)
              {
              if (x<bestfac) bestfac=x;
              if (x&1) break;
              x>>=1;
              }
            else
              return n;
            }
          }
    return bestfac;
    }

  /* returns the smallest composite of 2, 3, 5 which is >= n */
  DUCC0_NOINLINE static size_t good_size_real(size_t n)
    {
    if (n<=6) return n;

    size_t bestfac=2*n;
    for (size_t f5=1; f5<bestfac; f5*=5)
      {
      size_t x = f5;
      while (x<n) x *= 2;
      for (;;)
        {
        if (x<n)
          x*=3;
        else if (x>n)
          {
          if (x<bestfac) bestfac=x;
          if (x&1) break;
          x>>=1;
          }
        else
          return n;
        }
      }
    return bestfac;
    }
  };

//
// complex FFTPACK transforms
//

template<typename T0> class cfftp
  {
  private:
    struct fctdata
      {
      size_t fct;
      Cmplx<T0> *tw, *tws;
      };

    size_t length;
    aligned_array<Cmplx<T0>> mem;
    std::vector<fctdata> fact;

    void add_factor(size_t factor)
      { fact.push_back({factor, nullptr, nullptr}); }

template<bool fwd, typename T> void pass2 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+2*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,0) = CC(0,0,k)+CC(0,1,k);
      CH(0,k,1) = CC(0,0,k)-CC(0,1,k);
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,0) = CC(0,0,k)+CC(0,1,k);
      CH(0,k,1) = CC(0,0,k)-CC(0,1,k);
      for (size_t i=1; i<ido; ++i)
        {
        CH(i,k,0) = CC(i,0,k)+CC(i,1,k);
        special_mul<fwd>(CC(i,0,k)-CC(i,1,k),WA(0,i),CH(i,k,1));
        }
      }
  }

#define POCKETFFT_PREP3(idx) \
        T t0 = CC(idx,0,k), t1, t2; \
        PM (t1,t2,CC(idx,1,k),CC(idx,2,k)); \
        CH(idx,k,0)=t0+t1;
#define POCKETFFT_PARTSTEP3a(u1,u2,twr,twi) \
        { \
        T ca=t0+t1*twr; \
        T cb{-t2.i*twi, t2.r*twi}; \
        PM(CH(0,k,u1),CH(0,k,u2),ca,cb) ;\
        }
#define POCKETFFT_PARTSTEP3b(u1,u2,twr,twi) \
        { \
        T ca=t0+t1*twr; \
        T cb{-t2.i*twi, t2.r*twi}; \
        special_mul<fwd>(ca+cb,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(ca-cb,WA(u2-1,i),CH(i,k,u2)); \
        }
template<bool fwd, typename T> void pass3 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  constexpr T0 tw1r=-0.5,
               tw1i= (fwd ? -1: 1) * T0(0.8660254037844386467637231707529362L);

  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+3*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      POCKETFFT_PREP3(0)
      POCKETFFT_PARTSTEP3a(1,2,tw1r,tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      POCKETFFT_PREP3(0)
      POCKETFFT_PARTSTEP3a(1,2,tw1r,tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        POCKETFFT_PREP3(i)
        POCKETFFT_PARTSTEP3b(1,2,tw1r,tw1i)
        }
      }
  }

#undef POCKETFFT_PARTSTEP3b
#undef POCKETFFT_PARTSTEP3a
#undef POCKETFFT_PREP3

template<bool fwd, typename T> void pass4 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+4*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      T t1, t2, t3, t4;
      PM(t2,t1,CC(0,0,k),CC(0,2,k));
      PM(t3,t4,CC(0,1,k),CC(0,3,k));
      ROTX90<fwd>(t4);
      PM(CH(0,k,0),CH(0,k,2),t2,t3);
      PM(CH(0,k,1),CH(0,k,3),t1,t4);
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      T t1, t2, t3, t4;
      PM(t2,t1,CC(0,0,k),CC(0,2,k));
      PM(t3,t4,CC(0,1,k),CC(0,3,k));
      ROTX90<fwd>(t4);
      PM(CH(0,k,0),CH(0,k,2),t2,t3);
      PM(CH(0,k,1),CH(0,k,3),t1,t4);
      }
      for (size_t i=1; i<ido; ++i)
        {
        T t1, t2, t3, t4;
        T cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
        PM(t2,t1,cc0,cc2);
        PM(t3,t4,cc1,cc3);
        ROTX90<fwd>(t4);
        CH(i,k,0) = t2+t3;
        special_mul<fwd>(t1+t4,WA(0,i),CH(i,k,1));
        special_mul<fwd>(t2-t3,WA(1,i),CH(i,k,2));
        special_mul<fwd>(t1-t4,WA(2,i),CH(i,k,3));
        }
      }
  }

#define POCKETFFT_PREP5(idx) \
        T t0 = CC(idx,0,k), t1, t2, t3, t4; \
        PM (t1,t4,CC(idx,1,k),CC(idx,4,k)); \
        PM (t2,t3,CC(idx,2,k),CC(idx,3,k)); \
        CH(idx,k,0).r=t0.r+t1.r+t2.r; \
        CH(idx,k,0).i=t0.i+t1.i+t2.i;

#define POCKETFFT_PARTSTEP5a(u1,u2,twar,twbr,twai,twbi) \
        { \
        T ca,cb; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PM(CH(0,k,u1),CH(0,k,u2),ca,cb); \
        }

#define POCKETFFT_PARTSTEP5b(u1,u2,twar,twbr,twai,twbi) \
        { \
        T ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        special_mul<fwd>(ca+cb,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(ca-cb,WA(u2-1,i),CH(i,k,u2)); \
        }
template<bool fwd, typename T> void pass5 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  constexpr T0 tw1r= T0(0.3090169943749474241022934171828191L),
               tw1i= (fwd ? -1: 1) * T0(0.9510565162951535721164393333793821L),
               tw2r= T0(-0.8090169943749474241022934171828191L),
               tw2i= (fwd ? -1: 1) * T0(0.5877852522924731291687059546390728L);

  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+5*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      POCKETFFT_PREP5(0)
      POCKETFFT_PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      POCKETFFT_PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      POCKETFFT_PREP5(0)
      POCKETFFT_PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      POCKETFFT_PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        POCKETFFT_PREP5(i)
        POCKETFFT_PARTSTEP5b(1,4,tw1r,tw2r,+tw1i,+tw2i)
        POCKETFFT_PARTSTEP5b(2,3,tw2r,tw1r,+tw2i,-tw1i)
        }
      }
  }

#undef POCKETFFT_PARTSTEP5b
#undef POCKETFFT_PARTSTEP5a
#undef POCKETFFT_PREP5

#define POCKETFFT_PREP7(idx) \
        T t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7; \
        PM (t2,t7,CC(idx,1,k),CC(idx,6,k)); \
        PM (t3,t6,CC(idx,2,k),CC(idx,5,k)); \
        PM (t4,t5,CC(idx,3,k),CC(idx,4,k)); \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i;

#define POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,out1,out2) \
        { \
        T ca,cb; \
        ca.r=t1.r+x1*t2.r+x2*t3.r+x3*t4.r; \
        ca.i=t1.i+x1*t2.i+x2*t3.i+x3*t4.i; \
        cb.i=y1*t7.r y2*t6.r y3*t5.r; \
        cb.r=-(y1*t7.i y2*t6.i y3*t5.i); \
        PM(out1,out2,ca,cb); \
        }
#define POCKETFFT_PARTSTEP7a(u1,u2,x1,x2,x3,y1,y2,y3) \
        POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,CH(0,k,u1),CH(0,k,u2))
#define POCKETFFT_PARTSTEP7(u1,u2,x1,x2,x3,y1,y2,y3) \
        { \
        T da,db; \
        POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,da,db) \
        special_mul<fwd>(da,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(db,WA(u2-1,i),CH(i,k,u2)); \
        }

template<bool fwd, typename T> void pass7(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  constexpr T0 tw1r= T0(0.6234898018587335305250048840042398L),
               tw1i= (fwd ? -1 : 1) * T0(0.7818314824680298087084445266740578L),
               tw2r= T0(-0.2225209339563144042889025644967948L),
               tw2i= (fwd ? -1 : 1) * T0(0.9749279121818236070181316829939312L),
               tw3r= T0(-0.9009688679024191262361023195074451L),
               tw3i= (fwd ? -1 : 1) * T0(0.433883739117558120475768332848359L);

  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+7*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      POCKETFFT_PREP7(0)
      POCKETFFT_PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
      POCKETFFT_PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
      POCKETFFT_PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      POCKETFFT_PREP7(0)
      POCKETFFT_PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
      POCKETFFT_PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
      POCKETFFT_PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        POCKETFFT_PREP7(i)
        POCKETFFT_PARTSTEP7(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
        POCKETFFT_PARTSTEP7(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
        POCKETFFT_PARTSTEP7(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
        }
      }
  }

#undef POCKETFFT_PARTSTEP7
#undef POCKETFFT_PARTSTEP7a0
#undef POCKETFFT_PARTSTEP7a
#undef POCKETFFT_PREP7

template <bool fwd, typename T> void ROTX45(T &a) const
  {
  constexpr T0 hsqt2=T0(0.707106781186547524400844362104849L);
  if (fwd)
    { auto tmp_=a.r; a.r=hsqt2*(a.r+a.i); a.i=hsqt2*(a.i-tmp_); }
  else
    { auto tmp_=a.r; a.r=hsqt2*(a.r-a.i); a.i=hsqt2*(a.i+tmp_); }
  }
template <bool fwd, typename T> void ROTX135(T &a) const
  {
  constexpr T0 hsqt2=T0(0.707106781186547524400844362104849L);
  if (fwd)
    { auto tmp_=a.r; a.r=hsqt2*(a.i-a.r); a.i=hsqt2*(-tmp_-a.i); }
  else
    { auto tmp_=a.r; a.r=hsqt2*(-a.r-a.i); a.i=hsqt2*(tmp_-a.i); }
  }

template<bool fwd, typename T> void pass8 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+8*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      T a0, a1, a2, a3, a4, a5, a6, a7;
      PM(a1,a5,CC(0,1,k),CC(0,5,k));
      PM(a3,a7,CC(0,3,k),CC(0,7,k));
      PMINPLACE(a1,a3);
      ROTX90<fwd>(a3);

      ROTX90<fwd>(a7);
      PMINPLACE(a5,a7);
      ROTX45<fwd>(a5);
      ROTX135<fwd>(a7);

      PM(a0,a4,CC(0,0,k),CC(0,4,k));
      PM(a2,a6,CC(0,2,k),CC(0,6,k));
      PM(CH(0,k,0),CH(0,k,4),a0+a2,a1);
      PM(CH(0,k,2),CH(0,k,6),a0-a2,a3);
      ROTX90<fwd>(a6);
      PM(CH(0,k,1),CH(0,k,5),a4+a6,a5);
      PM(CH(0,k,3),CH(0,k,7),a4-a6,a7);
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      T a0, a1, a2, a3, a4, a5, a6, a7;
      PM(a1,a5,CC(0,1,k),CC(0,5,k));
      PM(a3,a7,CC(0,3,k),CC(0,7,k));
      PMINPLACE(a1,a3);
      ROTX90<fwd>(a3);

      ROTX90<fwd>(a7);
      PMINPLACE(a5,a7);
      ROTX45<fwd>(a5);
      ROTX135<fwd>(a7);

      PM(a0,a4,CC(0,0,k),CC(0,4,k));
      PM(a2,a6,CC(0,2,k),CC(0,6,k));
      PM(CH(0,k,0),CH(0,k,4),a0+a2,a1);
      PM(CH(0,k,2),CH(0,k,6),a0-a2,a3);
      ROTX90<fwd>(a6);
      PM(CH(0,k,1),CH(0,k,5),a4+a6,a5);
      PM(CH(0,k,3),CH(0,k,7),a4-a6,a7);
      }
      for (size_t i=1; i<ido; ++i)
        {
        T a0, a1, a2, a3, a4, a5, a6, a7;
        PM(a1,a5,CC(i,1,k),CC(i,5,k));
        PM(a3,a7,CC(i,3,k),CC(i,7,k));
        ROTX90<fwd>(a7);
        PMINPLACE(a1,a3);
        ROTX90<fwd>(a3);
        PMINPLACE(a5,a7);
        ROTX45<fwd>(a5);
        ROTX135<fwd>(a7);
        PM(a0,a4,CC(i,0,k),CC(i,4,k));
        PM(a2,a6,CC(i,2,k),CC(i,6,k));
        PMINPLACE(a0,a2);
        CH(i,k,0) = a0+a1;
        special_mul<fwd>(a0-a1,WA(3,i),CH(i,k,4));
        special_mul<fwd>(a2+a3,WA(1,i),CH(i,k,2));
        special_mul<fwd>(a2-a3,WA(5,i),CH(i,k,6));
        ROTX90<fwd>(a6);
        PMINPLACE(a4,a6);
        special_mul<fwd>(a4+a5,WA(0,i),CH(i,k,1));
        special_mul<fwd>(a4-a5,WA(4,i),CH(i,k,5));
        special_mul<fwd>(a6+a7,WA(2,i),CH(i,k,3));
        special_mul<fwd>(a6-a7,WA(6,i),CH(i,k,7));
        }
      }
   }


#define POCKETFFT_PREP11(idx) \
        T t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7, t8, t9, t10, t11; \
        PM (t2,t11,CC(idx,1,k),CC(idx,10,k)); \
        PM (t3,t10,CC(idx,2,k),CC(idx, 9,k)); \
        PM (t4,t9 ,CC(idx,3,k),CC(idx, 8,k)); \
        PM (t5,t8 ,CC(idx,4,k),CC(idx, 7,k)); \
        PM (t6,t7 ,CC(idx,5,k),CC(idx, 6,k)); \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r+t5.r+t6.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i+t5.i+t6.i;

#define POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,out1,out2) \
        { \
        T ca = t1 + t2*x1 + t3*x2 + t4*x3 + t5*x4 +t6*x5, \
          cb; \
        cb.i=y1*t11.r y2*t10.r y3*t9.r y4*t8.r y5*t7.r; \
        cb.r=-(y1*t11.i y2*t10.i y3*t9.i y4*t8.i y5*t7.i ); \
        PM(out1,out2,ca,cb); \
        }
#define POCKETFFT_PARTSTEP11a(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,CH(0,k,u1),CH(0,k,u2))
#define POCKETFFT_PARTSTEP11(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        { \
        T da,db; \
        POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,da,db) \
        special_mul<fwd>(da,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(db,WA(u2-1,i),CH(i,k,u2)); \
        }

template<bool fwd, typename T> void pass11 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa) const
  {
  constexpr T0 tw1r= T0(0.8412535328311811688618116489193677L),
               tw1i= (fwd ? -1 : 1) * T0(0.5406408174555975821076359543186917L),
               tw2r= T0(0.4154150130018864255292741492296232L),
               tw2i= (fwd ? -1 : 1) * T0(0.9096319953545183714117153830790285L),
               tw3r= T0(-0.1423148382732851404437926686163697L),
               tw3i= (fwd ? -1 : 1) * T0(0.9898214418809327323760920377767188L),
               tw4r= T0(-0.6548607339452850640569250724662936L),
               tw4i= (fwd ? -1 : 1) * T0(0.7557495743542582837740358439723444L),
               tw5r= T0(-0.9594929736144973898903680570663277L),
               tw5i= (fwd ? -1 : 1) * T0(0.2817325568414296977114179153466169L);

  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+11*c)]; };
  auto WA = [wa, ido](size_t x, size_t i)
    { return wa[i-1+x*(ido-1)]; };

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      POCKETFFT_PREP11(0)
      POCKETFFT_PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
      POCKETFFT_PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
      POCKETFFT_PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
      POCKETFFT_PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
      POCKETFFT_PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      POCKETFFT_PREP11(0)
      POCKETFFT_PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
      POCKETFFT_PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
      POCKETFFT_PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
      POCKETFFT_PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
      POCKETFFT_PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        POCKETFFT_PREP11(i)
        POCKETFFT_PARTSTEP11(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
        POCKETFFT_PARTSTEP11(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
        POCKETFFT_PARTSTEP11(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
        POCKETFFT_PARTSTEP11(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
        POCKETFFT_PARTSTEP11(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
        }
      }
  }

#undef PARTSTEP11
#undef PARTSTEP11a0
#undef PARTSTEP11a
#undef POCKETFFT_PREP11

template<bool fwd, typename T> void passg (size_t ido, size_t ip,
  size_t l1, T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const Cmplx<T0> * DUCC0_RESTRICT wa,
  const Cmplx<T0> * DUCC0_RESTRICT csarr) const
  {
  const size_t cdim=ip;
  size_t ipph = (ip+1)/2;
  size_t idl1 = ido*l1;

  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto CC = [cc,ido,cdim](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+cdim*c)]; };
  auto CX = [cc, ido, l1](size_t a, size_t b, size_t c) -> T&
    { return cc[a+ido*(b+l1*c)]; };
  auto CX2 = [cc, idl1](size_t a, size_t b) -> T&
    { return cc[a+idl1*b]; };
  auto CH2 = [ch, idl1](size_t a, size_t b) -> const T&
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
      T tmp = CH(i,k,0);
      for (size_t j=1; j<ipph; ++j)
        tmp+=CH(i,k,j);
      CX(i,k,0) = tmp;
      }
  for (size_t l=1, lc=ip-1; l<ipph; ++l, --lc)
    {
    // j=0
    for (size_t ik=0; ik<idl1; ++ik)
      {
      auto wal  = fwd ? csarr[  l].conj() : csarr[  l];
      auto wal2 = fwd ? csarr[2*l].conj() : csarr[2*l];
      CX2(ik,l ).r = CH2(ik,0).r+wal.r*CH2(ik,1).r+wal2.r*CH2(ik,2).r;
      CX2(ik,l ).i = CH2(ik,0).i+wal.r*CH2(ik,1).i+wal2.r*CH2(ik,2).i;
      CX2(ik,lc).r =-wal.i*CH2(ik,ip-1).i-wal2.i*CH2(ik,ip-2).i;
      CX2(ik,lc).i = wal.i*CH2(ik,ip-1).r+wal2.i*CH2(ik,ip-2).r;
      }

    size_t iwal=2*l;
    size_t j=3, jc=ip-3;
    for (; j<ipph-1; j+=2, jc-=2)
      {
      iwal+=l; if (iwal>ip) iwal-=ip;
      Cmplx<T0> xwal=fwd ? csarr[iwal].conj() : csarr[iwal];
      iwal+=l; if (iwal>ip) iwal-=ip;
      Cmplx<T0> xwal2=fwd ? csarr[iwal].conj() : csarr[iwal];
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
      Cmplx<T0> xwal=fwd ? csarr[iwal].conj() : csarr[iwal];
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
        T t1=CX2(ik,j), t2=CX2(ik,jc);
        PM(CX2(ik,j),CX2(ik,jc),t1,t2);
        }
  else
    {
    for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)
      for (size_t k=0; k<l1; ++k)
        {
        T t1=CX(0,k,j), t2=CX(0,k,jc);
        PM(CX(0,k,j),CX(0,k,jc),t1,t2);
        for (size_t i=1; i<ido; ++i)
          {
          T x1, x2;
          PM(x1,x2,CX(i,k,j),CX(i,k,jc));
          size_t idij=(j-1)*(ido-1)+i-1;
          special_mul<fwd>(x1,wa[idij],CX(i,k,j));
          idij=(jc-1)*(ido-1)+i-1;
          special_mul<fwd>(x2,wa[idij],CX(i,k,jc));
          }
        }
    }
  }

template<bool fwd, typename T> T *pass_all(T c[], T ch[], T0 fct) const
  {
  if (length==1) { c[0]*=fct; return c; }
  size_t l1=1;
  T *p1=c, *p2=ch;

  for(size_t k1=0; k1<fact.size(); k1++)
    {
    size_t ip=fact[k1].fct;
    size_t l2=ip*l1;
    size_t ido = length/l2;
    if     (ip==4)
      pass4<fwd> (ido, l1, p1, p2, fact[k1].tw);
    else if(ip==8)
      pass8<fwd>(ido, l1, p1, p2, fact[k1].tw);
    else if(ip==2)
      pass2<fwd>(ido, l1, p1, p2, fact[k1].tw);
    else if(ip==3)
      pass3<fwd> (ido, l1, p1, p2, fact[k1].tw);
    else if(ip==5)
      pass5<fwd> (ido, l1, p1, p2, fact[k1].tw);
    else if(ip==7)
      pass7<fwd> (ido, l1, p1, p2, fact[k1].tw);
    else if(ip==11)
      pass11<fwd> (ido, l1, p1, p2, fact[k1].tw);
    else
      {
      passg<fwd>(ido, ip, l1, p1, p2, fact[k1].tw, fact[k1].tws);
      std::swap(p1,p2);
      }
    std::swap(p1,p2);
    l1=l2;
    }
  if (fct!=1.)
    for (size_t i=0; i<length; ++i)
      p1[i] *= fct;
  return p1;
  }

template<bool fwd, typename T> void pass_all(T c[], T0 fct) const
  {
  if (length==1) { c[0]*=fct; return; }
  aligned_array<T> ch(length);
  auto p1 = pass_all<fwd>(c, ch.data(), T0(1));
  if (p1!=c)
    {
    if (fct!=1.)
      for (size_t i=0; i<length; ++i)
        c[i] = ch[i]*fct;
    else
      memcpy (c,p1,length*sizeof(T));
    }
  else
    if (fct!=1.)
      for (size_t i=0; i<length; ++i)
        c[i] *= fct;
  }

  public:
    template<typename T> void exec(T c[], T0 fct, bool fwd) const
      { fwd ? pass_all<true>(c, fct) : pass_all<false>(c, fct); }
    template<typename T> T *exec(T c[], T ch[], T0 fct, bool fwd) const
      { return fwd ? pass_all<true>(c, ch, fct) : pass_all<false>(c, ch, fct); }

  private:
    DUCC0_NOINLINE void factorize()
      {
      size_t len=length;
      while ((len&7)==0)
        { add_factor(8); len>>=3; }
      while ((len&3)==0)
        { add_factor(4); len>>=2; }
      if ((len&1)==0)
        {
        len>>=1;
        // factor 2 should be at the front of the factor list
        add_factor(2);
        std::swap(fact[0].fct, fact.back().fct);
        }
      for (size_t divisor=3; divisor*divisor<=len; divisor+=2)
        while ((len%divisor)==0)
          {
          add_factor(divisor);
          len/=divisor;
          }
      if (len>1) add_factor(len);
      }

    size_t twsize() const
      {
      size_t twsize=0, l1=1;
      for (size_t k=0; k<fact.size(); ++k)
        {
        size_t ip=fact[k].fct, ido= length/(l1*ip);
        twsize+=(ip-1)*(ido-1);
        if (ip>11)
          twsize+=ip;
        l1*=ip;
        }
      return twsize;
      }

    void comp_twiddle()
      {
      UnityRoots<T0,Cmplx<T0>> twiddle(length);
      size_t l1=1;
      size_t memofs=0;
      for (size_t k=0; k<fact.size(); ++k)
        {
        size_t ip=fact[k].fct, ido=length/(l1*ip);
        fact[k].tw=mem.data()+memofs;
        memofs+=(ip-1)*(ido-1);
        for (size_t j=1; j<ip; ++j)
          for (size_t i=1; i<ido; ++i)
            fact[k].tw[(j-1)*(ido-1)+i-1] = twiddle[j*l1*i];
        if (ip>11)
          {
          fact[k].tws=mem.data()+memofs;
          memofs+=ip;
          for (size_t j=0; j<ip; ++j)
            fact[k].tws[j] = twiddle[j*l1*ido];
          }
        l1*=ip;
        }
      }

  public:
    DUCC0_NOINLINE cfftp(size_t length_)
      : length(length_)
      {
      if (length==0) throw std::runtime_error("zero-length FFT requested");
      if (length==1) return;
      factorize();
      mem.resize(twsize());
      comp_twiddle();
      }

    size_t bufsize() const { return length; }
  };

//
// real-valued FFTPACK transforms
//

template<typename T0> class rfftp
  {
  private:
    struct fctdata
      {
      size_t fct;
      T0 *tw, *tws;
      };

    size_t length;
    aligned_array<T0> mem;
    std::vector<fctdata> fact;

    void add_factor(size_t factor)
      { fact.push_back({factor, nullptr, nullptr}); }

/* (a+ib) = conj(c+id) * (e+if) */
template<typename T1, typename T2, typename T3> inline void MULPM
  (T1 &a, T1 &b, T2 c, T2 d, T3 e, T3 f) const
  {  a=c*e+d*f; b=c*f-d*e; }

template<typename T> void radf2 (size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido,l1](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+l1*c)]; };
  auto CH = [ch,ido](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+2*c)]; };

  for (size_t k=0; k<l1; k++)
    PM (CH(0,0,k),CH(ido-1,1,k),CC(0,k,0),CC(0,k,1));
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(    0,1,k) = -CC(ido-1,k,1);
      CH(ido-1,0,k) =  CC(ido-1,k,0);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      T tr2, ti2;
      MULPM (tr2,ti2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1));
      PM (CH(i-1,0,k),CH(ic-1,1,k),CC(i-1,k,0),tr2);
      PM (CH(i  ,0,k),CH(ic  ,1,k),ti2,CC(i  ,k,0));
      }
  }

// a2=a+b; b2=i*(b-a);
#define POCKETFFT_REARRANGE(rx, ix, ry, iy) \
  {\
  auto t1=rx+ry, t2=ry-rx, t3=ix+iy, t4=ix-iy; \
  rx=t1; ix=t3; ry=t4; iy=t2; \
  }

template<typename T> void radf3(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  constexpr T0 taur=-0.5, taui=T0(0.8660254037844386467637231707529362L);

  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido,l1](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+l1*c)]; };
  auto CH = [ch,ido](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+3*c)]; };

  for (size_t k=0; k<l1; k++)
    {
    T cr2=CC(0,k,1)+CC(0,k,2);
    CH(0,0,k) = CC(0,k,0)+cr2;
    CH(0,2,k) = taui*(CC(0,k,2)-CC(0,k,1));
    CH(ido-1,1,k) = CC(0,k,0)+taur*cr2;
    }
  if (ido==1) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      T di2, di3, dr2, dr3;
      MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1)); // d2=conj(WA0)*CC1
      MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2)); // d3=conj(WA1)*CC2
      POCKETFFT_REARRANGE(dr2, di2, dr3, di3);
      CH(i-1,0,k) = CC(i-1,k,0)+dr2; // c add
      CH(i  ,0,k) = CC(i  ,k,0)+di2;
      T tr2 = CC(i-1,k,0)+taur*dr2; // c add
      T ti2 = CC(i  ,k,0)+taur*di2;
      T tr3 = taui*dr3;  // t3 = taui*i*(d3-d2)?
      T ti3 = taui*di3;
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr3); // PM(i) = t2+t3
      PM(CH(i  ,2,k),CH(ic  ,1,k),ti3,ti2); // PM(ic) = conj(t2-t3)
      }
  }

template<typename T> void radf4(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  constexpr T0 hsqt2=T0(0.707106781186547524400844362104849L);

  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido,l1](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+l1*c)]; };
  auto CH = [ch,ido](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+4*c)]; };

  for (size_t k=0; k<l1; k++)
    {
    T tr1,tr2;
    PM (tr1,CH(0,2,k),CC(0,k,3),CC(0,k,1));
    PM (tr2,CH(ido-1,1,k),CC(0,k,0),CC(0,k,2));
    PM (CH(0,0,k),CH(ido-1,3,k),tr2,tr1);
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      T ti1=-hsqt2*(CC(ido-1,k,1)+CC(ido-1,k,3));
      T tr1= hsqt2*(CC(ido-1,k,1)-CC(ido-1,k,3));
      PM (CH(ido-1,0,k),CH(ido-1,2,k),CC(ido-1,k,0),tr1);
      PM (CH(    0,3,k),CH(    0,1,k),ti1,CC(ido-1,k,2));
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      T ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      MULPM(cr2,ci2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1));
      MULPM(cr3,ci3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2));
      MULPM(cr4,ci4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3));
      PM(tr1,tr4,cr4,cr2);
      PM(ti1,ti4,ci2,ci4);
      PM(tr2,tr3,CC(i-1,k,0),cr3);
      PM(ti2,ti3,CC(i  ,k,0),ci3);
      PM(CH(i-1,0,k),CH(ic-1,3,k),tr2,tr1);
      PM(CH(i  ,0,k),CH(ic  ,3,k),ti1,ti2);
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr3,ti4);
      PM(CH(i  ,2,k),CH(ic  ,1,k),tr4,ti3);
      }
  }

template<typename T> void radf5(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  constexpr T0 tr11= T0(0.3090169943749474241022934171828191L),
               ti11= T0(0.9510565162951535721164393333793821L),
               tr12= T0(-0.8090169943749474241022934171828191L),
               ti12= T0(0.5877852522924731291687059546390728L);

  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido,l1](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+l1*c)]; };
  auto CH = [ch,ido](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+5*c)]; };

  for (size_t k=0; k<l1; k++)
    {
    T cr2, cr3, ci4, ci5;
    PM (cr2,ci5,CC(0,k,4),CC(0,k,1));
    PM (cr3,ci4,CC(0,k,3),CC(0,k,2));
    CH(0,0,k)=CC(0,k,0)+cr2+cr3;
    CH(ido-1,1,k)=CC(0,k,0)+tr11*cr2+tr12*cr3;
    CH(0,2,k)=ti11*ci5+ti12*ci4;
    CH(ido-1,3,k)=CC(0,k,0)+tr12*cr2+tr11*cr3;
    CH(0,4,k)=ti12*ci5-ti11*ci4;
    }
  if (ido==1) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
      {
      T di2, di3, di4, di5, dr2, dr3, dr4, dr5;
      MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1));
      MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2));
      MULPM (dr4,di4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3));
      MULPM (dr5,di5,WA(3,i-2),WA(3,i-1),CC(i-1,k,4),CC(i,k,4));
      POCKETFFT_REARRANGE(dr2, di2, dr5, di5);
      POCKETFFT_REARRANGE(dr3, di3, dr4, di4);
      CH(i-1,0,k)=CC(i-1,k,0)+dr2+dr3;
      CH(i  ,0,k)=CC(i  ,k,0)+di2+di3;
      T tr2=CC(i-1,k,0)+tr11*dr2+tr12*dr3;
      T ti2=CC(i  ,k,0)+tr11*di2+tr12*di3;
      T tr3=CC(i-1,k,0)+tr12*dr2+tr11*dr3;
      T ti3=CC(i  ,k,0)+tr12*di2+tr11*di3;
      T tr5 = ti11*dr5 + ti12*dr4;
      T ti5 = ti11*di5 + ti12*di4;
      T tr4 = ti12*dr5 - ti11*dr4;
      T ti4 = ti12*di5 - ti11*di4;
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr5);
      PM(CH(i  ,2,k),CH(ic  ,1,k),ti5,ti2);
      PM(CH(i-1,4,k),CH(ic-1,3,k),tr3,tr4);
      PM(CH(i  ,4,k),CH(ic  ,3,k),ti4,ti3);
      }
  }

#undef POCKETFFT_REARRANGE

template<typename T> void radfg(size_t ido, size_t ip, size_t l1,
  T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa, const T0 * DUCC0_RESTRICT csarr) const
  {
  const size_t cdim=ip;
  size_t ipph=(ip+1)/2;
  size_t idl1 = ido*l1;

  auto CC = [cc,ido,cdim](size_t a, size_t b, size_t c) -> T&
    { return cc[a+ido*(b+cdim*c)]; };
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> const T&
    { return ch[a+ido*(b+l1*c)]; };
  auto C1 = [cc,ido,l1] (size_t a, size_t b, size_t c) -> T&
    { return cc[a+ido*(b+l1*c)]; };
  auto C2 = [cc,idl1] (size_t a, size_t b) -> T&
    { return cc[a+idl1*b]; };
  auto CH2 = [ch,idl1] (size_t a, size_t b) -> T&
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
          T t1=C1(i,k,j ), t2=C1(i+1,k,j ),
            t3=C1(i,k,jc), t4=C1(i+1,k,jc);
          T x1=wa[idij]*t1 + wa[idij+1]*t2,
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
      T0 ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      T0 ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      T0 ar3=csarr[2*iang], ai3=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      T0 ar4=csarr[2*iang], ai4=csarr[2*iang+1];
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
      T0 ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      T0 ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1);
        CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1);
        }
      }
    for (; j<ipph; ++j,--jc)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      T0 ar=csarr[2*iang], ai=csarr[2*iang+1];
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

  if (ido==1) return;

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
  }

template<typename T> void radb2(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+2*c)]; };
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };

  for (size_t k=0; k<l1; k++)
    PM (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(ido-1,1,k));
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(ido-1,k,0) = T0( 2)*CC(ido-1,0,k);
      CH(ido-1,k,1) = T0(-2)*CC(0    ,1,k);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      T ti2, tr2;
      PM (CH(i-1,k,0),tr2,CC(i-1,0,k),CC(ic-1,1,k));
      PM (ti2,CH(i  ,k,0),CC(i  ,0,k),CC(ic  ,1,k));
      MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ti2,tr2);
      }
  }

template<typename T> void radb3(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  constexpr T0 taur=-0.5, taui=T0(0.8660254037844386467637231707529362L);

  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+3*c)]; };
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };

  for (size_t k=0; k<l1; k++)
    {
    T tr2=T0(2)*CC(ido-1,1,k);
    T cr2=CC(0,0,k)+taur*tr2;
    CH(0,k,0)=CC(0,0,k)+tr2;
    T ci3=T0(2)*taui*CC(0,2,k);
    PM (CH(0,k,2),CH(0,k,1),cr2,ci3);
    }
  if (ido==1) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
      {
      T tr2=CC(i-1,2,k)+CC(ic-1,1,k); // t2=CC(I) + conj(CC(ic))
      T ti2=CC(i  ,2,k)-CC(ic  ,1,k);
      T cr2=CC(i-1,0,k)+taur*tr2;     // c2=CC +taur*t2
      T ci2=CC(i  ,0,k)+taur*ti2;
      CH(i-1,k,0)=CC(i-1,0,k)+tr2;         // CH=CC+t2
      CH(i  ,k,0)=CC(i  ,0,k)+ti2;
      T cr3=taui*(CC(i-1,2,k)-CC(ic-1,1,k));// c3=taui*(CC(i)-conj(CC(ic)))
      T ci3=taui*(CC(i  ,2,k)+CC(ic  ,1,k));
      T di2, di3, dr2, dr3;
      PM(dr3,dr2,cr2,ci3); // d2= (cr2-ci3, ci2+cr3) = c2+i*c3
      PM(di2,di3,ci2,cr3); // d3= (cr2+ci3, ci2-cr3) = c2-i*c3
      MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2); // ch = WA*d2
      MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3);
      }
  }

template<typename T> void radb4(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  constexpr T0 sqrt2=T0(1.414213562373095048801688724209698L);

  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+4*c)]; };
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };

  for (size_t k=0; k<l1; k++)
    {
    T tr1, tr2;
    PM (tr2,tr1,CC(0,0,k),CC(ido-1,3,k));
    T tr3=T0(2)*CC(ido-1,1,k);
    T tr4=T0(2)*CC(0,2,k);
    PM (CH(0,k,0),CH(0,k,2),tr2,tr3);
    PM (CH(0,k,3),CH(0,k,1),tr1,tr4);
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      T tr1,tr2,ti1,ti2;
      PM (ti1,ti2,CC(0    ,3,k),CC(0    ,1,k));
      PM (tr2,tr1,CC(ido-1,0,k),CC(ido-1,2,k));
      CH(ido-1,k,0)=tr2+tr2;
      CH(ido-1,k,1)=sqrt2*(tr1-ti1);
      CH(ido-1,k,2)=ti2+ti2;
      CH(ido-1,k,3)=-sqrt2*(tr1+ti1);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      T ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      size_t ic=ido-i;
      PM (tr2,tr1,CC(i-1,0,k),CC(ic-1,3,k));
      PM (ti1,ti2,CC(i  ,0,k),CC(ic  ,3,k));
      PM (tr4,ti3,CC(i  ,2,k),CC(ic  ,1,k));
      PM (tr3,ti4,CC(i-1,2,k),CC(ic-1,1,k));
      PM (CH(i-1,k,0),cr3,tr2,tr3);
      PM (CH(i  ,k,0),ci3,ti2,ti3);
      PM (cr4,cr2,tr1,tr4);
      PM (ci2,ci4,ti1,ti4);
      MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ci2,cr2);
      MULPM (CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),ci3,cr3);
      MULPM (CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),ci4,cr4);
      }
  }

template<typename T> void radb5(size_t ido, size_t l1,
  const T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa) const
  {
  constexpr T0 tr11= T0(0.3090169943749474241022934171828191L),
               ti11= T0(0.9510565162951535721164393333793821L),
               tr12= T0(-0.8090169943749474241022934171828191L),
               ti12= T0(0.5877852522924731291687059546390728L);

  auto WA = [wa,ido](size_t x, size_t i) { return wa[i+x*(ido-1)]; };
  auto CC = [cc,ido](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+5*c)]; };
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };

  for (size_t k=0; k<l1; k++)
    {
    T ti5=CC(0,2,k)+CC(0,2,k);
    T ti4=CC(0,4,k)+CC(0,4,k);
    T tr2=CC(ido-1,1,k)+CC(ido-1,1,k);
    T tr3=CC(ido-1,3,k)+CC(ido-1,3,k);
    CH(0,k,0)=CC(0,0,k)+tr2+tr3;
    T cr2=CC(0,0,k)+tr11*tr2+tr12*tr3;
    T cr3=CC(0,0,k)+tr12*tr2+tr11*tr3;
    T ci4, ci5;
    MULPM(ci5,ci4,ti5,ti4,ti11,ti12);
    PM(CH(0,k,4),CH(0,k,1),cr2,ci5);
    PM(CH(0,k,3),CH(0,k,2),cr3,ci4);
    }
  if (ido==1) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2, ic=ido-2; i<ido; i+=2, ic-=2)
      {
      T tr2, tr3, tr4, tr5, ti2, ti3, ti4, ti5;
      PM(tr2,tr5,CC(i-1,2,k),CC(ic-1,1,k));
      PM(ti5,ti2,CC(i  ,2,k),CC(ic  ,1,k));
      PM(tr3,tr4,CC(i-1,4,k),CC(ic-1,3,k));
      PM(ti4,ti3,CC(i  ,4,k),CC(ic  ,3,k));
      CH(i-1,k,0)=CC(i-1,0,k)+tr2+tr3;
      CH(i  ,k,0)=CC(i  ,0,k)+ti2+ti3;
      T cr2=CC(i-1,0,k)+tr11*tr2+tr12*tr3;
      T ci2=CC(i  ,0,k)+tr11*ti2+tr12*ti3;
      T cr3=CC(i-1,0,k)+tr12*tr2+tr11*tr3;
      T ci3=CC(i  ,0,k)+tr12*ti2+tr11*ti3;
      T ci4, ci5, cr5, cr4;
      MULPM(cr5,cr4,tr5,tr4,ti11,ti12);
      MULPM(ci5,ci4,ti5,ti4,ti11,ti12);
      T dr2, dr3, dr4, dr5, di2, di3, di4, di5;
      PM(dr4,dr3,cr3,ci4);
      PM(di3,di4,ci3,cr4);
      PM(dr5,dr2,cr2,ci5);
      PM(di2,di5,ci2,cr5);
      MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2);
      MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3);
      MULPM(CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),di4,dr4);
      MULPM(CH(i,k,4),CH(i-1,k,4),WA(3,i-2),WA(3,i-1),di5,dr5);
      }
  }

template<typename T> void radbg(size_t ido, size_t ip, size_t l1,
  T * DUCC0_RESTRICT cc, T * DUCC0_RESTRICT ch,
  const T0 * DUCC0_RESTRICT wa, const T0 * DUCC0_RESTRICT csarr) const
  {
  const size_t cdim=ip;
  size_t ipph=(ip+1)/ 2;
  size_t idl1 = ido*l1;

  auto CC = [cc,ido,cdim](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+cdim*c)]; };
  auto CH = [ch,ido,l1](size_t a, size_t b, size_t c) -> T&
    { return ch[a+ido*(b+l1*c)]; };
  auto C1 = [cc,ido,l1](size_t a, size_t b, size_t c) -> const T&
    { return cc[a+ido*(b+l1*c)]; };
  auto C2 = [cc,idl1](size_t a, size_t b) -> T&
    { return cc[a+idl1*b]; };
  auto CH2 = [ch,idl1](size_t a, size_t b) -> T&
    { return ch[a+idl1*b]; };

  for (size_t k=0; k<l1; ++k)        // 102
    for (size_t i=0; i<ido; ++i)     // 101
      CH(i,k,0) = CC(i,0,k);
  for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)   // 108
    {
    size_t j2=2*j-1;
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,j ) = T0(2)*CC(ido-1,j2,k);
      CH(0,k,jc) = T0(2)*CC(0,j2+1,k);
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
      T0 ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      T0 ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      T0 ar3=csarr[2*iang], ai3=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      T0 ar4=csarr[2*iang], ai4=csarr[2*iang+1];
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
      T0 ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      T0 ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1);
        C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1);
        }
      }
    for(; j<ipph; ++j,--jc)
      {
      iang+=l; if(iang>ip) iang-=ip;
      T0 war=csarr[2*iang], wai=csarr[2*iang+1];
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

  if (ido==1) return;

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
        T t1=CH(i,k,j), t2=CH(i+1,k,j);
        CH(i  ,k,j) = wa[idij]*t1-wa[idij+1]*t2;
        CH(i+1,k,j) = wa[idij]*t2+wa[idij+1]*t1;
        idij+=2;
        }
      }
    }
  }

    template<typename T> void copy_and_norm(T *c, T *p1, size_t n, T0 fct) const
      {
      if (p1!=c)
        {
        if (fct!=1.)
          for (size_t i=0; i<n; ++i)
            c[i] = fct*p1[i];
        else
          memcpy (c,p1,n*sizeof(T));
        }
      else
        if (fct!=1.)
          for (size_t i=0; i<n; ++i)
            c[i] *= fct;
      }

  public:
    template<typename T> T *exec(T c[], T ch[], T0 fct, bool r2hc) const
      {
      if (length==1) { c[0]*=fct; return c; }
      size_t n=length, nf=fact.size();
      T *p1=c, *p2=ch;

      if (r2hc)
        for(size_t k1=0, l1=n; k1<nf;++k1)
          {
          size_t k=nf-k1-1;
          size_t ip=fact[k].fct;
          size_t ido=n / l1;
          l1 /= ip;
          if(ip==4)
            radf4(ido, l1, p1, p2, fact[k].tw);
          else if(ip==2)
            radf2(ido, l1, p1, p2, fact[k].tw);
          else if(ip==3)
            radf3(ido, l1, p1, p2, fact[k].tw);
          else if(ip==5)
            radf5(ido, l1, p1, p2, fact[k].tw);
          else
            { radfg(ido, ip, l1, p1, p2, fact[k].tw, fact[k].tws); std::swap (p1,p2); }
          std::swap (p1,p2);
          }
      else
        for(size_t k=0, l1=1; k<nf; k++)
          {
          size_t ip = fact[k].fct,
                 ido= n/(ip*l1);
          if(ip==4)
            radb4(ido, l1, p1, p2, fact[k].tw);
          else if(ip==2)
            radb2(ido, l1, p1, p2, fact[k].tw);
          else if(ip==3)
            radb3(ido, l1, p1, p2, fact[k].tw);
          else if(ip==5)
            radb5(ido, l1, p1, p2, fact[k].tw);
          else
            radbg(ido, ip, l1, p1, p2, fact[k].tw, fact[k].tws);
          std::swap (p1,p2);
          l1*=ip;
          }
      copy_and_norm(p1, p1, length, fct);
      return p1;
      }

    template<typename T> void exec(T c[], T0 fct, bool r2hc) const
      {
      if (length==1) { c[0]*=fct; return; }
      aligned_array<T> ch(length);
      auto p1 = exec(c, ch.data(), T0(1), r2hc);
      copy_and_norm(c,p1,length,fct);
      }

  private:
    void factorize()
      {
      size_t len=length;
      while ((len%4)==0)
        { add_factor(4); len>>=2; }
      if ((len%2)==0)
        {
        len>>=1;
        // factor 2 should be at the front of the factor list
        add_factor(2);
        std::swap(fact[0].fct, fact.back().fct);
        }
      for (size_t divisor=3; divisor*divisor<=len; divisor+=2)
        while ((len%divisor)==0)
          {
          add_factor(divisor);
          len/=divisor;
          }
      if (len>1) add_factor(len);
      }

    size_t twsize() const
      {
      size_t twsz=0, l1=1;
      for (size_t k=0; k<fact.size(); ++k)
        {
        size_t ip=fact[k].fct, ido=length/(l1*ip);
        twsz+=(ip-1)*(ido-1);
        if (ip>5) twsz+=2*ip;
        l1*=ip;
        }
      return twsz;
      }

    void comp_twiddle()
      {
      UnityRoots<T0,Cmplx<T0>> twid(length);
      size_t l1=1;
      T0 *ptr=mem.data();
      for (size_t k=0; k<fact.size(); ++k)
        {
        size_t ip=fact[k].fct, ido=length/(l1*ip);
        if (k<fact.size()-1) // last factor doesn't need twiddles
          {
          fact[k].tw=ptr; ptr+=(ip-1)*(ido-1);
          for (size_t j=1; j<ip; ++j)
            for (size_t i=1; i<=(ido-1)/2; ++i)
              {
              fact[k].tw[(j-1)*(ido-1)+2*i-2] = twid[j*l1*i].r;
              fact[k].tw[(j-1)*(ido-1)+2*i-1] = twid[j*l1*i].i;
              }
          }
        if (ip>5) // special factors required by *g functions
          {
          fact[k].tws=ptr; ptr+=2*ip;
          fact[k].tws[0] = 1.;
          fact[k].tws[1] = 0.;
          for (size_t i=2, ic=2*ip-2; i<=ic; i+=2, ic-=2)
            {
            fact[k].tws[i  ] = twid[i/2*(length/ip)].r;
            fact[k].tws[i+1] = twid[i/2*(length/ip)].i;
            fact[k].tws[ic]   = twid[i/2*(length/ip)].r;
            fact[k].tws[ic+1] = -twid[i/2*(length/ip)].i;
            }
          }
        l1*=ip;
        }
      }

  public:
    DUCC0_NOINLINE rfftp(size_t length_)
      : length(length_)
      {
      if (length==0) throw std::runtime_error("zero-length FFT requested");
      if (length==1) return;
      factorize();
      mem.resize(twsize());
      comp_twiddle();
      }

    size_t bufsize() const { return length; }
};

//
// complex Bluestein transforms
//

template<typename T0> class fftblue
  {
  private:
    size_t n, n2;
    cfftp<T0> plan;
    aligned_array<Cmplx<T0>> mem;
    Cmplx<T0> *bk, *bkf;

    template<bool fwd, typename T> void fft(Cmplx<T> c[], Cmplx<T> buf[], T0 fct) const
      {
      auto akf = &buf[0];
      auto akf2 = &buf[n2];

      /* initialize a_k and FFT it */
      for (size_t m=0; m<n; ++m)
        special_mul<fwd>(c[m],bk[m],akf[m]);
      auto zero = akf[0]*T0(0);
      for (size_t m=n; m<n2; ++m)
        akf[m]=zero;

      auto res = plan.exec (akf,akf2,T0(1),true);

      /* do the convolution */
      res[0] = res[0].template special_mul<!fwd>(bkf[0]);
      for (size_t m=1; m<(n2+1)/2; ++m)
        {
        res[m] = res[m].template special_mul<!fwd>(bkf[m]);
        res[n2-m] = res[n2-m].template special_mul<!fwd>(bkf[m]);
        }
      if ((n2&1)==0)
        res[n2/2] = res[n2/2].template special_mul<!fwd>(bkf[n2/2]);

      /* inverse FFT */
      res = plan.exec (res, (res==akf) ? akf2 : akf, T0(1), false);

      /* multiply by b_k */
      for (size_t m=0; m<n; ++m)
        c[m] = res[m].template special_mul<fwd>(bk[m])*fct;
      }

    template<bool fwd, typename T> void fft(Cmplx<T> c[], T0 fct) const
      {
      aligned_array<Cmplx<T>> buf(2*n2);
      fft<fwd>(c, buf.data(), fct);
      }

  public:
    DUCC0_NOINLINE fftblue(size_t length)
      : n(length), n2(util1d::good_size_cmplx(n*2-1)), plan(n2), mem(n+n2/2+1),
        bk(mem.data()), bkf(mem.data()+n)
      {
      /* initialize b_k */
      UnityRoots<T0,Cmplx<T0>> tmp(2*n);
      bk[0].Set(1, 0);

      size_t coeff=0;
      for (size_t m=1; m<n; ++m)
        {
        coeff+=2*m-1;
        if (coeff>=2*n) coeff-=2*n;
        bk[m] = tmp[coeff];
        }

      /* initialize the zero-padded, Fourier transformed b_k. Add normalisation. */
      aligned_array<Cmplx<T0>> tbkf(n2);
      T0 xn2 = T0(1)/T0(n2);
      tbkf[0] = bk[0]*xn2;
      for (size_t m=1; m<n; ++m)
        tbkf[m] = tbkf[n2-m] = bk[m]*xn2;
      for (size_t m=n;m<=(n2-n);++m)
        tbkf[m].Set(0.,0.);
      plan.exec(tbkf.data(),1.,true);
      for (size_t i=0; i<n2/2+1; ++i)
        bkf[i] = tbkf[i];
      }

    size_t bufsize_c() const { return 2*n2; }
    size_t bufsize_r() const { return 2*bufsize_c(); }

    template<typename T> void exec(Cmplx<T> c[], T0 fct, bool fwd) const
      { fwd ? fft<true>(c,fct) : fft<false>(c,fct); }
    template<typename T> Cmplx<T> *exec(Cmplx<T> c[], Cmplx<T> *buf, T0 fct, bool fwd) const
      {
      fwd ? fft<true>(c,buf,fct) : fft<false>(c,buf,fct);
      return c;
      }

    template<typename T> T *exec_r(T c[], T buf[], T0 fct, bool fwd)
      {
      auto tmp = reinterpret_cast<Cmplx<T> *>(&buf[2*n2]);
      auto buf2 = reinterpret_cast<Cmplx<T> *>(&buf[0]);
      if (fwd)
        {
        auto zero = T0(0)*c[0];
        for (size_t m=0; m<n; ++m)
          tmp[m].Set(c[m], zero);
        fft<true>(tmp,buf2,fct);
        c[0] = tmp[0].r;
        memcpy (reinterpret_cast<void *>(&c[1]),
                reinterpret_cast<void *>(&tmp[1]), (n-1)*sizeof(T));
        }
      else
        {
        tmp[0].Set(c[0],c[0]*0);
        memcpy (reinterpret_cast<void *>(&tmp[1]),
                reinterpret_cast<void *>(c+1), (n-1)*sizeof(T));
        if ((n&1)==0) tmp[n/2].i=T0(0)*c[0];
        for (size_t m=1; 2*m<n; ++m)
          tmp[n-m].Set(tmp[m].r, -tmp[m].i);
        fft<false>(tmp,buf2,fct);
        for (size_t m=0; m<n; ++m)
          c[m] = tmp[m].r;
        }
      return c;
      }
    template<typename T> void exec_r(T c[], T0 fct, bool fwd)
      {
      aligned_array<T> buf(4*n2);
      exec_r(c, buf.data(), fct, fwd);
      }
  };

//
// flexible (FFTPACK/Bluestein) complex 1D transform
//

template<typename T0> class pocketfft_c
  {
  private:
    std::unique_ptr<cfftp<T0>> packplan;
    std::unique_ptr<fftblue<T0>> blueplan;
    size_t len;

  public:
    DUCC0_NOINLINE pocketfft_c(size_t length)
      : len(length)
      {
      if (length==0) throw std::runtime_error("zero-length FFT requested");
      size_t tmp = (length<50) ? 0 : util1d::largest_prime_factor(length);
      if (tmp*tmp <= length)
        {
        packplan=std::unique_ptr<cfftp<T0>>(new cfftp<T0>(length));
        return;
        }
      double comp1 = util1d::cost_guess(length);
      double comp2 = 2*util1d::cost_guess(util1d::good_size_cmplx(2*length-1));
      comp2*=1.5; /* fudge factor that appears to give good overall performance */
      if (comp2<comp1) // use Bluestein
        blueplan=std::unique_ptr<fftblue<T0>>(new fftblue<T0>(length));
      else
        packplan=std::unique_ptr<cfftp<T0>>(new cfftp<T0>(length));
      }

    template<typename T> DUCC0_NOINLINE void exec(Cmplx<T> c[], T0 fct, bool fwd) const
      { packplan ? packplan->exec(c,fct,fwd) : blueplan->exec(c,fct,fwd); }
    template<typename T> DUCC0_NOINLINE Cmplx<T> *exec(Cmplx<T> c[], Cmplx<T> buf[], T0 fct, bool fwd) const
      { return packplan ? packplan->exec(c,buf,fct,fwd) : blueplan->exec(c,buf,fct,fwd); }

    size_t length() const { return len; }
    size_t bufsize() const
      { return packplan ? packplan->bufsize() : blueplan->bufsize_c(); }
  };

//
// flexible (FFTPACK/Bluestein) real-valued 1D transform
//

template<typename T0> class pocketfft_r
  {
  private:
    std::unique_ptr<rfftp<T0>> packplan;
    std::unique_ptr<fftblue<T0>> blueplan;
    size_t len;

  public:
    DUCC0_NOINLINE pocketfft_r(size_t length)
      : len(length)
      {
      if (length==0) throw std::runtime_error("zero-length FFT requested");
      size_t tmp = (length<50) ? 0 : util1d::largest_prime_factor(length);
      if (tmp*tmp <= length)
        {
        packplan=std::unique_ptr<rfftp<T0>>(new rfftp<T0>(length));
        return;
        }
      double comp1 = 0.5*util1d::cost_guess(length);
      double comp2 = 2*util1d::cost_guess(util1d::good_size_cmplx(2*length-1));
      comp2*=1.5; /* fudge factor that appears to give good overall performance */
      if (comp2<comp1) // use Bluestein
        blueplan=std::unique_ptr<fftblue<T0>>(new fftblue<T0>(length));
      else
        packplan=std::unique_ptr<rfftp<T0>>(new rfftp<T0>(length));
      }

    template<typename T> DUCC0_NOINLINE void exec(T c[], T0 fct, bool fwd) const
      { packplan ? packplan->exec(c,fct,fwd) : blueplan->exec_r(c,fct,fwd); }
    template<typename T> DUCC0_NOINLINE T *exec(T c[], T buf[], T0 fct, bool fwd) const
      { return packplan ? packplan->exec(c,buf,fct,fwd) : blueplan->exec_r(c,buf,fct,fwd); }

    size_t length() const { return len; }
    size_t bufsize() const
      { return packplan ? packplan->bufsize() : blueplan->bufsize_r(); }
  };

}

using detail_fft::pocketfft_c;
using detail_fft::pocketfft_r;
inline size_t good_size_complex(size_t n)
  { return detail_fft::util1d::good_size_cmplx(n); }
inline size_t good_size_real(size_t n)
  { return detail_fft::util1d::good_size_real(n); }

}

#endif
