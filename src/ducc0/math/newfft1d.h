#ifndef DUCC0_NEWFFT1D_H
#define DUCC0_NEWFFT1D_H

#include <memory>
#include <complex>
#include <cstring>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/math/cmplx.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/math/unity_roots.h"

#include <iostream>

namespace ducc0 {

namespace detail_newfft {

using namespace std;

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

vector<size_t> factorize(size_t N)
  {
  MR_assert(N>0, "need a positive number");
  vector<size_t> factors;
  while ((N&7)==0)
    { factors.push_back(8); N>>=3; }
  while ((N&3)==0)
    { factors.push_back(4); N>>=2; }
  if ((N&1)==0)
    {
    N>>=1;
    // factor 2 should be at the front of the factor list
    factors.push_back(2);
    swap(factors[0], factors.back());
    }
  for (size_t divisor=3; divisor*divisor<=N; divisor+=2)
  while ((N%divisor)==0)
    {
    factors.push_back(divisor);
    N/=divisor;
    }
  if (N>1) factors.push_back(N);
  return factors;
  }

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

template<typename T> using Troots = shared_ptr<const UnityRoots<T,Cmplx<T>>>;

template <typename Tfs> class cfftpass
  {
  public:
//     using Tfv2 = simd<Tfs, 2>;
//     using Tfv4 = simd<Tfs, 4>;
//     using Tfv8 = simd<Tfs, 8>;
//     using Tfv16= simd<Tfs,16>;
    using Tcs = Cmplx<Tfs>;
//     using Tcv2  = Cmplx<Tfv2>;
//     using Tcv4  = Cmplx<Tfv4>;
//     using Tcv8  = Cmplx<Tfv8>;
//     using Tcv16 = Cmplx<Tfv16>;
    virtual size_t bufsize() const = 0;
    virtual bool needs_copy() const = 0;
    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs *buf, bool fwd) = 0;
//     virtual Tcv2 *exec(Tcv2 *in, Tcv2 *copy, Tcv2 *buf, bool fwd) = 0;
//     virtual Tcv4 *exec(Tcv4 *in, Tcv4 *copy, Tcv4 *buf, bool fwd) = 0;
//     virtual Tcv8 *exec(Tcv8 *in, Tcv8 *copy, Tcv8 *buf, bool fwd) = 0;
//     virtual Tcv16 *exec(Tcv16 *in, Tcv16 *copy, Tcv16 *buf, bool fwd) = 0;
  };
template<typename T> using Tpass = shared_ptr<cfftpass<T>>;
template<typename Tfs> Tpass<Tfs> make_pass(size_t l1, size_t ido, size_t ip, const Troots<Tfs> &roots);
template<typename Tfs> Tpass<Tfs> make_pass(size_t ip)
  { return make_pass<Tfs> (1,1,ip,make_shared<UnityRoots<Tfs,Cmplx<Tfs>>>(ip)); }

template <typename Tfs> class cfftp1: public cfftpass<Tfs>
  {
  public:
    using Tcs = Cmplx<Tfs>;
    cfftp1() {}
    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return false; }
    virtual Tcs *exec(Tcs *in, Tcs * /*copy*/, Tcs * /*buf*/, bool /*fwd*/)
      { return in; }
  };

template <typename Tfs> class cfftp2: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=2;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Cmplx<T>&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Cmplx<T>&
        { return cc[a+ido*(b+ip*c)]; };

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
      return ch;
      }

  public:
    cfftp2(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };

template <typename Tfs> class cfftp3: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=3;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;
      constexpr Tfs tw1r=-0.5,
                    tw1i= (fwd ? -1: 1) * Tfs(0.8660254037844386467637231707529362L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP3(idx) \
        Tc t0 = CC(idx,0,k), t1, t2; \
        PM (t1,t2,CC(idx,1,k),CC(idx,2,k)); \
        CH(idx,k,0)=t0+t1;
#define POCKETFFT_PARTSTEP3a(u1,u2,twr,twi) \
        { \
        Tc ca=t0+t1*twr; \
        Tc cb{-t2.i*twi, t2.r*twi}; \
        PM(CH(0,k,u1),CH(0,k,u2),ca,cb) ;\
        }
#define POCKETFFT_PARTSTEP3b(u1,u2,twr,twi) \
        { \
        Tc ca=t0+t1*twr; \
        Tc cb{-t2.i*twi, t2.r*twi}; \
        special_mul<fwd>(ca+cb,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(ca-cb,WA(u2-1,i),CH(i,k,u2)); \
        }

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

#undef POCKETFFT_PARTSTEP3b
#undef POCKETFFT_PARTSTEP3a
#undef POCKETFFT_PREP3

      return ch;
      }

  public:
    cfftp3(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };

template <typename Tfs> class cfftp4: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=4;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;
      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

      if (ido==1)
        for (size_t k=0; k<l1; ++k)
          {
          Tc t1, t2, t3, t4;
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
          Tc t1, t2, t3, t4;
          PM(t2,t1,CC(0,0,k),CC(0,2,k));
          PM(t3,t4,CC(0,1,k),CC(0,3,k));
          ROTX90<fwd>(t4);
          PM(CH(0,k,0),CH(0,k,2),t2,t3);
          PM(CH(0,k,1),CH(0,k,3),t1,t4);
          }
          for (size_t i=1; i<ido; ++i)
            {
            Tc t1, t2, t3, t4;
            Tc cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
            PM(t2,t1,cc0,cc2);
            PM(t3,t4,cc1,cc3);
            ROTX90<fwd>(t4);
            CH(i,k,0) = t2+t3;
            special_mul<fwd>(t1+t4,WA(0,i),CH(i,k,1));
            special_mul<fwd>(t2-t3,WA(1,i),CH(i,k,2));
            special_mul<fwd>(t1-t4,WA(2,i),CH(i,k,3));
            }
          }
      return ch;
      }

  public:
    cfftp4(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };

template <typename Tfs> class cfftp5: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=5;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;
      constexpr Tfs tw1r= Tfs(0.3090169943749474241022934171828191L),
                    tw1i= (fwd ? -1: 1) * Tfs(0.9510565162951535721164393333793821L),
                    tw2r= Tfs(-0.8090169943749474241022934171828191L),
                    tw2i= (fwd ? -1: 1) * Tfs(0.5877852522924731291687059546390728L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP5(idx) \
        Tc t0 = CC(idx,0,k), t1, t2, t3, t4; \
        PM (t1,t4,CC(idx,1,k),CC(idx,4,k)); \
        PM (t2,t3,CC(idx,2,k),CC(idx,3,k)); \
        CH(idx,k,0).r=t0.r+t1.r+t2.r; \
        CH(idx,k,0).i=t0.i+t1.i+t2.i;

#define POCKETFFT_PARTSTEP5a(u1,u2,twar,twbr,twai,twbi) \
        { \
        Tc ca,cb; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PM(CH(0,k,u1),CH(0,k,u2),ca,cb); \
        }

#define POCKETFFT_PARTSTEP5b(u1,u2,twar,twbr,twai,twbi) \
        { \
        Tc ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        special_mul<fwd>(ca+cb,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(ca-cb,WA(u2-1,i),CH(i,k,u2)); \
        }

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

#undef POCKETFFT_PARTSTEP5b
#undef POCKETFFT_PARTSTEP5a
#undef POCKETFFT_PREP5

      return ch;
      }

  public:
    cfftp5(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };
template <typename Tfs> class cfftp7: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=7;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;
      constexpr Tfs tw1r= Tfs(0.6234898018587335305250048840042398L),
                    tw1i= (fwd ? -1 : 1) * Tfs(0.7818314824680298087084445266740578L),
                    tw2r= Tfs(-0.2225209339563144042889025644967948L),
                    tw2i= (fwd ? -1 : 1) * Tfs(0.9749279121818236070181316829939312L),
                    tw3r= Tfs(-0.9009688679024191262361023195074451L),
                    tw3i= (fwd ? -1 : 1) * Tfs(0.433883739117558120475768332848359L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP7(idx) \
        Tc t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7; \
        PM (t2,t7,CC(idx,1,k),CC(idx,6,k)); \
        PM (t3,t6,CC(idx,2,k),CC(idx,5,k)); \
        PM (t4,t5,CC(idx,3,k),CC(idx,4,k)); \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i;

#define POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,out1,out2) \
        { \
        Tc ca,cb; \
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
        Tc da,db; \
        POCKETFFT_PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,da,db) \
        special_mul<fwd>(da,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(db,WA(u2-1,i),CH(i,k,u2)); \
        }

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

#undef POCKETFFT_PARTSTEP7
#undef POCKETFFT_PARTSTEP7a0
#undef POCKETFFT_PARTSTEP7a
#undef POCKETFFT_PREP7

      return ch;
      }

  public:
    cfftp7(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };
template <typename Tfs> class cfftp8: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=8;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template <bool fwd, typename T> void ROTX45(T &a) const
      {
      constexpr Tfs hsqt2=Tfs(0.707106781186547524400844362104849L);
      if constexpr (fwd)
        { auto tmp_=a.r; a.r=hsqt2*(a.r+a.i); a.i=hsqt2*(a.i-tmp_); }
      else
        { auto tmp_=a.r; a.r=hsqt2*(a.r-a.i); a.i=hsqt2*(a.i+tmp_); }
      }
    template <bool fwd, typename T> void ROTX135(T &a) const
      {
      constexpr Tfs hsqt2=Tfs(0.707106781186547524400844362104849L);
      if constexpr (fwd)
        { auto tmp_=a.r; a.r=hsqt2*(a.i-a.r); a.i=hsqt2*(-tmp_-a.i); }
      else
        { auto tmp_=a.r; a.r=hsqt2*(-a.r-a.i); a.i=hsqt2*(tmp_-a.i); }
      }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

      if (ido==1)
        for (size_t k=0; k<l1; ++k)
          {
          Tc a0, a1, a2, a3, a4, a5, a6, a7;
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
          Tc a0, a1, a2, a3, a4, a5, a6, a7;
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
            Tc a0, a1, a2, a3, a4, a5, a6, a7;
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
      return ch;
      }

  public:
    cfftp8(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };

template <typename Tfs> class cfftp11: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    static constexpr size_t ip=11;
    aligned_array<Tcs> wa;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;
      constexpr Tfs tw1r= Tfs(0.8412535328311811688618116489193677L),
                    tw1i= (fwd ? -1 : 1) * Tfs(0.5406408174555975821076359543186917L),
                    tw2r= Tfs(0.4154150130018864255292741492296232L),
                    tw2i= (fwd ? -1 : 1) * Tfs(0.9096319953545183714117153830790285L),
                    tw3r= Tfs(-0.1423148382732851404437926686163697L),
                    tw3i= (fwd ? -1 : 1) * Tfs(0.9898214418809327323760920377767188L),
                    tw4r= Tfs(-0.6548607339452850640569250724662936L),
                    tw4i= (fwd ? -1 : 1) * Tfs(0.7557495743542582837740358439723444L),
                    tw5r= Tfs(-0.9594929736144973898903680570663277L),
                    tw5i= (fwd ? -1 : 1) * Tfs(0.2817325568414296977114179153466169L);

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

#define POCKETFFT_PREP11(idx) \
        Tc t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7, t8, t9, t10, t11; \
        PM (t2,t11,CC(idx,1,k),CC(idx,10,k)); \
        PM (t3,t10,CC(idx,2,k),CC(idx, 9,k)); \
        PM (t4,t9 ,CC(idx,3,k),CC(idx, 8,k)); \
        PM (t5,t8 ,CC(idx,4,k),CC(idx, 7,k)); \
        PM (t6,t7 ,CC(idx,5,k),CC(idx, 6,k)); \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r+t5.r+t6.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i+t5.i+t6.i;

#define POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,out1,out2) \
        { \
        Tc ca = t1 + t2*x1 + t3*x2 + t4*x3 + t5*x4 +t6*x5, \
           cb; \
        cb.i=y1*t11.r y2*t10.r y3*t9.r y4*t8.r y5*t7.r; \
        cb.r=-(y1*t11.i y2*t10.i y3*t9.i y4*t8.i y5*t7.i ); \
        PM(out1,out2,ca,cb); \
        }
#define POCKETFFT_PARTSTEP11a(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,CH(0,k,u1),CH(0,k,u2))
#define POCKETFFT_PARTSTEP11(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        { \
        Tc da,db; \
        POCKETFFT_PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,da,db) \
        special_mul<fwd>(da,WA(u1-1,i),CH(i,k,u1)); \
        special_mul<fwd>(db,WA(u2-1,i),CH(i,k,u2)); \
        }

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

#undef POCKETFFT_PARTSTEP11
#undef POCKETFFT_PARTSTEP11a0
#undef POCKETFFT_PARTSTEP11a
#undef POCKETFFT_PREP11
      return ch;
      }

  public:
    cfftp11(size_t l1_, size_t ido_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), wa((ip-1)*(ido-1))
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };

template <typename Tfs> class cfftpg: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t l1, ido;
    size_t ip;
    aligned_array<Tcs> wa;
    aligned_array<Tcs> csarr;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch)
      {
      using Tc = Cmplx<T>;
      size_t ipph = (ip+1)/2;
      size_t idl1 = ido*l1;

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };
      auto CX = [cc,this](size_t a, size_t b, size_t c) -> Tc&
        { return cc[a+ido*(b+l1*c)]; };
      auto CX2 = [cc, idl1](size_t a, size_t b) -> Tc&
        { return cc[a+idl1*b]; };
      auto CH2 = [ch, idl1](size_t a, size_t b) -> const Tc&
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
          Tc tmp = CH(i,k,0);
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
            Tc t1=CX2(ik,j), t2=CX2(ik,jc);
            PM(CX2(ik,j),CX2(ik,jc),t1,t2);
            }
      else
        {
        for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)
          for (size_t k=0; k<l1; ++k)
            {
            Tc t1=CX(0,k,j), t2=CX(0,k,jc);
            PM(CX(0,k,j),CX(0,k,jc),t1,t2);
            for (size_t i=1; i<ido; ++i)
              {
              Tc x1, x2;
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
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];
      for (size_t i=0; i<ip; ++i)
        csarr[i] = (*roots)[rfct*ido*l1*i];
      }

    virtual size_t bufsize() const { return 0; }
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs * /*buf*/, bool fwd)
      { return fwd ? exec_<true>(in, copy) : exec_<false>(in, copy); }
  };

template <typename Tfs> class bluepass: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    const size_t l1, ido, ip;
    const size_t ip2;
    const Tpass<Tfs> subplan;
    aligned_array<Tcs> wa, bk, bkf;
    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_
      (Cmplx<T> * DUCC0_RESTRICT cc,
       Cmplx<T> * DUCC0_RESTRICT ch,
       Cmplx<T> * DUCC0_RESTRICT buf)
      {
      using Tc = Cmplx<T>;

      auto akf = &buf[0];
      auto akf2 = &buf[ip2];

      auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
        { return ch[a+ido*(b+l1*c)]; };
      auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
        { return cc[a+ido*(b+ip*c)]; };

      for (size_t k=0; k<l1; ++k)
        for (size_t i=0; i<ido; ++i)
          {
          /* initialize a_k and FFT it */
          for (size_t m=0; m<ip; ++m)
            special_mul<fwd>(CC(i,m,k),bk[m],akf[m]);
          auto zero = akf[0]*Tfs(0);
          for (size_t m=ip; m<ip2; ++m)
            akf[m]=zero;

          auto res = subplan->exec (akf,akf2,&buf[2*ip2],true);

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
          res = subplan->exec (res, (res==akf) ? akf2 : akf, &buf[2*ip2], false);

          /* multiply by b_k */
          if (i==0)
            for (size_t m=0; m<ip; ++m)
              CH(0,k,m) = res[m].template special_mul<fwd>(bk[m]);
          else
            {
            CH(i,k,0) = res[0].template special_mul<fwd>(bk[0]);
            for (size_t m=1; m<ip; ++m)
              CH(i,k,m) = res[m].template special_mul<fwd>(bk[m] *WA(m-1,i));
            }
          }

      return ch;
      }

  public:
    bluepass(size_t l1_, size_t ido_, size_t ip_, const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), ip(ip_), ip2(util1d::good_size_cmplx(ip*2-1)),
        subplan(make_pass<Tfs>(ip2)), wa((ip-1)*(ido-1)), bk(ip), bkf(ip2/2+1)
      {
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];

      /* initialize b_k */
      bk[0].Set(1, 0);
      size_t coeff=0;
      // FIXME: reuse "roots" if possible
      auto roots2 = make_shared<const UnityRoots<Tfs,Tcs>>(2*ip);
      for (size_t m=1; m<ip; ++m)
        {
        coeff+=2*m-1;
        if (coeff>=2*ip) coeff-=2*ip;
        bk[m] = (*roots2)[coeff];
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
      auto res = subplan->exec(tbkf.data(), tbkf2.data(), buf.data(), true);
      for (size_t i=0; i<ip2/2+1; ++i)
        bkf[i] = res[i];
      }

    virtual size_t bufsize() const { return 2*ip2+subplan->bufsize(); } // FIXME: might be able to save some more
    virtual bool needs_copy() const { return true; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs *buf, bool fwd)
      { return fwd ? exec_<true>(in, copy, buf) : exec_<false>(in, copy, buf); }
  };

template <typename Tfs> class cfft_multipass: public cfftpass<Tfs>
  {
  private:
    using Tcs = Cmplx<Tfs>;
    const size_t l1, ido;
    size_t ip;
    vector<Tpass<Tfs>> passes;
    size_t bufsz;
    bool need_cpy;
    aligned_array<Tcs> wa;

    auto WA(size_t x, size_t i) const
      { return wa[i-1+x*(ido-1)]; }

    template<bool fwd, typename T> Cmplx<T> *exec_(Cmplx<T> *cc, Cmplx<T> *ch, Cmplx<T> *buf)
      {
      using Tc = Cmplx<T>;
      if ((l1==1) && (ido==1))
        {
        Cmplx<T> *p1=cc, *p2=ch;
        for(const auto &pass: passes)
          {
          auto res = pass->exec(p1, p2, buf, fwd);
          if (res==p2) swap (p1,p2);
          }
        return p1;
        }
      else
        {
        auto cc2 = &buf[0];
        auto ch2 = &buf[ip];
        auto buf2 = &buf[2*ip];

        auto CH = [ch,this](size_t a, size_t b, size_t c) -> Tc&
          { return ch[a+ido*(b+l1*c)]; };
        auto CC = [cc,this](size_t a, size_t b, size_t c) -> const Tc&
          { return cc[a+ido*(b+ip*c)]; };

        for (size_t k=0; k<l1; ++k)
          for (size_t i=0; i<ido; ++i)
            {
            for (size_t m=0; m<ip; ++m)
              cc2[m] = CC(i,m,k);

            Cmplx<T> *p1=cc2, *p2=ch2;
            for(const auto &pass: passes)
              {
              auto res = pass->exec(p1, p2, buf2, fwd);
              if (res==p2) swap (p1,p2);
              }

            if (i==0)
              for (size_t m=0; m<ip; ++m)
                CH(0,k,m) = p1[m];
            else
              {
              CH(i,k,0) = p1[0];
              for (size_t m=1; m<ip; ++m)
                CH(i,k,m) = p1[m].template special_mul<fwd>(WA(m-1,i));
              }
            }
        return ch;
        }
      }

  public:
    cfft_multipass(size_t l1_, size_t ido_, size_t ip_,
      const Troots<Tfs> &roots)
      : l1(l1_), ido(ido_), ip(ip_), bufsz(0), need_cpy(false)
      {
 //     MR_assert((roots->size()/ip)*ip==roots->size(), "mismatch");
      wa.resize((ip-1)*(ido-1));
      size_t N=ip*l1*ido;
      size_t rfct = roots->size()/N;
      MR_assert(roots->size()==N*rfct, "mismatch");
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<ido; ++i)
          wa[(j-1)*(ido-1)+i-1] = (*roots)[rfct*j*l1*i];

      auto factors = factorize(ip);
MR_assert(factors.size()>1, "uuups");
constexpr size_t lim=1024;
      if (ip<=lim)
        {
        size_t l1l=1;
        for (auto fct: factors)
          {
          passes.push_back(make_pass<Tfs>(l1l, ip/(fct*l1l), fct, roots));
          l1l*=fct;
          }
        }
      else
        {
        vector<size_t> packets;
        size_t acc=1;
        for (auto fct: factors)
          {
          if (fct>lim) // large prime factor, needs isolated Bluestein pass
            packets.push_back(fct);
          else if (acc*fct>lim)
            {
            packets.push_back(acc);
            acc=fct;
            }
          else
            acc*=fct;
          }
        if (acc>1) packets.push_back(acc);
//cout << "subdividing into: ";
//for (auto pkt: packets)
//  cout << pkt << " ";
//cout << endl;
        size_t l1l=1;
        for (auto pkt: packets)
          {
          passes.push_back(make_pass<Tfs>(l1l, ip/(pkt*l1l), pkt, roots));
          l1l*=pkt;
          }
        }
      for (const auto &pass: passes)
        {
        bufsz = max(bufsz, pass->bufsize());
        need_cpy |= pass->needs_copy();
        }
      if ((l1!=1)||(ido!=1))
        {
        bufsz += 2*ip;
        need_cpy = true;
        }
      }

    virtual size_t bufsize() const { return bufsz; }
    virtual bool needs_copy() const { return need_cpy; }

    virtual Tcs *exec(Tcs *in, Tcs *copy, Tcs *buf, bool fwd)
      { return fwd ? exec_<true>(in, copy, buf) : exec_<false>(in, copy, buf); }
  };

template<typename Tfs> Tpass<Tfs> make_pass(size_t l1, size_t ido, size_t ip, const Troots<Tfs> &roots)
  {
  MR_assert(ip>=1, "no zero-sized FFTs");
  if (ip==1) return make_shared<cfftp1<Tfs>>();
  auto factors=factorize(ip);
  if (factors.size()==1)
    {
    switch(ip)
      {
      case 2:
        return make_shared<cfftp2<Tfs>>(l1, ido, roots);
      case 3:
        return make_shared<cfftp3<Tfs>>(l1, ido, roots);
      case 4:
        return make_shared<cfftp4<Tfs>>(l1, ido, roots);
      case 5:
        return make_shared<cfftp5<Tfs>>(l1, ido, roots);
      case 7:
        return make_shared<cfftp7<Tfs>>(l1, ido, roots);
      case 8:
        return make_shared<cfftp8<Tfs>>(l1, ido, roots);
      case 11:
        return make_shared<cfftp11<Tfs>>(l1, ido, roots);
      default:
        if (ip<90)
          return make_shared<cfftpg<Tfs>>(l1, ido, ip, roots);
        else
          return make_shared<bluepass<Tfs>>(l1, ido, ip, roots);
      }
    }
  else // more than one factor, need a multipass
    return make_shared<cfft_multipass<Tfs>>(l1, ido, ip, roots);
  }

template<typename Tfs> class newcfft1d
  {
  private:
    using Tcs = Cmplx<Tfs>;
    size_t N;
    Tpass<Tfs> plan;

  public:
    newcfft1d(size_t n) : N(n), plan(make_pass<Tfs>(n)) {}

    void exec(complex<Tfs> *data, Tfs fct, bool fwd) const
      {
      auto in = reinterpret_cast<Tcs *>(data);
      aligned_array<Tcs> buf(N*plan->needs_copy()+plan->bufsize());
      auto res = plan->exec(in, buf.data(), buf.data()+N*plan->needs_copy(), fwd);
      if (res==in)
        {
        if (fct!=Tfs(1))
          for (size_t i=0; i<N; ++i) in[i]*=fct;
        }
      else
        {
        if (fct!=Tfs(1))
          for (size_t i=0; i<N; ++i) in[i]=res[i]*fct;
        else
          memcpy(in, res, N*sizeof(Tcs));
        }
      }
  };

}

using detail_newfft::newcfft1d;

}

#endif
