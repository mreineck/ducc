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

/*! \file sht_inner_loop.h
 *  Critical inner loop code for pherical harmonic transforms
 *
 *  Copyright (C) 2020-2025 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_SHT_INNER_LOOP_H
#define DUCC0_SHT_INNER_LOOP_H

namespace ducc0 {

namespace detail_sht {

namespace detail_sht_inner_loop {

using namespace std;

static constexpr double sht_fbig=0x1p+800,sht_fsmall=0x1p-800;
static constexpr double sht_fbighalf=0x1p+400;

struct ringdata
  {
  size_t mlim, idx, midx;
  double cth, sth;
  double wgt=1.;  // currently only used for spin 1/2 SHT accelerated via spin 0
  };

class YlmBase
  {
  public:
    size_t lmax, mmax, s;
    vector<double> powlimit;
    /* used if s==0 */
    vector<double> mfac;

  protected:
    /* used if s!=0 */
    vector<double> flm1, flm2, inv;

  public:
    vector<double> prefac;
    vector<int> fscale;

  protected:
    inline void normalize (double &val, int &scale, double xfmax)
      {
      while (abs(val)>xfmax) { val*=sht_fsmall; ++scale; }
      if (val!=0.)
        while (abs(val)<xfmax*sht_fsmall) { val*=sht_fbig; --scale; }
      }

  public:
    static vector<double> get_norm(size_t lmax, size_t spin)
      {
      /* sign convention for H=1 (LensPix paper) */
#if 1
       double spinsign = (spin>0) ? -1.0 : 1.0;
#else
       double spinsign = 1.0;
#endif

      if (spin==0)
        return vector<double>(lmax+1,1.);

      vector<double> res(lmax+1);
      spinsign = (spin&1) ? -spinsign : spinsign;
      for (size_t l=0; l<=lmax; ++l)
        res[l] = (l<spin) ? 0. : spinsign*0.5*sqrt((2*l+1)/(4*pi));
      return res;
      }

    /*! Returns a vector with \a lmax+1 entries containing
      normalisation factors that must be applied to Y_lm values computed for
      first derivatives. */
    static vector<double> get_d1norm(size_t lmax)
      {
      vector<double> res(lmax+1);
      for (size_t l=0; l<=lmax; ++l)
        res[l] = (l<1) ? 0. : 0.5*sqrt(l*(l+1.)*(2*l+1.)/(4*pi));
      return res;
      }

    YlmBase(size_t l_max, size_t m_max, size_t spin)
      : lmax(l_max), mmax(m_max), s(spin),
        powlimit(mmax+s+1),
        mfac((s==0) ? (mmax+1) : 0),
        flm1((s==0) ? 0 : (2*lmax+3)),
        flm2((s==0) ? 0 : (2*lmax+3)),
        inv((s==0) ? 0 : (lmax+2)),
        prefac((s==0) ? 0 : (mmax+1)),
        fscale((s==0) ? 0 : (mmax+1))
      {
      MR_assert(l_max>=spin,"incorrect l_max: must be >= spin");
      MR_assert(l_max>=m_max,"incorrect l_max: must be >= m_max");
      powlimit[0]=0.;
      constexpr double expo=-400*ln2;
      for (size_t i=1; i<=m_max+spin; ++i)
        powlimit[i]=exp(expo/i);

      if (s==0)
        {
        mfac[0] = inv_sqrt4pi;
        for (size_t i=1; i<=mmax; ++i)
          mfac[i] = mfac[i-1]*sqrt((2*i+1.)/(2*i));
        }
      else
        {
        inv[0]=0;
        for (size_t i=1; i<lmax+2; ++i) inv[i]=1./i;
        for (size_t i=0; i<2*lmax+3; ++i)
          {
          flm1[i] = sqrt(1./(i+1.));
          flm2[i] = sqrt(i/(i+1.));
          }
        vector<double> fac(2*lmax+1);
        vector<int> facscale(2*lmax+1);
        fac[0]=1; facscale[0]=0;
        for (size_t i=1; i<2*lmax+1; ++i)
          {
          fac[i]=fac[i-1]*sqrt(i);
          facscale[i]=facscale[i-1];
          normalize(fac[i],facscale[i],sht_fbighalf);
          }
        for (size_t i=0; i<=mmax; ++i)
          {
          size_t mlo_=min(s,i), mhi_=max(s,i);
          double tfac=fac[2*mhi_]/fac[mhi_+mlo_];
          int tscale=facscale[2*mhi_]-facscale[mhi_+mlo_];
          normalize(tfac,tscale,sht_fbighalf);
          tfac/=fac[mhi_-mlo_];
          tscale-=facscale[mhi_-mlo_];
          normalize(tfac,tscale,sht_fbighalf);
          prefac[i]=tfac;
          fscale[i]=tscale;
          }
        }
      }
  };

class Ylmgen: public YlmBase
  {
  public:
    struct dbl2 { double a, b; };

    size_t m;

    vector<double> alpha;
    vector<dbl2> coef;

    /* used if s==0 */
    vector<double> eps;

    /* used if s!=0 */
    size_t sinPow, cosPow;
    bool preMinus_p, preMinus_m;

    size_t mlo, mhi;

    Ylmgen(const YlmBase &base)
      : YlmBase(base),
        m(~size_t(0)),
        alpha((s==0) ? (lmax/2+2) : (lmax+3), 0.),
        coef((s==0) ? (lmax/2+2) : (lmax+3), {0.,0.}),
        eps((s==0) ? (lmax+4) : 0),
        mlo(~size_t(0)),
        mhi(~size_t(0))
      {}

    void prepare (size_t m_)
      {
      if (m_==m) return;
      m = m_;

      if (s==0)
        {
        eps[m] = 0.;
        for (size_t l=m+1; l<lmax+4; ++l)
          eps[l] = sqrt((double(l+m)*(l-m))/(double(2*l+1)*(2*l-1)));
        alpha[0] = 1./eps[m+1];
        alpha[1] = eps[m+1]/(eps[m+2]*eps[m+3]);
        for (size_t il=1, l=m+2; l<lmax+1; ++il, l+=2)
          alpha[il+1]= ((il&1) ? -1 : 1) / (eps[l+2]*eps[l+3]*alpha[il]);
        for (size_t il=0, l=m; l<lmax+2; ++il, l+=2)
          {
          coef[il].a = ((il&1) ? -1 : 1)*alpha[il]*alpha[il];
          double t1 = eps[l+2], t2 = eps[l+1];
          coef[il].b = -coef[il].a*(t1*t1+t2*t2);
          }
        }
      else
        {
        size_t mlo_=m, mhi_=s;
        if (mhi_<mlo_) swap(mhi_,mlo_);
        bool ms_similar = ((mhi==mhi_) && (mlo==mlo_));

        mlo = mlo_; mhi = mhi_;

        if (!ms_similar)
          {
          alpha[mhi] = 1.;
          coef[mhi].a = coef[mhi].b = 0.;
          for (size_t l=mhi; l<=lmax; ++l)
            {
            double t = flm1[l+m]*flm1[l-m]*flm1[l+s]*flm1[l-s];
            double lt = 2*l+1;
            double l1 = l+1;
            double flp10=l1*lt*t;
            double flp11=m*s*inv[l]*inv[l+1];
            t = flm2[l+m]*flm2[l-m]*flm2[l+s]*flm2[l-s];
            double flp12=t*l1*inv[l];
            if (l>mhi)
              alpha[l+1] = alpha[l-1]*flp12;
            else
              alpha[l+1] = 1.;
            coef[l+1].a = flp10*alpha[l]/alpha[l+1];
            coef[l+1].b = flp11*coef[l+1].a;
            }
          }

        preMinus_p = preMinus_m = false;
        if (mhi==m)
          {
          cosPow = mhi+s; sinPow = mhi-s;
          preMinus_p = preMinus_m = ((mhi-s)&1);
          }
        else
          {
          cosPow = mhi+m; sinPow = mhi-m;
          preMinus_m = ((mhi+m)&1);
          }
        }
      }
  };

using Tv=native_simd<double>;
static constexpr size_t VLEN=Tv::size();

#if ((!defined(DUCC0_NO_SIMD)) && defined(__AVX__) && (!defined(__AVX512F__)))
static_assert(Tv::size()==4, "must not happen");
static inline void vhsum_cmplx_special (Tv a, Tv b, Tv c, Tv d,
  complex<double> * DUCC0_RESTRICT cc)
  {
  auto tmp1=_mm256_hadd_pd(__m256d(a),__m256d(b)),
       tmp2=_mm256_hadd_pd(__m256d(c),__m256d(d));
  auto tmp3=_mm256_permute2f128_pd(tmp1,tmp2,49),
       tmp4=_mm256_permute2f128_pd(tmp1,tmp2,32);
  tmp1=tmp3+tmp4;
  cc[0]+=complex<double>(tmp1[0], tmp1[1]);
  cc[1]+=complex<double>(tmp1[2], tmp1[3]);
  }
#else
static inline void vhsum_cmplx_special (Tv a, Tv b, Tv c, Tv d,
  complex<double> * DUCC0_RESTRICT cc)
  {
  cc[0] += complex<double>(reduce(a,std::plus<>()),reduce(b,std::plus<>()));
  cc[1] += complex<double>(reduce(c,std::plus<>()),reduce(d,std::plus<>()));
  }
#endif

using dcmplx = complex<double>;

static constexpr double sht_ftol=0x1p-60;

constexpr size_t nv0 = 256/VLEN;
constexpr size_t nvx = 128/VLEN;
constexpr size_t lstep = 4096;  // MUST be divisible by 8!

using Tbv0 = std::array<Tv,nv0>;
using Tbs0 = std::array<double,nv0*VLEN>;

struct s0data_v
  { Tbv0 sth, corfac, scale, lam1, lam2, csq, p1r, p1i, p2r, p2i; };

struct s0data_s
  { Tbs0 sth, corfac, scale, lam1, lam2, csq, p1r, p1i, p2r, p2i; };

union s0data_u
  {
  s0data_v v;
  s0data_s s;
  };

using Tbvx = std::array<Tv,nvx>;
using Tbsx = std::array<double,nvx*VLEN>;

struct sxdata_v
  {
  Tbvx sth, cfp, cfm, scp, scm, l1p, l2p, l1m, l2m, cth,
       p1pr, p1pi, p2pr, p2pi, p1mr, p1mi, p2mr, p2mi;
  };

struct sxdata_s
  {
  Tbsx sth, cfp, cfm, scp, scm, l1p, l2p, l1m, l2m, cth,
       p1pr, p1pi, p2pr, p2pi, p1mr, p1mi, p2mr, p2mi;
  };

union sxdata_u
  {
  sxdata_v v;
  sxdata_s s;
  };

static inline void Tvnormalize (Tv & DUCC0_RESTRICT val_,
  Tv & DUCC0_RESTRICT scale_, double maxval)
  {
  // This copying is necessary for MSVC ... no idea why
  Tv val = val_;
  Tv scale = scale_;
  const Tv vfmin=sht_fsmall*maxval, vfmax=maxval;
  const Tv vfsmall=sht_fsmall, vfbig=sht_fbig;
  auto mask = abs(val)>vfmax;
  while (any_of(mask))
    {
    where(mask,val)*=vfsmall;
    where(mask,scale)+=1;
    mask = abs(val)>vfmax;
    }
  mask = (abs(val)<vfmin) & (val!=0);
  while (any_of(mask))
    {
    where(mask,val)*=vfbig;
    where(mask,scale)-=1;
    mask = (abs(val)<vfmin) & (val!=0);
    }
  val_ = val;
  scale_ = scale;
  }

static void mypow(Tv val, size_t npow, const vector<double> &powlimit,
  Tv & DUCC0_RESTRICT resd, Tv & DUCC0_RESTRICT ress)
  {
  Tv vminv=powlimit[npow];
  auto mask = abs(val)<vminv;
  if (none_of(mask)) // no underflows possible, use quick algoritm
    {
    Tv res=1;
    do
      {
      if (npow&1)
        res*=val;
      val*=val;
      }
    while(npow>>=1);
    resd=res;
    ress=0;
    }
  else
    {
    Tv scale=0, scaleint=0, res=1;
    Tvnormalize(val,scaleint,sht_fbighalf);
    do
      {
      if (npow&1)
        {
        res*=val;
        scale+=scaleint;
        Tvnormalize(res,scale,sht_fbighalf);
        }
      val*=val;
      scaleint+=scaleint;
      Tvnormalize(val,scaleint,sht_fbighalf);
      }
    while(npow>>=1);
    resd=res;
    ress=scale;
    }
  }

static inline void getCorfac(Tv scale, Tv & DUCC0_RESTRICT corfac)
  {
// not sure why, but MSVC miscompiles the default code
#if defined(_MSC_VER)
  for (size_t i=0; i<Tv::size(); ++i)
    corfac[i] = (scale[i]<0) ? 0. : ((scale[i]<1) ? 1. : sht_fbig);
#else
  corfac = Tv(1.);
  where(scale<-0.5,corfac)=0;
  where(scale>0.5,corfac)=sht_fbig;
#endif
  }

static inline bool rescale(Tv &v1, Tv &v2, Tv &s, Tv eps)
  {
  auto mask = abs(v2)>eps;
  if (any_of(mask))
    {
    where(mask,v1)*=sht_fsmall;
    where(mask,v2)*=sht_fsmall;
    where(mask,s)+=1;
    return true;
    }
  return false;
  }

DUCC0_NOINLINE static void init_lambda(const Ylmgen &gen,
  s0data_v & DUCC0_RESTRICT d, size_t nv2)
  {
  Tv mfac = (gen.m&1) ? -gen.mfac[gen.m]:gen.mfac[gen.m];
  for (size_t i=0; i<nv2; ++i)
    {
    d.lam1[i]=0;
    mypow(d.sth[i],gen.m,gen.powlimit,d.lam2[i],d.scale[i]);
    d.lam2[i] *= mfac;
    Tvnormalize(d.lam2[i],d.scale[i],sht_ftol);
    }
  }

DUCC0_NOINLINE static void iter_to_ieee(const Ylmgen &gen,
  s0data_v & DUCC0_RESTRICT d, size_t & DUCC0_RESTRICT l_, size_t & DUCC0_RESTRICT il_, size_t nv2,
  size_t lstart, size_t lstop)
  {
  size_t l=lstart, il=(lstart-gen.m)/2;
  bool below_limit = true;

  for (size_t i=0; i<nv2; ++i)
    {
    rescale(d.lam1[i], d.lam2[i], d.scale[i], sht_ftol);
    below_limit &= all_of(d.scale[i]<1);
    }

  while (below_limit)
    {
    if (l==lstop) { l_=l; il_=il; return; }
    if (l+4>gen.lmax) {l_=gen.lmax+1;return;}
    below_limit=true;
    Tv a1=gen.coef[il  ].a, b1=gen.coef[il  ].b;
    Tv a2=gen.coef[il+1].a, b2=gen.coef[il+1].b;
    for (size_t i=0; i<nv2; ++i)
      {
      d.lam1[i] = (a1*d.csq[i] + b1)*d.lam2[i] + d.lam1[i];
      d.lam2[i] = (a2*d.csq[i] + b2)*d.lam1[i] + d.lam2[i];
      if (rescale(d.lam1[i], d.lam2[i], d.scale[i], sht_ftol))
        below_limit &= all_of(d.scale[i]<1);
      }
    l+=4; il+=2;
    }
  l_=l; il_=il;
  }

DUCC0_NOINLINE static void alm2map_kernel(s0data_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &coef, const dcmplx * DUCC0_RESTRICT alm,
  size_t l, size_t il, size_t lmax, size_t nv2)
  {
  if constexpr(Tv::size()>4)  // this loop seems to help AVX512
    for (; l+6<=lmax; il+=4, l+=8)
      {
      Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag();
      Tv ar2=alm[l+1].real(), ai2=alm[l+1].imag();
      Tv ar3=alm[l+2].real(), ai3=alm[l+2].imag();
      Tv ar4=alm[l+3].real(), ai4=alm[l+3].imag();
      Tv ar5=alm[l+4].real(), ai5=alm[l+4].imag();
      Tv ar6=alm[l+5].real(), ai6=alm[l+5].imag();
      Tv ar7=alm[l+6].real(), ai7=alm[l+6].imag();
      Tv ar8=alm[l+7].real(), ai8=alm[l+7].imag();
      Tv a1=coef[il  ].a, b1=coef[il  ].b;
      Tv a2=coef[il+1].a, b2=coef[il+1].b;
      Tv a3=coef[il+2].a, b3=coef[il+2].b;
      Tv a4=coef[il+3].a, b4=coef[il+3].b;
      for (size_t i=0; i<nv2; ++i)
        {
        d.p1r[i] += d.lam2[i]*ar1;
        d.p1i[i] += d.lam2[i]*ai1;
        d.p2r[i] += d.lam2[i]*ar2;
        d.p2i[i] += d.lam2[i]*ai2;
        d.lam1[i] = (a1*d.csq[i] + b1)*d.lam2[i] + d.lam1[i];
        d.p1r[i] += d.lam1[i]*ar3;
        d.p1i[i] += d.lam1[i]*ai3;
        d.p2r[i] += d.lam1[i]*ar4;
        d.p2i[i] += d.lam1[i]*ai4;
        d.lam2[i] = (a2*d.csq[i] + b2)*d.lam1[i] + d.lam2[i];
        d.p1r[i] += d.lam2[i]*ar5;
        d.p1i[i] += d.lam2[i]*ai5;
        d.p2r[i] += d.lam2[i]*ar6;
        d.p2i[i] += d.lam2[i]*ai6;
        d.lam1[i] = (a3*d.csq[i] + b3)*d.lam2[i] + d.lam1[i];
        d.p1r[i] += d.lam1[i]*ar7;
        d.p1i[i] += d.lam1[i]*ai7;
        d.p2r[i] += d.lam1[i]*ar8;
        d.p2i[i] += d.lam1[i]*ai8;
        d.lam2[i] = (a4*d.csq[i] + b4)*d.lam1[i] + d.lam2[i];
        }
      }

  for (; l+2<=lmax; il+=2, l+=4)
    {
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag();
    Tv ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    Tv ar3=alm[l+2].real(), ai3=alm[l+2].imag();
    Tv ar4=alm[l+3].real(), ai4=alm[l+3].imag();
    Tv a1=coef[il  ].a, b1=coef[il  ].b;
    Tv a2=coef[il+1].a, b2=coef[il+1].b;
    for (size_t i=0; i<nv2; ++i)
      {
      d.p1r[i] += d.lam2[i]*ar1;
      d.p1i[i] += d.lam2[i]*ai1;
      d.p2r[i] += d.lam2[i]*ar2;
      d.p2i[i] += d.lam2[i]*ai2;
      d.lam1[i] = (a1*d.csq[i] + b1)*d.lam2[i] + d.lam1[i];
      d.p1r[i] += d.lam1[i]*ar3;
      d.p1i[i] += d.lam1[i]*ai3;
      d.p2r[i] += d.lam1[i]*ar4;
      d.p2i[i] += d.lam1[i]*ai4;
      d.lam2[i] = (a2*d.csq[i] + b2)*d.lam1[i] + d.lam2[i];
      }
    }
  for (; l<=lmax; ++il, l+=2)
    {
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag();
    Tv ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    Tv a=coef[il].a, b=coef[il].b;
    for (size_t i=0; i<nv2; ++i)
      {
      d.p1r[i] += d.lam2[i]*ar1;
      d.p1i[i] += d.lam2[i]*ai1;
      d.p2r[i] += d.lam2[i]*ar2;
      d.p2i[i] += d.lam2[i]*ai2;
      Tv tmp = (a*d.csq[i] + b)*d.lam2[i] + d.lam1[i];
      d.lam1[i] = d.lam2[i];
      d.lam2[i] = tmp;
      }
    }
  }

DUCC0_NOINLINE static void calc_alm2map (const dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, s0data_v & DUCC0_RESTRICT d, size_t nth, size_t lstart, size_t lstop)
  {
  size_t l,il=0;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee(gen, d, l, il, nv2, lstart, lstop);

  if (l>=lstop) return;

  auto &coef = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scale[i], d.corfac[i]);
    full_ieee &= all_of(d.scale[i]>=0);
    }

  while((!full_ieee) && (l<lstop))
    {
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag();
    Tv ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    Tv a=coef[il].a, b=coef[il].b;
    full_ieee=true;
    for (size_t i=0; i<nv2; ++i)
      {
      d.p1r[i] += d.lam2[i]*d.corfac[i]*ar1;
      d.p1i[i] += d.lam2[i]*d.corfac[i]*ai1;
      d.p2r[i] += d.lam2[i]*d.corfac[i]*ar2;
      d.p2i[i] += d.lam2[i]*d.corfac[i]*ai2;
      Tv tmp = (a*d.csq[i] + b)*d.lam2[i] + d.lam1[i];
      d.lam1[i] = d.lam2[i];
      d.lam2[i] = tmp;
      if (rescale(d.lam1[i], d.lam2[i], d.scale[i], sht_ftol))
        getCorfac(d.scale[i], d.corfac[i]);
      full_ieee &= all_of(d.scale[i]>=0);
      }
    l+=2; ++il;
    }

  if (l>=lstop) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.lam1[i] *= d.corfac[i];
    d.lam2[i] *= d.corfac[i];
    d.scale[i] = 0;
    }
  alm2map_kernel(d, coef, alm, l, il, lstop-1, nv2);
  }

DUCC0_NOINLINE static void map2alm_kernel(s0data_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &coef, dcmplx * DUCC0_RESTRICT alm, size_t l,
  size_t il, size_t lmax, size_t nv2)
  {
  for (; l+2<=lmax; il+=2, l+=4)
    {
    Tv a1=coef[il  ].a, b1=coef[il  ].b;
    Tv a2=coef[il+1].a, b2=coef[il+1].b;
    Tv atmp1[4] = {0,0,0,0};
    Tv atmp2[4] = {0,0,0,0};
    for (size_t i=0; i<nv2; ++i)
      {
      atmp1[0] += d.lam2[i]*d.p1r[i];
      atmp1[1] += d.lam2[i]*d.p1i[i];
      atmp1[2] += d.lam2[i]*d.p2r[i];
      atmp1[3] += d.lam2[i]*d.p2i[i];
      d.lam1[i] = (a1*d.csq[i] + b1)*d.lam2[i] + d.lam1[i];
      atmp2[0] += d.lam1[i]*d.p1r[i];
      atmp2[1] += d.lam1[i]*d.p1i[i];
      atmp2[2] += d.lam1[i]*d.p2r[i];
      atmp2[3] += d.lam1[i]*d.p2i[i];
      d.lam2[i] = (a2*d.csq[i] + b2)*d.lam1[i] + d.lam2[i];
      }
    vhsum_cmplx_special (atmp1[0], atmp1[1], atmp1[2], atmp1[3], &alm[l  ]);
    vhsum_cmplx_special (atmp2[0], atmp2[1], atmp2[2], atmp2[3], &alm[l+2]);
    }
  for (; l<=lmax; ++il, l+=2)
    {
    Tv a=coef[il].a, b=coef[il].b;
    Tv atmp[4] = {0,0,0,0};
    for (size_t i=0; i<nv2; ++i)
      {
      atmp[0] += d.lam2[i]*d.p1r[i];
      atmp[1] += d.lam2[i]*d.p1i[i];
      atmp[2] += d.lam2[i]*d.p2r[i];
      atmp[3] += d.lam2[i]*d.p2i[i];
      Tv tmp = (a*d.csq[i] + b)*d.lam2[i] + d.lam1[i];
      d.lam1[i] = d.lam2[i];
      d.lam2[i] = tmp;
      }
    vhsum_cmplx_special (atmp[0], atmp[1], atmp[2], atmp[3], &alm[l]);
    }
  }

DUCC0_NOINLINE static void calc_map2alm (dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, s0data_v & DUCC0_RESTRICT d, size_t nth, size_t lstart, size_t lstop)
  {
  size_t l,il=0;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee(gen, d, l, il, nv2, lstart, lstop);

  if (l>=lstop) return;

  auto &coef = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scale[i], d.corfac[i]);
    full_ieee &= all_of(d.scale[i]>=0);
    }

  while((!full_ieee) && (l<lstop))
    {
    Tv a=coef[il].a, b=coef[il].b;
    Tv atmp[4] = {0,0,0,0};
    full_ieee=1;
    for (size_t i=0; i<nv2; ++i)
      {
      atmp[0] += d.lam2[i]*d.corfac[i]*d.p1r[i];
      atmp[1] += d.lam2[i]*d.corfac[i]*d.p1i[i];
      atmp[2] += d.lam2[i]*d.corfac[i]*d.p2r[i];
      atmp[3] += d.lam2[i]*d.corfac[i]*d.p2i[i];
      Tv tmp = (a*d.csq[i] + b)*d.lam2[i] + d.lam1[i];
      d.lam1[i] = d.lam2[i];
      d.lam2[i] = tmp;
      if (rescale(d.lam1[i], d.lam2[i], d.scale[i], sht_ftol))
        getCorfac(d.scale[i], d.corfac[i]);
      full_ieee &= all_of(d.scale[i]>=0);
      }
    vhsum_cmplx_special (atmp[0], atmp[1], atmp[2], atmp[3], &alm[l]);
    l+=2; ++il;
    }

  if (l>=lstop) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.lam1[i] *= d.corfac[i];
    d.lam2[i] *= d.corfac[i];
    d.scale[i] = 0;
    }
  map2alm_kernel(d, coef, alm, l, il, lstop-1, nv2);
  }

DUCC0_NOINLINE static void init_lambda_spin (const Ylmgen &gen,
  sxdata_v & DUCC0_RESTRICT d, size_t nv2)
  {
  Tv prefac=gen.prefac[gen.m],
     prescale=gen.fscale[gen.m];
  for (size_t i=0; i<nv2; ++i)
    {
// FIXME: can we do this better?
    Tv cth2=max(Tv(1e-15),sqrt((1.+d.cth[i])*0.5));
    Tv sth2=max(Tv(1e-15),sqrt((1.-d.cth[i])*0.5));
    auto mask=d.sth[i]<0;
    where(mask&(d.cth[i]<0),cth2)*=-1.;
    where(mask&(d.cth[i]<0),sth2)*=-1.;

    Tv ccp, ccps, ssp, ssps, csp, csps, scp, scps;
    mypow(cth2,gen.cosPow,gen.powlimit,ccp,ccps);
    mypow(sth2,gen.sinPow,gen.powlimit,ssp,ssps);
    mypow(cth2,gen.sinPow,gen.powlimit,csp,csps);
    mypow(sth2,gen.cosPow,gen.powlimit,scp,scps);

    d.l1p[i] = 0;
    d.l1m[i] = 0;
    d.l2p[i] = prefac*ccp;
    d.scp[i] = prescale+ccps;
    d.l2m[i] = prefac*csp;
    d.scm[i] = prescale+csps;
    Tvnormalize(d.l2m[i],d.scm[i],sht_fbighalf);
    Tvnormalize(d.l2p[i],d.scp[i],sht_fbighalf);
    d.l2p[i] *= ssp;
    d.scp[i] += ssps;
    d.l2m[i] *= scp;
    d.scm[i] += scps;
    if (gen.preMinus_p)
      d.l2p[i] = -d.l2p[i];
    if (gen.preMinus_m)
      d.l2m[i] = -d.l2m[i];
    if (gen.s&1)
      d.l2p[i] = -d.l2p[i];

    Tvnormalize(d.l2m[i],d.scm[i],sht_ftol);
    Tvnormalize(d.l2p[i],d.scp[i],sht_ftol);
    }
  }

DUCC0_NOINLINE static void iter_to_ieee_spin (const Ylmgen &gen,
  sxdata_v & DUCC0_RESTRICT d, size_t & DUCC0_RESTRICT l_, size_t nv2,
  size_t lstart, size_t lstop)
  {
  const auto &fx = gen.coef;

  size_t l=lstart;
  bool below_limit = true;
  for (size_t i=0; i<nv2; ++i)
    {
    rescale(d.l1m[i], d.l2m[i], d.scm[i], sht_ftol);
    rescale(d.l1p[i], d.l2p[i], d.scp[i], sht_ftol);

    below_limit &= all_of(d.scm[i]<1) &&
                   all_of(d.scp[i]<1);
    }

  while (below_limit)
    {
    if (l==lstop) { l_=l; return; }
    if (l+2>gen.lmax) {l_=gen.lmax+1;return;}
    below_limit=true;
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      // The bitwise or operator is deliberate!
      // Silencing clang compiler warning by casting to int...
      if (int(rescale(d.l1p[i],d.l2p[i],d.scp[i],sht_ftol)) |
          rescale(d.l1m[i],d.l2m[i],d.scm[i],sht_ftol))
        below_limit &= all_of(d.scp[i]<1) &&
                       all_of(d.scm[i]<1);
      }
    l+=2;
    }

  l_=l;
  }

DUCC0_NOINLINE static void alm2map_spin_kernel(sxdata_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &fx, const dcmplx * DUCC0_RESTRICT alm,
  size_t l, size_t lmax, size_t nv2)
  {
  size_t lsave = l;

  if constexpr(Tv::size()>4)  // this loop seems to help AVX512
    while (l+2<=lmax)
      {
      Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
      Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
      Tv fx30=fx[l+3].a,fx31=fx[l+3].b;
      Tv fx40=fx[l+4].a,fx41=fx[l+4].b;
      Tv agr1=alm[2*l  ].real(), agi1=alm[2*l  ].imag(),
         acr1=alm[2*l+1].real(), aci1=alm[2*l+1].imag();
      Tv agr2=alm[2*l+2].real(), agi2=alm[2*l+2].imag(),
         acr2=alm[2*l+3].real(), aci2=alm[2*l+3].imag();
      Tv agr3=alm[2*l+4].real(), agi3=alm[2*l+4].imag(),
         acr3=alm[2*l+5].real(), aci3=alm[2*l+5].imag();
      Tv agr4=alm[2*l+6].real(), agi4=alm[2*l+6].imag(),
         acr4=alm[2*l+7].real(), aci4=alm[2*l+7].imag();
      for (size_t i=0; i<nv2; ++i)
        {
        d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
        d.p1pr[i] += agr1*d.l2p[i];
        d.p1pi[i] += agi1*d.l2p[i];
        d.p1mr[i] += acr1*d.l2p[i];
        d.p1mi[i] += aci1*d.l2p[i];

        d.p1pr[i] += aci2*d.l1p[i];
        d.p1pi[i] -= acr2*d.l1p[i];
        d.p1mr[i] -= agi2*d.l1p[i];
        d.p1mi[i] += agr2*d.l1p[i];
        d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];

        d.l1p[i] = (d.cth[i]*fx30 - fx31)*d.l2p[i] - d.l1p[i];
        d.p1pr[i] += agr3*d.l2p[i];
        d.p1pi[i] += agi3*d.l2p[i];
        d.p1mr[i] += acr3*d.l2p[i];
        d.p1mi[i] += aci3*d.l2p[i];

        d.p1pr[i] += aci4*d.l1p[i];
        d.p1pi[i] -= acr4*d.l1p[i];
        d.p1mr[i] -= agi4*d.l1p[i];
        d.p1mi[i] += agr4*d.l1p[i];
        d.l2p[i] = (d.cth[i]*fx40 - fx41)*d.l1p[i] - d.l2p[i];
        }
      l+=4;
      }

  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=alm[2*l  ].real(), agi1=alm[2*l  ].imag(),
       acr1=alm[2*l+1].real(), aci1=alm[2*l+1].imag();
    Tv agr2=alm[2*l+2].real(), agi2=alm[2*l+2].imag(),
       acr2=alm[2*l+3].real(), aci2=alm[2*l+3].imag();
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.p1pr[i] += agr1*d.l2p[i];
      d.p1pi[i] += agi1*d.l2p[i];
      d.p1mr[i] += acr1*d.l2p[i];
      d.p1mi[i] += aci1*d.l2p[i];

      d.p1pr[i] += aci2*d.l1p[i];
      d.p1pi[i] -= acr2*d.l1p[i];
      d.p1mr[i] -= agi2*d.l1p[i];
      d.p1mi[i] += agr2*d.l1p[i];
      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      }
    l+=2;
    }
  l=lsave;

  if constexpr(Tv::size()>4)  // this loop seems to help AVX512
    while (l+2<=lmax)
      {
      Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
      Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
      Tv fx30=fx[l+3].a,fx31=fx[l+3].b;
      Tv fx40=fx[l+4].a,fx41=fx[l+4].b;
      Tv agr1=alm[2*l  ].real(), agi1=alm[2*l  ].imag(),
         acr1=alm[2*l+1].real(), aci1=alm[2*l+1].imag();
      Tv agr2=alm[2*l+2].real(), agi2=alm[2*l+2].imag(),
         acr2=alm[2*l+3].real(), aci2=alm[2*l+3].imag();
      Tv agr3=alm[2*l+4].real(), agi3=alm[2*l+4].imag(),
         acr3=alm[2*l+5].real(), aci3=alm[2*l+5].imag();
      Tv agr4=alm[2*l+6].real(), agi4=alm[2*l+6].imag(),
         acr4=alm[2*l+7].real(), aci4=alm[2*l+7].imag();
      for (size_t i=0; i<nv2; ++i)
        {
        d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
        d.p2pr[i] -= aci1*d.l2m[i];
        d.p2pi[i] += acr1*d.l2m[i];
        d.p2mr[i] += agi1*d.l2m[i];
        d.p2mi[i] -= agr1*d.l2m[i];

        d.p2pr[i] += agr2*d.l1m[i];
        d.p2pi[i] += agi2*d.l1m[i];
        d.p2mr[i] += acr2*d.l1m[i];
        d.p2mi[i] += aci2*d.l1m[i];
        d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];

        d.l1m[i] = (d.cth[i]*fx30 + fx31)*d.l2m[i] - d.l1m[i];
        d.p2pr[i] -= aci3*d.l2m[i];
        d.p2pi[i] += acr3*d.l2m[i];
        d.p2mr[i] += agi3*d.l2m[i];
        d.p2mi[i] -= agr3*d.l2m[i];

        d.p2pr[i] += agr4*d.l1m[i];
        d.p2pi[i] += agi4*d.l1m[i];
        d.p2mr[i] += acr4*d.l1m[i];
        d.p2mi[i] += aci4*d.l1m[i];
        d.l2m[i] = (d.cth[i]*fx40 + fx41)*d.l1m[i] - d.l2m[i];
        }
      l+=4;
      }

  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=alm[2*l  ].real(), agi1=alm[2*l  ].imag(),
       acr1=alm[2*l+1].real(), aci1=alm[2*l+1].imag();
    Tv agr2=alm[2*l+2].real(), agi2=alm[2*l+2].imag(),
       acr2=alm[2*l+3].real(), aci2=alm[2*l+3].imag();
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      d.p2pr[i] -= aci1*d.l2m[i];
      d.p2pi[i] += acr1*d.l2m[i];
      d.p2mr[i] += agi1*d.l2m[i];
      d.p2mi[i] -= agr1*d.l2m[i];

      d.p2pr[i] += agr2*d.l1m[i];
      d.p2pi[i] += agi2*d.l1m[i];
      d.p2mr[i] += acr2*d.l1m[i];
      d.p2mi[i] += aci2*d.l1m[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      }
    l+=2;
    }
  }

DUCC0_NOINLINE static void calc_alm2map_spin (const dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth, size_t lstart, size_t lstop)
  {
  size_t l;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2, lstart, lstop);
  if (l>=lstop) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }

  while((!full_ieee) && (l<lstop))
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=alm[2*l  ].real(), agi1=alm[2*l  ].imag(),
       acr1=alm[2*l+1].real(), aci1=alm[2*l+1].imag();
    Tv agr2=alm[2*l+2].real(), agi2=alm[2*l+2].imag(),
       acr2=alm[2*l+3].real(), aci2=alm[2*l+3].imag();
    full_ieee=true;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];

      Tv l2p=d.l2p[i]*d.cfp[i], l2m=d.l2m[i]*d.cfm[i];
      Tv l1m=d.l1m[i]*d.cfm[i], l1p=d.l1p[i]*d.cfp[i];

      d.p1pr[i] += agr1*l2p + aci2*l1p;
      d.p1pi[i] += agi1*l2p - acr2*l1p;
      d.p1mr[i] += acr1*l2p - agi2*l1p;
      d.p1mi[i] += aci1*l2p + agr2*l1p;

      d.p2pr[i] += agr2*l1m - aci1*l2m;
      d.p2pi[i] += agi2*l1m + acr1*l2m;
      d.p2mr[i] += acr2*l1m + agi1*l2m;
      d.p2mi[i] += aci2*l1m - agr1*l2m;

      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sht_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sht_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    l+=2;
    }
  if (l>=lstop) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    d.scp[i] = d.scm[i] = 0;
    }
  alm2map_spin_kernel(d, fx, alm, l, lstop-1, nv2);
  }

DUCC0_NOINLINE static void map2alm_spin_kernel(sxdata_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &fx, dcmplx * DUCC0_RESTRICT alm,
  size_t l, size_t lmax, size_t nv2)
  {
  size_t lsave=l;
  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0, acr1=0, aci1=0;
    Tv agr2=0, agi2=0, acr2=0, aci2=0;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      agr1 += d.p2mi[i]*d.l2p[i];
      agi1 -= d.p2mr[i]*d.l2p[i];
      acr1 -= d.p2pi[i]*d.l2p[i];
      aci1 += d.p2pr[i]*d.l2p[i];
      agr2 += d.p2pr[i]*d.l1p[i];
      agi2 += d.p2pi[i]*d.l1p[i];
      acr2 += d.p2mr[i]*d.l1p[i];
      aci2 += d.p2mi[i]*d.l1p[i];
      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      }
    vhsum_cmplx_special (agr1,agi1,acr1,aci1,&alm[2*l]);
    vhsum_cmplx_special (agr2,agi2,acr2,aci2,&alm[2*l+2]);
    l+=2;
    }
  l=lsave;
  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0, acr1=0, aci1=0;
    Tv agr2=0, agi2=0, acr2=0, aci2=0;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      agr1 += d.p1pr[i]*d.l2m[i];
      agi1 += d.p1pi[i]*d.l2m[i];
      acr1 += d.p1mr[i]*d.l2m[i];
      aci1 += d.p1mi[i]*d.l2m[i];
      agr2 -= d.p1mi[i]*d.l1m[i];
      agi2 += d.p1mr[i]*d.l1m[i];
      acr2 += d.p1pi[i]*d.l1m[i];
      aci2 -= d.p1pr[i]*d.l1m[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      }
    vhsum_cmplx_special (agr1,agi1,acr1,aci1,&alm[2*l]);
    vhsum_cmplx_special (agr2,agi2,acr2,aci2,&alm[2*l+2]);
    l+=2;
    }
  }

DUCC0_NOINLINE static void calc_map2alm_spin (dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth, size_t lstart, size_t lstop)
  {
  size_t l;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2, lstart, lstop);
  if (l>=lstop) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }

  while((!full_ieee) && (l<lstop))
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0, acr1=0, aci1=0;
    Tv agr2=0, agi2=0, acr2=0, aci2=0;
    full_ieee=true;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      Tv l2p = d.l2p[i]*d.cfp[i], l2m = d.l2m[i]*d.cfm[i];
      Tv l1p = d.l1p[i]*d.cfp[i], l1m = d.l1m[i]*d.cfm[i];
      agr1 += d.p1pr[i]*l2m + d.p2mi[i]*l2p;
      agi1 += d.p1pi[i]*l2m - d.p2mr[i]*l2p;
      acr1 += d.p1mr[i]*l2m - d.p2pi[i]*l2p;
      aci1 += d.p1mi[i]*l2m + d.p2pr[i]*l2p;
      agr2 += d.p2pr[i]*l1p - d.p1mi[i]*l1m;
      agi2 += d.p2pi[i]*l1p + d.p1mr[i]*l1m;
      acr2 += d.p2mr[i]*l1p + d.p1pi[i]*l1m;
      aci2 += d.p2mi[i]*l1p - d.p1pr[i]*l1m;

      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sht_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sht_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    vhsum_cmplx_special (agr1,agi1,acr1,aci1,&alm[2*l]);
    vhsum_cmplx_special (agr2,agi2,acr2,aci2,&alm[2*l+2]);
    l+=2;
    }
  if (l>=lstop) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    d.scp[i] = d.scm[i] = 0;
    }
  map2alm_spin_kernel(d, fx, alm, l, lstop-1, nv2);
  }


DUCC0_NOINLINE static void alm2map_spin_gradonly_kernel(sxdata_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &fx, const dcmplx * DUCC0_RESTRICT alm,
  size_t l, size_t lmax, size_t nv2)
  {
  size_t lsave=l;
  if constexpr(Tv::size()>4)  // this loop seems to help AVX512
    while (l+2<=lmax)
      {
      Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
      Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
      Tv fx30=fx[l+3].a,fx31=fx[l+3].b;
      Tv fx40=fx[l+4].a,fx41=fx[l+4].b;
      Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag(),
         ar2=alm[l+1].real(), ai2=alm[l+1].imag(),
         ar3=alm[l+2].real(), ai3=alm[l+2].imag(),
         ar4=alm[l+3].real(), ai4=alm[l+3].imag();
      for (size_t i=0; i<nv2; ++i)
        {
        d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
        d.p1pr[i] += ar1*d.l2p[i];
        d.p1pi[i] += ai1*d.l2p[i];
        d.p1mr[i] -= ai2*d.l1p[i];
        d.p1mi[i] += ar2*d.l1p[i];
        d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];

        d.l1p[i] = (d.cth[i]*fx30 - fx31)*d.l2p[i] - d.l1p[i];
        d.p1pr[i] += ar3*d.l2p[i];
        d.p1pi[i] += ai3*d.l2p[i];
        d.p1mr[i] -= ai4*d.l1p[i];
        d.p1mi[i] += ar4*d.l1p[i];
        d.l2p[i] = (d.cth[i]*fx40 - fx41)*d.l1p[i] - d.l2p[i];
        }
      l+=4;
      }
  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag(),
       ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.p1pr[i] += ar1*d.l2p[i];
      d.p1pi[i] += ai1*d.l2p[i];

      d.p1mr[i] -= ai2*d.l1p[i];
      d.p1mi[i] += ar2*d.l1p[i];
      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      }
    l+=2;
    }
  l=lsave;
  if constexpr(Tv::size()>4)  // this loop seems to help AVX512
    while (l+2<=lmax)
      {
      Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
      Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
      Tv fx30=fx[l+3].a,fx31=fx[l+3].b;
      Tv fx40=fx[l+4].a,fx41=fx[l+4].b;
      Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag(),
         ar2=alm[l+1].real(), ai2=alm[l+1].imag(),
         ar3=alm[l+2].real(), ai3=alm[l+2].imag(),
         ar4=alm[l+3].real(), ai4=alm[l+3].imag();
      for (size_t i=0; i<nv2; ++i)
        {
        d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
        d.p2mr[i] += ai1*d.l2m[i];
        d.p2mi[i] -= ar1*d.l2m[i];
        d.p2pr[i] += ar2*d.l1m[i];
        d.p2pi[i] += ai2*d.l1m[i];
        d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];

        d.l1m[i] = (d.cth[i]*fx30 + fx31)*d.l2m[i] - d.l1m[i];
        d.p2mr[i] += ai3*d.l2m[i];
        d.p2mi[i] -= ar3*d.l2m[i];
        d.p2pr[i] += ar4*d.l1m[i];
        d.p2pi[i] += ai4*d.l1m[i];
        d.l2m[i] = (d.cth[i]*fx40 + fx41)*d.l1m[i] - d.l2m[i];
        }
      l+=4;
      }
  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag(),
       ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      d.p2mr[i] += ai1*d.l2m[i];
      d.p2mi[i] -= ar1*d.l2m[i];

      d.p2pr[i] += ar2*d.l1m[i];
      d.p2pi[i] += ai2*d.l1m[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      }
    l+=2;
    }
  }

DUCC0_NOINLINE static void calc_alm2map_spin_gradonly(const dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth, size_t lstart, size_t lstop)
  {
  size_t l;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2, lstart, lstop);
  if (l>=lstop) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }

  while((!full_ieee) && (l<lstop))
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag(),
       ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    full_ieee=true;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];

      Tv l2p=d.l2p[i]*d.cfp[i], l2m=d.l2m[i]*d.cfm[i];
      Tv l1m=d.l1m[i]*d.cfm[i], l1p=d.l1p[i]*d.cfp[i];

      d.p1pr[i] += ar1*l2p;
      d.p1pi[i] += ai1*l2p;
      d.p1mr[i] -= ai2*l1p;
      d.p1mi[i] += ar2*l1p;

      d.p2pr[i] += ar2*l1m;
      d.p2pi[i] += ai2*l1m;
      d.p2mr[i] += ai1*l2m;
      d.p2mi[i] -= ar1*l2m;

      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sht_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sht_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    l+=2;
    }
  if (l>=lstop) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    d.scp[i] = d.scm[i] = 0;
    }
  alm2map_spin_gradonly_kernel(d, fx, alm, l, lstop-1, nv2);
  }

DUCC0_NOINLINE static void map2alm_spin_gradonly_kernel(sxdata_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &fx, dcmplx * DUCC0_RESTRICT alm,
  size_t l, size_t lmax, size_t nv2)
  {
  size_t lsave=l;
  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0;
    Tv agr2=0, agi2=0;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      agr1 += d.p2mi[i]*d.l2p[i];
      agi1 -= d.p2mr[i]*d.l2p[i];
      agr2 += d.p2pr[i]*d.l1p[i];
      agi2 += d.p2pi[i]*d.l1p[i];
      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      }
    vhsum_cmplx_special (agr1,agi1,agr2,agi2,&alm[l]);
    l+=2;
    }
  l=lsave;
  while (l<=lmax)
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0;
    Tv agr2=0, agi2=0;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      agr1 += d.p1pr[i]*d.l2m[i];
      agi1 += d.p1pi[i]*d.l2m[i];
      agr2 -= d.p1mi[i]*d.l1m[i];
      agi2 += d.p1mr[i]*d.l1m[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      }
    vhsum_cmplx_special (agr1,agi1,agr2,agi2,&alm[l]);
    l+=2;
    }
  }

DUCC0_NOINLINE static void calc_map2alm_spin_gradonly (dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth, size_t lstart, size_t lstop)
  {
  size_t l;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2, lstart, lstop);
  if (l>=lstop) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }

  while((!full_ieee) && (l<lstop))
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0;
    Tv agr2=0, agi2=0;
    full_ieee=1;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      Tv l2p = d.l2p[i]*d.cfp[i], l2m = d.l2m[i]*d.cfm[i];
      Tv l1p = d.l1p[i]*d.cfp[i], l1m = d.l1m[i]*d.cfm[i];
      agr1 += d.p1pr[i]*l2m + d.p2mi[i]*l2p;
      agi1 += d.p1pi[i]*l2m - d.p2mr[i]*l2p;
      agr2 += d.p2pr[i]*l1p - d.p1mi[i]*l1m;
      agi2 += d.p2pi[i]*l1p + d.p1mr[i]*l1m;

      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sht_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sht_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    vhsum_cmplx_special (agr1,agi1,agr2,agi2,&alm[l]);
    l+=2;
    }
  if (l>=lstop) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    d.scp[i] = d.scm[i] = 0;
    }
  map2alm_spin_gradonly_kernel(d, fx, alm, l, lstop-1, nv2);
  }

template<typename T> DUCC0_NOINLINE static void fill_a2m(const Ylmgen &gen,
  const vector<ringdata> &rdata, const vmav<complex<T>,3> &phase,
  size_t mi, size_t &ith, s0data_u &d, array<size_t, nv0*VLEN> &idx,
  array<size_t, nv0*VLEN> &midx, Tbv0 &cth, array<double, nv0*VLEN> &wgt, size_t &nth)
  {
  constexpr size_t nval=nv0*VLEN;
  nth=0;
  while ((nth<nval)&&(ith<rdata.size()))
    {
    if (rdata[ith].mlim>=gen.m)
      {
      idx[nth] = rdata[ith].idx;
      midx[nth] = rdata[ith].midx;
      auto lcth = rdata[ith].cth;
      cth[nth/VLEN][nth%VLEN] = lcth;
      wgt[nth] = rdata[ith].wgt;
      if (abs(lcth)>0.99)
        d.s.csq[nth]=(1.-rdata[ith].sth)*(1.+rdata[ith].sth);
      else
        d.s.csq[nth]=lcth*lcth;
      d.s.sth[nth]=rdata[ith].sth;
      ++nth;
      }
    else
      phase(0, rdata[ith].idx, mi) = phase(0, rdata[ith].midx, mi) = 0;
    ++ith;
    }
  if (nth>0)
    {
    size_t nvec = (nth+VLEN-1)/VLEN;
    size_t i2 = nvec*VLEN;
    for (auto i=nth; i<i2; ++i)
      {
      d.s.csq[i]=d.s.csq[nth-1];
      d.s.sth[i]=d.s.sth[nth-1];
      }
    for (size_t i=0; i<nvec; ++i)
      d.v.p1r[i] = d.v.p1i[i] = d.v.p2r[i] = d.v.p2i[i] = 0;

    init_lambda(gen, d.v, nvec);
    }
  }

template<typename T> DUCC0_NOINLINE static void extract_a2m( size_t mi,
  s0data_u &d, const array<size_t, nv0*VLEN> &idx,
  const array<size_t, nv0*VLEN> &midx, const size_t &nth, const Tbv0 &cth,
  const array<double, nv0*VLEN> &wgt, const vmav<complex<T>,3> &phase)
  {
  size_t nvec = (nth+VLEN-1)/VLEN;
  for (size_t i=0; i<nvec; ++i)
    {
    auto t1r = d.v.p1r[i];
    auto t2r = d.v.p2r[i]*cth[i];
    auto t1i = d.v.p1i[i];
    auto t2i = d.v.p2i[i]*cth[i];
    d.v.p1r[i] = t1r+t2r;
    d.v.p1i[i] = t1i+t2i;
    d.v.p2r[i] = t1r-t2r;
    d.v.p2i[i] = t1i-t2i;
    }
  for (size_t i=0; i<nth; ++i)
    {
    //adjust for new algorithm
    phase(0, idx[i], mi) = complex<T>(T(wgt[i]*d.s.p1r[i]),T(wgt[i]*d.s.p1i[i]));
    if (idx[i]!=midx[i])
      phase(0, midx[i], mi) = complex<T>(T(wgt[i]*d.s.p2r[i]),T(wgt[i]*d.s.p2i[i]));
    }
  }
template<typename T> DUCC0_NOINLINE static void fill_a2m_spin(
  const Ylmgen &gen, const vector<ringdata> &rdata,
  const vmav<complex<T>,3> &phase, size_t mi, size_t &ith, sxdata_u &d,
  array<size_t, nvx*VLEN> &idx, array<size_t, nvx*VLEN> &midx, size_t &nth)
  {
  constexpr size_t nval=nvx*VLEN;
  nth=0;
  while ((nth<nval)&&(ith<rdata.size()))
    {
    if (rdata[ith].mlim>=gen.m)
      {
      idx[nth] = rdata[ith].idx;
      midx[nth] = rdata[ith].midx;
      d.s.cth[nth]=rdata[ith].cth; d.s.sth[nth]=rdata[ith].sth;
      ++nth;
      }
    else
      {
      phase(0, rdata[ith].idx, mi) = phase(0, rdata[ith].midx, mi) = 0;
      phase(1, rdata[ith].idx, mi) = phase(1, rdata[ith].midx, mi) = 0;
      }
    ++ith;
    }
  if (nth>0)
    {
    size_t nvec = (nth+VLEN-1)/VLEN;
    size_t i2 = nvec*VLEN;
    for (size_t i=nth; i<i2; ++i)
      {
      d.s.cth[i]=d.s.cth[nth-1];
      d.s.sth[i]=d.s.sth[nth-1];
      }
    for (size_t i=0; i<nvec; ++i)
      d.v.p1pr[i] = d.v.p1pi[i] = d.v.p2pr[i] = d.v.p2pi[i] =
      d.v.p1mr[i] = d.v.p1mi[i] = d.v.p2mr[i] = d.v.p2mi[i] = 0;

    init_lambda_spin(gen, d.v, nvec);
    }
  }
template<typename T> DUCC0_NOINLINE static void extract_a2m_spin(
  const Ylmgen &gen, size_t mi, sxdata_u &d, const array<size_t, nvx*VLEN> &idx,
  const array<size_t, nvx*VLEN> &midx, const size_t &nth,
  const vmav<complex<T>,3> &phase)
  {
  size_t nvec = (nth+VLEN-1)/VLEN;
  double fct = ((gen.mhi-gen.m+gen.s)&1) ? -1.: 1.;
  for (size_t i=0; i<nvec; ++i)
    {
    auto p1pr = d.v.p1pr[i]-d.v.p2mi[i], p1pi = d.v.p1pi[i]+d.v.p2mr[i],
         p2pr = d.v.p2pr[i]+d.v.p1mi[i], p2pi = d.v.p2pi[i]-d.v.p1mr[i],
         p1mr = d.v.p1mr[i]+d.v.p2pi[i], p1mi = d.v.p1mi[i]-d.v.p2pr[i],
         p2mr = d.v.p2mr[i]-d.v.p1pi[i], p2mi = d.v.p2mi[i]+d.v.p1pr[i];
    d.v.p1pr[i] = p1pr+p2pr;
    d.v.p1pi[i] = p1pi+p2pi;
    d.v.p1mr[i] = p1mr+p2mr;
    d.v.p1mi[i] = p1mi+p2mi;
    d.v.p2pr[i] = fct*(p1pr-p2pr);
    d.v.p2pi[i] = fct*(p1pi-p2pi);
    d.v.p2mr[i] = fct*(p1mr-p2mr);
    d.v.p2mi[i] = fct*(p1mi-p2mi);
    }
  for (size_t i=0; i<nth; ++i)
    {
    phase(0, idx[i], mi) = complex<T>(T(d.s.p1pr[i]), T(d.s.p1pi[i]));
    phase(1, idx[i], mi) = complex<T>(T(d.s.p1mr[i]), T(d.s.p1mi[i]));
    if (idx[i]!=midx[i])
      {
      phase(0, midx[i], mi) = complex<T>(T(d.s.p2pr[i]), T(d.s.p2pi[i]));
      phase(1, midx[i], mi) = complex<T>(T(d.s.p2mr[i]), T(d.s.p2mi[i]));
      }
    }
  }

template<typename T> DUCC0_NOINLINE static void inner_loop_a2m(SHT_mode mode,
  const vmav<complex<double>,2> &almtmp,
  const vmav<complex<T>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi)
  {
  if (gen.s==0)
    {
    // adjust the a_lm for the new algorithm
    MR_assert(almtmp.stride(1)==1, "bad stride");
    dcmplx * DUCC0_RESTRICT alm=almtmp.data();
    for (size_t il=0, l=gen.m; l<=gen.lmax; ++il,l+=2)
      {
      dcmplx al = alm[l];
      dcmplx al1 = (l+1>gen.lmax) ? 0. : alm[l+1];
      dcmplx al2 = (l+2>gen.lmax) ? 0. : alm[l+2];
      alm[l  ] = gen.alpha[il]*(gen.eps[l+1]*al + gen.eps[l+2]*al2);
      alm[l+1] = gen.alpha[il]*al1;
      }

    constexpr size_t nval=nv0*VLEN;
    if (gen.lmax+1-gen.m > lstep)
      {
      vector<s0data_u> v_d;
      vector<array<size_t, nval>> v_idx, v_midx;
      vector<Tbv0> v_cth;
      vector<array<double, nval>> v_wgt;
      vector<size_t> v_nth;

      size_t ith=0;
      while (ith<rdata.size())
        {
        v_d.push_back({}); v_idx.push_back({}); v_midx.push_back({});
        v_cth.push_back({}); v_nth.push_back(0); v_wgt.push_back({});
        fill_a2m(gen, rdata, phase, mi, ith, v_d.back(), v_idx.back(),
                 v_midx.back(), v_cth.back(), v_wgt.back(), v_nth.back());
        if (v_nth.back()==0)
          v_d.pop_back();
        }

      size_t lstart = gen.m;
      while (lstart<=gen.lmax)
        {
        size_t lstop = min(gen.lmax+1, lstart+lstep);
        for (size_t vi=0; vi<v_d.size(); ++vi)
          calc_alm2map (almtmp.data(), gen, v_d[vi].v, v_nth[vi], lstart, lstop);
        lstart = lstop;
        }

      for (size_t vi=0; vi<v_d.size(); ++vi)
        extract_a2m(mi, v_d[vi], v_idx[vi], v_midx[vi], v_nth[vi], v_cth[vi], v_wgt[vi], phase);
      }
    else
      {
      size_t ith=0;
      while (ith<rdata.size())
        {
        s0data_u d;
        array<size_t, nval> idx, midx;
        array<double, nval> wgt;
        Tbv0 cth;
        size_t nth=0;
        while ((nth<nval)&&(ith<rdata.size()))
          {
          fill_a2m(gen, rdata, phase, mi, ith, d, idx, midx, cth, wgt, nth);

          if (nth>0)
            {
            calc_alm2map (almtmp.data(), gen, d.v, nth, gen.m, gen.lmax+1);
            extract_a2m(mi, d, idx, midx, nth, cth, wgt, phase);
            }
          }
        }
      }
    }
  else
    {
    //adjust the a_lm for the new algorithm
    for (size_t l=gen.mhi; l<=gen.lmax+1; ++l)
      for (size_t i=0; i<almtmp.shape(1); ++i)
        almtmp(l,i)*=gen.alpha[l];

    if (gen.lmax+1-gen.mhi > lstep)
      {
      vector<sxdata_u> v_d;
      vector<array<size_t, nvx*VLEN>> v_idx, v_midx;
      vector<size_t> v_nth;
      size_t ith=0;
      while (ith<rdata.size())
        {
        v_d.push_back({}); v_idx.push_back({}); v_midx.push_back({});
        v_nth.push_back(0);
        fill_a2m_spin(gen, rdata, phase, mi, ith, v_d.back(), v_idx.back(), v_midx.back(), v_nth.back());
        if (v_nth.back()==0)
          v_d.pop_back();
        }

      size_t lstart = gen.mhi;
      while (lstart<=gen.lmax)
        {
        size_t lstop = min(gen.lmax+1, lstart+lstep);
        for (size_t vi=0; vi<v_d.size(); ++vi)
          {
          if (mode==STANDARD)
            calc_alm2map_spin(almtmp.data(), gen, v_d[vi].v, v_nth[vi], lstart, lstop);
          else // GRAD_ONLY or DERIV1
            calc_alm2map_spin_gradonly(almtmp.data(), gen, v_d[vi].v, v_nth[vi], lstart, lstop);
          }
        lstart = lstop;
        }

      for (size_t vi=0; vi<v_d.size(); ++vi)
        extract_a2m_spin(gen, mi, v_d[vi], v_idx[vi], v_midx[vi], v_nth[vi], phase);
      }
    else
      {
      sxdata_u d;
      array<size_t, nvx*VLEN> idx, midx;
      size_t nth;
      size_t ith=0;
      while (ith<rdata.size())
        {
        fill_a2m_spin(gen, rdata, phase, mi, ith, d, idx, midx, nth);
        if (nth>0)
          {
          if (mode==STANDARD)
            calc_alm2map_spin(almtmp.data(), gen, d.v, nth, gen.mhi, gen.lmax+1);
          else // GRAD_ONLY or DERIV1
            calc_alm2map_spin_gradonly(almtmp.data(), gen, d.v, nth, gen.mhi, gen.lmax+1);

          extract_a2m_spin(gen, mi, d, idx, midx, nth, phase);
          }
        }
      }
    }
  }

template<typename T> DUCC0_NOINLINE static void fill_m2a(const Ylmgen &gen,
  const vector<ringdata> &rdata, const cmav<complex<T>,3> &phase,
  size_t mi, size_t &ith, s0data_u &d, size_t &nth)
  {
  constexpr size_t nval = nv0*VLEN;
  nth=0;
  while ((nth<nval)&&(ith<rdata.size()))
    {
    if (rdata[ith].mlim>=gen.m)
      {
      if (abs(rdata[ith].cth)>0.99)
        d.s.csq[nth]=(1.-rdata[ith].sth)*(1.+rdata[ith].sth);
      else
        d.s.csq[nth]=rdata[ith].cth*rdata[ith].cth;
      d.s.sth[nth]=rdata[ith].sth;
      double wgt = rdata[ith].wgt;
      dcmplx ph1 = wgt*dcmplx(phase(0, rdata[ith].idx, mi));
      dcmplx ph2 = wgt*dcmplx((rdata[ith].idx==rdata[ith].midx) ? 0 : phase(0, rdata[ith].midx, mi));
      d.s.p1r[nth]=(ph1+ph2).real(); d.s.p1i[nth]=(ph1+ph2).imag();
      d.s.p2r[nth]=(ph1-ph2).real(); d.s.p2i[nth]=(ph1-ph2).imag();
      //adjust for new algorithm
      d.s.p2r[nth]*=rdata[ith].cth;
      d.s.p2i[nth]*=rdata[ith].cth;
      ++nth;
      }
    ++ith;
    }
  if (nth>0)
    {
    size_t nvec = (nth+VLEN-1)/VLEN;
    size_t i2 = nvec*VLEN;
    for (size_t i=nth; i<i2; ++i)
      {
      d.s.csq[i]=d.s.csq[nth-1];
      d.s.sth[i]=d.s.sth[nth-1];
      d.s.p1r[i]=d.s.p1i[i]=d.s.p2r[i]=d.s.p2i[i]=0.;
      }

    init_lambda(gen, d.v, nvec);
    }
  }

template<typename T> DUCC0_NOINLINE static void fill_m2a_spin(const Ylmgen &gen,
  const vector<ringdata> &rdata, const cmav<complex<T>,3> &phase,
  size_t mi, size_t &ith, sxdata_u &d, size_t &nth)
  {
  constexpr size_t nval = nvx*VLEN;
  nth=0;
  while ((nth<nval)&&(ith<rdata.size()))
    {
    if (rdata[ith].mlim>=gen.m)
      {
      d.s.cth[nth]=rdata[ith].cth; d.s.sth[nth]=rdata[ith].sth;
      dcmplx p1Q=phase(0, rdata[ith].idx, mi),
             p1U=phase(1, rdata[ith].idx, mi),
             p2Q=(rdata[ith].idx!=rdata[ith].midx) ? phase(0, rdata[ith].midx, mi):0.,
             p2U=(rdata[ith].idx!=rdata[ith].midx) ? phase(1, rdata[ith].midx, mi):0.;
      if ((gen.mhi-gen.m+gen.s)&1)
        { p2Q=-p2Q; p2U=-p2U; }
      d.s.p1pr[nth]=(p1Q+p2Q).real(); d.s.p1pi[nth]=(p1Q+p2Q).imag();
      d.s.p1mr[nth]=(p1U+p2U).real(); d.s.p1mi[nth]=(p1U+p2U).imag();
      d.s.p2pr[nth]=(p1Q-p2Q).real(); d.s.p2pi[nth]=(p1Q-p2Q).imag();
      d.s.p2mr[nth]=(p1U-p2U).real(); d.s.p2mi[nth]=(p1U-p2U).imag();
      ++nth;
      }
    ++ith;
    }
  if (nth>0)
    {
    size_t nvec = (nth+VLEN-1)/VLEN;
    size_t i2 = nvec*VLEN;
    for (size_t i=nth; i<i2; ++i)
      {
      d.s.cth[i]=d.s.cth[nth-1];
      d.s.sth[i]=d.s.sth[nth-1];
      d.s.p1pr[i]=d.s.p1pi[i]=d.s.p2pr[i]=d.s.p2pi[i]=0.;
      d.s.p1mr[i]=d.s.p1mi[i]=d.s.p2mr[i]=d.s.p2mi[i]=0.;
      }

    init_lambda_spin(gen, d.v, nvec);

    for (size_t i=0; i<nvec; ++i)
      {
      Tv tmp;
      tmp = d.v.p1pr[i]; d.v.p1pr[i] -= d.v.p2mi[i]; d.v.p2mi[i] += tmp;
      tmp = d.v.p1pi[i]; d.v.p1pi[i] += d.v.p2mr[i]; d.v.p2mr[i] -= tmp;
      tmp = d.v.p1mr[i]; d.v.p1mr[i] += d.v.p2pi[i]; d.v.p2pi[i] -= tmp;
      tmp = d.v.p1mi[i]; d.v.p1mi[i] -= d.v.p2pr[i]; d.v.p2pr[i] += tmp;
      }
    }
  }

template<typename T> DUCC0_NOINLINE static void inner_loop_m2a(SHT_mode mode,
  const vmav<complex<double>,2> &almtmp,
  const cmav<complex<T>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi)
  {
  if (gen.s==0)
    {
    if (gen.lmax+1-gen.m > lstep)
      {
      vector<s0data_u> v_d;
      vector<size_t> v_nth;

      size_t ith=0;
      while (ith<rdata.size())
        {
        v_d.push_back({});
        v_nth.push_back(0);
        fill_m2a(gen, rdata, phase, mi, ith, v_d.back(), v_nth.back());
        if (v_nth.back()==0)
          v_d.pop_back();
        }

      size_t lstart = gen.m;
      while (lstart<=gen.lmax)
        {
        size_t lstop = min(gen.lmax+1, lstart+lstep);
        for (size_t vi=0; vi<v_d.size(); ++vi)
          calc_map2alm (almtmp.data(), gen, v_d[vi].v, v_nth[vi], lstart, lstop);
        lstart = lstop;
        }
      }
    else
      {
      s0data_u d;
      size_t nth;

      size_t ith=0;
      while (ith<rdata.size())
        {
        fill_m2a(gen, rdata, phase, mi, ith, d, nth);
        if (nth>0)
          calc_map2alm (almtmp.data(), gen, d.v, nth, gen.m, gen.lmax+1);
        }
      }

    //adjust the a_lm for the new algorithm
    dcmplx * DUCC0_RESTRICT alm=almtmp.data();
    dcmplx alm2 = 0.;
    double alold=0;
    for (size_t il=0, l=gen.m; l<=gen.lmax; ++il,l+=2)
      {
      dcmplx al = alm[l];
      dcmplx al1 = (l+1>gen.lmax) ? 0. : alm[l+1];
      alm[l  ] = gen.alpha[il]*gen.eps[l+1]*al + alold*gen.eps[l]*alm2;
      alm[l+1] = gen.alpha[il]*al1;
      alm2=al;
      alold=gen.alpha[il];
      }
    }
  else
    {
    if (gen.lmax+1-gen.mhi > lstep)
      {
      vector<sxdata_u> v_d;
      vector<size_t> v_nth;
      size_t ith=0;
      while (ith<rdata.size())
        {
        v_d.push_back({});
        v_nth.push_back(0);
        fill_m2a_spin(gen, rdata, phase, mi, ith, v_d.back(), v_nth.back());
        if (v_nth.back()==0)
          v_d.pop_back();
        }

      size_t lstart = gen.mhi;
      while (lstart<=gen.lmax)
        {
        size_t lstop = min(gen.lmax+1, lstart+lstep);
        for (size_t vi=0; vi<v_d.size(); ++vi)
          {
          if (mode==STANDARD)
            calc_map2alm_spin(almtmp.data(), gen, v_d[vi].v, v_nth[vi], lstart, lstop);
          else
            calc_map2alm_spin_gradonly(almtmp.data(), gen, v_d[vi].v, v_nth[vi], lstart, lstop);
          }
        lstart = lstop;
        }
      }
    else
      {
      sxdata_u d;
      size_t nth;
      size_t ith=0;
      while (ith<rdata.size())
        {
        fill_m2a_spin(gen, rdata, phase, mi, ith, d, nth);
        if (nth>0)
          {
          if (mode==STANDARD)
            calc_map2alm_spin(almtmp.data(), gen, d.v, nth, gen.mhi, gen.lmax+1);
          else
            calc_map2alm_spin_gradonly(almtmp.data(), gen, d.v, nth, gen.mhi, gen.lmax+1);
          }
        }
      }

    //adjust the a_lm for the new algorithm
    for (size_t l=gen.mhi; l<=gen.lmax; ++l)
      for (size_t i=0; i<almtmp.shape(1); ++i)
        almtmp(l,i)*=gen.alpha[l];
    }
  }

}

using detail_sht_inner_loop::dcmplx;
using detail_sht_inner_loop::ringdata;
using detail_sht_inner_loop::YlmBase;
using detail_sht_inner_loop::Ylmgen;
using detail_sht_inner_loop::inner_loop_a2m;
using detail_sht_inner_loop::inner_loop_m2a;

}

}

#endif
