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

/*! \file sht.cc
 *  Functionality related to spherical harmonic transforms
 *
 *  Copyright (C) 2020-2021 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <vector>
#include <cmath>
#include <cstring>
#if ((!defined(DUCC0_NO_SIMD)) && defined(__AVX__) && (!defined(__AVX512F__)))
#include <x86intrin.h>
#endif
#include "ducc0/infra/simd.h"
#include "ducc0/sht/sht.h"
#include "ducc0/fft/fft1d.h"
#include "ducc0/math/math_utils.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/math/constants.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

static constexpr double sharp_fbig=0x1p+800,sharp_fsmall=0x1p-800;
static constexpr double sharp_fbighalf=0x1p+400;

struct ringdata
  {
  size_t mlim, idx, midx;
  double cth, sth;
  };

class YlmBase
  {
  public:
    size_t lmax, mmax, s;
    vector<double> powlimit;
    /* used if s==0 */
    vector<double> mfac;

  protected:
    /* used if s==0 */
    vector<double> root, iroot;

    /* used if s!=0 */
    vector<double> flm1, flm2, inv;

  public:
    vector<double> prefac;
    vector<int> fscale;

  protected:
    inline void normalize (double &val, int &scale, double xfmax)
      {
      while (abs(val)>xfmax) { val*=sharp_fsmall; ++scale; }
      if (val!=0.)
        while (abs(val)<xfmax*sharp_fsmall) { val*=sharp_fbig; --scale; }
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
        root((s==0) ? (2*lmax+8) : 0),
        iroot((s==0) ? (2*lmax+8) : 0),
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
        for (size_t i=0; i<2*lmax+8; ++i)
          {
          root[i] = sqrt(i);
          iroot[i] = (i==0) ? 0. : 1./root[i];
          }
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
          normalize(fac[i],facscale[i],sharp_fbighalf);
          }
        for (size_t i=0; i<=mmax; ++i)
          {
          size_t mlo_=min(s,i), mhi_=max(s,i);
          double tfac=fac[2*mhi_]/fac[mhi_+mlo_];
          int tscale=facscale[2*mhi_]-facscale[mhi_+mlo_];
          normalize(tfac,tscale,sharp_fbighalf);
          tfac/=fac[mhi_-mlo_];
          tscale-=facscale[mhi_-mlo_];
          normalize(tfac,tscale,sharp_fbighalf);
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
          eps[l] = root[l+m]*root[l-m]*iroot[2*l+1]*iroot[2*l-1];
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

struct ringhelper
  {
  using dcmplx = complex<double>;
  double phi0_;
  vector<dcmplx> shiftarr;
  size_t s_shift;
  unique_ptr<pocketfft_r<double>> plan;
  vector<double> buf;
  size_t length;
  bool norot;
  ringhelper() : phi0_(0), s_shift(0), length(0), norot(false) {}
  void update(size_t nph, size_t mmax, double phi0)
    {
    norot = (abs(phi0)<1e-14);
    if (!norot)
      if ((mmax!=s_shift-1) || (!approx(phi0,phi0_,1e-15)))
      {
      shiftarr.resize(mmax+1);
      s_shift = mmax+1;
      phi0_ = phi0;
      MultiExp<double, dcmplx> mexp(phi0, mmax+1);
      for (size_t m=0; m<=mmax; ++m)
        shiftarr[m] = mexp[m];
      }
    if (nph!=length)
      {
      plan=make_unique<pocketfft_r<double>>(nph);
      buf.resize(plan->bufsize());
      length=nph;
      }
    }
  template<typename T> DUCC0_NOINLINE void phase2ring (size_t nph,
    double phi0, vmav<double,1> &data, size_t mmax, const cmav<complex<T>,1> &phase)
    {
    update (nph, mmax, phi0);

    if (nph>=2*mmax+1)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          {
          data(2*m)=phase(m).real();
          data(2*m+1)=phase(m).imag();
          }
      else
        for (size_t m=0; m<=mmax; ++m)
          {
          dcmplx tmp = dcmplx(phase(m))*shiftarr[m];
          data(2*m)=tmp.real();
          data(2*m+1)=tmp.imag();
          }
      for (size_t m=2*(mmax+1); m<nph+2; ++m)
        data(m)=0.;
      }
    else
      {
      data(0)=phase(0).real();
      fill(&data(1),&data(nph+2),0.);

      for (size_t m=1, idx1=1, idx2=nph-1; m<=mmax; ++m,
           idx1=(idx1+1==nph) ? 0 : idx1+1, idx2=(idx2==0) ? nph-1 : idx2-1)
        {
        dcmplx tmp = phase(m);
        if(!norot) tmp*=shiftarr[m];
        if (idx1<(nph+2)/2)
          {
          data(2*idx1)+=tmp.real();
          data(2*idx1+1)+=tmp.imag();
          }
        if (idx2<(nph+2)/2)
          {
          data(2*idx2)+=tmp.real();
          data(2*idx2+1)-=tmp.imag();
          }
        }
      }
    data(1)=data(0);
    plan->exec_copyback(&(data(1)), buf.data(), 1., false);
    }
  template<typename T> DUCC0_NOINLINE void ring2phase (size_t nph, double phi0,
    vmav<double,1> &data, size_t mmax, vmav<complex<T>,1> &phase)
    {
    update (nph, mmax, -phi0);

    plan->exec_copyback(&(data(1)), buf.data(), 1., true);
    data(0)=data(1);
    data(1)=data(nph+1)=0.;

    if (mmax<=nph/2)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          phase(m) = complex<T>(T(data(2*m)), T(data(2*m+1)));
      else
        for (size_t m=0; m<=mmax; ++m)
          phase(m) = complex<T>(dcmplx(data(2*m), data(2*m+1)) * shiftarr[m]);
      }
    else
      {
      for (size_t m=0, idx=0; m<=mmax; ++m, idx=(idx+1==nph) ? 0 : idx+1)
        {
        dcmplx val;
        if (idx<(nph-idx))
          val = dcmplx(data(2*idx), data(2*idx+1));
        else
          val = dcmplx(data(2*(nph-idx)), -data(2*(nph-idx)+1));
        if (!norot)
          val *= shiftarr[m];
        phase(m)=complex<T>(val);
        }
      }
    }
  };


using Tv=native_simd<double>;
static constexpr size_t VLEN=Tv::size();

#if ((!defined(DUCC0_NO_SIMD)) && defined(__AVX__) && (!defined(__AVX512F__)))
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

static constexpr double sharp_ftol=0x1p-60;

constexpr size_t nv0 = 128/VLEN;
constexpr size_t nvx = 64/VLEN;

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
#if defined(_MSC_VER)
  s0data_u() {}
#endif
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
#if defined(_MSC_VER)
  sxdata_u() {}
#endif
  };

static inline void Tvnormalize (Tv & DUCC0_RESTRICT val_,
  Tv & DUCC0_RESTRICT scale_, double maxval)
  {
  // This copying is necessary for MSVC ... no idea why
  Tv val = val_;
  Tv scale = scale_;
  const Tv vfmin=sharp_fsmall*maxval, vfmax=maxval;
  const Tv vfsmall=sharp_fsmall, vfbig=sharp_fbig;
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
    Tvnormalize(val,scaleint,sharp_fbighalf);
    do
      {
      if (npow&1)
        {
        res*=val;
        scale+=scaleint;
        Tvnormalize(res,scale,sharp_fbighalf);
        }
      val*=val;
      scaleint+=scaleint;
      Tvnormalize(val,scaleint,sharp_fbighalf);
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
    corfac[i] = (scale[i]<0) ? 0. : ((scale[i]<1) ? 1. : sharp_fbig);
#else
  corfac = Tv(1.);
  where(scale<-0.5,corfac)=0;
  where(scale>0.5,corfac)=sharp_fbig;
#endif
  }

static inline bool rescale(Tv &v1, Tv &v2, Tv &s, Tv eps)
  {
  auto mask = abs(v2)>eps;
  if (any_of(mask))
    {
    where(mask,v1)*=sharp_fsmall;
    where(mask,v2)*=sharp_fsmall;
    where(mask,s)+=1;
    return true;
    }
  return false;
  }

DUCC0_NOINLINE static void iter_to_ieee(const Ylmgen &gen,
  s0data_v & DUCC0_RESTRICT d, size_t & DUCC0_RESTRICT l_, size_t & DUCC0_RESTRICT il_, size_t nv2)
  {
  size_t l=gen.m, il=0;
  Tv mfac = (gen.m&1) ? -gen.mfac[gen.m]:gen.mfac[gen.m];
  bool below_limit = true;
  for (size_t i=0; i<nv2; ++i)
    {
    d.lam1[i]=0;
    mypow(d.sth[i],gen.m,gen.powlimit,d.lam2[i],d.scale[i]);
    d.lam2[i] *= mfac;
    Tvnormalize(d.lam2[i],d.scale[i],sharp_ftol);
    below_limit &= all_of(d.scale[i]<1);
    }

  while (below_limit)
    {
    if (l+4>gen.lmax) {l_=gen.lmax+1;return;}
    below_limit=1;
    Tv a1=gen.coef[il  ].a, b1=gen.coef[il  ].b;
    Tv a2=gen.coef[il+1].a, b2=gen.coef[il+1].b;
    for (size_t i=0; i<nv2; ++i)
      {
      d.lam1[i] = (a1*d.csq[i] + b1)*d.lam2[i] + d.lam1[i];
      d.lam2[i] = (a2*d.csq[i] + b2)*d.lam1[i] + d.lam2[i];
      if (rescale(d.lam1[i], d.lam2[i], d.scale[i], sharp_ftol))
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
  const Ylmgen &gen, s0data_v & DUCC0_RESTRICT d, size_t nth)
  {
  size_t l,il=0,lmax=gen.lmax;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee(gen, d, l, il, nv2);
  if (l>lmax) return;

  auto &coef = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scale[i], d.corfac[i]);
    full_ieee &= all_of(d.scale[i]>=0);
    }

  while((!full_ieee) && (l<=lmax))
    {
    Tv ar1=alm[l  ].real(), ai1=alm[l  ].imag();
    Tv ar2=alm[l+1].real(), ai2=alm[l+1].imag();
    Tv a=coef[il].a, b=coef[il].b;
    full_ieee=1;
    for (size_t i=0; i<nv2; ++i)
      {
      d.p1r[i] += d.lam2[i]*d.corfac[i]*ar1;
      d.p1i[i] += d.lam2[i]*d.corfac[i]*ai1;
      d.p2r[i] += d.lam2[i]*d.corfac[i]*ar2;
      d.p2i[i] += d.lam2[i]*d.corfac[i]*ai2;
      Tv tmp = (a*d.csq[i] + b)*d.lam2[i] + d.lam1[i];
      d.lam1[i] = d.lam2[i];
      d.lam2[i] = tmp;
      if (rescale(d.lam1[i], d.lam2[i], d.scale[i], sharp_ftol))
        getCorfac(d.scale[i], d.corfac[i]);
      full_ieee &= all_of(d.scale[i]>=0);
      }
    l+=2; ++il;
    }
  if (l>lmax) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.lam1[i] *= d.corfac[i];
    d.lam2[i] *= d.corfac[i];
    }
  alm2map_kernel(d, coef, alm, l, il, lmax, nv2);
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
  const Ylmgen &gen, s0data_v & DUCC0_RESTRICT d, size_t nth)
  {
  size_t l,il=0,lmax=gen.lmax;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee(gen, d, l, il, nv2);
  if (l>lmax) return;

  auto &coef = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scale[i], d.corfac[i]);
    full_ieee &= all_of(d.scale[i]>=0);
    }

  while((!full_ieee) && (l<=lmax))
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
      if (rescale(d.lam1[i], d.lam2[i], d.scale[i], sharp_ftol))
        getCorfac(d.scale[i], d.corfac[i]);
      full_ieee &= all_of(d.scale[i]>=0);
      }
    vhsum_cmplx_special (atmp[0], atmp[1], atmp[2], atmp[3], &alm[l]);
    l+=2; ++il;
    }
  if (l>lmax) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.lam1[i] *= d.corfac[i];
    d.lam2[i] *= d.corfac[i];
    }
  map2alm_kernel(d, coef, alm, l, il, lmax, nv2);
  }

DUCC0_NOINLINE static void iter_to_ieee_spin (const Ylmgen &gen,
  sxdata_v & DUCC0_RESTRICT d, size_t & DUCC0_RESTRICT l_, size_t nv2)
  {
  const auto &fx = gen.coef;
  Tv prefac=gen.prefac[gen.m],
     prescale=gen.fscale[gen.m];
  bool below_limit=true;
  for (size_t i=0; i<nv2; ++i)
    {
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
    Tvnormalize(d.l2m[i],d.scm[i],sharp_fbighalf);
    Tvnormalize(d.l2p[i],d.scp[i],sharp_fbighalf);
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

    Tvnormalize(d.l2m[i],d.scm[i],sharp_ftol);
    Tvnormalize(d.l2p[i],d.scp[i],sharp_ftol);

    below_limit &= all_of(d.scm[i]<1) &&
                   all_of(d.scp[i]<1);
    }

  size_t l=gen.mhi;

  while (below_limit)
    {
    if (l+2>gen.lmax) {l_=gen.lmax+1;return;}
    below_limit=1;
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    for (size_t i=0; i<nv2; ++i)
      {
      d.l1p[i] = (d.cth[i]*fx10 - fx11)*d.l2p[i] - d.l1p[i];
      d.l1m[i] = (d.cth[i]*fx10 + fx11)*d.l2m[i] - d.l1m[i];
      d.l2p[i] = (d.cth[i]*fx20 - fx21)*d.l1p[i] - d.l2p[i];
      d.l2m[i] = (d.cth[i]*fx20 + fx21)*d.l1m[i] - d.l2m[i];
      if (rescale(d.l1p[i],d.l2p[i],d.scp[i],sharp_ftol) |
          rescale(d.l1m[i],d.l2m[i],d.scm[i],sharp_ftol))
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
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth)
  {
  size_t l,lmax=gen.lmax;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2);
  if (l>lmax) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }

  while((!full_ieee) && (l<=lmax))
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
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sharp_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sharp_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    l+=2;
    }
//  if (l>lmax) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    }
  alm2map_spin_kernel(d, fx, alm, l, lmax, nv2);

  for (size_t i=0; i<nv2; ++i)
    {
    Tv tmp;
    tmp = d.p1pr[i]; d.p1pr[i] -= d.p2mi[i]; d.p2mi[i] += tmp;
    tmp = d.p1pi[i]; d.p1pi[i] += d.p2mr[i]; d.p2mr[i] -= tmp;
    tmp = d.p1mr[i]; d.p1mr[i] += d.p2pi[i]; d.p2pi[i] -= tmp;
    tmp = d.p1mi[i]; d.p1mi[i] -= d.p2pr[i]; d.p2pr[i] += tmp;
    }
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
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth)
  {
  size_t l,lmax=gen.lmax;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2);
  if (l>lmax) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }
  for (size_t i=0; i<nv2; ++i)
    {
    Tv tmp;
    tmp = d.p1pr[i]; d.p1pr[i] -= d.p2mi[i]; d.p2mi[i] += tmp;
    tmp = d.p1pi[i]; d.p1pi[i] += d.p2mr[i]; d.p2mr[i] -= tmp;
    tmp = d.p1mr[i]; d.p1mr[i] += d.p2pi[i]; d.p2pi[i] -= tmp;
    tmp = d.p1mi[i]; d.p1mi[i] -= d.p2pr[i]; d.p2pr[i] += tmp;
    }

  while((!full_ieee) && (l<=lmax))
    {
    Tv fx10=fx[l+1].a,fx11=fx[l+1].b;
    Tv fx20=fx[l+2].a,fx21=fx[l+2].b;
    Tv agr1=0, agi1=0, acr1=0, aci1=0;
    Tv agr2=0, agi2=0, acr2=0, aci2=0;
    full_ieee=1;
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
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sharp_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sharp_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    vhsum_cmplx_special (agr1,agi1,acr1,aci1,&alm[2*l]);
    vhsum_cmplx_special (agr2,agi2,acr2,aci2,&alm[2*l+2]);
    l+=2;
    }
  if (l>lmax) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    }
  map2alm_spin_kernel(d, fx, alm, l, lmax, nv2);
  }


DUCC0_NOINLINE static void alm2map_deriv1_kernel(sxdata_v & DUCC0_RESTRICT d,
  const vector<Ylmgen::dbl2> &fx, const dcmplx * DUCC0_RESTRICT alm,
  size_t l, size_t lmax, size_t nv2)
  {
  size_t lsave=l;
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

DUCC0_NOINLINE static void calc_alm2map_deriv1(const dcmplx * DUCC0_RESTRICT alm,
  const Ylmgen &gen, sxdata_v & DUCC0_RESTRICT d, size_t nth)
  {
  size_t l,lmax=gen.lmax;
  size_t nv2 = (nth+VLEN-1)/VLEN;
  iter_to_ieee_spin(gen, d, l, nv2);
  if (l>lmax) return;

  const auto &fx = gen.coef;
  bool full_ieee=true;
  for (size_t i=0; i<nv2; ++i)
    {
    getCorfac(d.scp[i], d.cfp[i]);
    getCorfac(d.scm[i], d.cfm[i]);
    full_ieee &= all_of(d.scp[i]>=0) &&
                 all_of(d.scm[i]>=0);
    }

  while((!full_ieee) && (l<=lmax))
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
      if (rescale(d.l1p[i], d.l2p[i], d.scp[i], sharp_ftol))
        getCorfac(d.scp[i], d.cfp[i]);
      full_ieee &= all_of(d.scp[i]>=0);
      if (rescale(d.l1m[i], d.l2m[i], d.scm[i], sharp_ftol))
        getCorfac(d.scm[i], d.cfm[i]);
      full_ieee &= all_of(d.scm[i]>=0);
      }
    l+=2;
    }
//  if (l>lmax) return;

  for (size_t i=0; i<nv2; ++i)
    {
    d.l1p[i] *= d.cfp[i];
    d.l2p[i] *= d.cfp[i];
    d.l1m[i] *= d.cfm[i];
    d.l2m[i] *= d.cfm[i];
    }
  alm2map_deriv1_kernel(d, fx, alm, l, lmax, nv2);

  for (size_t i=0; i<nv2; ++i)
    {
    Tv tmp;
    tmp = d.p1pr[i]; d.p1pr[i] -= d.p2mi[i]; d.p2mi[i] += tmp;
    tmp = d.p1pi[i]; d.p1pi[i] += d.p2mr[i]; d.p2mr[i] -= tmp;
    tmp = d.p1mr[i]; d.p1mr[i] += d.p2pi[i]; d.p2pi[i] -= tmp;
    tmp = d.p1mi[i]; d.p1mi[i] -= d.p2pr[i]; d.p2pr[i] += tmp;
    }
  }


template<typename T> DUCC0_NOINLINE static void inner_loop_a2m(SHT_mode mode,
  vmav<complex<double>,2> &almtmp,
  vmav<complex<T>,3> &phase, const vector<ringdata> &rdata,
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
    s0data_u d;
    array<size_t, nval> idx, midx;
    Tbv0 cth;
    size_t ith=0;
    while (ith<rdata.size())
      {
      size_t nth=0;
      while ((nth<nval)&&(ith<rdata.size()))
        {
        if (rdata[ith].mlim>=gen.m)
          {
          idx[nth] = rdata[ith].idx;
          midx[nth] = rdata[ith].midx;
          auto lcth = rdata[ith].cth;
          cth[nth/VLEN][nth%VLEN] = lcth;
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
        calc_alm2map (almtmp.data(), gen, d.v, nth);
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
          phase(0, idx[i], mi) = complex<T>(T(d.s.p1r[i]),T(d.s.p1i[i]));
          if (idx[i]!=midx[i])
            phase(0, midx[i], mi) = complex<T>(T(d.s.p2r[i]),T(d.s.p2i[i]));
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

    constexpr size_t nval=nvx*VLEN;
    sxdata_u d;
    array<size_t, nval> idx, midx;
    size_t ith=0;
    while (ith<rdata.size())
      {
      size_t nth=0;
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
        (mode==ALM2MAP) ?
          calc_alm2map_spin  (almtmp.data(), gen, d.v, nth) :
          calc_alm2map_deriv1(almtmp.data(), gen, d.v, nth);
        double fct = ((gen.mhi-gen.m+gen.s)&1) ? -1.: 1.;
        for (size_t i=0; i<nvec; ++i)
          {
          auto p1pr=d.v.p1pr[i], p1pi=d.v.p1pi[i],
               p2pr=d.v.p2pr[i], p2pi=d.v.p2pi[i],
               p1mr=d.v.p1mr[i], p1mi=d.v.p1mi[i],
               p2mr=d.v.p2mr[i], p2mi=d.v.p2mi[i];
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
          dcmplx q1(d.s.p1pr[i], d.s.p1pi[i]),
                 q2(d.s.p2pr[i], d.s.p2pi[i]),
                 u1(d.s.p1mr[i], d.s.p1mi[i]),
                 u2(d.s.p2mr[i], d.s.p2mi[i]);
          phase(0, idx[i], mi) = complex<T>(T(d.s.p1pr[i]), T(d.s.p1pi[i]));
          phase(1, idx[i], mi) = complex<T>(T(d.s.p1mr[i]), T(d.s.p1mi[i]));
          if (idx[i]!=midx[i])
            {
            phase(0, midx[i], mi) = complex<T>(T(d.s.p2pr[i]), T(d.s.p2pi[i]));
            phase(1, midx[i], mi) = complex<T>(T(d.s.p2mr[i]), T(d.s.p2mi[i]));
            }
          }
        }
      }
    }
  }

template<typename T> DUCC0_NOINLINE static void inner_loop_m2a(
  vmav<complex<double>,2> &almtmp,
  const cmav<complex<T>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi)
  {
  if (gen.s==0)
    {
    constexpr size_t nval=nv0*VLEN;
    size_t ith=0;
    while (ith<rdata.size())
      {
      s0data_u d;
      size_t nth=0;
      while ((nth<nval)&&(ith<rdata.size()))
        {
        if (rdata[ith].mlim>=gen.m)
          {
          d.s.csq[nth]=rdata[ith].cth*rdata[ith].cth; d.s.sth[nth]=rdata[ith].sth;
          dcmplx ph1=phase(0, rdata[ith].idx, mi);
          dcmplx ph2=(rdata[ith].idx==rdata[ith].midx) ? 0 : phase(0, rdata[ith].midx, mi);
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
        size_t i2=((nth+VLEN-1)/VLEN)*VLEN;
        for (size_t i=nth; i<i2; ++i)
          {
          d.s.csq[i]=d.s.csq[nth-1];
          d.s.sth[i]=d.s.sth[nth-1];
          d.s.p1r[i]=d.s.p1i[i]=d.s.p2r[i]=d.s.p2i[i]=0.;
          }
        calc_map2alm (almtmp.data(), gen, d.v, nth);
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
    constexpr size_t nval=nvx*VLEN;
    size_t ith=0;
    while (ith<rdata.size())
      {
      sxdata_u d;
      size_t nth=0;
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
        size_t i2=((nth+VLEN-1)/VLEN)*VLEN;
        for (size_t i=nth; i<i2; ++i)
          {
          d.s.cth[i]=d.s.cth[nth-1];
          d.s.sth[i]=d.s.sth[nth-1];
          d.s.p1pr[i]=d.s.p1pi[i]=d.s.p2pr[i]=d.s.p2pi[i]=0.;
          d.s.p1mr[i]=d.s.p1mi[i]=d.s.p2mr[i]=d.s.p2mi[i]=0.;
          }
        calc_map2alm_spin(almtmp.data(), gen, d.v, nth);
        }
      }
    //adjust the a_lm for the new algorithm
    for (size_t l=gen.mhi; l<=gen.lmax; ++l)
      {
      almtmp(l,0)*=gen.alpha[l];
      almtmp(l,1)*=gen.alpha[l];
      }
    }
  }

size_t get_mmax(const cmav<size_t,1> &mval, size_t lmax)
  {
  size_t nm=mval.shape(0);
  size_t mmax=0;
  vector<bool> present(lmax+1, false);
  for (size_t mi=0; mi<nm; ++mi)
    {
    size_t m=mval(mi);
    MR_assert(m<=lmax, "mmax too large");
    MR_assert(!present[m], "m value present more than once");
    present[m]=true;
    mmax = max(mmax,m);
    }
  return mmax;
  }

DUCC0_NOINLINE size_t get_mlim (size_t lmax, size_t spin, double sth, double cth)
  {
  double ofs=lmax*0.01;
  if (ofs<100.) ofs=100.;
  double b = -2*double(spin)*abs(cth);
  double t1 = lmax*sth+ofs;
  double c = double(spin)*spin-t1*t1;
  double discr = b*b-4*c;
  if (discr<=0) return lmax;
  double res=(-b+sqrt(discr))/2.;
  res = min(res, double(lmax));
  return size_t(res+0.5);
  }

vector<ringdata> make_ringdata(const cmav<double,1> &theta, size_t lmax,
  size_t spin)
  {
  size_t nrings = theta.shape(0);
  struct ringinfo
    {
    double theta, cth, sth;
    size_t idx;
    };
  vector<ringinfo> tmp(nrings);
  for (size_t i=0; i<nrings; ++i)
    tmp[i] = { theta(i), cos(theta(i)), sin(theta(i)), i };
  sort(tmp.begin(), tmp.end(), [](const ringinfo &a, const ringinfo &b)
    { return (a.sth<b.sth); });

  vector<ringdata> res;
  size_t pos=0;
  while (pos<nrings)
    {
    if ((pos+1<nrings) && approx(tmp[pos].cth,-tmp[pos+1].cth,1e-12))
      {
      double cth = (tmp[pos].theta<tmp[pos+1].theta) ? tmp[pos].cth : -tmp[pos+1].cth;
      double sth = (tmp[pos].theta<tmp[pos+1].theta) ? tmp[pos].sth :  tmp[pos+1].sth;
      res.push_back({get_mlim(lmax, spin, sth, cth), tmp[pos].idx, tmp[pos+1].idx, cth, sth});
      pos += 2;
      }
    else
      {
      res.push_back({get_mlim(lmax, spin, tmp[pos].sth, tmp[pos].cth),
        tmp[pos].idx, tmp[pos].idx, tmp[pos].cth, tmp[pos].sth});
      ++pos;
      }
    }
  return res;
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
static vector<double> get_dh_weights(size_t nrings)
  {
  vector<double> weight(nrings);

  weight[0]=2.;
  for (size_t k=1; k<=(nrings/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k);
  weight[2*(nrings/2)-1]=(nrings-3.)/(2*(nrings/2)-1) -1.;
  pocketfft_r<double> plan(nrings);
  plan.exec(weight.data(), 1., false);
  weight[0] = 0.;  // ensure that this is an exact zero
  return weight;
  }

void get_gridweights(const string &type, vmav<double,1> &wgt)
  {
  size_t nrings=wgt.shape(0);
  if (type=="GL") // Gauss-Legendre
    {
    ducc0::GL_Integrator integ(nrings);
    auto xwgt = integ.weights();
    for (size_t m=0; m<nrings; ++m)
      wgt(m) = 2*pi*xwgt[m];
    }
  else if (type=="F1") // Fejer 1
    {
    /* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
    vector<double> xwgt(nrings);
    xwgt[0]=2.;
    UnityRoots<double,dcmplx> roots(2*nrings);
    for (size_t k=1; k<=(nrings-1)/2; ++k)
      {
      auto tmp = roots[k];
      xwgt[2*k-1]=2./(1.-4.*k*k)*tmp.real();
      xwgt[2*k  ]=2./(1.-4.*k*k)*tmp.imag();
      }
    if ((nrings&1)==0) xwgt[nrings-1]=0.;
    pocketfft_r<double> plan(nrings);
    plan.exec(xwgt.data(), 1., false);
    for (size_t m=0; m<(nrings+1)/2; ++m)
      wgt(m)=wgt(nrings-1-m)=xwgt[m]*2*pi/nrings;
    }
  else if (type=="CC") // Clenshaw-Curtis
    {
    /* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
    MR_assert(nrings>2, "too few rings for Clenshaw-Curtis grid");
    size_t n=nrings-1;
    double dw=-1./(n*n-1.+(n&1));
    vector<double> xwgt(nrings);
    xwgt[0]=2.+dw;
    for (size_t k=1; k<=(n/2-1); ++k)
      xwgt[2*k-1]=2./(1.-4.*k*k) + dw;
    //FIXME if (n>1) ???
    xwgt[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
    pocketfft_r<double> plan(n);
    plan.exec(xwgt.data(), 1., false);
    for (size_t m=0; m<(nrings+1)/2; ++m)
      wgt(m)=wgt(nrings-1-m)=xwgt[m]*2*pi/n;
    }
  else if (type=="F2") // Fejer 2
    {
    auto xwgt = get_dh_weights(nrings+1);
    for (size_t m=0; m<nrings; ++m)
      wgt(m) = xwgt[m+1]*2*pi/(nrings+1);
    }
  else if (type=="DH") // Driscoll-Healy
    {
    auto xwgt = get_dh_weights(nrings);
    for (size_t m=0; m<nrings; ++m)
      wgt(m) = xwgt[m]*2*pi/nrings;
    }
  else
    MR_fail("unsupported grid type");
  }

vmav<double,1> get_gridweights(const string &type, size_t nrings)
  {
  vmav<double,1> wgt({nrings});
  get_gridweights(type, wgt);
  return wgt;
  }


bool downsampling_ok(const cmav<double,1> &theta, size_t lmax,
  bool &npi, bool &spi, size_t &ntheta_out)
  {
  size_t ntheta = theta.shape(0);
  if (ntheta<=500) return false; // not worth thinking about shortcuts
  npi = abs_approx(theta(0), 0., 1e-14);
  spi = abs_approx(theta(ntheta-1), pi, 1e-14);
  size_t nthetafull = 2*ntheta-npi-spi;
  double dtheta = 2*pi/nthetafull;
  for (size_t i=0; i<ntheta; ++i)
    if (!abs_approx(theta(i),(0.5*(1-npi)+i)*dtheta, 1e-14))
      return false;
  size_t npairs = ntheta*(2-(npi==spi))/2;
  ntheta_out = good_size_complex(lmax+1)+1;
  if (2*npairs<1.2*ntheta_out)  // not worth taking the shortcut
    return false;
  return true;
  }

template<typename T> void resample_theta(const cmav<complex<T>,3> &legi, bool npi, bool spi,
  vmav<complex<T>,3> &lego, bool npo, bool spo, size_t spin, size_t nthreads, bool adjoint)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  if ((npi==npo)&&(spi==spo)&&(legi.shape(1)==lego.shape(1)))  // shortcut
    {
    mav_apply([](complex<T> &a, complex<T> b) {a=b;}, nthreads, lego, legi);
    return;
    }
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-npi-spi;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-npo-spo;
  auto dthi = T(2*pi/nfull_in);
  auto dtho = T(2*pi/nfull_out);
  auto shift = T(0.5*(dtho*(1-npo)-dthi*(1-npi)));
  size_t nfull = max(nfull_in, nfull_out);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(nfull_in), plan_out(nfull_out);
  MultiExp<T,complex<T>> phase(adjoint ? -shift : shift, (shift==0.) ? 1 : nrings_in+2);
  execDynamic((nm+1)/2, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({nfull}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), plan_out.bufsize())}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{2*rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{2*rng.lo,MAXIDX}}));
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          for (size_t i=0, im=nfull_in-1+npi; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,2*j);
            complex<T> v2 = ((2*j+1)<llegi.shape(1)) ? llegi(i,2*j+1) : 0;
            tmp(i) = v1 + v2;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct * (v1-v2);
            else
              tmp(i) = (adjoint ? T(1) : T(0.5)) * (tmp(i) + fct*(v1-v2)); // sic!
            }
          plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), !adjoint);
          if (shift!=0)
            for (size_t i=1, im=nfull_in-1; (i<nrings_in+1)&&(i<=im); ++i,--im)
              {
              if (i!=im)
                tmp(i) *= phase[i];
              tmp(im) *= conj(phase[i]);
              }

          // zero padding/truncation
          if (nfull_out>nfull_in) // pad
            {
            size_t dist = nfull_out-nfull_in;
            size_t nmove = nfull_in/2;
            for (size_t i=nfull_out-1; i>nfull_out-1-nmove; --i)
              tmp(i) = tmp(i-dist);
            for (size_t i=nfull_out-nmove-dist; i<nfull_out-nmove; ++i)
              tmp(i) = 0;
            }
          if (nfull_out<nfull_in) // truncate
            {
            size_t dist = nfull_in-nfull_out;
            size_t nmove = nfull_out/2;
            for (size_t i=nfull_in-nmove; i<nfull_in; ++i)
              tmp(i-dist) = tmp(i);
            }
          plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), adjoint);
          auto norm = T(1./(2*(adjoint ? nfull_out : nfull_in)));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-1+npo-i;
            if (im==nfull_out) im=0;
            T fct2 = (adjoint && (im==i)) ? T(0.5) : 1;
            complex<T> v1 = fct2*tmp(i);
            complex<T> v2 = fct2*fct*tmp(im);
            llego(i,2*j) = norm * (v1 + v2);
            if ((2*j+1)<llego.shape(1))
              llego(i,2*j+1) = norm * (v1 - v2);
            }
          }
        }
      }
    });
  }

template<typename T> void alm2leg(  // associated Legendre transform
  const cmav<complex<T>,2> &alm, // (ncomp, lmidx)
  vmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode)
  {
  // sanity checks
  auto nrings=theta.shape(0);
  MR_assert(nrings==leg.shape(1), "nrings mismatch");
  auto nm=mval.shape(0);
  MR_assert(nm==mstart.shape(0), "nm mismatch");
  MR_assert(nm==leg.shape(2), "nm mismatch");
  auto nalm=alm.shape(0);
  auto mmax = get_mmax(mval, lmax);
  if (mode==ALM2MAP_DERIV1)
    {
    spin=1;
    MR_assert(nalm==1, "need one a_lm component");
    MR_assert(leg.shape(0)==2, "need two Legendre components");
    }
  else
    {
    size_t ncomp = (spin==0) ? 1 : 2;
    MR_assert(nalm==ncomp, "incorrect number of a_lm components");
    MR_assert(leg.shape(0)==ncomp, "incorrect number of Legendre components");
    }

  bool npi, spi;
  size_t ntheta_tmp;
  if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
    {
    vmav<double,1> theta_tmp({ntheta_tmp});
    for (size_t i=0; i<ntheta_tmp; ++i)
      theta_tmp(i) = i*pi/(ntheta_tmp-1);
    if (ntheta_tmp<=nrings)
      {
      auto leg_tmp(subarray<3>(leg, {{},{0,ntheta_tmp},{}}));
      alm2leg(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode);
      resample_theta(leg_tmp, true, true, leg, npi, spi, spin, nthreads, false);
      }
    else
      {
//FIXME noncrit?
      vmav<complex<T>,3> leg_tmp({leg.shape(0),ntheta_tmp,leg.shape(2)});
      alm2leg(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode);
      resample_theta(leg_tmp, true, true, leg, npi, spi, spin, nthreads, false);
      }
    return;
    }

  auto norm_l = (mode==ALM2MAP_DERIV1) ? Ylmgen::get_d1norm (lmax) :
                                         Ylmgen::get_norm (lmax, spin);
  auto rdata = make_ringdata(theta, lmax, spin);
  YlmBase base(lmax, mmax, spin);

  ducc0::execDynamic(nm, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    Ylmgen gen(base);
    vmav<complex<double>,2> almtmp({lmax+2,nalm});

    while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
      {
      auto m=mval(mi);
      auto lmin=max(spin,m);
      for (size_t ialm=0; ialm<nalm; ++ialm)
        {
        for (size_t l=m; l<lmin; ++l)
          almtmp(l,ialm) = 0;
        for (size_t l=lmin; l<=lmax; ++l)
          almtmp(l,ialm) = alm(ialm,mstart(mi)+l*lstride)*T(norm_l[l]);
        almtmp(lmax+1,ialm) = 0;
        }
      gen.prepare(m);
      inner_loop_a2m (mode, almtmp, leg, rdata, gen, mi);
      }
    }); /* end of parallel region */
  }

template<typename T> void leg2alm(  // associated Legendre transform
  vmav<complex<T>,2> &alm, // (ncomp, lmidx)
  const cmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads)
  {
  // sanity checks
  auto nrings=theta.shape(0);
  MR_assert(nrings==leg.shape(1), "nrings mismatch");
  auto nm=mval.shape(0);
  MR_assert(nm==mstart.shape(0), "nm mismatch");
  MR_assert(nm==leg.shape(2), "nm mismatch");
  auto mmax = get_mmax(mval, lmax);
  size_t ncomp = (spin==0) ? 1 : 2;
  MR_assert(alm.shape(0)==ncomp, "incorrect number of a_lm components");
  MR_assert(leg.shape(0)==ncomp, "incorrect number of Legendre components");

  bool npi, spi;
  size_t ntheta_tmp;
  if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
    {
    vmav<double,1> theta_tmp({ntheta_tmp});
    for (size_t i=0; i<ntheta_tmp; ++i)
      theta_tmp(i) = i*pi/(ntheta_tmp-1);
    auto leg_tmp(vmav<complex<T>,3>::build_noncritical({leg.shape(0), ntheta_tmp, leg.shape(2)}));
    resample_theta(leg, npi, spi, leg_tmp, true, true, spin, nthreads, true);
    leg2alm(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads);
    return;
    }

  auto norm_l = Ylmgen::get_norm (lmax, spin);
  auto rdata = make_ringdata(theta, lmax, spin);
  YlmBase base(lmax, mmax, spin);

  ducc0::execDynamic(nm, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    Ylmgen gen(base);
    vmav<complex<double>,2> almtmp({lmax+2,ncomp});

    while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
      {
      auto m=mval(mi);
      gen.prepare(m);
      for (size_t l=m; l<almtmp.shape(0); ++l)
        for (size_t ialm=0; ialm<ncomp; ++ialm)
          almtmp(l,ialm) = 0.;
      inner_loop_m2a (almtmp, leg, rdata, gen, mi);
      auto lmin=max(spin,m);
      for (size_t l=m; l<lmin; ++l)
        for (size_t ialm=0; ialm<ncomp; ++ialm)
          alm(ialm,mstart(mi)+l*lstride) = 0;
      for (size_t l=lmin; l<=lmax; ++l)
        for (size_t ialm=0; ialm<ncomp; ++ialm)
          alm(ialm,mstart(mi)+l*lstride) = complex<T>(almtmp(l,ialm)*norm_l[l]);
      }
    }); /* end of parallel region */
  }

template<typename T> void leg2map(  // FFT
  vmav<T,2> &map, // (ncomp, pix)
  const cmav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t ncomp=map.shape(0);
  MR_assert(ncomp==leg.shape(0), "number of components mismatch");
  size_t nrings=leg.shape(1);
  MR_assert(nrings>=1, "need at least one ring");
  MR_assert((nrings==nphi.shape(0)) && (nrings==ringstart.shape(0))
         && (nrings==phi0.shape(0)), "inconsistent number of rings");
  size_t nphmax=0;
  for (size_t i=0; i<nrings; ++i)
    nphmax=max(nphi(i),nphmax);
  MR_assert(leg.shape(2)>=1, "bad mmax");
  size_t mmax=leg.shape(2)-1;
  execDynamic(nrings, nthreads, 64, [&](Scheduler &sched)
    {
    ringhelper helper;
    vmav<double,1> ringtmp({nphmax+2});
    while (auto rng=sched.getNext()) for(auto ith=rng.lo; ith<rng.hi; ++ith)
      {
      for (size_t icomp=0; icomp<ncomp; ++icomp)
        {
        auto ltmp = subarray<1>(leg, {{icomp}, {ith}, {}});
        helper.phase2ring (nphi(ith),phi0(ith),ringtmp,mmax,ltmp);
        for (size_t i=0; i<nphi(ith); ++i)
          map(icomp,ringstart(ith)+i*pixstride) = T(ringtmp(i+1));
        }
      }
    }); /* end of parallel region */
  }

template<typename T> void map2leg(  // FFT
  const cmav<T,2> &map, // (ncomp, pix)
  vmav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t ncomp=map.shape(0);
  MR_assert(ncomp==leg.shape(0), "number of components mismatch");
  size_t nrings=leg.shape(1);
  MR_assert(nrings>=1, "need at least one ring");
  MR_assert((nrings==nphi.shape(0)) && (nrings==ringstart.shape(0))
         && (nrings==phi0.shape(0)), "inconsistent number of rings");
  size_t nphmax=0;
  for (size_t i=0; i<nrings; ++i)
    nphmax=max(nphi(i),nphmax);
  MR_assert(leg.shape(2)>=1, "bad mmax");
  size_t mmax=leg.shape(2)-1;
  execDynamic(nrings, nthreads, 64, [&](Scheduler &sched)
    {
    ringhelper helper;
    vmav<double,1> ringtmp({nphmax+2});
    while (auto rng=sched.getNext()) for(auto ith=rng.lo; ith<rng.hi; ++ith)
      {
      for (size_t icomp=0; icomp<ncomp; ++icomp)
        {
        for (size_t i=0; i<nphi(ith); ++i)
          ringtmp(i+1) = map(icomp,ringstart(ith)+i*pixstride);
        auto ltmp = subarray<1>(leg, {{icomp}, {ith}, {}});
        helper.ring2phase (nphi(ith),phi0(ith),ringtmp,mmax,ltmp);
        }
      }
    }); /* end of parallel region */
  }

template<typename T> void resample_to_prepared_CC(const cmav<complex<T>,3> &legi, bool npi, bool spi,
  vmav<complex<T>,3> &lego, size_t spin, size_t lmax, size_t nthreads)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-npi-spi;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-2;
  bool need_first_resample = !(npi&&spi&&(nrings_in>=2*lmax+2));
  size_t nfull = need_first_resample ? 2*nfull_out : nfull_in;

  vector<complex<T>> shift(npi ? 0 : nrings_in+1);
  if (!npi)
    {
    UnityRoots<T,complex<T>> roots(2*nfull_in);
    for (size_t i=0; i<shift.size(); ++i)
      shift[i] = roots[i];
    }
  auto wgt = get_gridweights("CC", nfull/2+1);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(need_first_resample ? nfull_in : 1),
                 plan_out(nfull_out), plan_full(nfull);
  execDynamic((nm+1)/2, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({max(nfull,nfull_in)}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), max(plan_out.bufsize(), plan_full.bufsize()))}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{2*rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{2*rng.lo,MAXIDX}}));
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          for (size_t i=0, im=nfull_in-1+npi; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,2*j);
            complex<T> v2 = ((2*j+1)<llegi.shape(1)) ? llegi(i,2*j+1) : 0;
            tmp(i) = v1 + v2;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct * (v1-v2);
            else
              tmp(i) = T(0.5)*(tmp(i)+fct*(v1-v2));
            }
          if (need_first_resample)
            {
            plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);

            // shift
            if (!npi)
              for (size_t i=1, im=nfull_in-1; (i<nrings_in+1)&&(i<=im); ++i,--im)
                {
                if (i!=im)
                  tmp(i) *= conj(shift[i]);
                tmp(im) *= shift[i];
                }
  
            // zero padding to full-resolution CC grid
            if (nfull>nfull_in) // pad
              {
              size_t dist = nfull-nfull_in;
              size_t nmove = nfull_in/2;
              for (size_t i=nfull-1; i+1+nmove>nfull; --i)
                tmp(i) = tmp(i-dist);
              for (size_t i=nfull-nmove-dist; i+nmove<nfull; ++i)
                tmp(i) = 0;
              }
            if (nfull<nfull_in) // truncate
              {
              size_t dist = nfull_in-nfull;
              size_t nmove = nfull/2;
              for (size_t i=nfull_in-nmove; i<nfull_in; ++i)
                tmp(i-dist) = tmp(i);
              }
            plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
            }
          for (size_t i=0, im=nfull; i<=im; ++i, --im)
            {
            tmp(i) *= T(wgt(i));
            if ((i==0) || (i==im)) tmp(i)*=2;
            if ((im<nfull) && (im!=i))
              tmp(im) *= T(wgt(i));
            }
          plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);
          if (nfull_out<nfull) // truncate
            {
            size_t dist = nfull-nfull_out;
            size_t nmove = nfull_out/2;
            for (size_t i=nfull-nmove; i<nfull; ++i)
            tmp(i-dist) = tmp(i);
            }
          plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
          auto norm = T(.5/(nfull_out*((need_first_resample ? nfull_in : 1))));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-i;
            if (im==nfull_out) im=0;
            auto norm2 = norm * (T(1)-T(0.5)*(i==im));
            llego(i,2*j  ) = norm2 * (tmp(i) + fct*tmp(im));
            if ((2*j+1)<llego.shape(1))
              llego(i,2*j+1) = norm2 * (tmp(i) - fct*tmp(im));
            }
          }
        }
      }
    });
  }

template<typename T> void resample_from_prepared_CC(const cmav<complex<T>,3> &legi, vmav<complex<T>,3> &lego, bool npo, bool spo, size_t spin, size_t lmax, size_t nthreads)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-2;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-npo-spo;
  bool need_second_resample = !(npo&&spo&&(nrings_out>=2*lmax+2));
  size_t nfull = need_second_resample ? 2*nfull_in : nfull_out;

  vector<complex<T>> shift(npo ? 0 : nrings_out+1);
  if (!npo)
    {
    UnityRoots<T,complex<T>> roots(2*nfull_out);
    for (size_t i=0; i<shift.size(); ++i)
      shift[i] = roots[i];
    }
  auto wgt = get_gridweights("CC", nfull/2+1);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(nfull_in),
                 plan_out(need_second_resample ? nfull_out : 1), plan_full(nfull);
  execDynamic((nm+1)/2, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({max(nfull,nfull_out)}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), max(plan_out.bufsize(), plan_full.bufsize()))}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{2*rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{2*rng.lo,MAXIDX}}));
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          for (size_t i=0, im=nfull_in; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,2*j);
            complex<T> v2 = ((2*j+1)<llegi.shape(1)) ? llegi(i,2*j+1) : 0;
            tmp(i) = v1 + v2;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct * (v1-v2);
            else
              tmp(i) = T(0.5)*(tmp(i)+fct*(v1-v2));
            }
          plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
          // zero padding to full-resolution CC grid
          if (nfull>nfull_in) // pad
            {
            size_t dist = nfull-nfull_in;
            size_t nmove = nfull_in/2;
            for (size_t i=nfull-1; i+1+nmove>nfull; --i)
              tmp(i) = tmp(i-dist);
            for (size_t i=nfull-nmove-dist; i+nmove<nfull; ++i)
              tmp(i) = 0;
            }
          MR_assert(nfull>=nfull_in, "must not happen");
          plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);
          for (size_t i=0, im=nfull; i<=im; ++i, --im)
            {
            tmp(i) *= T(wgt(i));
            if ((i==0) || (i==im)) tmp(i)*=2;
            if ((im<nfull) && (im!=i))
              tmp(im) *= T(wgt(i));
            }

          if (need_second_resample)
            {
            plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
            if (nfull_out>nfull) // pad
              {
              size_t dist = nfull_out-nfull;
              size_t nmove = nfull/2;
              for (size_t i=nfull_out-1; i+1+nmove>nfull_out; --i)
                tmp(i) = tmp(i-dist);
              for (size_t i=nfull_out-nmove-dist; i+nmove<nfull_out; ++i)
                tmp(i) = 0;
              }
            if (nfull_out<nfull) // truncate
              {
              size_t dist = nfull-nfull_out;
              size_t nmove = nfull_out/2;
              for (size_t i=nfull-nmove; i<nfull; ++i)
                tmp(i-dist) = tmp(i);
              }
            // shift
            if (!npo)
              for (size_t i=1, im=nfull_out-1; (i<nrings_out+1)&&(i<=im); ++i,--im)
                {
                if (i!=im)
                  tmp(i) *= conj(shift[i]);
                tmp(im) *= shift[i];
                }
            plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);
            }
          auto norm = T(.5/(nfull_in*((need_second_resample ? nfull_out : 1))));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-1+npo-i;
            if (im==nfull_out) im=0;
            auto norm2 = norm * (T(1)-T(0.5)*(i==im));
            llego(i,2*j) = norm2 * (tmp(i) + fct*tmp(im));
            if ((2*j+1)<llego.shape(1))
              llego(i,2*j+1) = norm2 * (tmp(i) - fct*tmp(im));
            }
          }
        }
      }
    });
  }

void sanity_checks(
  const mav_info<2> &alm, // (ncomp, *)
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  const mav_info<2> &map, // (ncomp, *)
  const cmav<double,1> &theta, // (nrings)
  const mav_info<1> &phi0, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  size_t spin,
  SHT_mode mode)
  {
  size_t nm = mstart.shape(0);
  MR_assert(nm>0, "mstart too small");
  size_t mmax = nm-1;
  MR_assert(lmax>=mmax, "lmax must be >= mmax");
  size_t nrings = theta.shape(0);
  MR_assert(nrings>0, "need at least one ring");
  MR_assert((phi0.shape(0)==nrings) &&
            (nphi.shape(0)==nrings) &&
            (ringstart.shape(0)==nrings),
    "inconsistency in the number of rings");
  size_t ncomp = 1+(spin>0);
  if (mode==ALM2MAP_DERIV1)
    MR_assert((alm.shape(0)==1) && (map.shape(0)==2),
      "inconsistent number of components");
  else
    MR_assert((alm.shape(0)==ncomp) && (map.shape(0)==ncomp),
      "inconsistent number of components");
  }

template<typename T> void synthesis(
  const cmav<complex<T>,2> &alm, // (ncomp, *)
  vmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  SHT_mode mode)
  {
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, spin, mode);
  vmav<size_t,1> mval({mstart.shape(0)});
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;

  bool npi, spi;
  size_t ntheta_tmp;
  if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
    {
    vmav<double,1> theta_tmp({ntheta_tmp});
    for (size_t i=0; i<ntheta_tmp; ++i)
      theta_tmp(i) = i*pi/(ntheta_tmp-1);
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0),max(theta.shape(0),ntheta_tmp),mstart.shape(0)}));
    auto legi(subarray<3>(leg, {{},{0,ntheta_tmp},{}}));
    auto lego(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));
    alm2leg(alm, legi, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode);
    resample_theta(legi, true, true, lego, npi, spi, spin, nthreads, false);
    leg2map(map, lego, nphi, phi0, ringstart, pixstride, nthreads);
    }
  else
    {
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0),theta.shape(0),mstart.shape(0)}));
    alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, mode);
    leg2map(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
    }
  }

void get_ringtheta_2d(const string &type, vmav<double, 1> &theta)
  {
  auto nrings = theta.shape(0);

  if (type=="GL") // Gauss-Legendre
    {
    ducc0::GL_Integrator integ(nrings);
    auto cth = integ.coords();
    for (size_t m=0; m<nrings; ++m)
      theta(m) = acos(-cth[m]);
    }
  else if (type=="F1") // Fejer 1
    for (size_t m=0; m<(nrings+1)/2; ++m)
      {
      theta(m)=pi*(m+0.5)/nrings;
      theta(nrings-1-m)=pi-theta(m);
      }
  else if (type=="CC") // Clenshaw-Curtis
    for (size_t m=0; m<(nrings+1)/2; ++m)
      {
      theta(m)=max(1e-15,pi*m/(nrings-1.));
      theta(nrings-1-m)=pi-theta(m);
      }
  else if (type=="F2") // Fejer 2
    for (size_t m=0; m<nrings; ++m)
      theta(m)=pi*(m+1)/(nrings+1.);
  else if (type=="DH") // Driscoll-Healy
    for (size_t m=0; m<nrings; ++m)
      theta(m) = m*pi/nrings;
  else if (type=="MW") // McEwen-Wiaux
    for (size_t m=0; m<nrings; ++m)
      theta(m)=pi*(2.*m+1.)/(2.*nrings-1.);
  else if (type=="MWflip") // McEwen-Wiaux mirrored
    for (size_t m=0; m<nrings; ++m)
      theta(m)=pi*(2.*m)/(2.*nrings-1.);
  else
    MR_fail("unsupported grid type");
  }

template<typename T> void synthesis_2d(const cmav<complex<T>,2> &alm, vmav<T,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads,
  SHT_mode mode)
  {
  auto nphi = cmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0 = cmav<double,1>::build_uniform({map.shape(1)}, 0.);
  vmav<size_t,1> mstart({mmax+1});
  for (size_t i=0, ofs=0; i<=mmax; ++i)
    {
    mstart(i) = ofs-i;
    ofs += lmax+1-i;
    }
  vmav<size_t,1> ringstart({map.shape(1)});
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  vmav<T,2> map2(map.data(), {map.shape(0), map.shape(1)*map.shape(2)},
                {map.stride(0), 1});
  vmav<double,1> theta({map.shape(1)});
  get_ringtheta_2d(geometry, theta);
  synthesis(alm, map2, spin, lmax, mstart, 1, theta, nphi, phi0, ringstart, pixstride, nthreads,
  mode);
  }
template void synthesis_2d(const cmav<complex<double>,2> &alm, vmav<double,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads,
  SHT_mode mode);
template void synthesis_2d(const cmav<complex<float>,2> &alm, vmav<float,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads,
  SHT_mode mode);

template<typename T> void adjoint_synthesis(
  vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, spin, MAP2ALM);
  vmav<size_t,1> mval({mstart.shape(0)});
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;

  bool npi, spi;
  size_t ntheta_tmp;
  if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
    {
    vmav<double,1> theta_tmp({ntheta_tmp});
    for (size_t i=0; i<ntheta_tmp; ++i)
      theta_tmp(i) = i*pi/(ntheta_tmp-1);
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0),max(theta.shape(0),ntheta_tmp),mstart.shape(0)}));
    auto legi(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));
    auto lego(subarray<3>(leg, {{},{0,ntheta_tmp},{}}));
    map2leg(map, legi, nphi, phi0, ringstart, pixstride, nthreads);
    resample_theta(legi, npi, spi, lego, true, true, spin, nthreads, true);
    leg2alm(alm, lego, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads);
    }
  else
    {
    auto leg(vmav<complex<T>,3>::build_noncritical({alm.shape(0),theta.shape(0),mstart.shape(0)}));
    map2leg(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
    leg2alm(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
    }
  }

template<typename T> void adjoint_synthesis_2d(vmav<complex<T>,2> &alm,
  const cmav<T,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads)
  {
  auto nphi = vmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0 = vmav<double,1>::build_uniform({map.shape(1)}, 0.);
  vmav<size_t,1> mstart({mmax+1});
  for (size_t i=0, ofs=0; i<=mmax; ++i)
    {
    mstart(i) = ofs-i;
    ofs += lmax+1-i;
    }
  vmav<size_t,1> ringstart({map.shape(1)});
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  cmav<T,2> map2(map.data(), {map.shape(0), map.shape(1)*map.shape(2)},
                {map.stride(0), 1});
  vmav<double,1> theta({map.shape(1)});
  get_ringtheta_2d(geometry, theta);
  adjoint_synthesis(alm, map2, spin, lmax, mstart, 1, theta, nphi, phi0, ringstart, pixstride, nthreads);
  }
template void adjoint_synthesis_2d(vmav<complex<double>,2> &alm,
  const cmav<double,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);
template void adjoint_synthesis_2d(vmav<complex<float>,2> &alm,
  const cmav<float,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);

template<typename T> void analysis_2d(
  vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const string &geometry,
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t nrings_min = lmax+1;
  if (geometry=="CC")
    nrings_min = lmax+2;
  else if (geometry=="DH")
    nrings_min = 2*lmax+2;
  else if (geometry=="F2")
    nrings_min = 2*lmax+1;
  MR_assert(map.shape(1)>=nrings_min,
    "too few rings for analysis up to requested lmax");

  vmav<size_t,1> mval({mstart.shape(0)});
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;
  vmav<double,1> theta({nphi.shape(0)});
  get_ringtheta_2d(geometry, theta);
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, spin, MAP2ALM);
  if ((geometry=="CC")||(geometry=="F1")||(geometry=="MW")||(geometry=="MWflip"))
    {
    bool npi, spi;
    if (geometry=="CC")
      { npi=spi=true; }
    else if (geometry=="F1")
      { npi=spi=false; }
    else if (geometry=="MW")
      { npi=false; spi=true; }
    else
      { npi=true; spi=false; }

    size_t ntheta_leg = good_size_complex(lmax+1)+1;
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0), max(ntheta_leg,theta.shape(0)), mstart.shape(0)}));
    auto legi(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));
    auto lego(subarray<3>(leg, {{},{0,ntheta_leg},{}}));
    map2leg(map, legi, nphi, phi0, ringstart, pixstride, nthreads);
    for (size_t i=0; i<legi.shape(0); ++i)
      for (size_t j=0; j<legi.shape(1); ++j)
        {
        auto wgt1 = T(1./nphi(j));
        for (size_t k=0; k<legi.shape(2); ++k)
          legi(i,j,k) *= wgt1;
        }
         
    resample_to_prepared_CC(legi, npi, spi, lego, spin, lmax, nthreads);
    vmav<double,1> newtheta({ntheta_leg});
    for (size_t i=0; i<ntheta_leg; ++i)
      newtheta(i) = (pi*i)/(ntheta_leg-1);
    leg2alm(alm, lego, spin, lmax, mval, mstart, lstride, newtheta, nthreads);
    return;
    }
  else
    {
    auto wgt = get_gridweights(geometry, theta.shape(0));
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0), theta.shape(0), mstart.shape(0)}));
    map2leg(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
    for (size_t i=0; i<leg.shape(0); ++i)
      for (size_t j=0; j<leg.shape(1); ++j)
        {
        auto wgt1 = T(wgt(j)/nphi(j));
        for (size_t k=0; k<leg.shape(2); ++k)
          leg(i,j,k) *= wgt1;
        }
    leg2alm(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
    }
  }

template<typename T> void analysis_2d(vmav<complex<T>,2> &alm,
  const cmav<T,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads)
  {
  auto nphi = cmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0 = cmav<double,1>::build_uniform({map.shape(1)}, 0.);
  vmav<size_t,1> mstart({mmax+1});
  for (size_t i=0, ofs=0; i<=mmax; ++i)
    {
    mstart(i) = ofs-i;
    ofs += lmax+1-i;
    }
  vmav<size_t,1> ringstart({map.shape(1)});
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  cmav<T,2> map2(map.data(), {map.shape(0), map.shape(1)*map.shape(2)},
                {map.stride(0), 1});

  analysis_2d(alm, map2, spin, lmax, mstart, 1, geometry, nphi, phi0, ringstart, pixstride, nthreads);
  }
template void analysis_2d(vmav<complex<double>,2> &alm,
  const cmav<double,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);
template void analysis_2d(vmav<complex<float>,2> &alm,
  const cmav<float,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);

template<typename T> void adjoint_analysis_2d(
  const cmav<complex<T>,2> &alm, // (ncomp, *)
  vmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const string &geometry,
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t nrings_min = lmax+1;
  if (geometry=="CC")
    nrings_min = lmax+2;
  else if (geometry=="DH")
    nrings_min = 2*lmax+2;
  else if (geometry=="F2")
    nrings_min = 2*lmax+1;
  MR_assert(map.shape(1)>=nrings_min,
    "too few rings for adjoint analysis up to requested lmax");

  vmav<size_t,1> mval({mstart.shape(0)});
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;
  vmav<double,1> theta({nphi.shape(0)});
  get_ringtheta_2d(geometry, theta);
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, spin, MAP2ALM);
  if ((geometry=="CC")||(geometry=="F1")||(geometry=="MW")||(geometry=="MWflip"))
    {
    bool npo, spo;
    if (geometry=="CC")
      { npo=spo=true; }
    else if (geometry=="F1")
      { npo=spo=false; }
    else if (geometry=="MW")
      { npo=false; spo=true; }
    else
      { npo=true; spo=false; }

    size_t ntheta_leg = good_size_complex(lmax+1)+1;
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0), max(ntheta_leg,theta.shape(0)), mstart.shape(0)}));
    auto legi(subarray<3>(leg, {{},{0,ntheta_leg},{}}));
    auto lego(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));

    vmav<double,1> theta_tmp({ntheta_leg});
    for (size_t i=0; i<ntheta_leg; ++i)
      theta_tmp(i) = (pi*i)/(ntheta_leg-1);
    alm2leg(alm, legi, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads);
    resample_from_prepared_CC(legi, lego, npo, spo, spin, lmax, nthreads);
    for (size_t i=0; i<lego.shape(0); ++i)
      for (size_t j=0; j<lego.shape(1); ++j)
        {
        auto wgt1 = T(1./nphi(j));
        for (size_t k=0; k<lego.shape(2); ++k)
          lego(i,j,k) *= wgt1;
        }
    leg2map(map, lego, nphi, phi0, ringstart, pixstride, nthreads);
    return;
    }
  else
    {
    auto wgt = get_gridweights(geometry, theta.shape(0));
    auto leg(vmav<complex<T>,3>::build_noncritical({map.shape(0), theta.shape(0), mstart.shape(0)}));
    alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
    for (size_t i=0; i<leg.shape(0); ++i)
      for (size_t j=0; j<leg.shape(1); ++j)
        {
        auto wgt1 = T(wgt(j)/nphi(j));
        for (size_t k=0; k<leg.shape(2); ++k)
          leg(i,j,k) *= wgt1;
        }
    leg2map(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
    }
  }

template<typename T> void adjoint_analysis_2d(const cmav<complex<T>,2> &alm, vmav<T,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads)
  {
  auto nphi = vmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0 = vmav<double,1>::build_uniform({map.shape(1)}, 0.);
  vmav<size_t,1> mstart({mmax+1});
  for (size_t i=0, ofs=0; i<=mmax; ++i)
    {
    mstart(i) = ofs-i;
    ofs += lmax+1-i;
    }
  vmav<size_t,1> ringstart({map.shape(1)});
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  vmav<T,2> map2(map.data(), {map.shape(0), map.shape(1)*map.shape(2)},
                {map.stride(0), 1});
  vmav<double,1> theta({map.shape(1)});
  adjoint_analysis_2d(alm, map2, spin, lmax, mstart, 1, geometry, nphi, phi0,
    ringstart, pixstride, nthreads);
  }
template void adjoint_analysis_2d(const cmav<complex<double>,2> &alm, vmav<double,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads);
template void adjoint_analysis_2d(const cmav<complex<float>,2> &alm, vmav<float,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads);
}}
