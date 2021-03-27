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
#include "ducc0/sharp/sht.h"
#include "ducc0/math/fft1d.h"
#include "ducc0/math/fft.h"
#include "ducc0/math/math_utils.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

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


#define VZERO(var) do { memset(&(var),0,sizeof(var)); } while(0)

template<typename T> DUCC0_NOINLINE static void inner_loop_a2m(SHT_mode mode,
  mav<complex<double>,2> &almtmp,
  mav<complex<T>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi)
  {
  if (gen.s==0)
    {
    // adjust the a_lm for the new algorithm
    MR_assert(almtmp.stride(1)==1, "bad stride");
    dcmplx * DUCC0_RESTRICT alm=almtmp.vdata();
    for (size_t il=0, l=gen.m; l<=gen.lmax; ++il,l+=2)
      {
      dcmplx al = alm[l];
      dcmplx al1 = (l+1>gen.lmax) ? 0. : alm[l+1];
      dcmplx al2 = (l+2>gen.lmax) ? 0. : alm[l+2];
      alm[l  ] = gen.alpha[il]*(gen.eps[l+1]*al + gen.eps[l+2]*al2);
      alm[l+1] = gen.alpha[il]*al1;
      }

    constexpr size_t nval=nv0*VLEN;
    size_t ith=0;
    std::array<size_t,nval> itgt;
    while (ith<rdata.size())
      {
      s0data_u d;
      VZERO(d.s.p1r); VZERO(d.s.p1i); VZERO(d.s.p2r); VZERO(d.s.p2i);
      size_t nth=0;
      while ((nth<nval)&&(ith<rdata.size()))
        {
        if (rdata[ith].mlim>=gen.m)
          {
          itgt[nth] = ith;
          d.s.csq[nth]=rdata[ith].cth*rdata[ith].cth;
          d.s.sth[nth]=rdata[ith].sth;
          ++nth;
          }
        else
          phase.v(rdata[ith].idx, mi, 0) = phase.v(rdata[ith].midx, mi, 0) = 0;
        ++ith;
        }
      if (nth>0)
        {
        size_t i2=((nth+VLEN-1)/VLEN)*VLEN;
        for (auto i=nth; i<i2; ++i)
          {
          d.s.csq[i]=d.s.csq[nth-1];
          d.s.sth[i]=d.s.sth[nth-1];
          d.s.p1r[i]=d.s.p1i[i]=d.s.p2r[i]=d.s.p2i[i]=0.;
          }
        calc_alm2map (almtmp.cdata(), gen, d.v, nth);
        for (size_t i=0; i<nth; ++i)
          {
          auto tgt=itgt[i];
          //adjust for new algorithm
          d.s.p2r[i]*=rdata[tgt].cth;
          d.s.p2i[i]*=rdata[tgt].cth;
          dcmplx r1(d.s.p1r[i], d.s.p1i[i]),
                 r2(d.s.p2r[i], d.s.p2i[i]);
          phase.v(rdata[tgt].idx, mi, 0) = complex<T>(r1+r2);
          if (rdata[tgt].idx!=rdata[tgt].midx)
            phase.v(rdata[tgt].midx, mi, 0) = complex<T>(r1-r2);
          }
        }
      }
    }
  else
    {
    //adjust the a_lm for the new algorithm
    for (size_t l=gen.mhi; l<=gen.lmax+1; ++l)
      for (size_t i=0; i<almtmp.shape(1); ++i)
        almtmp.v(l,i)*=gen.alpha[l];

    constexpr size_t nval=nvx*VLEN;
    size_t ith=0;
    std::array<size_t,nval> itgt;
    while (ith<rdata.size())
      {
      sxdata_u d;
      VZERO(d.s.p1pr); VZERO(d.s.p1pi); VZERO(d.s.p2pr); VZERO(d.s.p2pi);
      VZERO(d.s.p1mr); VZERO(d.s.p1mi); VZERO(d.s.p2mr); VZERO(d.s.p2mi);
      size_t nth=0;
      while ((nth<nval)&&(ith<rdata.size()))
        {
        if (rdata[ith].mlim>=gen.m)
          {
          itgt[nth] = ith;
          d.s.cth[nth]=rdata[ith].cth; d.s.sth[nth]=rdata[ith].sth;
          ++nth;
          }
        else
          {
          phase.v(rdata[ith].idx, mi, 0) = phase.v(rdata[ith].midx, mi, 0) = 0;
          phase.v(rdata[ith].idx, mi, 1) = phase.v(rdata[ith].midx, mi, 1) = 0;
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
// FIXME are those two lines needed?
          d.s.p1pr[i]=d.s.p1pi[i]=d.s.p2pr[i]=d.s.p2pi[i]=0.;
          d.s.p1mr[i]=d.s.p1mi[i]=d.s.p2mr[i]=d.s.p2mi[i]=0.;
          }
        (mode==ALM2MAP) ?
          calc_alm2map_spin  (almtmp.cdata(), gen, d.v, nth) :
          calc_alm2map_deriv1(almtmp.cdata(), gen, d.v, nth);
        for (size_t i=0; i<nth; ++i)
          {
          auto tgt=itgt[i];
          dcmplx q1(d.s.p1pr[i], d.s.p1pi[i]),
                 q2(d.s.p2pr[i], d.s.p2pi[i]),
                 u1(d.s.p1mr[i], d.s.p1mi[i]),
                 u2(d.s.p2mr[i], d.s.p2mi[i]);
          phase.v(rdata[tgt].idx, mi, 0) = complex<T>(q1+q2);
          phase.v(rdata[tgt].idx, mi, 1) = complex<T>(u1+u2);
          if (rdata[tgt].idx!=rdata[tgt].midx)
            {
            auto *phQ = &(phase.v(rdata[tgt].midx, mi, 0)),
                 *phU = &(phase.v(rdata[tgt].midx, mi, 1));
            *phQ = complex<T>(q1-q2);
            *phU = complex<T>(u1-u2);
            if ((gen.mhi-gen.m+gen.s)&1)
              { *phQ=-(*phQ); *phU=-(*phU); }
            }
          }
        }
      }
    }
  }

template<typename T> DUCC0_NOINLINE static void inner_loop_m2a(
  mav<complex<double>,2> &almtmp,
  const mav<complex<T>,3> &phase, const vector<ringdata> &rdata,
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
          dcmplx ph1=phase(rdata[ith].idx, mi, 0);
          dcmplx ph2=(rdata[ith].idx==rdata[ith].midx) ? 0 : phase(rdata[ith].midx, mi, 0);
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
        calc_map2alm (almtmp.vdata(), gen, d.v, nth);
        }
      }
    //adjust the a_lm for the new algorithm
    dcmplx * DUCC0_RESTRICT alm=almtmp.vdata();
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
          dcmplx p1Q=phase(rdata[ith].idx, mi, 0),
                 p1U=phase(rdata[ith].idx, mi, 1),
                 p2Q=(rdata[ith].idx!=rdata[ith].midx) ? phase(rdata[ith].midx, mi, 0):0.,
                 p2U=(rdata[ith].idx!=rdata[ith].midx) ? phase(rdata[ith].midx, mi, 1):0.;
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
        calc_map2alm_spin(almtmp.vdata(), gen, d.v, nth);
        }
      }
    //adjust the a_lm for the new algorithm
    for (size_t l=gen.mhi; l<=gen.lmax; ++l)
      {
      almtmp.v(l,0)*=gen.alpha[l];
      almtmp.v(l,1)*=gen.alpha[l];
      }
    }
  }

#undef VZERO

template<typename T> DUCC0_NOINLINE void inner_loop(SHT_mode mode,
  mav<complex<double>,2> &almtmp,
  mav<complex<T>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi)
  {
  (mode==MAP2ALM) ? inner_loop_m2a(almtmp, phase, rdata, gen, mi)
                  : inner_loop_a2m(mode, almtmp, phase, rdata, gen, mi);
  }
template void inner_loop(SHT_mode mode,
  mav<complex<double>,2> &almtmp,
  mav<complex<double>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi);
template void inner_loop(SHT_mode mode,
  mav<complex<double>,2> &almtmp,
  mav<complex<float>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi);

size_t get_mmax(const mav<size_t,1> &mval, size_t lmax)
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

#if 0

vector<ringdata> make_ringdata(const mav<double,1> &theta, size_t lmax,
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

template<typename T> void alm2leg(  // associated Legendre transform
  const mav<complex<T>,2> &alm, // (lmidx, ncomp)
  mav<complex<T>,3> &leg, // (nrings, nm, ncomp)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads,
  SHT_mode mode)
  {
  // sanity checks
  auto nrings=theta.shape(0);
  MR_assert(nrings==leg.shape(0), "nrings mismatch");
  auto nm=mval.shape(0);
  MR_assert(nm==mstart.shape(0), "nm mismatch");
  MR_assert(nm==leg.shape(1), "nm mismatch");
  auto nalm=alm.shape(1);
  auto mmax = get_mmax(mval, lmax);
  if (mode==ALM2MAP_DERIV1)
    {
    spin=1;
    MR_assert(nalm==1, "need one a_lm component");
    MR_assert(leg.shape(2)==2, "need two Legendre components");
    }
  else
    {
    size_t ncomp = (spin==0) ? 1 : 2;
    MR_assert(nalm==ncomp, "incorrect number of a_lm components");
    MR_assert(leg.shape(2)==ncomp, "incorrect number of Legendre components");
    }

  auto norm_l = (mode==ALM2MAP_DERIV1) ? Ylmgen::get_d1norm (lmax) :
                                         Ylmgen::get_norm (lmax, spin);
  auto rdata = make_ringdata(theta, lmax, spin);
  YlmBase base(lmax, mmax, spin);

  ducc0::execDynamic(mval.shape(0), nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    Ylmgen gen(base);
    mav<complex<double>,2> almtmp({lmax+2,nalm});

    while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
      {
      auto m=mval(mi);
      auto lmin=max(spin,m);
      for (size_t l=m; l<lmin; ++l)
        for (size_t ialm=0; ialm<nalm; ++ialm)
          almtmp.v(l,ialm) = 0;
      for (size_t l=lmin; l<=lmax; ++l)
        for (size_t ialm=0; ialm<nalm; ++ialm)
          almtmp.v(l,ialm) = alm(mstart(mi)+l,ialm)*norm_l[l];
      for (size_t ialm=0; ialm<nalm; ++ialm)
        almtmp.v(lmax+1,ialm) = 0;
      gen.prepare(m);
      inner_loop_a2m (mode, almtmp, leg, rdata, gen, mi);
      }
    }); /* end of parallel region */
  }

template<typename T> void leg2alm(  // associated Legendre transform
  const mav<complex<T>,3> &leg, // (lmidx, ncomp)
  mav<complex<T>,2> &alm, // (nrings, nm, ncomp)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads)
  {
  // sanity checks
  auto nrings=theta.shape(0);
  MR_assert(nrings==leg.shape(0), "nrings mismatch");
  auto nm=mval.shape(0);
  MR_assert(nm==mstart.shape(0), "nm mismatch");
  MR_assert(nm==leg.shape(1), "nm mismatch");
  auto nalm=alm.shape(1);
  auto mmax = get_mmax(mval, lmax);
  size_t ncomp = (spin==0) ? 1 : 2;
  MR_assert(alm.shape(1)==ncomp, "incorrect number of a_lm components");
  MR_assert(leg.shape(2)==ncomp, "incorrect number of Legendre components");

  auto norm_l = Ylmgen::get_norm (lmax, spin);
  auto rdata = make_ringdata(theta, lmax, spin);
  YlmBase base(lmax, mmax, spin);

  ducc0::execDynamic(mval.shape(0), nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    Ylmgen gen(base);
    mav<complex<double>,2> almtmp({lmax+2,nalm});

    while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
      {
      auto m=mval(mi);
      gen.prepare(m);
      for (size_t l=m; l<almtmp.shape(0); ++l)
        for (size_t ialm=0; ialm<nalm; ++ialm)
          almtmp.v(l,ialm) = 0.;
      inner_loop_m2a (almtmp, leg, rdata, gen, mi);
      auto lmin=max(spin,m);
      for (size_t l=m; l<lmin; ++l)
        for (size_t ialm=0; ialm<nalm; ++ialm)
          alm.v(mstart(mi)+l,ialm) = 0;
      for (size_t l=lmin; l<=lmax; ++l)
        for (size_t ialm=0; ialm<nalm; ++ialm)
          alm.v(mstart(mi)+l,ialm) = almtmp(l,ialm)*norm_l[l];
      }
    }); /* end of parallel region */
  }

void clenshaw_curtis_weights(mav<double,1> &weight)
  {
  auto nrings = weight.shape(0);
  MR_assert(nrings>=2, "too few rings for CC geometry");
  size_t n=nrings-1;
  double dw=-1./(n*n-1.+(n&1));
  vector<double> wgt(n);
  wgt[0]=2.+dw;
  for (size_t k=1; k<=(n/2-1); ++k)
    wgt[2*k-1]=2./(1.-4.*k*k) + dw;
  wgt[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
  pocketfft_r<double> plan(n);
  plan.exec(wgt.data(), 1., false);

  for (size_t m=0; m<n; ++m)
    weight.v(m) = double(wgt[m]*2*pi/n);
  weight.v(n) = weight(0);
  }

void fejer1_weights (mav<double,1> &weight)
  {
  auto nrings = weight.shape(0);

  weight.v(0)=2.;
  for (size_t k=1; k<=(nrings-1)/2; ++k)
    {
    weight.v(2*k-1)=2./(1.-4.*k*k)*cos((k*pi)/nrings);
    weight.v(2*k  )=2./(1.-4.*k*k)*sin((k*pi)/nrings);
    }
  if ((nrings&1)==0) weight.v(nrings-1)=0.;
  {
  pocketfft_r<double> plan(nrings);
  plan.exec(weight.vdata(), 1., false);
  }

  for (size_t m=0; m<(nrings+1)/2; ++m)
    weight.v(m)=weight.v(nrings-1-m)=weight(m)*2*pi/nrings;
  }

void resample_theta(const mav<complex<double>,3> &legi, bool npi, bool spi,
  mav<complex<double>,3> &lego, bool npo, bool spo, size_t spin, size_t nthreads)
  {
  MR_assert(legi.shape(1)==lego.shape(1), "dimension mismatch");
  MR_assert(legi.shape(2)==lego.shape(2), "dimension mismatch");
  size_t nrings_in = legi.shape(0);
  size_t nfull_in = 2*nrings_in-npi-spi;
  size_t nrings_out = lego.shape(0);
  size_t nfull_out = 2*nrings_out-npo-spo;
  double dthi = 2*pi/nfull_in;
  double dtho = 2*pi/nfull_out;
  double shift = 0.5*(dtho*(1-npo)-dthi*(1-npi));
  size_t nfull = max(nfull_in, nfull_out);
  auto nm = legi.shape(1);
  auto nm2 = nm/2;
  auto tmp(mav<complex<double>,3>::build_noncritical({nfull, (nm+1)/2, legi.shape(2)}, UNINITIALIZED));
  fmav<complex<double>> ftmp_in(tmp.subarray<3>({0,0,0},{nfull_in,MAXIDX,MAXIDX}));
  fmav<complex<double>> ftmp_out(tmp.subarray<3>({0,0,0},{nfull_out,MAXIDX,MAXIDX}));
  double fct = ((spin&1)==0) ? 1 : -1;
  // fill dark side
  execParallel(0, nrings_in, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo, im=nfull_in-lo-1+npi; (i<hi)&&(i<=im); ++i,--im)
      {
      for (size_t j=0; j<nm2; ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          tmp.v(i,j,k) = legi(i,2*j,k) + legi(i,2*j+1,k);
          if ((im<nfull_in) && (i!=im))
            tmp.v(im,j,k) = fct * (legi(i,2*j,k) - legi(i,2*j+1,k));
          }
      if ((nm&1)==1)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          tmp.v(i,tmp.shape(1)-1,k) = legi(i,nm-1,k);
          if ((im<nfull_in) && (i!=im))
            tmp.v(im,tmp.shape(1)-1,k) = fct * legi(i,nm-1,k);
          }
      }
    });

  c2c(ftmp_in,ftmp_in,{0},true,1.,nthreads);

  if (shift!=0)
    execParallel(1, nrings_in+1, nthreads, [&](size_t lo, size_t hi)
      {
      for (size_t i=lo, im=nfull_in-lo; (i<hi)&&(i<=im); ++i,--im)
        {
        auto phase=std::polar(1., i*shift);
        for (size_t j=0; j<tmp.shape(1); ++j)
          for (size_t k=0; k<tmp.shape(2); ++k)
            {
            if (i!=im)
              tmp.v(i,j,k) *= phase;
            tmp.v(im,j,k) *= conj(phase);
            }
        }
      });

  // zero padding/truncation
  if (nfull_out>nfull_in) // pad
    {
    size_t dist = nfull_out-nfull_in;
    size_t nmove = nfull_in/2;
    for (size_t i=nfull_out-1; i>nfull_out-1-nmove; --i)
      for (size_t j=0; j<tmp.shape(1); ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          tmp.v(i,j,k) = tmp(i-dist,j,k);
          tmp.v(i-dist,j,k) = 0;
          }
    }
  if (nfull_out<nfull_in) // truncate
    {
    size_t dist = nfull_in-nfull_out;
    size_t nmove = nfull_out/2;
    for (size_t i=nfull_in-nmove; i<nfull_in; ++i)
      for (size_t j=0; j<tmp.shape(1); ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          tmp.v(i-dist,j,k) = tmp(i,j,k);
    }

  c2c(ftmp_out,ftmp_out,{0},false,1.,nthreads);

  double norm = 1./(2*sqrt(nfull_in*nfull_out));
  execParallel(0, nrings_out, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo; i<hi; ++i)
      {
      size_t im = nfull_out-1+npo-i;
      if (im==nfull_out) im=0;
      for (size_t j=0; j<nm2; ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          lego.v(i,2*j  ,k) = norm * (tmp(i,j,k) + fct*tmp(im,j,k));
          lego.v(i,2*j+1,k) = norm * (tmp(i,j,k) - fct*tmp(im,j,k));
          }
      if ((nm&1)==1)
        for (size_t k=0; k<tmp.shape(2); ++k)
          lego.v(i,nm-1,k) = norm * (tmp(i,tmp.shape(1)-1,k) + fct*tmp(im,tmp.shape(1)-1,k));
      }
    });
  }

void prep_for_analysis(mav<complex<double>,3> &leg, size_t spin, size_t nthreads)
  {
  auto nrings = leg.shape(0);
  mav<double,1> wgt({2*nrings-1});
  clenshaw_curtis_weights(wgt);
  auto nm = leg.shape(1);
  auto nm2 = nm/2;
  size_t nfull = 2*nrings-2;
  auto tmp(mav<complex<double>,3>::build_noncritical({nfull, (nm+1)/2, leg.shape(2)}, UNINITIALIZED));
  fmav<complex<double>> ftmp(tmp);
  double fct = ((spin&1)==0) ? 1 : -1;
  execParallel(0, nrings, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo, im=nfull-lo; i<hi; ++i,--im)
      {
      for (size_t j=0; j<nm2; ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          tmp.v(i,j,k) = leg(i,2*j,k) + leg(i,2*j+1,k);
          if ((im<nfull) && (i!=im))
            tmp.v(im,j,k) = fct * (leg(i,2*j,k) - leg(i,2*j+1,k));
          }
      if ((nm&1)==1)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          tmp.v(i,tmp.shape(1)-1,k) = leg(i,nm-1,k);
          if ((im<nfull) && (i!=im))
            tmp.v(im,tmp.shape(1)-1,k) = fct * leg(i,nm-1,k);
          }
      }
    });

  c2c(ftmp,ftmp,{0},true,1.,nthreads);

  vector<complex<double>> shift(nrings+1);
  UnityRoots<double,complex<double>> roots(4*nrings-4);
  for (size_t i=1; i<shift.size(); ++i)
    shift[i] = roots[i];
  execParallel(1, nrings+1, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo, im=nfull-lo; (i<hi)&&(i<=im); ++i,--im)
      for (size_t j=0; j<tmp.shape(1); ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          if (i!=im)
            tmp.v(i,j,k) *= shift[i];
          tmp.v(im,j,k) *= conj(shift[i]);
          }
    });
  c2c(ftmp,ftmp,{0},false,1.,nthreads);

  double norm = 1./(2*tmp.shape(0)*tmp.shape(0));
  execParallel(0, nrings+1, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo, im=nfull-lo-1; (i<hi)&&(i<im); ++i,--im)
      {
      auto factor = wgt(1+2*i)*norm;
      for (size_t j=0; j<tmp.shape(1); ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          tmp.v(i,j,k) *= factor;
          if (i!=im) tmp.v(im,j,k) *= factor;
          }
      }
    });
  c2c(ftmp,ftmp,{0},true,1.,nthreads);
  execParallel(1, nrings+1, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo, im=nfull-lo; (i<hi)&&(i<=im); ++i,--im)
      for (size_t j=0; j<tmp.shape(1); ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          if (i!=im)
            tmp.v(i,j,k) *= conj(shift[i]);
          tmp.v(im,j,k) *= shift[i];
          }
    });
  c2c(ftmp,ftmp,{0},false,1.,nthreads);

  execParallel(0, nrings, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo, im=nfull-lo; i<hi; ++i,--im)
      {
      for (size_t j=0; j<nm2; ++j)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          leg.v(i,2*j,k) = wgt(2*i)*leg(i,2*j,k) + tmp(i,j,k);
          leg.v(i,2*j+1,k) = wgt(2*i)*leg(i,2*j+1,k) + tmp(i,j,k);
          if ((im<nfull) && (i!=im))
            {
            leg.v(i,2*j,k) += fct*tmp(im,j,k);
            leg.v(i,2*j+1,k) -= fct*tmp(im,j,k);
            }
          }
      if ((nm&1)==1)
        for (size_t k=0; k<tmp.shape(2); ++k)
          {
          leg.v(i,nm-1,k) = wgt(2*i)*leg(i,nm-1,k) + tmp(i,tmp.shape(1)-1,k);
          if ((im<nfull) && (i!=im))
            leg.v(i,nm-1,k) += fct*tmp(im,tmp.shape(1)-1,k);
          }
      }
    });
  }
#if 1
void prep_for_analysis2(mav<complex<double>,3> &leg, size_t lmax, size_t spin, size_t nthreads)
  {
  auto legtmp2(mav<complex<double>,3>::build_noncritical({leg.shape(0), leg.shape(1), leg.shape(2)}));
  auto legtmp = legtmp2.subarray<3>({0,0,0},{leg.shape(0)-1, leg.shape(1), leg.shape(2)});
  resample_theta(leg, true, true, legtmp, false, false, spin, nthreads);

  mav<double,1> wgt({2*leg.shape(0)-1});
  clenshaw_curtis_weights(wgt);

  for (size_t i=0; i<legtmp.shape(0); ++i)
    for (size_t j=0; j<legtmp.shape(1); ++j)
      for (size_t k=0; k<legtmp.shape(2); ++k)
        legtmp.v(i,j,k) *= wgt(1+2*i);

  resample_theta(legtmp, false, false, legtmp2, true, true, spin, nthreads);

  for (size_t i=0; i<leg.shape(0); ++i)
    {
    double wgtx=1;
    if ((i==0)||(i==leg.shape(0)-1)) wgtx=0.5;
    for (size_t j=0; j<leg.shape(1); ++j)
      for (size_t k=0; k<leg.shape(2); ++k)
        leg.v(i,j,k) = leg(i,j,k)*wgt(2*i) + wgtx*legtmp2(i,j,k);
    }
  }
#else
void prep_for_analysis2(mav<complex<double>,3> &leg, size_t lmax, size_t spin, size_t nthreads)
  {
  auto nfull = 2*good_size_complex(2*lmax+1);
  auto nrings = nfull/2 + 1;

  mav<complex<double>,3> legtmp({nrings, leg.shape(1), leg.shape(2)});
  resample_theta(leg, true, true, legtmp, true, true, spin, nthreads);

  mav<double,1> wgt({legtmp.shape(0)});
  clenshaw_curtis_weights(wgt);

  for (size_t i=0; i<legtmp.shape(0); ++i)
    for (size_t j=0; j<legtmp.shape(1); ++j)
      for (size_t k=0; k<legtmp.shape(2); ++k)
        legtmp.v(i,j,k) *= wgt(i);

  resample_theta(legtmp, true, true, leg, true, true, spin, nthreads);
  }
#endif

#endif

}}
