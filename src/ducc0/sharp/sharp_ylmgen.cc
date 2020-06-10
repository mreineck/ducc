/*
 *  This file is part of libsharp2.
 *
 *  libsharp2 is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp2 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp2; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* libsharp2 is being developed at the Max-Planck-Institut fuer Astrophysik */

/*
 *  Helper code for efficient calculation of Y_lm(theta,phi=0)
 *
 *  Copyright (C) 2005-2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "ducc0/sharp/sharp_internal.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"

namespace ducc0 {

namespace detail_sharp {

using namespace std;

#pragma GCC visibility push(hidden)

static inline void normalize (double &val, int &scale, double xfmax)
  {
  while (abs(val)>xfmax) { val*=sharp_fsmall; ++scale; }
  if (val!=0.)
    while (abs(val)<xfmax*sharp_fsmall) { val*=sharp_fbig; --scale; }
  }

sharp_Ylmgen::sharp_Ylmgen (size_t l_max, size_t m_max, size_t spin)
  {
  lmax = l_max;
  mmax = m_max;
  MR_assert(l_max>=spin,"incorrect l_max: must be >= spin");
  MR_assert(l_max>=m_max,"incorrect l_max: must be >= m_max");
  s = spin;
  MR_assert((sharp_minscale<=0)&&(sharp_maxscale>0),
    "bad value for min/maxscale");
  cf.resize(sharp_maxscale-sharp_minscale+1);
  cf[-sharp_minscale]=1.;
  for (int sc=-sharp_minscale-1; sc>=0; --sc)
    cf[size_t(sc)]=cf[size_t(sc+1)]*sharp_fsmall;
  for (int sc=-sharp_minscale+1; sc<(sharp_maxscale-sharp_minscale+1); ++sc)
    cf[size_t(sc)]=cf[size_t(sc-1)]*sharp_fbig;
  powlimit.resize(m_max+spin+1);
  powlimit[0]=0.;
  constexpr double expo=-400*ln2;
  for (size_t i=1; i<=m_max+spin; ++i)
    powlimit[i]=exp(expo/i);

  m = ~size_t(0);
  if (spin==0)
    {
    mfac.resize(mmax+1);
    mfac[0] = inv_sqrt4pi;
    for (size_t i=1; i<=mmax; ++i)
      mfac[i] = mfac[i-1]*sqrt((2*i+1.)/(2*i));
    root.resize(2*lmax+8);
    iroot.resize(2*lmax+8);
    for (size_t i=0; i<2*lmax+8; ++i)
      {
      root[i] = sqrt(i);
      iroot[i] = (i==0) ? 0. : 1./root[i];
      }
    eps.resize(lmax+4);
    alpha.resize(lmax/2+2);
    coef.resize(lmax/2+2);
    }
  else
    {
    m=mlo=mhi=~size_t(0);
    coef.resize(lmax+3);
    for (size_t i=0; i<lmax+3; ++i)
      coef[i].a=coef[i].b=0.;
    alpha.resize(lmax+3);
    inv.resize(lmax+2);
    inv[0]=0;
    for (size_t i=1; i<lmax+2; ++i) inv[i]=1./i;
    flm1.resize(2*lmax+3);
    flm2.resize(2*lmax+3);
    for (size_t i=0; i<2*lmax+3; ++i)
      {
      flm1[i] = sqrt(1./(i+1.));
      flm2[i] = sqrt(i/(i+1.));
      }
    prefac.resize(mmax+1);
    fscale.resize(mmax+1);
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
      size_t mlo_=s, mhi_=i;
      if (mhi_<mlo_) swap(mhi_,mlo_);
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

void sharp_Ylmgen::prepare (size_t m_)
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

vector<double> sharp_Ylmgen::get_norm (size_t lmax, size_t spin)
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

vector<double> sharp_Ylmgen::get_d1norm (size_t lmax)
  {
  vector<double> res(lmax+1);

  for (size_t l=0; l<=lmax; ++l)
    res[l] = (l<1) ? 0. : 0.5*sqrt(l*(l+1.)*(2*l+1.)/(4*pi));
  return res;
  }

#pragma GCC visibility pop

}}
