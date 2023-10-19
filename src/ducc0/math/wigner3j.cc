/*
 *  This file is part of ducc0.
 *
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

/*
 *  ducc0 is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file ducc0/math/wigner3j.cc
 *  Computation of Wigner-3j symbols
 *  Algorithm implemented according to Schulten & Gordon:
 *  J. Math. Phys. 16, p. 10 (1975)
 *
 *  Copyright (C) 2009-2023 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <cmath>
#include <cstdlib>
#include <vector>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/math/wigner3j.h"

namespace ducc0 {

namespace detail_wigner3j {

using namespace std;

static inline int nearest_int (double arg)
  { return int(round(arg)); }

static inline bool intcheck (double val)
  { return abs(val-round(val))<1e-13; }

static inline double xpow (int m, double val)
  { return (m&1) ? -val : val; }

auto wigner3j_checks_and_sizes(double l2, double l3, double m2, double m3)
  {
  MR_assert (l2>=abs(m2),"l2<abs(m2)");
  MR_assert (l3>=abs(m3),"l3<abs(m3)");
  MR_assert (intcheck(l2+abs(m2)),"l2+abs(m2) is not integer");
  MR_assert (intcheck(l3+abs(m3)),"l3+abs(m3) is not integer");
  const double m1 = -m2 -m3;
  const double l1min = max(abs(l2-l3),abs(m1)),
               l1max = l2 + l3;
  MR_assert (intcheck(l1max-l1min), "l1max-l1min is not integer");
  MR_assert (l1max>=l1min, "l1max is smaller than l1min");
  const int ncoef = nearest_int(l1max-l1min)+1;

  return make_tuple(m1, l1min, l1max, ncoef);
  }

auto wigner3j_checks_and_sizes_int(int l2, int l3, int m2, int m3)
  {
  MR_assert (l2>=abs(m2),"l2<abs(m2)");
  MR_assert (l3>=abs(m3),"l3<abs(m3)");
  const int m1 = -m2 -m3;
  const int l1min = max(abs(l2-l3),abs(m1)),
                   l1max = l2 + l3;
  MR_assert (l1max>=l1min, "l1max is smaller than l1min");
  const int ncoef = l1max-l1min+1;

  return make_tuple(m1, l1min, l1max, ncoef);
  }

// version for m2==m3==0
void wigner3j_00_internal (double l2, double l3, double l1min, double l1max,
                           int ncoef, vmav<double,1> &res)
  {
  constexpr double srhuge=0x1p+250,
                   tiny=0x1p-500, srtiny=0x1p-250;

  const double l2ml3sq = (l2-l3)*(l2-l3),
               pre1 = (l2+l3+1.)*(l2+l3+1.);

  MR_assert(res.shape(0)==size_t(ncoef), "bad size of result array");

  res(0) = srtiny;
  double sumfor = (2.*l1min+1.) * res(0)*res(0);

  for (int i=0; i+2<ncoef; i+=2)
    {
    double l1 = l1min+i+1,
           l1sq = l1*l1;

    res(i+1) = 0.;

    double l1p1 = l1+1;
    double l1p1sq = l1p1*l1p1;

    const double tmp1 = sqrt(((l1sq-l2ml3sq)*(pre1-l1sq))
                             /((l1p1sq-l2ml3sq)*(pre1-l1p1sq)));
    res(i+2) = -res(i)*tmp1;

    sumfor += (2.*l1p1+1.)*res(i+2)*res(i+2);
    if (abs(res(i+2))>=srhuge)
      {
      for (int k=0; k<=i+2; k+=2)
        res(k)*=srtiny;
      sumfor*=tiny;
      }
    }

  bool last_coeff_is_negative = (((ncoef+1)/2)&1) == 0;
  double cnorm=1./sqrt(sumfor);
  // follow sign convention: sign(f(l_max)) = (-1)**(l2-l3+m2+m3)
  bool last_coeff_should_be_negative = nearest_int(abs(l2-l3))&1;
  if (last_coeff_is_negative != last_coeff_should_be_negative)
    cnorm = -cnorm;

  for (int k=0; k<ncoef; k+=2)
    res(k)*=cnorm;
  }


template<size_t bufsize> void wigner3j_internal_block
  (double l2, double l3, double m2, double m3,
   double m1, double l1min, double l1max, int ncoef,
   vmav<double,1> &res)
  {
  constexpr double srhuge=0x1p+250,
                   tiny=0x1p-500, srtiny=0x1p-250;

  const double l2ml3sq = (l2-l3)*(l2-l3),
               pre1 = (l2+l3+1.)*(l2+l3+1.),
               m1sq = m1*m1,
               pre2 = m1*(l2*(l2+1.)-l3*(l3+1.)),
               m3mm2 = m3-m2;

  using Tv = native_simd<double>;
  constexpr size_t vlen=Tv::size();
  constexpr size_t nvec=bufsize/vlen;
  static_assert(nvec*vlen==bufsize, "illegal bufsize");
  union Tbuf
    {
    array<double, bufsize> s;
    array<Tv, nvec> v;
    };
  Tbuf l1ladder, c1v, c2v, newfacv;

  double c1=0x1p+1000;
  double oldfac=0.;
  for (size_t m=0; m<bufsize; ++m)
    l1ladder.s[m] = m;

  MR_assert(res.shape(0)==size_t(ncoef), "bad size of result array");

  int i=0;
  res(i) = srtiny;
  double sumfor = (2.*l1min+1.) * res(i)*res(i);

  // do first iteration separately
  if (i+1<ncoef)
    {
    ++i;

    const double l1 = l1min+i;
    const double l1sq = l1*l1,
                 c1old = abs(c1),
                 newfac = sqrt((l1sq-l2ml3sq)*(pre1-l1sq)*(l1sq-m1sq));
    c1 = (l1>1.000001) ? (2.*l1-1.)*(pre2-(l1sq-l1)*m3mm2)/((l1-1.)*newfac)
                       : -(2.*l1-1.)*l1*(m3mm2)/newfac;
    res(i) = res(i-1)*c1;
  
    oldfac=newfac;
  
    sumfor += (2.*l1+1.)*res(i)*res(i);
    if (abs(res(i))>=srhuge)
      {
      for (int k=0; k<=i; ++k)
        res(k)*=srtiny;
      sumfor*=tiny;
      }
    }

  while(true)
    {
// prepare buffers
    for (size_t m=0; m<nvec; ++m)
      {
      auto l1v = double(l1min+i+1)+l1ladder.v[m];
      auto l1sq = l1v*l1v;
      newfacv.v[m] = sqrt((l1sq-l2ml3sq)*(pre1-l1sq)*(l1sq-m1sq));
      auto tmp1 = native_simd<double>(1.)/((l1v-1.)*newfacv.v[m]);
      c1v.v[m] = (2.*l1v-1.)*(pre2-(l1sq-l1v)*m3mm2) * tmp1;
      c2v.v[m] = l1v*tmp1;
      }

    int ilim = min(ncoef, i+1+int(bufsize));
    int vidx=0;
    while(i+1<ilim)
      {
      ++i;
      const double l1 = l1min+i,
                   c1old = abs(c1);
      c1 = c1v.s[vidx];
      res(i) = res(i-1)*c1 - res(i-2)*c2v.s[vidx]*oldfac;

      oldfac=newfacv.s[vidx];

      sumfor += (2.*l1+1.)*res(i)*res(i);
      if (abs(res(i))>=srhuge)
        {
        for (int k=0; k<=i; ++k)
          res(k)*=srtiny;
        sumfor*=tiny;
        }
      if (c1old<=abs(c1)) goto bailout_fwd;
      ++vidx;
      }
    if (i+1>=ncoef) goto bailout_fwd;
    }

bailout_fwd:

  double sumbac=0.;
  bool last_coeff_is_negative=false;
  double fct_fwd=1., fct_bwd=1.;
  int nstep2=ncoef;

  if (i+1<ncoef) /* we have to iterate from the other side */
    {
    const double x1=res(i-2), x2=res(i-1), x3=res(i);
    nstep2 = i-2;

    i=ncoef-1;
    res(i) = srtiny;
    sumbac = (2.*l1max+1.) * res(i)*res(i);

    {
    --i;

    const double l1 = l1min+i,
                 l1p1sq = (l1+1.)*(l1+1.),
                 newfac = sqrt((l1p1sq-l2ml3sq)*(pre1-l1p1sq)*(l1p1sq-m1sq));

    res(i) = res(i+1)*(2.*l1+3.)*(pre2-(l1p1sq+l1+1.)*m3mm2)
             /((l1+2.)*newfac);

    oldfac=newfac;

    sumbac += (2.*l1+1.)*res(i)*res(i);
    if (abs(res(i))>=srhuge)
      {
      for (int k=i; k<ncoef; ++k)
        res(k)*=srtiny;
      sumbac*=tiny;
      }
    }

    for (size_t m=0; m<bufsize; ++m)
      l1ladder.s[m] = -int(m);
  
    while(true)
      {
      // prepare buffers
      for (size_t m=0; m<nvec; ++m)
        {
        auto l1p1v = double(l1min+i)+l1ladder.v[m];
        auto l1p1sq = l1p1v*l1p1v;
        newfacv.v[m] = sqrt((l1p1sq-l2ml3sq)*(pre1-l1p1sq)*(l1p1sq-m1sq));
        auto tmp1 = native_simd<double>(1.)/((l1p1v+1.)*newfacv.v[m]);
        c1v.v[m] = (2.*l1p1v+1.)*(pre2-(l1p1sq+l1p1v)*m3mm2) * tmp1;
        c2v.v[m] = l1p1v*tmp1;
        }
  
      int ilim = max(nstep2, i-int(bufsize));
      int vidx=0;
      while(i>ilim)
        {
        --i;
        const double l1 = l1min+i;
        res(i) = res(i+1)*c1v.s[vidx] - res(i+2)*c2v.s[vidx]*oldfac;
        oldfac=newfacv.s[vidx];
  
        sumbac += (2.*l1+1.)*res(i)*res(i);
        if (abs(res(i))>=srhuge)
          {
          for (int k=i; k<ncoef; ++k)
            res(k)*=srtiny;
          sumbac*=tiny;
          }
        ++vidx;
        }
      if (i<=nstep2) goto bailout_bwd;
      }

bailout_bwd:

    for (size_t i=nstep2; i<min(ncoef,nstep2+3); ++i)
      {
      auto l1=l1min+i;
      sumbac -= (2.*l1+1.)*res(i)*res(i);
      }

    const double ratio = (x1*res(i)+x2*res(i+1)+x3*res(i+2))
                         /(x1*x1+x2*x2+x3*x3);
    if (abs(ratio)<1.)
      { fct_bwd = 1./ratio; sumbac/=ratio*ratio; last_coeff_is_negative=ratio<0; }
    else
      { fct_fwd = ratio; sumfor*=ratio*ratio; }
    }
  else
    {
    last_coeff_is_negative = res(ncoef-1)<0.;
    }

  double cnorm=1./sqrt(sumfor+sumbac);
  // follow sign convention: sign(f(l_max)) = (-1)**(l2-l3+m2+m3)
  bool last_coeff_should_be_negative = nearest_int(abs(l2-l3+m2+m3))&1;
  if (last_coeff_is_negative != last_coeff_should_be_negative)
    cnorm = -cnorm;

  for (int k=0; k<nstep2; ++k)
    res(k)*=cnorm*fct_fwd;
  for (int k=nstep2; k<ncoef; ++k)
    res(k)*=cnorm*fct_bwd;
  }

// sign convention: sign(f(l_max)) = (-1)**(l2-l3+m2+m3)
void wigner3j_internal (double l2, double l3, double m2, double m3,
                        double m1, double l1min, double l1max, int ncoef,
                        vmav<double,1> &res)
  {
  if ((m2==0.) && (m3==0.))
    return wigner3j_00_internal (l2, l3, l1min, l1max, ncoef, res);

  if constexpr (native_simd<double>::size()>=4)
    return wigner3j_internal_block<16> (l2, l3, m2, m3, m1, l1min, l1max, ncoef, res);

  constexpr double srhuge=0x1p+250,
                   tiny=0x1p-500, srtiny=0x1p-250;

  const double l2ml3sq = (l2-l3)*(l2-l3),
               pre1 = (l2+l3+1.)*(l2+l3+1.),
               m1sq = m1*m1,
               pre2 = m1*(l2*(l2+1.)-l3*(l3+1.)),
               m3mm2 = m3-m2;

  int i=0;
  double c1=0x1p+1000;
  double oldfac=0.;

  MR_assert(res.shape(0)==size_t(ncoef), "bad size of result array");

  res(i) = srtiny;
  double sumfor = (2.*l1min+1.) * res(i)*res(i);

  while(true)
    {
    if (i+1==ncoef) break; /* all done */
    ++i;

    const double l1 = l1min+i,
                 l1sq = l1*l1,
                 c1old = abs(c1),
                 newfac = sqrt((l1sq-l2ml3sq)*(pre1-l1sq)*(l1sq-m1sq));

    if (i>1)
      {
      const double tmp1 = 1./((l1-1.)*newfac);
      c1 = (2.*l1-1.)*(pre2-(l1sq-l1)*m3mm2) * tmp1;
      res(i) = res(i-1)*c1 - res(i-2)*l1*oldfac*tmp1;
      }
    else
      {
      c1 = (l1>1.000001) ? (2.*l1-1.)*(pre2-(l1sq-l1)*m3mm2)/((l1-1.)*newfac)
                         : -(2.*l1-1.)*l1*m3mm2/newfac;
      res(i) = res(i-1)*c1;
      }

    oldfac=newfac;

    sumfor += (2.*l1+1.)*res(i)*res(i);
    if (abs(res(i))>=srhuge)
      {
      for (int k=0; k<=i; ++k)
        res(k)*=srtiny;
      sumfor*=tiny;
      }
    if (c1old<=abs(c1)) break;
    }

  double sumbac=0.;
  bool last_coeff_is_negative=false;
  double fct_fwd=1., fct_bwd=1.;
  int nstep2=ncoef;

  if (i+1<ncoef) /* we have to iterate from the other side */
    {
    const double x1=res(i-2), x2=res(i-1), x3=res(i);
    nstep2 = i-2;

    i=ncoef-1;
    res(i) = srtiny;
    sumbac = (2.*l1max+1.) * res(i)*res(i);

    do
      {
      --i;

      const double l1 = l1min+i,
                   l1p1sq = (l1+1.)*(l1+1.),
                   newfac = sqrt((l1p1sq-l2ml3sq)*(pre1-l1p1sq)*(l1p1sq-m1sq));

      if (i<ncoef-2)
        res(i) = (res(i+1) * (2.*l1+3.)*(pre2-(l1p1sq+l1+1.)*m3mm2)
                 -res(i+2) * (l1+1.)*oldfac)
                 / ((l1+2.)*newfac);
      else
        res(i) = res(i+1)*(2.*l1+3.)*(pre2-(l1p1sq+l1+1.)*m3mm2)
                 /((l1+2.)*newfac);

      oldfac=newfac;

      sumbac += (2.*l1+1.)*res(i)*res(i);
      if (abs(res(i))>=srhuge)
        {
        for (int k=i; k<ncoef; ++k)
          res(k)*=srtiny;
        sumbac*=tiny;
        }
      }
    while (i>nstep2);

    for (size_t i=nstep2; i<min(ncoef,nstep2+3); ++i)
      {
      auto l1=l1min+i;
      sumbac -= (2.*l1+1.)*res(i)*res(i);
      }

    const double ratio = (x1*res(i)+x2*res(i+1)+x3*res(i+2))
                         /(x1*x1+x2*x2+x3*x3);
    if (abs(ratio)<1.)
      { fct_bwd = 1./ratio; sumbac/=ratio*ratio; last_coeff_is_negative=ratio<0; }
    else
      { fct_fwd = ratio; sumfor*=ratio*ratio; }
    }
  else
    {
    last_coeff_is_negative = res(ncoef-1)<0.;
    }

  double cnorm=1./sqrt(sumfor+sumbac);
  // follow sign convention: sign(f(l_max)) = (-1)**(l2-l3+m2+m3)
  bool last_coeff_should_be_negative = nearest_int(abs(l2-l3+m2+m3))&1;
  if (last_coeff_is_negative != last_coeff_should_be_negative)
    cnorm = -cnorm;

  for (int k=0; k<nstep2; ++k)
    res(k)*=cnorm*fct_fwd;
  for (int k=nstep2; k<ncoef; ++k)
    res(k)*=cnorm*fct_bwd;
  }

void wigner3j (double l2, double l3, double m2, double m3, vmav<double,1> &res)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes(l2, l3, m2, m3);
  wigner3j_internal (l2, l3, m2, m3, m1, l1min, l1max, ncoef, res);  
  }
void wigner3j (double l2, double l3, double m2, double m3, vector<double> &res)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes(l2, l3, m2, m3);
  res.resize(ncoef);
  vmav<double,1> tmp(res.data(), {size_t(ncoef)});
  wigner3j_internal (l2, l3, m2, m3, m1, l1min, l1max, ncoef, tmp);
  }

void wigner3j_int (int l2, int l3, int m2, int m3, int &l1min_, vmav<double,1> &res)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes_int (l2, l3, m2, m3);
  wigner3j_internal (l2, l3, m2, m3, double(m1), double(l1min), double(l1max), ncoef, res);
  l1min_ = l1min;
  }
void wigner3j_int (int l2, int l3, int m2, int m3, int &l1min_, vector<double> &res)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes_int (l2, l3, m2, m3);
  res.resize(ncoef);
  vmav<double,1> tmp(res.data(), {size_t(ncoef)});
  wigner3j_internal (l2, l3, m2, m3, double(m1), double(l1min), double(l1max), ncoef, tmp);
  l1min_ = l1min;
  }
int wigner3j_ncoef_int (int l2, int l3, int m2, int m3)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes_int (l2, l3, m2, m3);
  return ncoef;
  }

}}
