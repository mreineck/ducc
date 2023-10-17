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

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/math/wigner3j.h"
#include "ducc0/infra/timers.h"

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

void wigner3j_00_internal (double l2, double l3, double l1min, double l1max,
                           int ncoef, vmav<double,1> &res)
  {
  constexpr double srhuge=0x1p+450,
                   tiny=0x1p-900, srtiny=0x1p-450;

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

  bool last_coeff_is_negative = res(ncoef-1)<0.;

  double cnorm=1./sqrt(sumfor);
  // follow sign convention: sign(f(l_max)) = (-1)**(l2-l3+m2+m3)
  bool last_coeff_should_be_negative = nearest_int(abs(l2-l3))&1;
  if (last_coeff_is_negative != last_coeff_should_be_negative)
    cnorm = -cnorm;

  for (int k=0; k<ncoef; k+=2)
    res(k)*=cnorm;
  }

// sign convention: sign(f(l_max)) = (-1)**(l2-l3+m2+m3)
void wigner3j_internal (double l2, double l3, double m2, double m3,
                        double m1, double l1min, double l1max, int ncoef,
                        vmav<double,1> &res)
  {
  if ((m2==0.) && (m3==0.))
    return wigner3j_00_internal (l2, l3, l1min, l1max, ncoef, res);

  constexpr double srhuge=0x1p+450,
                   tiny=0x1p-900, srtiny=0x1p-450;

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
                         : -(2.*l1-1.)*l1*(m3mm2)/newfac;
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
    if (abs(ratio)>1.)
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

void wigner3j_internal_tweaked (double l2, double l3, double m2, double m3,
                        double m1, double l1min, double l1max, int ncoef,
                        vmav<double,1> &res)
  {
  if ((m2==0.) && (m3==0.))
    return wigner3j_00_internal (l2, l3, l1min, l1max, ncoef, res);

  MR_assert(res.shape(0)==size_t(ncoef), "bad size of result array");

  constexpr double srhuge=0x1p+450, tiny=0x1p-900, srtiny=0x1p-450;

  const double l2ml3sq = (l2-l3)*(l2-l3),
               pre1 = (l2+l3+1.)*(l2+l3+1.),
               m1sq = m1*m1,
               pre2 = m1*(l2*(l2+1.)-l3*(l3+1.)),
               m3mm2 = m3-m2;

  int i=0;
  double c1=0x1p+1000;
  double oldfac=0.;

  res(i) = srtiny;
  double sumfor = (2.*l1min+1.) * res(i)*res(i);

  double fct_fwd=1, fct_bwd=1;
  size_t vidx=0;
  constexpr size_t vlen = native_simd<double>::size();
  native_simd<double> l1ofs, c1v, c2v, newfacv;

  for (size_t ii=0; ii<vlen; ++ii)
    l1ofs[ii] = ii;

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
    if (i+1==ncoef) break; /* all done */
    ++i;

    const double l1 = l1min+i,
                 c1old = abs(c1);
    if (vidx==0)
      {
      auto l1 = l1min+i+l1ofs;
      auto l1sq = l1*l1;
      newfacv = sqrt((l1sq-l2ml3sq)*(pre1-l1sq)*(l1sq-m1sq));
      auto tmp1 = native_simd<double>(1.)/((l1-1.)*newfacv);
      c1v = (2.*l1-1.)*(pre2-(l1sq-l1)*m3mm2) * tmp1;
      c2v = l1*tmp1;
      }
    c1 = c1v[vidx];
    double newfac=newfacv[vidx];

    res(i) = res(i-1)*c1v[vidx] - res(i-2)*c2v[vidx]*oldfac;

    oldfac=newfac;

    sumfor += (2.*l1+1.)*res(i)*res(i);
    if (abs(res(i))>=srhuge)
      {
      for (int k=0; k<=i; ++k)
        res(k)*=srtiny;
      sumfor*=tiny;
      }
    if (c1old<=abs(c1)) break;

    vidx = (vidx+1)%vlen;
    }

  double sumbac=0.;
bool last_negative;
  int nstep2=ncoef;
  if (i+1<ncoef) /* we have to iterate from the other side */
    {
    last_negative = false;
    for (size_t ii=0; ii<vlen; ++ii)
      l1ofs[ii] = -double(ii);

    vidx=0;
    const double x1=res(i-2), x2=res(i-1), x3=res(i);
    nstep2 = i-2;

    i=ncoef-1;
    res(i) = srtiny;
    sumbac = (2.*l1max+1.) * res(i)*res(i);

    do
      {
      --i;
      const double l1 = l1min+i;

      if (vidx==0)
        {
        auto l1p1 = l1min+i+l1ofs+1.;
        auto l1p1sq = l1p1*l1p1;
        newfacv = sqrt((l1p1sq-l2ml3sq)*(pre1-l1p1sq)*(l1p1sq-m1sq));
        auto tmp1 = native_simd<double>(1.)/((l1p1+1.)*newfacv);
        c1v = (2.*l1p1+1.)*(pre2-(l1p1sq+l1p1)*m3mm2) * tmp1;
        c2v = l1p1*tmp1;
        }

      if (i<ncoef-2)
        res(i) = res(i+1)*c1v[vidx] - res(i+2)*c2v[vidx]*oldfac;
      else
        {
        const double l1p1sq = (l1+1.)*(l1+1.),
                     newfac = sqrt((l1p1sq-l2ml3sq)*(pre1-l1p1sq)*(l1p1sq-m1sq));
        res(i) = res(i+1)*(2.*l1+3.)*(pre2-(l1p1sq+l1+1.)*m3mm2)
                 /((l1+2.)*newfac);
        }
      oldfac=newfacv[vidx];

      sumbac += (2.*l1+1.)*res(i)*res(i);
      if (abs(res(i))>=srhuge)
        {
        for (int k=i; k<ncoef; ++k)
          res(k)*=srtiny;
        sumbac*=tiny;
        }

      vidx = (vidx+1)%vlen;
      }
    while (i>nstep2);

  for (size_t i=nstep2; i<min(ncoef,nstep2+3); ++i)
    {
    auto l1=l1min+i;
    sumbac -= (2.*l1+1.)*res(i)*res(i);
    }

    double ratio = (x1*res(i)+x2*res(i+1)+x3*res(i+2))
                         /(x1*x1+x2*x2+x3*x3);
    if (abs(ratio)>1.)
      { fct_bwd = 1./ratio; sumbac/=ratio*ratio; last_negative=ratio<0; }
    else
      { fct_fwd = ratio; sumfor*=ratio*ratio; }
    }
  else
    {
    last_negative = res(ncoef-1)<0.;
    }

  double cnorm=1./sqrt(sumfor+sumbac);
  bool last_should_be_negative = nearest_int(l2-l3+m2+m3)&1;
  if (last_negative != last_should_be_negative)
    cnorm = -cnorm;

  for (int k=0; k<nstep2; ++k)
    res(k)*=cnorm*fct_fwd;
  for (int k=nstep2; k<ncoef; ++k)
    res(k)*=cnorm*fct_bwd;
  }
#if 1
void wigner3j_internal_tweaked2 (double l2, double l3, double m2, double m3,
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

  int i=0;
  double c1=1e300;
  double oldfac=0.;

  MR_assert(res.shape(0)==size_t(ncoef), "bad size of result array");

  res(i) = srtiny;
  double sumfor = (2.*l1min+1.) * res(i)*res(i);

  double oldval=res(i), oldoldval=0;

  vector<int> scalefwd;
  scalefwd.push_back(0);
double ratio=0;
  double newval;
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
      newval = oldval*c1 - oldoldval*l1*oldfac*tmp1;
      }
    else
      {
      c1 = (l1>1.000001) ? (2.*l1-1.)*(pre2-(l1sq-l1)*m3mm2)/((l1-1.)*newfac)
                         : -(2.*l1-1.)*l1*(m3mm2)/newfac;
      newval = oldval*c1;
      }

    oldfac=newfac;

    sumfor += (2.*l1+1.)*newval*newval;
    if (abs(newval)>=srhuge)
      {
      scalefwd.push_back(i);
      sumfor*=tiny;
      newval*=srtiny;
      oldval*=srtiny;
      oldoldval*=srtiny;
      }
    res(i) = newval;
    if (c1old<=abs(c1)) break;
    oldoldval = oldval;
    oldval = newval;
    }

  vector<int> scalebwd;
  double sumbac=0.;
  if (i+1<ncoef) /* we have to iterate from the other side */
    {
//    const double x1=res(i-2), x2=res(i-1), x3=res(i);
    const double x1=oldoldval, x2=oldval, x3=newval;
    const int nstep2 = i-2;

    i=ncoef-1;
    res(i) = srtiny;
    oldval = srtiny;
    sumbac = (2.*l1max+1.) * res(i)*res(i);

    do
      {
      --i;

      const double l1 = l1min+i,
                   l1p1sq = (l1+1.)*(l1+1.),
                   newfac = sqrt((l1p1sq-l2ml3sq)*(pre1-l1p1sq)*(l1p1sq-m1sq));

      if (i<ncoef-2)
        newval = (oldval * (2.*l1+3.)*(pre2-(l1p1sq+l1+1.)*m3mm2)
                 -oldoldval * (l1+1.)*oldfac)
                 / ((l1+2.)*newfac);
      else
        newval = oldval*(2.*l1+3.)*(pre2-(l1p1sq+l1+1.)*m3mm2)
                 /((l1+2.)*newfac);

      oldfac=newfac;

      if (i>nstep2+2)
        sumbac += (2.*l1+1.)*newval*newval;
      if (abs(newval)>=srhuge)
        {
        scalebwd.push_back(i);
//        for (int k=i; k<ncoef; ++k)
//          res(k)*=srtiny;
        newval*=srtiny;
        oldval*=srtiny;
        sumbac*=tiny;
        }
      res(i) = newval;
      if (i>nstep2)
        {
        oldoldval = oldval;
        oldval = newval;
        }
      }
    while (i>nstep2);

    ratio = (x1*newval+x2*oldval+x3*oldoldval)
           /(x1*x1+x2*x2+x3*x3);
//    for (int k=0; k<nstep2; ++k)
//      res(k)*=ratio;
    sumfor*=ratio*ratio;
    }

  double cnorm=1./sqrt(sumfor+sumbac);
  // FIXME: this is a very shoddy fix! Try to come up with something better!
  double dtest = (res(ncoef-1)>=0) ? 1 : -1;
  double dtest2 = (abs(nearest_int(l2-l3-m1)) & 1) ? -1 : 1;
//  if (xpow(nearest_int(l2-l3-m1),dtest)<0.)
  if (dtest*dtest2<0.)
    cnorm = -cnorm;

  //double fct = cnorm;
  //for (size_t i=scalefwd.size()-1; i>0; --i)
    //{
    //for (size_t j=scalefwd[i-1]; j<scalefwd[i]; ++j)
      //res(j) *= fct;
    //fct *=srtiny;
    //}
  //fct = cnorm*ratio;
  //for (size_t i=scalebwd.size()-1; i>0; --i)
    //{
    //for (size_t j=scalebwd[i]; j<scalebwd[i-1]; ++j)
      //res(j) *= fct;
    //fct *=srtiny;
    //}
//  for (int i=nstep2; i<ncoef; ++i)
//    {
//}
  }
#endif
#if 0
class wig3j
  {
  private:
    double l2, l3, m2, m3, l1min, l1max;
    size_t ncoef;

    double A(double l) const
      {
      return sqrt((l*l - (l2-l3)*(l2-l3)) * ((l2+l3+1)*(l2+l3+1) - l*l) * (l*l - (m2+m3)*(m2+m3)));
      }
    double B(double l) const
      {
      return (2*l+1) * ((m2+m3)*(l2*(l2+1)-l3*(l3+1)) - (m2-m3)*l*(l+1));
      }
    double Xpsi(double l) const
      {
      return l*A(l+1);
      }
    double Ypsi(double l) const
      {
      return B(l);
      }
    double Zpsi(double l) const
      {
      return (l+1)*A(l);
      }
    double normalization(const vmav<double,1> &res) const
      {
      double norm = 0;
      for (size_t i=0; i<res.shape(0); ++i)
        norm += res(i)*res(i) * (2*(l1min+i)+1);
      return sqrt(abs(norm));
      }

    double f_jmax_sign() const
      {
      return (abs(int(round(j2-j3+m2+m3)))&1) ? -1 : 1;
      }

    size_t rpsi(size_t idxmid, vmav<double,1> &psi) const
      {
      size_t idx = ncoef-1;
      double l = l1max;
      while (true)
        {
        if (idx==ncoef-1)
          psi(idx) = -Zpsi(l)/Ypsi(l);
        else
          psi(idx) = -Zpsi(l)/(Ypsi(l) + Xpsi(l)*psi(idx+1));
        if (!isfinite(psi(idx)))
          {
          psi(idx) = 1.;
          return idx;
          }
        if (abs(psi(idx))>=1.)
          return idx;
        if (idx==idxmid)
          return idx;
        --idx;
        l = l1min+idx;
        }
      }
    size_t spsi(size_t idxmid, vmav<double,1> &psi) const
      {
      size_t idx = 0;
      double l = l1min;
      while (true)
        {
        if (idx==0)
          psi(idx) = -Xpsi(l)/Ypsi(l);
        else
          psi(idx) = -Xpsi(l)/(Ypsi(l) + Zpsi(l)*psi(idx-1));
        if (!isfinite(psi(idx)))
          {
          psi(idx) = 1.;
          return idx;
          }
        if (abs(psi(idx))>=1.)
          return idx;
        if (idx==idxmid)
          return idx;
        ++idx;
        l = l1min+idx;
        }
      }
    void psiauxplus(size_t idxminus, size_t idxc, vmav<double,1> &psi) const
      {
      size_t startidx = idxminus;
      if (idxminus==0)
        {
        psi(0) = 1.;
        if (l1min==0.)
          psi(1) = -(m3-m2+2*B(0.))/A(1.);
        else
          psi(1) = -(B(l1min))/(l1min*A(l1min+1.));
        startidx = 1;
        }

      for (size_t idx=startidx; idx<idxc; ++idx)
        {
        auto Xn = Xpsi(l1min+idx);
        if (Xn==0.)
          psi(idx+1) = 0.;
        else
          psi(idx+1) = -(Ypsi(idx+l1min)*psi(idx) + Zpsi(idx+l1min)*psi(idx-1))/Xn
        }
      }

  public:
    wig3j (double l2_, double l3_, double m2_, double m3_)
      : l2(l2_), l3(l3_), m2(m2_), m3(m3_)
      {
      MR_assert (l2>=abs(m2),"l2<abs(m2)");
      MR_assert (l3>=abs(m3),"l3<abs(m3)");
      MR_assert (intcheck(l2+abs(m2)),"l2+abs(m2) is not integer");
      MR_assert (intcheck(l3+abs(m3)),"l3+abs(m3) is not integer");
      const double m1 = -m2 -m3;
      l1min = max(abs(l2-l3),abs(m1));
      l1max = l2 + l3;
      MR_assert (intcheck(l1max-l1min), "l1max-l1min is not integer");
      MR_assert (l1max>=l1min, "l1max is smaller than l1min");
      ncoef = nearest_int(l1max-l1min)+1;
      }

    void calc(vmav<double,1> &w3j) const
      {
      MR_assert(w3j.shape(0)==ncoef, "bad resultarray size");
      if (ncoef==1)
        {
        w3j(0) = f_jmax_sign()/sqrt(l1min+l2+l3+1);
        return;
        }
      else if (ncoef==2)
        {
        w3j(0) = w3j(1) = 1.;
        spsi(w3j);
        double norm = 1./normalization(w3j);
        norm *= (sign(w3j(1))!=f_jmax_sign()) ? -1 : 1;
        for (size_t i=0; i<w3j.shape(0); ++i)
          w3j(i) *= norm;
        return;
        }
      }
#endif
//Returns all non-zero wigner-3j symbols
// il2 (in) : l2
// il3 (in) : l3
// im2 (in) : m2
// im3 (in) : m3
// l1min_out (out) : min value for l1
// l1max_out (out) : max value for l1
// thrcof (out) : array with the values of the wigner-3j
// size (in) : size allocated for thrcof
int drc3jj(int il2,int il3,int im2, int im3,int *l1min_out,
	   int *l1max_out,double *thrcof,int size)
{
  int sign1,sign2,nfin,im1,l1max,l1min,ii,lstep;
  int converging,nstep2,nfinp1,index,nlim;
  double newfac,c1,c2,sum1,sum2,a1,a2,a1s,a2s,dv,denom,c1old,oldfac,l1,l2,l3,m1,m2,m3;
  double x,x1,x2,x3,y,y1,y2,y3,sumfor,sumbac,sumuni,cnorm,thresh,ratio;
  double huge=sqrt(1.79E308/20.0);
  double srhuge=sqrt(huge);
  double tiny=1./huge;
  double srtiny=1./srhuge;

  im1=-im2-im3;
  l2=(double)il2; l3=(double)il3;
  m1=(double)im1; m2=(double)im2; m3=(double)im3;
  
  if((abs(il2+im2-il3+im3))%2==0)
    sign2=1;
  else
    sign2=-1;
  
  //l1 bounds
  l1max=il2+il3;
  l1min=max((abs(il2-il3)),(abs(im1)));
  *l1max_out=l1max;
  *l1min_out=l1min;

  if((il2-abs(im2)<0)||(il3-abs(im3)<0)) {
    for(ii=0;ii<=l1max-l1min;ii++)
      thrcof[ii]=0;
    return 0;
  }
MR_assert(l1max-l1min>=0, "aargh1");
  
  if(l1max==l1min) { //If it's only one value:
    thrcof[0]=sign2/sqrt(l1min+l2+l3+1);
    return 0;
  }
  else {
    nfin=l1max-l1min+1;
MR_assert(nfin<=size, "aargh2");
      {
      l1=l1min;
      newfac=0.;
      c1=0.;
      sum1=(l1+l1+1)*tiny;
      thrcof[0]=srtiny;
      
      lstep=0;
      converging=1;
      while((lstep<nfin-1)&&(converging)) { //Forward series
	lstep++;
	l1++; //order
	
	oldfac=newfac;
	a1=(l1+l2+l3+1)*(l1-l2+l3)*(l1+l2-l3)*(-l1+l2+l3+1);
	a2=(l1+m1)*(l1-m1);
	newfac=sqrt(a1*a2);
	
	if(l1>1) {
	  dv=-l2*(l2+1)*m1+l3*(l3+1)*m1+l1*(l1-1)*(m3-m2);
	  denom=(l1-1)*newfac;
	  if(lstep>1)
	    c1old=fabs(c1);
	  c1=-(l1+l1-1)*dv/denom;
	}
	else {
	  c1=-(l1+l1-1)*l1*(m3-m2)/newfac;
	}
	
	if(lstep<=1) {
	  x=srtiny*c1;
	  thrcof[1]=x;
	  sum1+=tiny*(l1+l1+1)*c1*c1;
	}
	else {
	  c2=-l1*oldfac/denom;
	  x=c1*thrcof[lstep-1]+c2*thrcof[lstep-2];
	  thrcof[lstep]=x;
	  sumfor=sum1;
	  sum1+=(l1+l1+1)*x*x;
	  if(lstep<nfin-1) {
	    if(fabs(x)>=srhuge) {
	      for(ii=0;ii<=lstep;ii++) {
		if(fabs(thrcof[ii])<srtiny)
		  thrcof[ii]=0;
		thrcof[ii]/=srhuge;
	      }
	      sum1/=huge;
	      sumfor/=huge;
	      x/=srhuge;
	    }
	    
	    if(c1old<=fabs(c1))
	      converging=0;
	  }
	}
      }
      
      if(nfin>2) {
	x1=x;
	x2=thrcof[lstep-1];
	x3=thrcof[lstep-2];
	nstep2=nfin-lstep-1+3;
	
	nfinp1=nfin+1;
	l1=l1max;
	thrcof[nfin-1]=srtiny;
	sum2=tiny*(l1+l1+1);
	
	l1+=2;
	lstep=0;
	while(lstep<nstep2-1) { //Backward series
	  lstep++;
	  l1--;
	  
	  oldfac=newfac;
	  a1s=(l1+l2+l3)*(l1-l2+l3-1)*(l1+l2-l3-1)*(-l1+l2+l3+2);
	  a2s=(l1+m1-1)*(l1-m1-1);
	  newfac=sqrt(a1s*a2s);
	  
	  dv=-l2*(l2+1)*m1+l3*(l3+1)*m1+l1*(l1-1)*(m3-m2);
	  denom=l1*newfac;
	  c1=-(l1+l1-1)*dv/denom;
	  if(lstep<=1) {
	    y=srtiny*c1;
	    thrcof[nfin-2]=y;
	    sumbac=sum2;
	    sum2+=tiny*(l1+l1-3)*c1*c1;
	  }
	  else {
	    c2=-(l1-1)*oldfac/denom;
	    y=c1*thrcof[nfin-lstep]+c2*thrcof[nfinp1-lstep]; //is the index ok??
	    if(lstep!=nstep2-1) {
	      thrcof[nfin-lstep-1]=y; //is the index ok??
	      sumbac=sum2;
	      sum2+=(l1+l1-3)*y*y;
	      if(fabs(y)>=srhuge) {
		for(ii=0;ii<=lstep;ii++) {
		  index=nfin-ii-1; //is the index ok??
		  if(fabs(thrcof[index])<srtiny)
		    thrcof[index]=0;
		  thrcof[index]=thrcof[index]/srhuge;
		}
		sum2/=huge;
		sumbac/=huge;
	      }
	    }
	  }
	}
	
	y3=y;
	y2=thrcof[nfin-lstep]; //is the index ok??
	y1=thrcof[nfinp1-lstep]; //is the index ok??
	
	ratio=(x1*y1+x2*y2+x3*y3)/(x1*x1+x2*x2+x3*x3);
	nlim=nfin-nstep2+1;
	
	if(fabs(ratio)<1) {
	  nlim++;
	  ratio=1./ratio;
	  for(ii=nlim-1;ii<nfin;ii++) //is the index ok??
	    thrcof[ii]*=ratio;
	  sumuni=ratio*ratio*sumbac+sumfor;
	}
	else {
	  for(ii=0;ii<nlim;ii++)
	    thrcof[ii]*=ratio;
	  sumuni=ratio*ratio*sumfor+sumbac;
	}
      }
      else
	sumuni=sum1;
      
      cnorm=1./sqrt(sumuni);
//      sign1 = copysign(1., thrcof[nfin-1]);
      if(thrcof[nfin-1]<0) sign1=-1;
      else sign1=1;
      
      if(sign1*sign2<=0)
	cnorm=-cnorm;
      if(fabs(cnorm)>=1) {
	for(ii=0;ii<nfin;ii++)
	  thrcof[ii]*=cnorm;
	return 0;
      }
      else {
	thresh=tiny/fabs(cnorm);
	for(ii=0;ii<nfin;ii++) {
	  if(fabs(thrcof[ii])<thresh)
	    thrcof[ii]=0;
	  thrcof[ii]*=cnorm;
	}
	return 0;
      }
    } //Size is good
  } //Doing for many l1s
  
  return 2;
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
void wigner3j_namaster (double l2, double l3, double m2, double m3, vector<double> &res)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes(l2, l3, m2, m3);
  res.resize(ncoef);
  int l1min2, l1max2;
  drc3jj (l2, l3, m2, m3, &l1min2, &l1max2, res.data(), ncoef);
  }
void wigner3j_tweaked (double l2, double l3, double m2, double m3, vector<double> &res)
  {
  auto [m1, l1min, l1max, ncoef] = wigner3j_checks_and_sizes(l2, l3, m2, m3);
  res.resize(ncoef);
  vmav<double,1> tmp(res.data(), {size_t(ncoef)});
  wigner3j_internal_tweaked (l2, l3, m2, m3, m1, l1min, l1max, ncoef, tmp);
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
