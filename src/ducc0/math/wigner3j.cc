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

void wigner3j_internal (double l2, double l3, double m2, double m3,
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
  if (i+1<ncoef) /* we have to iterate from the other side */
    {
    const double x1=res(i-2), x2=res(i-1), x3=res(i);
    const int nstep2 = i-2;

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

      if (i>nstep2+2)
        sumbac += (2.*l1+1.)*res(i)*res(i);
      if (abs(res(i))>=srhuge)
        {
        for (int k=i; k<ncoef; ++k)
          res(k)*=srtiny;
        sumbac*=tiny;
        }
      }
    while (i>nstep2);

    const double ratio = (x1*res(i)+x2*res(i+1)+x3*res(i+2))
                         /(x1*x1+x2*x2+x3*x3);
    for (int k=0; k<nstep2; ++k)
      res(k)*=ratio;
    sumfor*=ratio*ratio;
    }

  double cnorm=1./sqrt(sumfor+sumbac);
  // FIXME: this is a very shoddy fix! Try to come up with something better!
  double dtest = (res(ncoef-1)>=0) ? 1 : -1;
  if (xpow(nearest_int(l2-l3-m1),dtest)<0.)
    cnorm = -cnorm;

  for (int k=0; k<ncoef; ++k)
    res(k)*=cnorm;
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
