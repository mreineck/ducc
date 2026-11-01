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

/* Copyright (C) 2020-2026 Max-Planck-Society
   Author: Martin Reinecke */

#include <cmath>
#include <limits>
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/math/pswf_rokhlin.h"

namespace DUCC0_NAMESPACE {

namespace detail_gridding_kernel {

using namespace std;

// Beatty estimate for the PSWF parameter, taken from Finufft
static double get_beta(size_t W, double sigma)
  {
  return pi * W * (1. - 1./(2.*sigma)) - 0.05;
  }
// rdyn estimate from FINUFFT paper (eq. 3.21)
// Doesn't match perfectly, because it's for the ES kernel.
static double get_rdyn(size_t W, double c, double sigma)
  {
  auto tmp = 1./(2*sigma-1);
  return exp(c*(1.-sqrt(1.-tmp*tmp)));

//  PSWF0 pswf(c);
  // Here we make use of the fact pswf(0)==1
//  return 1./pswf(pi*W/(2*sigma*c));
  }
static double tol_kernel(size_t W, double sigma, size_t ndim)
  {
  // factor currently used in ducc (too conservative)
//  double tolfac = ndim;
  double tolfac = 0.18 * pow(1.4,(ndim - 1));
  constexpr double nsoff=1.5;
  return tolfac/exp((W-nsoff)*pi* sqrt(1. - 1./sigma));
  }
static double tol_rdyn(size_t W, double sigma, size_t ndim, bool singleprec)
  {
  double c = get_beta(W, sigma);
  // Dynamic range of the deconvolution correction function
  // (I may not have all factors correct here yet)
//  double rdyn = get_corfunc2(W, c, 0.)/get_corfunc2(W, c, 1./(2*sigma));
  double rdyn = get_rdyn(W, c, sigma);
  double epsmach = singleprec ? numeric_limits<float>::epsilon() : numeric_limits<double>::epsilon();
  return epsmach*rdyn*ndim;
//  return epsmach*pow(rdyn, ndim);
  }
static bool tol_is_reached(size_t W, double sigma, size_t ndim,
  bool singleprec, double epsilon)
  {
  double t1 = tol_kernel(W, sigma, ndim);
  if (t1>epsilon)
    return false;
  double t2 = tol_rdyn(W, sigma, ndim, singleprec);
  return (t1+t2) <= epsilon;
  }

double PSWF_get_best_sigma(double sigma_lo, double sigma_hi, size_t W,
  size_t ndim, bool singleprec, double epsilon)
  {
  if (tol_is_reached(W, sigma_lo, ndim, singleprec, epsilon))
    return sigma_lo;
  if (!tol_is_reached(W, sigma_hi, ndim, singleprec, epsilon))
    return -1.;
//  MR_assert(tol_is_reached(W, sigma_hi, ndim, singleprec, epsilon), "tolerance not achievable", W, " ", sigma_lo," ",sigma_hi," ",ndim," ",singleprec, " ",epsilon);
  while (sigma_hi-sigma_lo > 1e-3)
    {
    double sigmamid = (sigma_lo+sigma_hi)/2.;
    if (tol_is_reached(W, sigmamid, ndim, singleprec, epsilon))
      sigma_hi = sigmamid;
    else
      sigma_lo = sigmamid;
    }
  return sigma_hi;
 }

shared_ptr<PolynomialKernel> PSWF_selectKernel(size_t supp, double ofactor)
  {
  PSWF0 pswf(get_beta(supp, ofactor));
  return make_shared<PolynomialKernel>(supp, supp+3, pswf, GLFullCorrection(supp, pswf));
  }

double bestEpsilon(size_t ndim, bool singleprec,
  double ofactor_min, double ofactor_max)
  {
  MR_assert((ndim>=1) && (ndim<=3), "bad dimensionality");
//FIXME!
return 0;
  }

}}
