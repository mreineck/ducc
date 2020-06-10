/*
 *  This file is part of the MR utility library.
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

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_ES_KERNEL_H
#define DUCC0_ES_KERNEL_H

#include <cmath>
#include <vector>
#include <array>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gl_integrator.h"

namespace ducc0 {

namespace detail_es_kernel {

using namespace std;

/* This class implements the "exponential of semicircle" gridding kernel
   described in https://arxiv.org/abs/1808.06736 */
class ES_Kernel
  {
  private:
    double beta;
    int p;
    vector<double> x, wgt, psi;
    size_t supp;

  public:
    ES_Kernel(size_t supp_, double ofactor, size_t nthreads)
      : beta(get_beta(supp_,ofactor)*supp_),
        p(int(1.5*supp_+2)), supp(supp_)
      {
      GL_Integrator integ(2*p,nthreads);
      x = integ.coordsSymmetric();
      wgt = integ.weightsSymmetric();
      psi=x;
      for (auto &v:psi)
        v=operator()(v);
      }
    ES_Kernel(size_t supp_, size_t nthreads)
      : ES_Kernel(supp_, 2., nthreads){}

    template<typename T> T operator()(T v) const
      {
      auto tmp = (1-v)*(1+v);
      auto tmp2 = tmp>=0;
      return tmp2*exp(T(beta)*(sqrt(tmp*tmp2)-1));
      }
    /* Compute correction factors for the ES gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    double corfac(double v) const
      {
      double tmp=0;
      for (int i=0; i<p; ++i)
        tmp += wgt[i]*psi[i]*cos(pi*supp*v*x[i]);
      return 2./(supp*tmp);
      }
    /* Compute correction factors for the ES gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    vector<double> correction_factors(size_t n, size_t nval, size_t nthreads)
      {
      vector<double> res(nval);
      double xn = 1./n;
      execStatic(nval, nthreads, 0, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          res[i] = corfac(i*xn);
        });
      return res;
      }
    static double get_beta(size_t supp, double ofactor=2)
      {
      MR_assert((supp>=2) && (supp<=15), "unsupported support size");
      if (ofactor>=2)
        {
        static const array<double,16> opt_beta {-1, 0.14, 1.70, 2.08, 2.205, 2.26,
          2.29, 2.307, 2.316, 2.3265, 2.3324, 2.282, 2.294, 2.304, 2.3138, 2.317};
        MR_assert(supp<opt_beta.size(), "bad support size");
        return opt_beta[supp];
        }
      if (ofactor>=1.175)
        {
        // empirical, but pretty accurate approximation
        static const array<double,16> betacorr{0,0,-0.51,-0.21,-0.1,-0.05,-0.025,-0.0125,0,0,0,0,0,0,0,0};
        auto x0 = 1./(2*ofactor);
        auto bcstrength=1.+(x0-0.25)*2.5;
        return 2.32+bcstrength*betacorr[supp]+(0.25-x0)*3.1;
        }
      MR_fail("oversampling factor is too small");
      }

    static size_t get_supp(double epsilon, double ofactor=2)
      {
      double epssq = epsilon*epsilon;
      if (ofactor>=2)
        {
        static const array<double,16> maxmaperr { 1e8, 0.19, 2.98e-3, 5.98e-5,
          1.11e-6, 2.01e-8, 3.55e-10, 5.31e-12, 8.81e-14, 1.34e-15, 2.17e-17,
          2.12e-19, 2.88e-21, 3.92e-23, 8.21e-25, 7.13e-27 };

        for (size_t i=2; i<maxmaperr.size(); ++i)
          if (epssq>maxmaperr[i]) return i;
        MR_fail("requested epsilon too small - minimum is 1e-13");
        }
      if (ofactor>=1.175)
        {
        for (size_t w=2; w<16; ++w)
          {
          auto estimate = 12*exp(-2.*w*ofactor); // empirical, not very good approximation
          if (epssq>estimate) return w;
          }
        MR_fail("requested epsilon too small");
        }
      MR_fail("oversampling factor is too small");
      }
  };

}

using detail_es_kernel::ES_Kernel;

}

#endif
