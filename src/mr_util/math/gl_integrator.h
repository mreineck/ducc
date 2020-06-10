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

#ifndef DUCC0_GL_INTEGRATOR_H
#define DUCC0_GL_INTEGRATOR_H

#include <cmath>
#include "mr_util/math/constants.h"
#include "mr_util/infra/error_handling.h"
#include "mr_util/infra/threading.h"

namespace mr {

namespace detail_gl_integrator {

using namespace std;

class GL_Integrator
  {
  private:
    size_t n_;
    vector<double> x, w;

    static inline double one_minus_x2 (double x)
      { return (std::abs(x)>0.1) ? (1.+x)*(1.-x) : 1.-x*x; }

  public:
    GL_Integrator(size_t n, size_t nthreads=1)
      : n_(n)
      {
      MR_assert(n>=1, "number of points must be at least 1");
      constexpr double eps = 3e-14;
      size_t m = (n+1)>>1;
      x.resize(m);
      w.resize(m);

      double dn=double(n);
      const double t0 = 1 - (1-1./dn) / (8.*dn*dn);
      const double t1 = 1./(4.*dn+2.);

      execDynamic(m, nthreads, 1, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo+1; i<rng.hi+1; ++i)
          {
          double x0 = cos(pi * double((i<<2)-1) * t1) * t0;

          bool dobreak=false;
          size_t j=0;
          double dpdx;
          while(1)
            {
            double P_1 = 1.0;
            double P0 = x0;
            double dx, x1;

            for (size_t k=2; k<=n; k++)
              {
              double P_2 = P_1;
              P_1 = P0;
//              P0 = ((2*k-1)*x0*P_1-(k-1)*P_2)/k;
              P0 = x0*P_1 + (k-1.)/k * (x0*P_1-P_2);
              }

            dpdx = (P_1 - x0*P0) * n / one_minus_x2(x0);

            /* Newton step */
            x1 = x0 - P0/dpdx;
            dx = x0-x1;
            x0 = x1;
            if (dobreak) break;

            if (std::abs(dx)<=eps) dobreak=1;
            MR_assert(++j<100, "convergence problem");
            }

          x[m-i] = x0;
          w[m-i] = 2. / (one_minus_x2(x0) * dpdx * dpdx);
          }
        });
      if (n&1) x[0] = 0.; // set to exact zero
      }

    template<typename Func> auto integrate(Func f) -> decltype(f(0.))
      {
      using T = decltype(f(0.));
      T res=0;
      size_t istart=0;
      if (n_&1)
        {
        res = f(x[0])*w[0];
        istart=1;
        }
      for (size_t i=istart; i<x.size(); ++i)
        res += (f(x[i])+f(-x[i]))*w[i];
      return res;
      }

    template<typename Func> auto integrateSymmetric(Func f) -> decltype(f(0.))
      {
      using T = decltype(f(0.));
      T res=f(x[0])*w[0];
      if (n_&1) res *= 0.5;
      for (size_t i=1; i<x.size(); ++i)
        res += f(x[i])*w[i];
      return res*2;
      }

    vector<double> coords() const
      {
      vector<double> res(n_);
      for (size_t i=0; i<x.size(); ++i)
        {
        res[i]=-x[x.size()-1-i];
        res[n_-1-i] = x[x.size()-1-i];
        }
      return res;
      }
    const vector<double> &coordsSymmetric() const
      { return x; }

    vector<double> weights() const
      {
      vector<double> res(n_);
      for (size_t i=0; i<w.size(); ++i)
        res[i]=res[n_-1-i]=w[w.size()-1-i];
      return res;
      }
    vector<double> weightsSymmetric() const
      {
      auto res = w;
      if (n_&1) res[0]*=0.5;
      for (auto &v:res) v*=2;
      return res;
      }
  };

}

using detail_gl_integrator::GL_Integrator;

}

#endif
