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

/* Copyright (C) 2019 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef MRUTIL_GL_INTEGRATOR_H
#define MRUTIL_GL_INTEGRATOR_H

#include <cmath>
#include "mr_util/error_handling.h"
#include "mr_util/threading.h"

namespace mr {

namespace gl_integrator {

namespace detail {

using namespace std;
using namespace mr::threading;

class GL_Integrator
  {
  private:
    int m;
    vector<double> x, w;

    static inline double one_minus_x2 (double x)
      { return (abs(x)>0.1) ? (1.+x)*(1.-x) : 1.-x*x; }

  public:
    GL_Integrator(int n, size_t nthreads=1)
      {
      MR_assert(n>=1, "number of points must be at least 1");
      constexpr double pi = 3.141592653589793238462643383279502884197;
      constexpr double eps = 3e-14;
      m = (n+1)>>1;
      x.resize(m);
      w.resize(m);

      const double t0 = 1 - (1-1./n) / (8.*n*n);
      const double t1 = 1./(4.*n+2.);

      execDynamic(m, nthreads, 1, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo+1; i<rng.hi+1; ++i)
          {
          double x0 = cos(pi * ((i<<2)-1) * t1) * t0;

          int dobreak=0;
          int j=0;
          double dpdx;
          while(1)
            {
            double P_1 = 1.0;
            double P0 = x0;
            double dx, x1;

            for (int k=2; k<=n; k++)
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

            if (abs(dx)<=eps) dobreak=1;
            MR_assert(++j<100, "convergence problem");
            }

          x[m-i] = x0;
          w[m-i] = 2. / (one_minus_x2(x0) * dpdx * dpdx);
          }
        });
      if (n&1) x[0] = 0.; // set to exact zero
      }

    template<typename Func> double integrate(Func f)
      {
      double res=0, istart=0;
      if (x[0]==0.)
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
      if (x[0]==0.) res *= 0.5;
      for (size_t i=1; i<x.size(); ++i)
        res += f(x[i])*w[i];
      return res*2;
      }
  };

}

using detail::GL_Integrator;

}}

#endif
