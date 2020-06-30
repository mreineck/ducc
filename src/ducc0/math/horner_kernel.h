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

/* Copyright (C) 2020 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_HORNER_KERNEL_H
#define DUCC0_HORNER_KERNEL_H

#include <cmath>
#include <array>
#include "ducc0/infra/simd.h"
#include "ducc0/infra/useful_macros.h"

namespace ducc0 {

namespace detail_horner_kernel {

using namespace std;
constexpr double pi=3.141592653589793238462643383279502884197;

/*! Class providing fast piecewise polynomial approximation of a function which
    is defined on the interval [-1;1]

    W is the number of equal-length intervals into which [-1;1] is subdivided.
    D is the degree of the approximating polynomials.
    T is the type at which the approximation is calculated;
      should be float or double. */
template<size_t W, size_t D, typename T> class HornerKernel
  {
  private:
    using Tsimd = native_simd<T>;
    static constexpr auto vlen = Tsimd::size();
    static constexpr auto nvec = (W+vlen-1)/vlen;

    array<array<Tsimd,nvec>,D+1> coeff;

    union {
      array<Tsimd,nvec> v;
      array<T,W> s;
      } res;

  public:
    template<typename Func> HornerKernel(Func func)
      {
      for (size_t i=0; i<=D; ++i)
        for (size_t j=0; j<nvec; ++j)
          coeff[i][j] = 0;
      array<double,D+1> chebroot;
      for (size_t i=0; i<=D; ++i)
        chebroot[i] = cos((2*i+1.)*pi/(2*D+2));
      for (size_t i=0; i<W; ++i)
        {
        double l = -1+2.*i/double(W);
        double r = -1+2.*(i+1)/double(W);
        // function values at Chebyshev nodes
        array<double,D+1> y;
        for (size_t j=0; j<=D; ++j)
          y[j] = func(chebroot[j]*(r-l)*0.5 + (r+l)*0.5);
        // Chebyshev coefficients
        array<double, D+1> lcf;
        for (size_t j=0; j<=D; ++j)
          {
          lcf[j] = 0;
          for (size_t k=0; k<=D; ++k)
            lcf[j] += 2./(D+1)*y[k]*cos(j*(2*k+1)*pi/(2*D+2));
          }
        lcf[0] *= 0.5;
        // Polynomial coefficients
        array<array<double,D+1>,D+1> C;
        for (size_t j=0; j<=D; ++j)
          for (size_t k=0; k<=D; ++k)
            C[j][k] = 0;
        C[0][0] = 1.;
        C[1][1] = 1.;
        for (size_t j=2; j<=D; ++j)
          {
          C[j][0] = -C[j-2][0];
          for (size_t k=1; k<=j; ++k)
            C[j][k] = 2*C[j-1][k-1] - C[j-2][k];
          }
        array<double, D+1> lcf2;
        for (size_t j=0; j<=D; ++j) lcf2[j] = 0;
        for (size_t j=0; j<=D; ++j)
          for (size_t k=0; k<=D; ++k)
            lcf2[k] += C[j][k]*lcf[j];
        for (size_t j=0; j<=D; ++j)
          coeff[j][i/vlen][i%vlen] = T(lcf2[D-j]);
        }
      }

    /*! Returns the function approximation at W different locations with the
        abscissas x, x+2./W, x+4./W, ..., x+(2.*W-2)/W.

        x must lie in [-1; -1+2./W].  */
    const array<T,W> & DUCC0_NOINLINE eval(T x)
      {
      x = (x+1)*W-1;
      for (size_t i=0; i<nvec; ++i)
        {
        auto tval = coeff[0][i];
        for (size_t j=1; j<=D; ++j)
          tval = tval*x+coeff[j][i];
        res.v[i] = tval;
        }
      return res.s; 
      }
  };

}

using detail_horner_kernel::HornerKernel;

}

#endif
