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

/* Copyright (C) 2020-2022 Max-Planck-Society
   Author: Martin Reinecke */

#include "ducc0/math/gridding_kernel.h"

namespace ducc0 {

namespace detail_gridding_kernel {

using namespace std;

vector<double> getCoeffs(size_t W, size_t D, const function<double(double)> &func)
  {
  vector<double> coeff(W*(D+1));
  vector<double> chebroot(D+1);
  for (size_t i=0; i<=D; ++i)
    chebroot[i] = cos((2*i+1.)*pi/(2*D+2));
  vector<double> y(D+1), lcf(D+1), C((D+1)*(D+1)), lcf2(D+1);
  for (size_t i=0; i<W; ++i)
    {
    double l = -1+2.*i/double(W);
    double r = -1+2.*(i+1)/double(W);
    // function values at Chebyshev nodes
    double avg = 0;
    for (size_t j=0; j<=D; ++j)
      {
      y[j] = func(chebroot[j]*(r-l)*0.5 + (r+l)*0.5);
      avg += y[j];
      }
    avg/=(D+1);
    for (size_t j=0; j<=D; ++j)
      y[j] -= avg;
    // Chebyshev coefficients
    for (size_t j=0; j<=D; ++j)
      {
      lcf[j] = 0;
      for (size_t k=0; k<=D; ++k)
        lcf[j] += 2./(D+1)*y[k]*cos(j*(2*k+1)*pi/(2*D+2));
      }
    lcf[0] *= 0.5;
    // Polynomial coefficients
    fill(C.begin(), C.end(), 0.);
    C[0] = 1.;
    C[1*(D+1) + 1] = 1.;
    for (size_t j=2; j<=D; ++j)
      {
      C[j*(D+1) + 0] = -C[(j-2)*(D+1) + 0];
      for (size_t k=1; k<=j; ++k)
        C[j*(D+1) + k] = 2*C[(j-1)*(D+1) + k-1] - C[(j-2)*(D+1) + k];
      }
    for (size_t j=0; j<=D; ++j) lcf2[j] = 0;
    for (size_t j=0; j<=D; ++j)
      for (size_t k=0; k<=D; ++k)
        lcf2[k] += C[j*(D+1) + k]*lcf[j];
    lcf2[0] += avg;
    for (size_t j=0; j<=D; ++j)
      coeff[j*W + i] = lcf2[D-j];
    }
  return coeff;
  }

shared_ptr<PolynomialKernel> selectKernel(size_t idx)
  {
  MR_assert(idx<KernelDB.size(), "no appropriate kernel found");
  auto supp = KernelDB[idx].W;
  auto beta = KernelDB[idx].beta*supp;
  auto e0 = KernelDB[idx].e0;
  auto lam = [beta,e0](double v){return esknew(v, beta, e0);};
  return make_shared<PolynomialKernel>(supp, supp+3, lam, GLFullCorrection(supp, lam));
  }

}}
