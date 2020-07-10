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

#ifndef DUCC0_GRIDDING_KERNEL_H
#define DUCC0_GRIDDING_KERNEL_H

#include <vector>
#include "ducc0/infra/simd.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/math/least_misfit.h"
#include "ducc0/math/es_kernel.h"

namespace ducc0 {

namespace detail_gridding_kernel {

using namespace std;
constexpr double pi=3.141592653589793238462643383279502884197;


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
    for (size_t j=0; j<=D; ++j)
      y[j] = func(chebroot[j]*(r-l)*0.5 + (r+l)*0.5);
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
    for (size_t j=0; j<=D; ++j)
      coeff[j*W + i] = lcf2[D-j];
    }
  return coeff;
  }


/*! A GriddingKernel is considered to be a symmetric real-valued function
    defined on the interval [-1; 1].
    This range is subdivided into W equal-sized parts. */
template<typename T> class GriddingKernel
  {
  public:
    virtual ~GriddingKernel() {}

    virtual size_t support() const = 0;

    /*! Returns the function approximation at W different locations with the
        abscissas x, x+2./W, x+4./W, ..., x+(2.*W-2)/W.
        x must lie in [-1; -1+2./W].
        NOTE: res must point to memory large enough to hold
        ((W+vlen-1)/vlen) objects of type native_simd<T>!
        */
    virtual void eval(T x, native_simd<T> *res) const = 0;
    /*! Returns the function approximation at location x.
        x must lie in [-1; 1].  */
    virtual T eval_single(T x) const = 0;


    /* Computes the correction function at a given coordinate.
       Useful coordinates lie in the range [0; 0.5]. */
    virtual double corfunc(double x) const = 0;

    /* Computes the correction function values at a coordinates
       [0, dx, 2*dx, ..., (n-1)*dx]  */
    virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const = 0;
  };

class PiecewiseKernelCorrection
  {
  private:
    static constexpr size_t ideg=10; // integration points per interval
    static_assert((ideg&1)==0, "ideg must be even");
    vector<double> x, wgtpsi;
    size_t supp;

  public:
    PiecewiseKernelCorrection(size_t W, const function<double(double)> &func)
      : supp(W)
      {
      // we know that the kernel is piecewise smooth in all W sections but not
      // necessarily at borders between sections. Therefore we integrate piece
      // by piece.
      GL_Integrator integ(ideg, 1);
      auto xgl = integ.coords();
      auto wgl = integ.weights();
      x.resize((supp*ideg)/2);
      wgtpsi.resize((supp*ideg)/2);
      for (size_t i=0; i<x.size(); ++i)
        {
        size_t iiv = i/ideg;
        x[i] = -1. + iiv*2./supp + (1.+xgl[i%ideg])/supp;
        wgtpsi[i] = wgl[i%ideg]*func(x[i]);
        }
      }

    /* Compute correction factors for gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    double corfunc(double v) const
      {
      double tmp=0;
      for (size_t i=0; i<x.size(); ++i)
        tmp += wgtpsi[i]*cos(pi*supp*v*x[i]);
      return 1./tmp;
      }
    /* Compute correction factors for gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      {
      vector<double> res(n);
      execStatic(n, nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          res[i] = corfunc(i*dx);
        });
      return res;
      }
  };

class GLFullCorrection
  {
  private:
    vector<double> x, wgtpsi;
    size_t supp;

  public:
    GLFullCorrection(size_t W, const function<double(double)> &func)
      : supp(W)
      {
      size_t p = size_t(1.5*W)+2;
      GL_Integrator integ(2*p,1);
      x = integ.coordsSymmetric();
      wgtpsi = integ.weightsSymmetric();
      for (size_t i=0; i<x.size(); ++i)
        wgtpsi[i] *= func(x[i])*supp*0.5;
      }

    /* Compute correction factors for gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    double corfunc(double v) const
      {
      double tmp=0;
      for (size_t i=0; i<x.size(); ++i)
        tmp += wgtpsi[i]*cos(pi*supp*v*x[i]);
      return 1./tmp;
      }
    /* Compute correction factors for gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      {
      vector<double> res(n);
      execStatic(n, nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          res[i] = corfunc(i*dx);
        });
      return res;
      }
  };

template<typename T> class HornerKernel: public GriddingKernel<T>
  {
  private:
    static constexpr size_t MAXW=16, MINDEG=0, MAXDEG=20;
    using Tsimd = native_simd<T>;
    static constexpr auto vlen = Tsimd::size();
    size_t W, D, nvec;

    vector<Tsimd> coeff;
    void (HornerKernel<T>::* evalfunc) (T, native_simd<T> *) const;

    PiecewiseKernelCorrection corr;

    template<size_t NV, size_t DEG> void eval_intern(T x, native_simd<T> *res) const
      {
      x = (x+1)*W-1;
      for (size_t i=0; i<NV; ++i)
        {
        auto tval = coeff[i];
        for (size_t j=1; j<=DEG; ++j)
          tval = tval*x + coeff[j*NV+i];
        res[i] = tval;
        }
      }

    void eval_intern_general(T x, native_simd<T> *res) const
      {
      x = (x+1)*W-1;
      for (size_t i=0; i<nvec; ++i)
        {
        auto tval = coeff[i];
        for (size_t j=1; j<=D; ++j)
          tval = tval*x+coeff[j*nvec+i];
        res[i] = tval;
        }
      }

    template<size_t NV, size_t DEG> auto evfhelper2() const
      {
      if (DEG==D)
        return &HornerKernel::eval_intern<NV,DEG>;
      if (DEG>MAXDEG)
        return &HornerKernel::eval_intern_general;
      return evfhelper2<NV, ((DEG>MAXDEG) ? DEG : DEG+1)>();
      }

    template<size_t NV> auto evfhelper1() const
      {
      if (nvec==NV) return evfhelper2<NV,0>();
      if (nvec*vlen>MAXW) return &HornerKernel::eval_intern_general;
      return evfhelper1<((NV*vlen>MAXW) ? NV : NV+1)>();
      }

    static vector<Tsimd> makeCoeff(size_t W, size_t D,
      const function<double(double)> &func)
      {
      auto nvec = ((W+vlen-1)/vlen);
      vector<Tsimd> coeff(nvec*(D+1), 0);
      auto coeff_raw = getCoeffs(W,D,func);
      for (size_t j=0; j<=D; ++j)
        {
        for (size_t i=0; i<W; ++i)
          coeff[j*nvec + i/vlen][i%vlen] = T(coeff_raw[j*W+i]);
        for (size_t i=W; i<vlen*nvec; ++i)
          coeff[j*nvec + i/vlen][i%vlen] = T(0);
        }
      return coeff;
      }
    static vector<Tsimd> coeffFromPolyKernel(const PolyKernel &krn)
      {
      auto W = krn.W;
      auto D = krn.D;
      auto nvec = ((W+vlen-1)/vlen);
      vector<Tsimd> coeff(nvec*(D+1), 42.);
      vector<double> coeff_raw(W*(D+1), 42.);
      size_t Whalf = W/2;
      for (size_t j=0; j<=D; ++j)
        {
        double flip = (j&1) ? -1:1;
        for (size_t i=Whalf; i<W; ++i)
          {
          double val = krn.coeff[(i-Whalf)*(D+1)+j];
          coeff_raw[j*W+i] = val;
          coeff_raw[j*W+W-1-i] = flip*val;
          }
        }
      for (size_t j=0; j<=D; ++j)
        {
        for (size_t i=0; i<W; ++i)
          coeff[(D-j)*nvec + i/vlen][i%vlen] = T(coeff_raw[j*W+i]);
        for (size_t i=W; i<vlen*nvec; ++i)
          coeff[(D-j)*nvec + i/vlen][i%vlen] = T(0);
        }
      return coeff;
      }

  public:
    using GriddingKernel<T>::eval;
    using GriddingKernel<T>::eval_single;
    using GriddingKernel<T>::corfunc;

    HornerKernel(size_t W_, size_t D_, const function<double(double)> &func)
      : W(W_), D(D_), nvec((W+vlen-1)/vlen),
        coeff(makeCoeff(W_, D_, func)), evalfunc(evfhelper1<1>()),
        corr(W_, [this](T v){return eval_single(v);})
      {}

    HornerKernel(const PolyKernel &krn)
      : W(krn.W), D(krn.coeff.size()/((W+1)/2)-1), nvec((W+vlen-1)/vlen),
        coeff(coeffFromPolyKernel(krn)), evalfunc(evfhelper1<1>()),
        corr(W, [this](T v){return eval_single(v);})
      {}

    virtual size_t support() const { return W; }

    virtual void eval(T x, native_simd<T> *res) const
      { (this->*evalfunc)(x, res); }
    /*! Returns the function approximation at location x.
        x must lie in [-1; 1].  */

    virtual T eval_single(T x) const
      {
      auto nth = min(W-1, size_t(max(T(0), (x+1)*W*T(0.5))));
      x = (x+1)*W-2*nth-1;
      auto i = nth/vlen;
      auto imod = nth%vlen;
      auto tval = coeff[i][imod];
      for (size_t j=1; j<=D; ++j)
        tval = tval*x + coeff[j*nvec+i][imod];
      return tval;
      }

    virtual double corfunc(double x) const {return corr.corfunc(x); }

    /* Computes the correction function values at a coordinates
       [0, dx, 2*dx, ..., (n-1)*dx]  */
    virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      { return corr.corfunc(n, dx, nthreads); }
  };

template<typename T> class ES_Kernel: public GriddingKernel<T>
  {
  private:
    using Tsimd = native_simd<T>;
    static constexpr auto vlen = Tsimd::size();
    size_t W, nvec;
    double beta;

    GLFullCorrection corr;

  public:
    using GriddingKernel<T>::eval;
    using GriddingKernel<T>::eval_single;
    using GriddingKernel<T>::corfunc;

    ES_Kernel(size_t W_, double beta_)
      : W(W_), nvec((W+vlen-1)/vlen), beta(beta_),
        corr(W, [this](T v){return eval_single(v);}) {}

    virtual size_t support() const { return W; }

    virtual void eval(T x, native_simd<T> *res) const
      {
      T dist = T(2./W);
      for (size_t i=0; i<W; ++i)
        res[i/vlen][i%vlen] = x+i*dist;
      for (size_t i=W; i<nvec*vlen; ++i)
        res[i/vlen][i%vlen] = 0;
      for (size_t i=0; i<nvec; ++i)
        res[i] = esk(res[i], T(beta));
      }

    /*! Returns the function approximation at location x.
        x must lie in [-1; 1].  */
    virtual T eval_single(T x) const
      { return esk(x, T(beta)); }

    virtual double corfunc(double x) const {return corr.corfunc(x); }

    /* Computes the correction function values at a coordinates
       [0, dx, 2*dx, ..., (n-1)*dx]  */
    virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      { return corr.corfunc(n, dx, nthreads); }
  };

template<typename T> auto selectESKernel(double ofactor, double epsilon)
  {
  auto supp = esk_support(epsilon, ofactor);
  auto beta = esk_beta(supp, ofactor)*supp;
//  return make_shared<HornerKernel<T>>(supp, supp+3, [beta](double v){return esk(v, beta);});
  return make_shared<ES_Kernel<T>>(supp, beta);
  }

}

using detail_gridding_kernel::GriddingKernel;
using detail_gridding_kernel::HornerKernel;
using detail_gridding_kernel::ES_Kernel;
using detail_gridding_kernel::selectESKernel;

}

#endif
