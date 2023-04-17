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

#ifndef DUCC0_GRIDDING_KERNEL_H
#define DUCC0_GRIDDING_KERNEL_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <vector>
#include <memory>
#include <cmath>
#include <type_traits>
#include <limits>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/math/constants.h"

namespace ducc0 {

namespace detail_gridding_kernel {

using namespace std;

vector<double> getCoeffs(size_t W, size_t D, const function<double(double)> &func);

/*! A GriddingKernel is considered to be a symmetric real-valued function
    defined on the interval [-1; 1].
    This range is subdivided into W equal-sized parts. */
class GriddingKernel
  {
  public:
    virtual ~GriddingKernel() {}

    virtual size_t support() const = 0;

    /* Computes the correction function at a given coordinate.
       Useful coordinates lie in the range [0; 0.5]. */
    virtual double corfunc(double x) const = 0;

    /* Computes the correction function values at a coordinates
       [0, dx, 2*dx, ..., (n-1)*dx]  */
    virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const = 0;

    /* Returns the kernel's value at x */
    virtual double eval(double x) const = 0;
  };

class KernelCorrection
  {
  protected:
    vector<double> x, wgtpsi;
    size_t supp;

  public:
    /* Compute correction factors for gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    template<typename T> [[gnu::always_inline]] T corfunc(T v) const
      {
      T tmp=0;
      for (size_t i=0; i<x.size(); ++i)
        tmp += wgtpsi[i]*cos((pi*supp*x[i])*v);
      return T(1)/tmp;
      }
    /* Compute correction factors for gridding kernel
       This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
    vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      {
      vector<double> res(n);
// The commented lines would add vectorization support,
// but this doesn't appear beneficial.
//      constexpr size_t vlen = native_simd<double>::size();
//      native_simd<double> itimesdx;
//      for (size_t i=0; i<vlen; ++i) itimesdx[i]=i*dx;
      execStatic(n, nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext())
          {
          auto i = rng.lo;
          //for (; i+vlen<=rng.hi; i+=vlen)
            //{
            //auto v = corfunc(itimesdx+i*dx);
            //v.copy_to(&res[i],element_aligned_tag());
            //}
          for(; i<rng.hi; ++i)
            res[i] = corfunc(i*dx);
          }
        });
      return res;
      }

    const vector<double> &X() const { return x; }
    const vector<double> &Wgtpsi() const { return wgtpsi; }
    size_t Supp() const { return supp; }
  };

class GLFullCorrection: public KernelCorrection
  {
  public:
    GLFullCorrection(size_t W, const function<double(double)> &func)
      {
      supp = W;
      size_t p = size_t(1.5*W)+2;
      GL_Integrator integ(2*p);
      x = integ.coordsSymmetric();
      wgtpsi = integ.weightsSymmetric();
      for (size_t i=0; i<x.size(); ++i)
        wgtpsi[i] *= func(x[i])*supp*0.5;
      }
  };

/*! This class implements the \a GriddingKernel interface by approximating the
    provided function with \a W polynomials of degree \a D. */
class PolynomialKernel: public GriddingKernel
  {
  private:
    size_t W, D;
    vector<double> coeff;
    KernelCorrection corr;

  public:
    PolynomialKernel(size_t W_, size_t D_, const function<double(double)> &func,
      const KernelCorrection &corr_)
      : W(W_), D(D_),
        coeff(getCoeffs(W_, D_, func)),
        corr(corr_)
      {}

    virtual size_t support() const { return W; }

    virtual double corfunc(double x) const { return corr.corfunc(x); }

    /* Computes the correction function values at a coordinates
       [0, dx, 2*dx, ..., (n-1)*dx]  */
    virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      { return corr.corfunc(n, dx, nthreads); }

    const vector<double> &Coeff() const { return coeff; }
    size_t degree() const { return D; }

    const KernelCorrection &Corr() const { return corr; }

    double eval(double x) const
      {
      if (abs(x)>=1) return 0.;
      double xrel = W*0.5*(x+1.);
      size_t nth = size_t(xrel);
      nth = min<size_t>(nth, W-1);
      double locx = ((xrel-nth)-0.5)*2; // should be in [-1; 1]
      double res = coeff[nth];
      for (size_t i=1; i<=D; ++i)
        res = res*locx+coeff[i*W+nth];
      return res;
      }
  };

/*! This class is initialized with a \a PolynomialKernel object and provides
    low level methods for extremely fast kernel evaluations. */
template<size_t W, typename Tsimd> class TemplateKernel
  {
  private:
    static constexpr auto D=W+3+(W&1);
    using T = typename Tsimd::value_type;
    using Tvl = typename Tsimd::Tv;
    static constexpr auto vlen = Tsimd::size();
    static constexpr auto nvec = (W+vlen-1)/vlen;

    std::array<Tsimd,(D+1)*nvec> coeff;
    const T *scoeff;
    static constexpr auto sstride = nvec*vlen;

    void transferCoeffs(const vector<double> &input, size_t d_input)
      {
      auto ofs = D-d_input;
      if (ofs>0)
        for (size_t i=0; i<nvec; ++i)
          coeff[i] = 0;
      for (size_t j=0; j<=d_input; ++j)
        {
        for (size_t i=0; i<W; ++i)
          coeff[(j+ofs)*nvec + i/vlen][i%vlen] = T(input[j*W+i]);
        for (size_t i=W; i<vlen*nvec; ++i)
          coeff[(j+ofs)*nvec + i/vlen][i%vlen] = T(0);
        }
      }

  public:
    TemplateKernel(const PolynomialKernel &krn)
      : scoeff(reinterpret_cast<T *>(&coeff[0]))
      {
      MR_assert(W==krn.support(), "support mismatch");
      MR_assert(D>=krn.degree(), "degree mismatch");
      transferCoeffs(krn.Coeff(), krn.degree());
      }

    constexpr size_t support() const { return W; }

    double eval(double x) const
      {
      if (abs(x)>=1) return 0.;
      double xrel = W*0.5*(x+1.);
      size_t nth = size_t(xrel);
      nth = min<size_t>(nth, W-1);
      double locx = ((xrel-nth)-0.5)*2; // should be in [-1; 1]
      double res = scoeff[nth];
      for (size_t i=1; i<=D; ++i)
        res = res*locx+scoeff[i*sstride+nth];
      return res;
      }

    [[gnu::always_inline]] void eval2s(T x, T y, T z, size_t nth, Tsimd * DUCC0_RESTRICT res) const
      {
      z = (z-nth)*2+(W-1);
      T x2=x*x, y2=y*y, z2=z*z;
      if constexpr (nvec==1)
        {
        Tvl tvalx = coeff[0], tvaly = coeff[0], tvalz = coeff[0];
        Tvl tvalx2 = coeff[1], tvaly2 = coeff[1], tvalz2 = coeff[1];
        for (size_t j=2; j<D; j+=2)
          {
          tvalx = tvalx*x2 + Tvl(coeff[j]);
          tvaly = tvaly*y2 + Tvl(coeff[j]);
          tvalz = tvalz*z2 + Tvl(coeff[j]);
          tvalx2 = tvalx2*x2 + Tvl(coeff[j+1]);
          tvaly2 = tvaly2*y2 + Tvl(coeff[j+1]);
          tvalz2 = tvalz2*z2 + Tvl(coeff[j+1]);
          }
        res[0] = (x*tvalx+tvalx2)*T(z*tvalz[nth]+tvalz2[nth]);
        res[1] = y*tvaly+tvaly2;
        }
      else
        {
        T zfac;
        {
        Tvl tvalx = coeff[0], tvaly = coeff[0];
        Tvl tvalx2 = coeff[nvec], tvaly2 = coeff[nvec];
        auto ptrz = scoeff+nth;
        auto tvalz = *ptrz, tvalz2 = ptrz[sstride];
        for (size_t j=2; j<D; j+=2)
          {
          tvalx = tvalx*x2 + Tvl(coeff[j*nvec]);
          tvaly = tvaly*y2 + Tvl(coeff[j*nvec]);
          tvalz = tvalz*z2 + ptrz[j*sstride];
          tvalx2 = tvalx2*x2 + Tvl(coeff[(j+1)*nvec]);
          tvaly2 = tvaly2*y2 + Tvl(coeff[(j+1)*nvec]);
          tvalz2 = tvalz2*z2 + ptrz[(j+1)*sstride];
          }
        zfac = tvalz*z+tvalz2;
        res[0] = (tvalx*x+tvalx2)*zfac;
        res[nvec] = tvaly*y+tvaly2;
        }
        for (size_t i=1; i<nvec; ++i)
          {
          Tvl tvalx = coeff[i], tvaly = coeff[i];
          Tvl tvalx2 = coeff[i+nvec], tvaly2 = coeff[i+nvec];
          for (size_t j=2; j<D; j+=2)
            {
            tvalx = tvalx*x2 + Tvl(coeff[i+j*nvec]);
            tvaly = tvaly*y2 + Tvl(coeff[i+j*nvec]);
            tvalx2 = tvalx2*x2 + Tvl(coeff[i+(j+1)*nvec]);
            tvaly2 = tvaly2*y2 + Tvl(coeff[i+(j+1)*nvec]);
            }
          res[i] = (tvalx*x+tvalx2)*zfac;
          res[nvec+i] = tvaly*y+tvaly2;
          }
        }
      }
    [[gnu::always_inline]] void eval2(T x, T y, Tsimd * DUCC0_RESTRICT res) const
      {
      T x2=x*x, y2=y*y;
      if constexpr (nvec==1)
        {
        Tvl tvalx = coeff[0], tvaly = coeff[0];
        Tvl tvalx2 = coeff[1], tvaly2 = coeff[1];
        for (size_t j=2; j<D; j+=2)
          {
          tvalx = tvalx*x2 + Tvl(coeff[j]);
          tvaly = tvaly*y2 + Tvl(coeff[j]);
          tvalx2 = tvalx2*x2 + Tvl(coeff[j+1]);
          tvaly2 = tvaly2*y2 + Tvl(coeff[j+1]);
          }
        res[0] = x*tvalx+tvalx2;
        res[1] = y*tvaly+tvaly2;
        }
      else
        for (size_t i=0; i<nvec; ++i)
          {
          Tvl tvalx = coeff[i], tvaly = coeff[i];
          Tvl tvalx2 = coeff[i+nvec], tvaly2 = coeff[i+nvec];
          for (size_t j=2; j<D; j+=2)
            {
            tvalx = tvalx*x2 + Tvl(coeff[i+j*nvec]);
            tvaly = tvaly*y2 + Tvl(coeff[i+j*nvec]);
            tvalx2 = tvalx2*x2 + Tvl(coeff[i+(j+1)*nvec]);
            tvaly2 = tvaly2*y2 + Tvl(coeff[i+(j+1)*nvec]);
            }
          res[i] = tvalx*x+tvalx2;
          res[nvec+i] = tvaly*y+tvaly2;
          }
      }
    [[gnu::always_inline]] void eval1(T x, Tsimd * DUCC0_RESTRICT res) const
      {
      T x2=x*x;
      if constexpr (nvec==1)
        {
        Tvl tvalx = coeff[0];
        Tvl tvalx2 = coeff[1];
        for (size_t j=2; j<D; j+=2)
          {
          tvalx = tvalx*x2 + Tvl(coeff[j]);
          tvalx2 = tvalx2*x2 + Tvl(coeff[j+1]);
          }
        res[0] = x*tvalx+tvalx2;
        }
      else
        for (size_t i=0; i<nvec; ++i)
          {
          Tvl tvalx = coeff[i];
          Tvl tvalx2 = coeff[i+nvec];
          for (size_t j=2; j<D; j+=2)
            {
            tvalx = tvalx*x2 + Tvl(coeff[i+j*nvec]);
            tvalx2 = tvalx2*x2 + Tvl(coeff[i+(j+1)*nvec]);
            }
          res[i] = tvalx*x+tvalx2;
          }
      }
    [[gnu::always_inline]] void eval3(T x, T y, T z, Tsimd * DUCC0_RESTRICT res) const
      {
      T x2=x*x, y2=y*y, z2=z*z;
      if constexpr (nvec==1)
        {
        Tvl tvalx = coeff[0], tvaly = coeff[0], tvalz = coeff[0];
        Tvl tvalx2 = coeff[1], tvaly2 = coeff[1], tvalz2 = coeff[1];
        for (size_t j=2; j<D; j+=2)
          {
          tvalx = tvalx*x2 + Tvl(coeff[j]);
          tvaly = tvaly*y2 + Tvl(coeff[j]);
          tvalz = tvalz*z2 + Tvl(coeff[j]);
          tvalx2 = tvalx2*x2 + Tvl(coeff[j+1]);
          tvaly2 = tvaly2*y2 + Tvl(coeff[j+1]);
          tvalz2 = tvalz2*z2 + Tvl(coeff[j+1]);
          }
        res[0] = tvalx*x+tvalx2;
        res[1] = tvaly*y+tvaly2;
        res[2] = tvalz*z+tvalz2;
        }
      else
        {
        for (size_t i=0; i<nvec; ++i)
          {
          Tvl tvalx = coeff[i], tvaly = coeff[i], tvalz = coeff[i];
          Tvl tvalx2 = coeff[i+nvec], tvaly2 = coeff[i+nvec], tvalz2 = coeff[i+nvec];
          for (size_t j=2; j<D; j+=2)
            {
            tvalx = tvalx*x2 + Tvl(coeff[i+j*nvec]);
            tvaly = tvaly*y2 + Tvl(coeff[i+j*nvec]);
            tvalz = tvalz*z2 + Tvl(coeff[i+j*nvec]);
            tvalx2 = tvalx2*x2 + Tvl(coeff[i+(j+1)*nvec]);
            tvaly2 = tvaly2*y2 + Tvl(coeff[i+(j+1)*nvec]);
            tvalz2 = tvalz2*z2 + Tvl(coeff[i+(j+1)*nvec]);
            }
          res[i] = tvalx*x+tvalx2;
          res[nvec+i] = tvaly*y+tvaly2;
          res[2*nvec+i] = tvalz*z+tvalz2;
          }
        }
      }
  };

struct KernelParams
  {
  size_t W;
  double ofactor, epsilon, beta, e0;
  size_t ndim;
  bool singleprec;
  };

shared_ptr<PolynomialKernel> selectKernel(size_t idx);
const KernelParams &getKernel(size_t idx);

template<typename T> constexpr inline size_t Wmax()
  { return is_same<T,float>::value ? 8 : 16; }

extern const vector<KernelParams> KernelDB;

/*! Returns the 2-parameter ES kernel for the given oversampling factor,
 *  dimensionality, and error that has the smallest support. */
template<typename T> auto selectKernel(double ofactor, size_t ndim, double epsilon)
  {
  constexpr bool singleprec = is_same<T, float>::value;
  size_t Wmin = Wmax<T>();
  size_t idx = KernelDB.size();
  for (size_t i=0; i<KernelDB.size(); ++i)
    {
    if  ((KernelDB[i].ndim==ndim) && (KernelDB[i].singleprec==singleprec)
      && (KernelDB[i].ofactor<=ofactor) && (KernelDB[i].epsilon<=epsilon)
      && (KernelDB[i].W<=Wmin))
      {
      idx = i;
      Wmin = KernelDB[i].W;
      }
    }
  return selectKernel(idx);
  }

template<typename T> auto getAvailableKernels(double epsilon,
  size_t ndim, double ofactor_min=1.1, double ofactor_max=2.6)
  {
  constexpr bool singleprec = is_same<T, float>::value;
  vector<double> ofc(20, ofactor_max);
  vector<size_t> idx(20, KernelDB.size());
  size_t Wlim = Wmax<T>();
  for (size_t i=0; i<KernelDB.size(); ++i)
    {
    auto ofactor = KernelDB[i].ofactor;
    size_t W = KernelDB[i].W;
    if ((KernelDB[i].ndim==ndim) && (KernelDB[i].singleprec==singleprec)
      && (W<=Wlim) && (KernelDB[i].epsilon<=epsilon)
      && (ofactor<=ofc[W]) && (ofactor>=ofactor_min))
      {
      ofc[W] = ofactor;
      idx[W] = i;
      }
    }
  vector<size_t> res;
  for (auto v: idx)
    if (v<KernelDB.size()) res.push_back(v);
  MR_assert(!res.empty(), "no appropriate kernel found");
  return res;
  }

double bestEpsilon(size_t ndim, bool singleprec,
  double ofactor_min=1.1, double ofactor_max=2.6);

}

using detail_gridding_kernel::GriddingKernel;
using detail_gridding_kernel::getKernel;
using detail_gridding_kernel::selectKernel;
using detail_gridding_kernel::getAvailableKernels;
using detail_gridding_kernel::bestEpsilon;
using detail_gridding_kernel::PolynomialKernel;
using detail_gridding_kernel::TemplateKernel;
using detail_gridding_kernel::KernelParams;

}

#endif
