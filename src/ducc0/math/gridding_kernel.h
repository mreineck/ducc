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
#include "ducc0/math/constants.h"

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

class KernelCorrection
  {
  protected:
    vector<double> x, wgtpsi;
    size_t supp;

  public:
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

// class PiecewiseKernelCorrection: public KernelCorrection
//   {
//   private:
//     static constexpr size_t ideg=10; // integration points per interval
//     static_assert((ideg&1)==0, "ideg must be even");
//
//   public:
//     PiecewiseKernelCorrection(size_t W, const function<double(double)> &func)
//       {
//       supp = W;
//       // we know that the kernel is piecewise smooth in all W sections but not
//       // necessarily at borders between sections. Therefore we integrate piece
//       // by piece.
//       GL_Integrator integ(ideg, 1);
//       auto xgl = integ.coords();
//       auto wgl = integ.weights();
//       x.resize((supp*ideg)/2);
//       wgtpsi.resize((supp*ideg)/2);
//       for (size_t i=0; i<x.size(); ++i)
//         {
//         size_t iiv = i/ideg;
//         x[i] = -1. + iiv*2./supp + (1.+xgl[i%ideg])/supp;
//         wgtpsi[i] = wgl[i%ideg]*func(x[i]);
//         }
//       }
//   };

class GLFullCorrection: public KernelCorrection
  {
  public:
    GLFullCorrection(size_t W, const function<double(double)> &func)
      {
      supp = W;
      size_t p = size_t(1.5*W)+2;
      GL_Integrator integ(2*p,1);
      x = integ.coordsSymmetric();
      wgtpsi = integ.weightsSymmetric();
      for (size_t i=0; i<x.size(); ++i)
        wgtpsi[i] *= func(x[i])*supp*0.5;
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

    KernelCorrection corr;

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

  public:
    using GriddingKernel<T>::eval;
    using GriddingKernel<T>::eval_single;
    using GriddingKernel<T>::corfunc;

    HornerKernel(size_t W_, size_t D_, const function<double(double)> &func,
      const KernelCorrection &corr_)
      : W(W_), D(D_), nvec((W+vlen-1)/vlen),
        coeff(makeCoeff(W_, D_, func)), evalfunc(evfhelper1<1>()),
        corr(corr_)
      {}

//     HornerKernel(size_t W_, size_t D_, const function<double(double)> &func)
//       : W(W_), D(D_), nvec((W+vlen-1)/vlen),
//         coeff(makeCoeff(W_, D_, func)), evalfunc(evfhelper1<1>()),
//         corr(PiecewiseKernelCorrection(W_, [this](T v){return eval_single(v);})) {}

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

// template<typename T> T esk (T v, T beta)
//   {
//   auto tmp = (1-v)*(1+v);
//   auto tmp2 = tmp>=0;
//   return tmp2*exp(T(beta)*(sqrt(tmp*tmp2)-1));
//   }
//
// template<typename T> class exponator
//   {
//   public:
//     T operator()(T val) { return exp(val); }
//   };
//
// template<typename T> native_simd<T> esk (native_simd<T> v, T beta)
//   {
//   auto tmp = (T(1)-v)*(T(1)+v);
//   auto mask = tmp<T(0);
//   where(mask,tmp)*=T(0);
//   auto res = (beta*(sqrt(tmp)-T(1))).apply(exponator<T>());
//   where (mask,res)*=T(0);
//   return res;
//   }
//
// double esk_beta(size_t supp, double ofactor)
//   {
//   MR_assert((supp>=2) && (supp<=15), "unsupported support size");
//   if (ofactor>=2)
//     {
//     static const array<double,16> opt_beta {-1, 0.14, 1.70, 2.08, 2.205, 2.26,
//       2.29, 2.307, 2.316, 2.3265, 2.3324, 2.282, 2.294, 2.304, 2.3138, 2.317};
//     MR_assert(supp<opt_beta.size(), "bad support size");
//     return opt_beta[supp];
//     }
//   if (ofactor>=1.175)
//     {
//     // empirical, but pretty accurate approximation
//     static const array<double,16> betacorr{0,0,-0.51,-0.21,-0.1,-0.05,-0.025,-0.0125,0,0,0,0,0,0,0,0};
//     auto x0 = 1./(2*ofactor);
//     auto bcstrength=1.+(x0-0.25)*2.5;
//     return 2.32+bcstrength*betacorr[supp]+(0.25-x0)*3.1;
//     }
//   MR_fail("oversampling factor is too small");
//   }
//
// size_t esk_support(double epsilon, double ofactor)
//   {
//   double epssq = epsilon*epsilon;
//   if (ofactor>=2)
//     {
//     static const array<double,16> maxmaperr { 1e8, 0.19, 2.98e-3, 5.98e-5,
//       1.11e-6, 2.01e-8, 3.55e-10, 5.31e-12, 8.81e-14, 1.34e-15, 2.17e-17,
//       2.12e-19, 2.88e-21, 3.92e-23, 8.21e-25, 7.13e-27 };
//
//     for (size_t i=2; i<maxmaperr.size(); ++i)
//       if (epssq>maxmaperr[i]) return i;
//     MR_fail("requested epsilon too small - minimum is 1e-13");
//     }
//   if (ofactor>=1.175)
//     {
//     for (size_t w=2; w<16; ++w)
//       {
//       auto estimate = 12*exp(-2.*w*ofactor); // empirical, not very good approximation
//       if (epssq>estimate) return w;
//       }
//     MR_fail("requested epsilon too small");
//     }
//   MR_fail("oversampling factor is too small");
//   }
//
// template<typename T> class ES_Kernel: public GriddingKernel<T>
//   {
//   private:
//     using Tsimd = native_simd<T>;
//     static constexpr auto vlen = Tsimd::size();
//     size_t W, nvec;
//     double beta;
//
//     KernelCorrection corr;
//
//   public:
//     using GriddingKernel<T>::eval;
//     using GriddingKernel<T>::eval_single;
//     using GriddingKernel<T>::corfunc;
//
//     ES_Kernel(size_t W_, double beta_)
//       : W(W_), nvec((W+vlen-1)/vlen), beta(beta_),
//         corr(GLFullCorrection(W, [this](T v){return eval_single(v);})) {}
//
//     virtual size_t support() const { return W; }
//
//     virtual void eval(T x, native_simd<T> *res) const
//       {
//       T dist = T(2./W);
//       for (size_t i=0; i<W; ++i)
//         res[i/vlen][i%vlen] = x+i*dist;
//       for (size_t i=W; i<nvec*vlen; ++i)
//         res[i/vlen][i%vlen] = 0;
//       for (size_t i=0; i<nvec; ++i)
//         res[i] = esk(res[i], T(beta));
//       }
//
//     /*! Returns the function approximation at location x.
//         x must lie in [-1; 1].  */
//     virtual T eval_single(T x) const
//       { return esk(x, T(beta)); }
//
//     virtual double corfunc(double x) const {return corr.corfunc(x); }
//
//     /* Computes the correction function values at a coordinates
//        [0, dx, 2*dx, ..., (n-1)*dx]  */
//     virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const
//       { return corr.corfunc(n, dx, nthreads); }
//   };

struct NESdata
  {
  size_t W;
  double ofactor, epsilon, beta, e0;
  };

const vector<NESdata> NEScache {
{4, 1.2, 0.014220694094126433, 1.308837890625, 0.5867762586805555},
{4, 1.3, 0.006769640172910669, 1.3671827060964923, 0.5973925406718384},
{4, 1.4, 0.003533653831957708, 1.4722493489583337, 0.5831955295138888},
{4, 1.5, 0.0026235582107206005, 1.6395670572916667, 0.5563075086805555},
{4, 1.6, 0.0017121400767454193, 1.6540527343749998, 0.5653244357638889},
{4, 1.7, 0.0013010783159010215, 1.7276204427083333, 0.5575661892361113},
{4, 1.8, 0.0009624786407884259, 1.7889811197916665, 0.5529871961805555},
{4, 1.9, 0.0007548051734958207, 1.9175618489583335, 0.5383062065972222},
{4, 2.0, 0.0005186220477960485, 1.906819661458333, 0.5468674045138888},
{5, 1.2, 0.004692051606881723, 1.4755045572916667, 0.546531032986111},
{5, 1.3, 0.0015994417148989, 1.5659993489583335, 0.5445670572916667},
{5, 1.4, 0.0006214730873093, 1.6250813802083333, 0.5462489149305556},
{5, 1.5, 0.0003348397337550639, 1.7045084635416663, 0.5432541232638888},
{5, 1.6, 0.00022454757237806803, 1.7499186197916665, 0.5453483072916666},
{5, 1.7, 0.00015812674485747515, 1.826741536458333, 0.5390440538194445},
{5, 1.8, 0.00010873993249987502, 1.9045410156249998, 0.5339767795138889},
{5, 1.9, 7.816000067531175e-05, 2.000732421875, 0.5274121093749999},
{5, 2.0, 5.97410287941553e-05, 2.0415852864583335, 0.5283344184027777},
{6, 1.2, 0.0009035394602860672, 1.4868977864583333, 0.543134765625},
{6, 1.3, 0.00032849065945147126, 1.6190592447916665, 0.5346712239583331},
{6, 1.4, 0.00011680261930031713, 1.7110188802083333, 0.5316221788194445},
{6, 1.5, 5.658116308415308e-05, 1.811604817708333, 0.5271299913194444},
{6, 1.6, 3.1513795171211674e-05, 1.8744303385416665, 0.5271842447916666},
{6, 1.7, 2.2417415186694496e-05, 1.9217936197916665, 0.5263053385416667},
{6, 1.8, 1.4854060644126216e-05, 1.9885253906250002, 0.5234733072916666},
{6, 1.9, 9.18465942998492e-06, 2.054280598958333, 0.5211295572916668},
{6, 2.0, 6.693833134345978e-06, 2.0909016927083335, 0.5225075954861111},
{7, 1.2, 0.0002861239634129858, 1.5651855468750002, 0.5294954427083335},
{7, 1.3, 7.531069590728102e-05, 1.669189453125, 0.5267719184027777},
{7, 1.4, 2.5700003129672653e-05, 1.7779134114583328, 0.5225401475694444},
{7, 1.5, 9.621175530737558e-06, 1.8669433593749996, 0.5200336371527778},
{7, 1.6, 4.723927940521181e-06, 1.943440755208333, 0.5191221788194444},
{7, 1.7, 2.960839122807533e-06, 1.982014973958333, 0.5196430121527779},
{7, 1.8, 1.754023497971143e-06, 2.046793619791667, 0.5172016059027778},
{7, 1.9, 1.1320469853975162e-06, 2.0868326822916665, 0.5179069010416666},
{7, 2.0, 7.750449146111461e-07, 2.1097819010416665, 0.520380859375},
{8, 1.2, 8.38719628704345e-05, 1.6249186197916665, 0.5212163628472221},
{8, 1.3, 1.4776178056293107e-05, 1.7333170572916663, 0.5189051649305555},
{8, 1.4, 4.593178465697842e-06, 1.8236490885416665, 0.5172233072916667},
{8, 1.5, 1.6176997163066937e-06, 1.916834513346354, 0.5150358751085068},
{8, 1.6, 7.572788341219199e-07, 1.9797363281250002, 0.5151508246527778},
{8, 1.7, 4.307772667559197e-07, 2.0407714843750004, 0.5133930121527777},
{8, 1.8, 2.4097873028482365e-07, 2.0912272135416665, 0.5127528211805555},
{8, 1.9, 1.3679155629994797e-07, 2.1239420572916665, 0.5139680989583335},
{8, 2.0, 8.512914903147771e-08, 2.177001953125, 0.5135666232638889},
{9, 1.2, 2.3455113812509457e-05, 1.6530761718749998, 0.5174186197916665},
{9, 1.3, 3.577757303750482e-06, 1.769124348958333, 0.514402126736111},
{9, 1.4, 9.455791814559009e-07, 1.8578287760416663, 0.5134364149305555},
{9, 1.5, 3.07993711241319e-07, 1.9369303385416663, 0.5122211371527777},
{9, 1.6, 1.1023667360041581e-07, 1.9875488281250002, 0.5141742621527777},
{9, 1.7, 6.006581600850492e-08, 2.0648859795479457, 0.5117597062520739},
{9, 1.8, 3.098125788380842e-08, 2.1207897812031766, 0.5110015076779959},
{9, 1.9, 1.6745057082558624e-08, 2.1499837239583335, 0.5116134982638889},
{9, 2.0, 9.06679659361064e-09, 2.1802571614583335, 0.5130023871527779},
{10, 1.2, 6.514594099237372e-06, 1.689846296428521, 0.513470891740015},
{10, 1.3, 8.21598917069439e-07, 1.7967932684503998, 0.51201551001592},
{10, 1.4, 1.7441518859330718e-07, 1.8850957031515838, 0.5112867891779155},
{10, 1.5, 5.4102060793230046e-08, 1.9621461370365634, 0.5104270958394896},
{10, 1.6, 1.6965859337170533e-08, 2.0254720052083335, 0.5104090711805556},
{10, 1.7, 8.295375084241509e-09, 2.0603835662117986, 0.5120321940089277},
{10, 1.8, 4.038070760493696e-09, 2.1337787635267325, 0.5097844830789858},
{10, 1.9, 2.0082193994394397e-09, 2.175836444185279, 0.5098410572828435},
{10, 2.0, 9.631655783064303e-10, 2.2007649739583335, 0.5110058593750001},
{11, 1.2, 1.8033899515187442e-06, 1.7108438029240443, 0.5110418476094097},
{11, 1.3, 1.9354647673781677e-07, 1.8055165608723958, 0.5110068766276041},
{11, 1.4, 3.278945796840094e-08, 1.8974355061848953, 0.5099313015407987},
{11, 1.5, 9.127798361742706e-09, 1.97070880159825, 0.5095923783620026},
{11, 1.6, 2.617198123701698e-09, 2.047444661458333, 0.5083691406249999},
{11, 1.7, 1.075500321319766e-09, 2.056559244791667, 0.5113639322916665},
{11, 1.8, 5.187591832863408e-10, 2.139721745951372, 0.5092240510422396},
{11, 1.9, 2.4186233385152035e-10, 2.200907220005266, 0.50773069118988},
{11, 2.0, 1.102672941901129e-10, 2.2157389322916665, 0.5095627170138887},
{12, 1.2, 5.268577886684257e-07, 1.7285003662109373, 0.5090266248914931},
{12, 1.3, 4.431744248245887e-08, 1.8093312581380208, 0.5105952284071181},
{12, 1.4, 6.590393759039512e-09, 1.8982391357421873, 0.5097007242838542},
{12, 1.5, 1.4928257178765997e-09, 1.9644215901692705, 0.5099760606553819},
{12, 1.6, 4.1571388648334064e-10, 2.046686808268229, 0.5083111572265624},
{12, 1.7, 1.608469361639032e-10, 2.066197713216146, 0.5105565728081597},
{12, 1.8, 5.90873259979258e-11, 2.125391642252604, 0.5098539903428819},
{12, 1.9, 2.6338814622488886e-11, 2.1977284749348964, 0.5077855767144097},
{12, 2.0, 1.2259090028976973e-11, 2.2395579020182295, 0.5077794731987847},
{13, 1.2, 2.2962002693763652e-07, 1.7522222222222223, 0.5063277777777777},
{13, 1.3, 1.0172544685993453e-08, 1.8269907633463538, 0.5087668863932294},
{13, 1.4, 1.3196746034290688e-09, 1.9140777587890623, 0.5081673855251735},
{13, 1.5, 2.2344483741062344e-10, 1.9728342692057286, 0.508975084092882},
{13, 1.6, 6.157302103198562e-11, 2.053156534830729, 0.5077062310112848},
{13, 1.7, 1.8549504854543277e-11, 2.1001942952473955, 0.5079991997612847},
{13, 1.8, 7.727455097399163e-12, 2.131098429361979, 0.5091676839192708},
{13, 1.9, 3.4877797068614593e-12, 2.192693074544271, 0.5078032090928819},
{13, 2.0, 1.4786702004869023e-12, 2.229151407877604, 0.5079537624782986},
{14, 1.2, 4.699427023519108e-08, 1.7449086507161453, 0.5073725721571182},
{14, 1.3, 2.761397451519925e-09, 1.8298339843749996, 0.5078374565972221},
{14, 1.4, 2.505210740740388e-10, 1.9203033447265625, 0.5075895860460069},
{14, 1.5, 4.105033785048668e-11, 1.9843190511067705, 0.5078445773654514},
{14, 1.6, 9.650795230961899e-12, 1.9902292887369795, 0.5130997043185765},
{14, 1.7, 2.6120765732851265e-12, 2.1137135823567714, 0.5068381754557291},
{14, 1.8, 8.437599841875124e-13, 2.1053721110026045, 0.5120607503255208},
{14, 1.9, 4.2322499570494777e-13, 2.1659596761067714, 0.5098282199435763},
{14, 2.0, 1.6256755487542861e-13, 2.228500366210937, 0.5080405680338542},
{15, 1.2, 1.3693634821648664e-08, 1.7414499918619788, 0.5076078965928819},
{15, 1.3, 8.322560291773562e-10, 1.7947133382161455, 0.5112523735894097},
{15, 1.4, 5.0003794669077473e-11, 1.8675791422526038, 0.5115582275390627},
{15, 1.5, 8.838887554462863e-12, 1.964930216471354, 0.5092205810546875},
{15, 1.6, 1.304664388513137e-12, 2.0278981526692714, 0.5096763102213542},
{15, 1.7, 3.6987552531089614e-13, 2.1216888427734375, 0.5061674669053818},
{15, 1.8, 1.1437622352932627e-13, 2.131169637044271, 0.5090835910373264},
{15, 1.9, 5.099664374381715e-14, 2.173151652018229, 0.5092965359157985},
{15, 2.0, 1.745295348486557e-14, 2.2203826904296875, 0.5088821750217013}
};

template<typename T> T esknew (T v, T beta, T e0)
  {
  auto tmp = (1-v)*(1+v);
  auto tmp2 = tmp>=0;
  return tmp2*exp(beta*(pow(tmp*tmp2, e0)-1));
  }

template<typename T> auto selectNESKernel(double ofactor, double epsilon)
  {
  size_t Wmin=1000;
  size_t idx = NEScache.size();
  for (size_t i=0; i<NEScache.size(); ++i)
    if ((NEScache[i].ofactor<=ofactor) && (1.3*NEScache[i].epsilon<=epsilon) && (NEScache[i].W<=Wmin))
      {
      idx = i;
      Wmin = NEScache[i].W;
      }
  MR_assert(idx<NEScache.size(), "no appropriate kernel found");
  auto supp = NEScache[idx].W;
  auto beta = NEScache[idx].beta*supp;
  auto e0 = NEScache[idx].e0;
  auto lam = [beta,e0](double v){return esknew(v, beta, e0);};
  return make_shared<HornerKernel<T>>(supp, supp+3, lam, GLFullCorrection(supp, lam));
  }

}

using detail_gridding_kernel::GriddingKernel;
using detail_gridding_kernel::selectNESKernel;

}

#endif
