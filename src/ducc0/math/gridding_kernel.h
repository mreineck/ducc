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
#include <memory>
#include <cmath>
#include <type_traits>
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
    const T *scoeff;
    size_t sstride;
    void (HornerKernel<T>::* evalfunc) (T, native_simd<T> *) const;
    T (HornerKernel<T>::* evalsinglefunc) (T) const;

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

    template<size_t DEG> T eval_single_intern(T x) const
      {
      auto nth = min(W-1, size_t(max(T(0), (x+1)*W*T(0.5))));
      x = (x+1)*W-2*nth-1;
      auto ptr = scoeff+nth;
      auto tval = *ptr;
      for (size_t j=1; j<=DEG; ++j)
        tval = tval*x + ptr[j*sstride];
      return tval;
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

    T eval_single_intern_general(T x) const
      {
      auto nth = min(W-1, size_t(max(T(0), (x+1)*W*T(0.5))));
      x = (x+1)*W-2*nth-1;
      auto ptr = scoeff+nth;
      auto tval = *ptr;
      for (size_t j=1; j<=D; ++j)
        tval = tval*x + ptr[j*sstride];
      return tval;
      }

    template<size_t NV, size_t DEG> void evfhelper2()
      {
      if (DEG==D)
        {
        evalfunc = &HornerKernel::eval_intern<NV,DEG>;
        evalsinglefunc = &HornerKernel::eval_single_intern<DEG>;
        }
      else if (DEG>MAXDEG)
        MR_fail("requested polynomial degree too high");
      else
        evfhelper2<NV, ((DEG>MAXDEG) ? DEG : DEG+1)>();
      }

    template<size_t NV> void evfhelper1()
      {
      if (nvec==NV)
        evfhelper2<NV,(NV-1)*vlen+4>();
      else if (nvec*vlen>MAXW)
        MR_fail("requested kernel support too high");
      else
        evfhelper1<((NV*vlen>MAXW) ? NV : NV+1)>();
      }

    void wire_eval()
      { evfhelper1<1>(); }

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
        coeff(makeCoeff(W_, D_, func)), scoeff(reinterpret_cast<T *>(&coeff[0])),
        sstride(vlen*nvec), corr(corr_)
      { wire_eval(); }

    virtual size_t support() const { return W; }

    virtual void eval(T x, native_simd<T> *res) const
      { (this->*evalfunc)(x, res); }

    virtual T eval_single(T x) const
      { return (this->*evalsinglefunc)(x); }

    virtual double corfunc(double x) const { return corr.corfunc(x); }

    /* Computes the correction function values at a coordinates
       [0, dx, 2*dx, ..., (n-1)*dx]  */
    virtual vector<double> corfunc(size_t n, double dx, int nthreads=1) const
      { return corr.corfunc(n, dx, nthreads); }

    const vector<Tsimd> &Coeff() const { return coeff; }
    size_t degree() const { return D; }
  };

template<size_t W, typename T> class TemplateKernel
  {
  private:
    static constexpr auto D=W+3;
    using Tsimd = native_simd<T>;
    static constexpr auto vlen = Tsimd::size();
    static constexpr auto nvec = (W+vlen-1)/vlen;

    std::array<Tsimd,(D+1)*nvec> coeff;
    const T *scoeff;
    static constexpr auto sstride = nvec*vlen;

  public:
    TemplateKernel(const HornerKernel<T> &krn)
      : scoeff(reinterpret_cast<T *>(&coeff[0]))
      {
      MR_assert(W==krn.support(), "support mismatch");
      MR_assert(D==krn.degree(), "degree mismatch");
      for (size_t i=0; i<coeff.size(); ++i) coeff[i] = krn.Coeff()[i];
      }

    constexpr size_t support() const { return W; }

    [[gnu::always_inline]] void eval2s(T x, T y, T z, size_t nth, native_simd<T> * DUCC0_RESTRICT res) const
      {
      z = (z-nth)*2+(W-1);
      if constexpr (nvec==1)
        {
        auto tvalx = coeff[0];
        auto tvaly = coeff[0];
        auto tvalz = coeff[0];
        for (size_t j=1; j<=D; ++j)
          {
          tvalx = tvalx*x + coeff[j];
          tvaly = tvaly*y + coeff[j];
          tvalz = tvalz*z + coeff[j];
          }
        res[0] = tvalx*T(tvalz[nth]);
        res[1] = tvaly;
        }
      else
        {
        auto ptrz = scoeff+nth;
        auto tvalz = *ptrz;
        for (size_t j=1; j<=D; ++j)
          tvalz = tvalz*z + ptrz[j*sstride];
        for (size_t i=0; i<nvec; ++i)
          {
          auto tvalx = coeff[i];
          auto tvaly = coeff[i];
          for (size_t j=1; j<=D; ++j)
            {
            tvalx = tvalx*x + coeff[j*nvec+i];
            tvaly = tvaly*y + coeff[j*nvec+i];
            }
          res[i] = tvalx*tvalz;
          res[i+nvec] = tvaly;
          }
        }
      }
    [[gnu::always_inline]] void eval2(T x, T y, native_simd<T> * DUCC0_RESTRICT res) const
      {
      if constexpr (nvec==1)
        {
        auto tvalx = coeff[0];
        auto tvaly = coeff[0];
        for (size_t j=1; j<=D; ++j)
          {
          tvalx = tvalx*x + coeff[j];
          tvaly = tvaly*y + coeff[j];
          }
        res[0] = tvalx;
        res[1] = tvaly;
        }
      else
        {
        for (size_t i=0; i<nvec; ++i)
          {
          auto tvalx = coeff[i];
          auto tvaly = coeff[i];
          for (size_t j=1; j<=D; ++j)
            {
            tvalx = tvalx*x + coeff[j*nvec+i];
            tvaly = tvaly*y + coeff[j*nvec+i];
            }
          res[i] = tvalx;
          res[i+nvec] = tvaly;
          }
        }
      }
  };

struct KernelParams
  {
  size_t W;
  double ofactor, epsilon, beta, e0;
  };

const vector<KernelParams> KernelDB {
{ 4, 1.15,   0.025654879, 1.3873426689, 0.5436851297},
{ 4, 1.20,   0.013809249, 1.3008419165, 0.5902137484},
{ 4, 1.25,  0.0085840685, 1.3274088935, 0.5953499486},
{ 4, 1.30,  0.0057322498, 1.3617063353, 0.5965631622},
{ 4, 1.35,  0.0042494419, 1.3845499880, 0.5990241291},
{ 4, 1.40,  0.0033459552, 1.4405325088, 0.5924776015},
{ 4, 1.45,  0.0028187359, 1.4635220066, 0.5929442711},
{ 4, 1.50,  0.0023843943, 1.5539689162, 0.5772217314},
{ 4, 1.55,  0.0020343796, 1.5991008653, 0.5721765215},
{ 4, 1.60,  0.0017143851, 1.6581546365, 0.5644747137},
{ 4, 1.65,  0.0014730848, 1.7135331415, 0.5572788589},
{ 4, 1.70,  0.0012554492, 1.7464330378, 0.5548742415},
{ 4, 1.75,  0.0010610904, 1.7887326906, 0.5509877716},
{ 4, 1.80, 0.00090885567, 1.8122309426, 0.5502273972},
{ 4, 1.85,  0.0007757401, 1.8304451327, 0.5503967160},
{ 4, 1.90,  0.0006740398, 1.8484487383, 0.5502376937},
{ 4, 1.95, 0.00058655391, 1.8742215688, 0.5489738941},
{ 4, 2.00, 0.00051911189, 1.9069436300, 0.5468009434},
{ 4, 2.05, 0.00047127936, 1.9287029587, 0.5459425678},
{ 4, 2.10, 0.00042991098, 1.9468344976, 0.5456431243},
{ 4, 2.15, 0.00039952939, 1.9598362025, 0.5457007164},
{ 4, 2.20, 0.00036728958, 1.9734445042, 0.5460368586},
{ 4, 2.25, 0.00034355459, 1.9833876672, 0.5464865366},
{ 4, 2.30, 0.00032238422, 1.9922532404, 0.5470505385},
{ 4, 2.35, 0.00030354772, 2.0001857065, 0.5476974129},
{ 4, 2.40, 0.00029003195, 2.0059275365, 0.5482583426},
{ 4, 2.45, 0.00027493243, 2.0124393824, 0.5489923951},
{ 4, 2.50, 0.00026418063, 2.0171876964, 0.5495887377},
{ 5, 1.15,  0.0088036926, 1.4211620799, 0.5484370222},
{ 5, 1.20,  0.0045432118, 1.4604589193, 0.5502520137},
{ 5, 1.25,  0.0025659469, 1.5114537479, 0.5482371505},
{ 5, 1.30,  0.0014949902, 1.5662004850, 0.5453959646},
{ 5, 1.35, 0.00092874124, 1.5940645314, 0.5464869375},
{ 5, 1.40,  0.0005820084, 1.6193311874, 0.5477484983},
{ 5, 1.45, 0.00041837131, 1.6702721179, 0.5446584432},
{ 5, 1.50, 0.00032139657, 1.7106607912, 0.5430584562},
{ 5, 1.55, 0.00025831183, 1.7411526213, 0.5424476190},
{ 5, 1.60, 0.00021156623, 1.7694517444, 0.5419943230},
{ 5, 1.65, 0.00018112326, 1.8069777863, 0.5396536287},
{ 5, 1.70, 0.00015177086, 1.8378820613, 0.5382171164},
{ 5, 1.75, 0.00012345178, 1.8819830388, 0.5352849778},
{ 5, 1.80, 0.00010093043, 1.9225188886, 0.5327117935},
{ 5, 1.85, 8.5743423e-05, 1.9766846627, 0.5286610959},
{ 5, 1.90, 7.5167678e-05, 2.0116590189, 0.5267194291},
{ 5, 1.95, 6.5915521e-05, 2.0401734131, 0.5256331063},
{ 5, 2.00, 5.7747201e-05, 2.0640495669, 0.5251476146},
{ 5, 2.05, 5.1546264e-05, 2.0815539113, 0.5250991831},
{ 5, 2.10, 4.6026099e-05, 2.0967549727, 0.5253193374},
{ 5, 2.15, 4.1922392e-05, 2.1078775421, 0.5256707657},
{ 5, 2.20,  3.755449e-05, 2.1195059411, 0.5262644187},
{ 5, 2.25, 3.4356546e-05, 2.1278007625, 0.5268876862},
{ 5, 2.30, 3.1539667e-05, 2.1348794082, 0.5276099184},
{ 5, 2.35, 2.9093026e-05, 2.1406731238, 0.5284272705},
{ 5, 2.40, 2.7399686e-05, 2.1442622022, 0.5291533269},
{ 5, 2.45, 2.5638396e-05, 2.1486533599, 0.5300021540},
{ 5, 2.50, 2.4438353e-05, 2.1554191248, 0.5302677770},
{ 6, 1.15,  0.0018919684, 1.4284593523, 0.5456388809},
{ 6, 1.20, 0.00087379161, 1.4871949080, 0.5434184013},
{ 6, 1.25, 0.00052387586, 1.5596009923, 0.5384733141},
{ 6, 1.30, 0.00030833805, 1.6293990176, 0.5339545697},
{ 6, 1.35, 0.00018595126, 1.6818294794, 0.5315541173},
{ 6, 1.40, 0.00010913759, 1.7181576557, 0.5313982413},
{ 6, 1.45,  7.446073e-05, 1.7747725210, 0.5283789353},
{ 6, 1.50, 5.3826324e-05, 1.8169800206, 0.5271499621},
{ 6, 1.55, 4.0746477e-05, 1.8477830463, 0.5269687744},
{ 6, 1.60, 3.1441179e-05, 1.8721012077, 0.5274488807},
{ 6, 1.65, 2.6100718e-05, 1.8868745350, 0.5282375527},
{ 6, 1.70, 2.1528068e-05, 1.9342762857, 0.5255216043},
{ 6, 1.75, 1.7177115e-05, 1.9721895688, 0.5238810155},
{ 6, 1.80, 1.3650115e-05, 1.9987007558, 0.5232375255},
{ 6, 1.85, 1.0598995e-05, 2.0219705218, 0.5229660080},
{ 6, 1.90, 8.8157904e-06, 2.0671787180, 0.5203913566},
{ 6, 1.95, 7.6286922e-06, 2.1003673879, 0.5190199946},
{ 6, 2.00, 6.5649967e-06, 2.1272513974, 0.5182973590},
{ 6, 2.05, 5.7476558e-06, 2.1458295051, 0.5181563583},
{ 6, 2.10, 5.0756513e-06, 2.1511783437, 0.5195284847},
{ 6, 2.15, 4.4661935e-06, 2.1743106617, 0.5184286324},
{ 6, 2.20, 3.8877561e-06, 2.1867440456, 0.5188571997},
{ 6, 2.25, 3.4672484e-06, 2.1957177257, 0.5193179074},
{ 6, 2.30, 3.1012426e-06, 2.2033754263, 0.5198649665},
{ 6, 2.35, 2.7894219e-06, 2.2096913080, 0.5204820251},
{ 6, 2.40, 2.5794626e-06, 2.2135312754, 0.5210295408},
{ 6, 2.45, 2.3571404e-06, 2.2317879841, 0.5204756852},
{ 6, 2.50, 2.1615297e-06, 2.2535598472, 0.5194274130},
{ 7, 1.15, 0.00078476028, 1.5248706519, 0.5288306317},
{ 7, 1.20, 0.00027127166, 1.5739348793, 0.5287992619},
{ 7, 1.25, 0.00012594628, 1.6245240723, 0.5279217770},
{ 7, 1.30, 7.0214545e-05, 1.6835745981, 0.5257484101},
{ 7, 1.35, 4.1972457e-05, 1.7343424414, 0.5239793844},
{ 7, 1.40,  2.378019e-05, 1.7845017738, 0.5224266045},
{ 7, 1.45, 1.3863408e-05, 1.8180597789, 0.5221834768},
{ 7, 1.50, 9.1605353e-06, 1.8680822720, 0.5206277502},
{ 7, 1.55,  6.479159e-06, 1.9188980015, 0.5183134674},
{ 7, 1.60, 4.6544571e-06, 1.9536166143, 0.5178695891},
{ 7, 1.65, 3.5489761e-06, 1.9786267068, 0.5178430252},
{ 7, 1.70, 2.7030348e-06, 2.0027666534, 0.5178577604},
{ 7, 1.75, 2.0533894e-06, 2.0289949199, 0.5176300336},
{ 7, 1.80, 1.6069122e-06, 2.0596412946, 0.5167551932},
{ 7, 1.85, 1.2936794e-06, 2.0720606842, 0.5178747891},
{ 7, 1.90, 1.0768664e-06, 2.0908981740, 0.5181009847},
{ 7, 1.95, 9.0890421e-07, 2.1086185697, 0.5184537843},
{ 7, 2.00, 7.7488775e-07, 2.1278284187, 0.5186377792},
{ 7, 2.05, 6.8025539e-07, 2.1300505355, 0.5201567726},
{ 7, 2.10, 6.0222531e-07, 2.1361214247, 0.5212397206},
{ 7, 2.15, 5.0130101e-07, 2.2231545475, 0.5137738214},
{ 7, 2.20, 4.2248762e-07, 2.2408449906, 0.5136504150},
{ 7, 2.25, 3.6494171e-07, 2.2556844458, 0.5135180892},
{ 7, 2.30, 3.1538194e-07, 2.2684881766, 0.5135795865},
{ 7, 2.35, 2.7282924e-07, 2.2815394648, 0.5136137508},
{ 7, 2.40, 2.4350524e-07, 2.2939223534, 0.5134777347},
{ 7, 2.45, 2.1263032e-07, 2.3041489588, 0.5138203874},
{ 7, 2.50, 1.9134836e-07, 2.3076482212, 0.5145035417},
{ 8, 1.15, 0.00026818611, 1.5681246490, 0.5223052481},
{ 8, 1.20, 7.8028732e-05, 1.6209261450, 0.5219287175},
{ 8, 1.25, 2.7460918e-05, 1.6851585171, 0.5199250590},
{ 8, 1.30, 1.3421658e-05, 1.7442373315, 0.5182155619},
{ 8, 1.35, 7.5158217e-06, 1.7876782642, 0.5176319503},
{ 8, 1.40, 4.2472384e-06, 1.8294321912, 0.5171860211},
{ 8, 1.45, 2.5794802e-06, 1.8716918210, 0.5161733611},
{ 8, 1.50, 1.6131994e-06, 1.9213040541, 0.5145350888},
{ 8, 1.55, 1.0974814e-06, 1.9637229131, 0.5134005827},
{ 8, 1.60,  7.531955e-07, 2.0002761373, 0.5128849282},
{ 8, 1.65, 5.5097346e-07, 2.0275645736, 0.5127082324},
{ 8, 1.70, 4.0136726e-07, 2.0498410409, 0.5130237662},
{ 8, 1.75,  2.906467e-07, 2.0731585170, 0.5131757153},
{ 8, 1.80, 2.1834922e-07, 2.0907418726, 0.5136046561},
{ 8, 1.85, 1.6329905e-07, 2.1164552354, 0.5133333878},
{ 8, 1.90, 1.2828598e-07, 2.1261570160, 0.5143004427},
{ 8, 1.95, 1.0171134e-07, 2.1363206613, 0.5152354910},
{ 8, 2.00, 8.1881369e-08, 2.1397013368, 0.5166895497},
{ 8, 2.05, 6.9121193e-08, 2.1466071700, 0.5176145380},
{ 8, 2.10, 5.9525932e-08, 2.1510526592, 0.5186914118},
{ 8, 2.15, 5.2942463e-08, 2.2365737543, 0.5125850104},
{ 8, 2.20, 4.3612361e-08, 2.2635555483, 0.5116114910},
{ 8, 2.25, 3.6764793e-08, 2.2808513000, 0.5112823144},
{ 8, 2.30, 3.0899101e-08, 2.2961118291, 0.5111472899},
{ 8, 2.35, 2.5951523e-08, 2.3025419974, 0.5117804832},
{ 8, 2.40, 2.2598633e-08, 2.3146967576, 0.5117074796},
{ 8, 2.45, 1.9029665e-08, 2.3186279028, 0.5125332192},
{ 8, 2.50, 1.6752523e-08, 2.3321309669, 0.5124348616},
{ 9, 1.15, 6.7523935e-05, 1.5927981851, 0.5188680345},
{ 9, 1.20, 2.2336088e-05, 1.6523068294, 0.5177819621},
{ 9, 1.25, 8.0261034e-06, 1.7103888450, 0.5164129862},
{ 9, 1.30, 3.2272675e-06, 1.7768638337, 0.5141821303},
{ 9, 1.35, 1.6398132e-06, 1.8259273732, 0.5131939428},
{ 9, 1.40, 8.5542435e-07, 1.8706775936, 0.5126332399},
{ 9, 1.45, 4.8998062e-07, 1.9079562176, 0.5122630978},
{ 9, 1.50, 2.8357238e-07, 1.9376265737, 0.5127460716},
{ 9, 1.55, 1.7632448e-07, 1.9831130904, 0.5114276594},
{ 9, 1.60, 1.1241387e-07, 2.0031047508, 0.5126932345},
{ 9, 1.65, 8.0252028e-08, 2.0285383278, 0.5127726006},
{ 9, 1.70,  5.741767e-08, 2.0574910347, 0.5124426828},
{ 9, 1.75, 4.0256578e-08, 2.0895174008, 0.5117693191},
{ 9, 1.80, 2.8882533e-08, 2.1256913951, 0.5104744301},
{ 9, 1.90, 1.5495159e-08, 2.1522089772, 0.5119205380},
{ 9, 1.95, 1.1746262e-08, 2.1630258913, 0.5127106930},
{ 9, 2.00, 9.1018891e-09, 2.1705676763, 0.5137317339},
{10, 1.15, 2.3448628e-05, 1.6209190610, 0.5149826251},
{10, 1.20, 5.9153174e-06, 1.6923420111, 0.5130947984},
{10, 1.25, 2.0151277e-06, 1.7478675248, 0.5122652231},
{10, 1.30, 7.6037409e-07, 1.8037273215, 0.5111938042},
{10, 1.35, 3.5081762e-07, 1.8371602342, 0.5120315623},
{10, 1.40, 1.7691912e-07, 1.8539150977, 0.5144182834},
{10, 1.45, 9.5898587e-08, 1.9083076984, 0.5123796414},
{10, 1.50, 5.1649488e-08, 1.9549482618, 0.5110787344},
{10, 1.55, 2.9344166e-08, 1.9935039498, 0.5103977246},
{10, 1.60, 1.6984065e-08, 2.0134235964, 0.5114751650},
{10, 1.65, 1.1201377e-08, 2.0278278839, 0.5124699585},
{10, 1.70, 7.7392472e-09, 2.0550888710, 0.5123896568},
{10, 1.75, 5.4226206e-09, 2.0984256136, 0.5108595057},
{10, 1.80, 3.8051062e-09, 2.1302650200, 0.5100239564},
{10, 1.85, 2.6039483e-09, 2.1585185874, 0.5095352903},
{10, 1.90, 1.8492238e-09, 2.1707445630, 0.5101429537},
{10, 1.95, 1.3147032e-09, 2.1817982009, 0.5108284721},
{10, 2.00, 9.6449676e-10, 2.2018582745, 0.5109329270},
{11, 1.15, 7.9162516e-06, 1.6472836222, 0.5119343348},
{11, 1.20, 1.7270227e-06, 1.7157016687, 0.5104680904},
{11, 1.25,  5.331949e-07, 1.7723332878, 0.5096753721},
{11, 1.30, 1.8775393e-07, 1.8205994670, 0.5094502380},
{11, 1.35, 7.7356019e-08, 1.8518214951, 0.5103702455},
{11, 1.40, 3.2528373e-08, 1.8880512818, 0.5108048393},
{11, 1.45, 1.6597763e-08, 1.9165471477, 0.5113840356},
{11, 1.50, 8.8600675e-09, 1.9625071236, 0.5103156252},
{11, 1.55, 4.8306858e-09, 2.0049198756, 0.5093020852},
{11, 1.60, 2.5766957e-09, 2.0295371707, 0.5098854450},
{11, 1.65,  1.559903e-09, 2.0444486400, 0.5107653615},
{11, 1.70, 9.9258899e-10, 2.0627527204, 0.5113810966},
{11, 1.75, 6.9033069e-10, 2.0881262613, 0.5114471968},
{11, 1.80, 4.9444651e-10, 2.1384158198, 0.5092846283},
{11, 1.85, 3.3291902e-10, 2.1714286439, 0.5084380285},
{11, 1.90, 2.3006797e-10, 2.1889103178, 0.5086174417},
{11, 1.95, 1.5880468e-10, 2.2013464966, 0.5091627247},
{11, 2.00, 1.1177184e-10, 2.2147511730, 0.5096421117},
{12, 1.15, 2.7535895e-06, 1.6661837519, 0.5098172147},
{12, 1.20, 5.2570038e-07, 1.7294557459, 0.5089239596},
{12, 1.25,  1.378658e-07, 1.7698182384, 0.5099240718},
{12, 1.30, 4.4329167e-08, 1.8092042442, 0.5106074270},
{12, 1.35, 1.7038991e-08, 1.8619112597, 0.5093832337},
{12, 1.40, 6.5438748e-09, 1.9069147481, 0.5089479889},
{12, 1.45, 2.9874764e-09, 1.9318398074, 0.5098082325},
{12, 1.50, 1.4920459e-09, 1.9628483155, 0.5100985753},
{12, 1.55, 8.0989276e-10, 2.0129847811, 0.5085327805},
{12, 1.60, 4.1660575e-10, 2.0517921747, 0.5079102398},
{12, 1.65, 2.3539727e-10, 2.0698388400, 0.5085131064},
{12, 1.70, 1.3497289e-10, 2.0887365361, 0.5090417146},
{12, 1.75, 8.3256938e-11, 2.1069557330, 0.5095920671},
{12, 1.80, 5.8834619e-11, 2.1359415217, 0.5091887069},
{12, 1.90, 2.6412908e-11, 2.2006369514, 0.5075889699},
{12, 1.95, 1.7189689e-11, 2.2146741638, 0.5080017404},
{12, 2.00, 1.2174796e-11, 2.2431392199, 0.5075191177},
{13, 1.15, 9.2170387e-07, 1.6836301171, 0.5079475797},
{13, 1.20, 1.6209683e-07, 1.7348077147, 0.5084318642},
{13, 1.25, 3.7397438e-08, 1.7964828447, 0.5071972221},
{13, 1.30, 1.0186845e-08, 1.8283438535, 0.5086467978},
{13, 1.35, 3.6754613e-09, 1.8810972719, 0.5075439352},
{13, 1.40, 1.3183319e-09, 1.9191093732, 0.5077468429},
{13, 1.45, 5.3619083e-10, 1.9470729711, 0.5082824666},
{13, 1.50, 2.2322801e-10, 1.9739868782, 0.5088959158},
{13, 1.55, 1.1526271e-10, 2.0033259275, 0.5090866602},
{13, 1.60, 6.1426379e-11, 2.0510989715, 0.5078563876},
{13, 1.65, 3.4052024e-11, 2.0818527907, 0.5074775828},
{13, 1.70, 1.8584229e-11, 2.1045643925, 0.5077204961},
{13, 1.75, 1.0810235e-11, 2.1243271913, 0.5081633997},
{13, 1.80,  7.377578e-12, 2.1467756379, 0.5081996246},
{13, 1.85, 4.9615599e-12, 2.1851483818, 0.5071383353},
{13, 1.90, 3.2630681e-12, 2.2086376720, 0.5068763959},
{13, 1.95,   2.08439e-12, 2.2299240544, 0.5068023663},
{13, 2.00, 1.3665993e-12, 2.2491841265, 0.5068976036},
{14, 1.15, 3.3179964e-07, 1.6903795555, 0.5072137985},
{14, 1.20, 4.7410658e-08, 1.7360564953, 0.5082272767},
{14, 1.25, 9.7512113e-09, 1.7969192746, 0.5071477747},
{14, 1.30, 2.3054484e-09, 1.8341158636, 0.5079636910},
{14, 1.35, 6.9021044e-10, 1.8675148638, 0.5085594139},
{14, 1.40, 2.4957022e-10, 1.9167647959, 0.5078690718},
{14, 1.45, 1.0233715e-10, 1.9577546112, 0.5073185014},
{14, 1.50, 4.0219827e-11, 1.9908513900, 0.5074206790},
{14, 1.55, 1.8311452e-11, 2.0199455000, 0.5076279818},
{14, 1.60, 9.3257786e-12, 2.0564687260, 0.5072732788},
{14, 1.65, 5.0810838e-12, 2.0910978691, 0.5066391006},
{14, 1.70, 2.6125118e-12, 2.1169092067, 0.5066457965},
{14, 1.75, 1.3838674e-12, 2.1429649758, 0.5066883335},
{14, 1.80, 8.4432728e-13, 2.1058949451, 0.5120095847},
{14, 1.85, 5.6112166e-13, 2.1848347409, 0.5069391840},
{14, 1.90, 3.6520481e-13, 2.2144100981, 0.5063398541},
{14, 1.95, 2.2835769e-13, 2.2387205994, 0.5061044697},
{14, 2.00, 1.5660252e-13, 2.2662033397, 0.5057027272},
{15, 1.15, 1.1497856e-07, 1.6942402725, 0.5068315822},
{15, 1.20, 1.3576023e-08, 1.7512660599, 0.5067157295},
{15, 1.25, 2.4191681e-09, 1.7894073869, 0.5077522206},
{15, 1.30, 5.6426243e-10, 1.8433332696, 0.5070776169},
{15, 1.35, 1.5386302e-10, 1.8866247570, 0.5068449590},
{15, 1.40, 4.5274677e-11, 1.9207795531, 0.5073294602},
{15, 1.45, 1.8387362e-11, 1.9015963774, 0.5125151370},
{15, 1.50, 6.7738438e-12, 2.0031350908, 0.5063691270},
{15, 1.55,  2.664519e-12, 2.0276909655, 0.5068592217},
{15, 1.60, 1.2501824e-12, 2.0594275589, 0.5068953514},
{15, 1.65, 7.1442756e-13, 2.0958989255, 0.5061994449},
{15, 1.70, 3.6768121e-13, 2.1256130851, 0.5059364487},
{15, 1.75, 1.8650547e-13, 2.1497846837, 0.5060649594},
{15, 1.80, 1.1458006e-13, 2.1307035522, 0.5091470333},
{15, 1.85, 7.0782501e-14, 2.1464045447, 0.5101577708},
{15, 1.90,  4.407902e-14, 2.2222258168, 0.5056705159},
{15, 1.95, 2.6655793e-14, 2.2419149837, 0.5057005374},
{15, 2.00, 1.7108261e-14, 2.2390528894, 0.5072248580},
// Some of the support=16 kernels have such a high dynamic range that cutoff
// errors become significant. Let's disable these.
//{16, 1.15, 4.5498331e-08, 1.7085132020, 0.5052624971},
//{16, 1.20, 4.3986374e-09, 1.7630310045, 0.5055047860},
//{16, 1.25, 6.1272681e-10, 1.7998874282, 0.5066934971},
{16, 1.30, 1.1509596e-10, 1.7892839755, 0.5122877693},
{16, 1.35, 3.2440049e-11, 1.8914441282, 0.5063521839},
{16, 1.40, 8.4329616e-12, 1.9296369098, 0.5065170208},
{16, 1.45, 3.1161739e-12, 1.9674735425, 0.5063244338},
{16, 1.50, 1.2100308e-12, 2.0130787701, 0.5055587965},
{16, 1.55, 4.6082202e-13, 2.0438032614, 0.5056309683},
{16, 1.60, 1.7883238e-13, 2.0329561822, 0.5089045671},
{16, 1.65, 9.2853815e-14, 2.0494514743, 0.5103582604},
{16, 1.70, 5.6614567e-14, 2.0925119791, 0.5083767402},
{16, 1.75,  2.875391e-14, 2.1461524027, 0.5062037834},
{16, 1.80, 1.6578982e-14, 2.1490040175, 0.5082721830},
{16, 1.85, 1.1782751e-14, 2.1811826814, 0.5072570059},
{16, 1.90, 8.9196865e-15, 2.1981176583, 0.5075840871},
{16, 1.95, 6.6530006e-15, 2.2340011350, 0.5060133105},
{16, 2.00, 5.0563492e-15, 2.2621631913, 0.5056924675}
};

template<typename T> T esknew (T v, T beta, T e0)
  {
  auto tmp = (1-v)*(1+v);
  auto tmp2 = tmp>=0;
  return tmp2*exp(beta*(pow(tmp*tmp2, e0)-1));
  }

template<typename T> auto selectKernel(size_t idx)
  {
  MR_assert(idx<KernelDB.size(), "no appropriate kernel found");
  auto supp = KernelDB[idx].W;
  auto beta = KernelDB[idx].beta*supp;
  auto e0 = KernelDB[idx].e0;
  auto lam = [beta,e0](double v){return esknew(v, beta, e0);};
  return make_shared<HornerKernel<T>>(supp, supp+3, lam, GLFullCorrection(supp, lam));
  }

/*! Returns the best matching 2-parameter ES kernel for the given oversampling
    factor and error. */
template<typename T> auto selectKernel(double ofactor, double epsilon)
  {
  size_t Wmin = is_same<T, float>::value ? 8 : 1000;
  size_t idx = KernelDB.size();
  for (size_t i=0; i<KernelDB.size(); ++i)
    if ((KernelDB[i].ofactor<=ofactor) && (KernelDB[i].epsilon<=epsilon) && (KernelDB[i].W<=Wmin))
      {
      idx = i;
      Wmin = KernelDB[i].W;
      }
  return selectKernel<T>(idx);
  }
template<typename T> auto selectKernel(double ofactor, double epsilon, size_t idx)
  {
  return (idx<KernelDB.size()) ?
    selectKernel<T>(idx) : selectKernel<T>(ofactor, epsilon);
  }

template<typename T> auto getAvailableKernels(double epsilon,
  double ofactor_min=1.1, double ofactor_max=2.6)
  {
  vector<double> ofc(20, ofactor_max);
  vector<size_t> idx(20, KernelDB.size());
  size_t Wlim = is_same<T, float>::value ? 8 : 16;
  for (size_t i=0; i<KernelDB.size(); ++i)
    {
    auto ofactor = KernelDB[i].ofactor;
    size_t W = KernelDB[i].W;
    if ((W<=Wlim) && (KernelDB[i].epsilon<=epsilon)
     && (ofactor<ofc[W]) && (ofactor>=ofactor_min))
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

}

using detail_gridding_kernel::GriddingKernel;
using detail_gridding_kernel::selectKernel;
using detail_gridding_kernel::getAvailableKernels;
using detail_gridding_kernel::HornerKernel;
using detail_gridding_kernel::TemplateKernel;
using detail_gridding_kernel::KernelParams;
using detail_gridding_kernel::KernelDB;

}

#endif
