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

/*! \file sht.cc
 *  Functionality related to spherical harmonic transforms
 *
 *  Copyright (C) 2020-2025 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <vector>
#include <cmath>
#include <cstring>
#if ((!defined(DUCC0_NO_SIMD)) && defined(__AVX__) && (!defined(__AVX512F__)))
#include <x86intrin.h>
#endif
#include "ducc0/infra/simd.h"
#include "ducc0/sht/sht.h"
#include "ducc0/sht/sphere_interpol.h"
#include "ducc0/fft/fft.h"
#include "ducc0/nufft/nufft.h"
#include "ducc0/math/math_utils.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/solvers.h"
#include "ducc0/sht/sht_utils.h"
#include "ducc0/infra/timers.h"
#include "ducc0/sht/sht_inner_loop.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

struct ringhelper
  {
  using dcmplx = complex<double>;
  double phi0_;
  vector<dcmplx> shiftarr;
  size_t s_shift;
  unique_ptr<pocketfft_r<double>> plan;
  vector<double> buf;
  size_t length;
  bool norot;
  ringhelper() : phi0_(0), s_shift(0), length(0), norot(false) {}
  void update(size_t nph, size_t mmax, double phi0)
    {
    norot = (abs(phi0)<1e-14);
    if (!norot)
      if ((mmax!=s_shift-1) || (!approx(phi0,phi0_,1e-15)))
      {
      shiftarr.resize(mmax+1);
      s_shift = mmax+1;
      phi0_ = phi0;
      MultiExp<double, dcmplx> mexp(phi0, mmax+1);
      for (size_t m=0; m<=mmax; ++m)
        shiftarr[m] = mexp[m];
      }
    if (nph!=length)
      {
      plan=make_unique<pocketfft_r<double>>(nph);
      buf.resize(plan->bufsize());
      length=nph;
      }
    }
  template<typename T> DUCC0_NOINLINE void phase2ring (size_t nph,
    double phi0, const vmav<double,1> &data, size_t mmax, const cmav<complex<T>,1> &phase)
    {
    update (nph, mmax, phi0);

    if (nph>=2*mmax+1)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          {
          data(2*m)=phase(m).real();
          data(2*m+1)=phase(m).imag();
          }
      else
        for (size_t m=0; m<=mmax; ++m)
          {
          dcmplx tmp = dcmplx(phase(m))*shiftarr[m];
          data(2*m)=tmp.real();
          data(2*m+1)=tmp.imag();
          }
      for (size_t m=2*(mmax+1); m<nph+2; ++m)
        data(m)=0.;
      }
    else
      {
      data(0)=phase(0).real();
      fill(&data(1),&data(nph+2),0.);

      for (size_t m=1, idx1=(nph==1) ? 0 : 1, idx2=nph-1; m<=mmax; ++m,
           idx1=(idx1+1==nph) ? 0 : idx1+1, idx2=(idx2==0) ? nph-1 : idx2-1)
        {
        dcmplx tmp = phase(m);
        if(!norot) tmp*=shiftarr[m];
        if (idx1<(nph+2)/2)
          {
          data(2*idx1)+=tmp.real();
          data(2*idx1+1)+=tmp.imag();
          }
        if (idx2<(nph+2)/2)
          {
          data(2*idx2)+=tmp.real();
          data(2*idx2+1)-=tmp.imag();
          }
        }
      }
    data(1)=data(0);
    plan->exec_copyback(&(data(1)), buf.data(), 1., false);
    }
  template<typename T> DUCC0_NOINLINE void ring2phase (size_t nph, double phi0,
    const vmav<double,1> &data, size_t mmax, const vmav<complex<T>,1> &phase)
    {
    update (nph, mmax, -phi0);

    plan->exec_copyback(&(data(1)), buf.data(), 1., true);
    data(0)=data(1);
    data(1)=data(nph+1)=0.;

    if (mmax<=nph/2)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          phase(m) = complex<T>(T(data(2*m)), T(data(2*m+1)));
      else
        for (size_t m=0; m<=mmax; ++m)
          phase(m) = complex<T>(dcmplx(data(2*m), data(2*m+1)) * shiftarr[m]);
      }
    else
      {
      for (size_t m=0, idx=0; m<=mmax; ++m, idx=(idx+1==nph) ? 0 : idx+1)
        {
        dcmplx val;
        if (idx<(nph-idx))
          val = dcmplx(data(2*idx), data(2*idx+1));
        else
          val = dcmplx(data(2*(nph-idx)), -data(2*(nph-idx)+1));
        if (!norot)
          val *= shiftarr[m];
        phase(m)=complex<T>(val);
        }
      }
    }
  };

size_t get_mmax(const cmav<size_t,1> &mval, size_t lmax)
  {
  size_t nm=mval.shape(0);
  size_t mmax=0;
  vector<bool> present(lmax+1, false);
  for (size_t mi=0; mi<nm; ++mi)
    {
    size_t m=mval(mi);
    MR_assert(m<=lmax, "mmax too large");
    MR_assert(!present[m], "m value present more than once");
    present[m]=true;
    mmax = max(mmax,m);
    }
  return mmax;
  }

DUCC0_NOINLINE size_t get_mlim (size_t lmax, size_t spin, double sth, double cth)
  {
  double ofs=lmax*0.01;
  if (ofs<100.) ofs=100.;
  double b = -2*double(spin)*abs(cth);
  double t1 = lmax*sth+ofs;
  double c = double(spin)*spin-t1*t1;
  double discr = b*b-4*c;
  if (discr<=0) return lmax;
  double res=(-b+sqrt(discr))/2.;
  res = min(res, double(lmax));
  return size_t(res+0.5);
  }

vector<ringdata> make_ringdata(const cmav<double,1> &theta, size_t lmax,
  size_t spin)
  {
  size_t nrings = theta.shape(0);
  struct ringinfo
    {
    double theta, cth, sth;
    size_t idx;
    };
  vector<ringinfo> tmp(nrings);
  for (size_t i=0; i<nrings; ++i)
    tmp[i] = { theta(i), cos(theta(i)), sin(theta(i)), i };
  sort(tmp.begin(), tmp.end(), [](const ringinfo &a, const ringinfo &b)
    { return (a.sth<b.sth); });

  vector<ringdata> res;
  size_t pos=0;
  while (pos<nrings)
    {
    if ((pos+1<nrings) && (fabs(tmp[pos].theta+tmp[pos+1].theta-pi)<=5e-15))
      {
      if (tmp[pos].theta<tmp[pos+1].theta)
        res.push_back({get_mlim(lmax, spin, tmp[pos].sth, tmp[pos].cth),
          tmp[pos].idx, tmp[pos+1].idx, tmp[pos].cth, tmp[pos].sth});
      else
        res.push_back({get_mlim(lmax, spin, tmp[pos+1].sth, tmp[pos+1].cth),
          tmp[pos+1].idx, tmp[pos].idx, tmp[pos+1].cth, tmp[pos+1].sth});
      pos += 2;
      }
    else
      {
      res.push_back({get_mlim(lmax, spin, tmp[pos].sth, tmp[pos].cth),
        tmp[pos].idx, tmp[pos].idx, tmp[pos].cth, tmp[pos].sth});
      ++pos;
      }
    }
  return res;
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
static vector<double> get_dh_weights(size_t nrings)
  {
  vector<double> weight(nrings);

  weight[0]=2.;
  for (size_t k=1; k<=(nrings/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k);
  weight[2*(nrings/2)-1]=(nrings-3.)/(2*(nrings/2)-1) -1.;
  pocketfft_r<double> plan(nrings);
  plan.exec(weight.data(), 1., false);
  weight[0] = 0.;  // ensure that this is an exact zero
  return weight;
  }

size_t maximum_safe_l(const string &geometry, size_t ntheta)
  {
  if ((geometry=="GL")||(geometry=="F1")||(geometry=="MW")||(geometry=="MWflip"))
    {
    MR_assert(ntheta>0, "need at least one ring");
    return ntheta-1;
    }
  else if (geometry=="CC")
    {
    MR_assert(ntheta>1, "need at least two rings");
    return ntheta-2;
    }
  else if (geometry=="DH")
    {
    MR_assert(ntheta>1, "need at least two rings");
    return (ntheta-2)/2;
    }
  else if (geometry=="F2")
    {
    MR_assert(ntheta>0, "need at least one ring");
    return (ntheta-1)/2;
    }
  MR_fail("unsupported grid type");
  }

void get_gridweights(const string &type, const vmav<double,1> &wgt)
  {
  size_t nrings=wgt.shape(0);
  if (type=="GL") // Gauss-Legendre
    {
    ducc0::GL_Integrator integ(nrings);
    auto xwgt = integ.weights();
    for (size_t m=0; m<nrings; ++m)
      wgt(m) = 2*pi*xwgt[m];
    }
  else if (type=="F1") // Fejer 1
    {
    /* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
    vector<double> xwgt(nrings);
    xwgt[0]=2.;
    UnityRoots<double,dcmplx> roots(2*nrings);
    for (size_t k=1; k<=(nrings-1)/2; ++k)
      {
      auto tmp = roots[k];
      xwgt[2*k-1]=2./(1.-4.*k*k)*tmp.real();
      xwgt[2*k  ]=2./(1.-4.*k*k)*tmp.imag();
      }
    if ((nrings&1)==0) xwgt[nrings-1]=0.;
    pocketfft_r<double> plan(nrings);
    plan.exec(xwgt.data(), 1., false);
    for (size_t m=0; m<(nrings+1)/2; ++m)
      wgt(m)=wgt(nrings-1-m)=xwgt[m]*2*pi/nrings;
    }
  else if (type=="CC") // Clenshaw-Curtis
    {
    /* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
    MR_assert(nrings>1, "too few rings for Clenshaw-Curtis grid");
    size_t n=nrings-1;
    double dw=-1./(n*n-1.+(n&1));
    vector<double> xwgt(nrings);
    xwgt[0]=2.+dw;
    for (size_t k=1; k<n/2; ++k)
      xwgt[2*k-1]=2./(1.-4.*k*k) + dw;
    if (n>1)
      xwgt[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
    pocketfft_r<double> plan(n);
    plan.exec(xwgt.data(), 1., false);
    for (size_t m=0; m<(nrings+1)/2; ++m)
      wgt(m)=wgt(nrings-1-m)=xwgt[m]*2*pi/n;
    }
  else if (type=="F2") // Fejer 2
    {
    auto xwgt = get_dh_weights(nrings+1);
    for (size_t m=0; m<nrings; ++m)
      wgt(m) = xwgt[m+1]*2*pi/(nrings+1);
    }
  else if (type=="DH") // Driscoll-Healy
    {
    auto xwgt = get_dh_weights(nrings);
    for (size_t m=0; m<nrings; ++m)
      wgt(m) = xwgt[m]*2*pi/nrings;
    }
  else
    MR_fail("unsupported grid type");
  }

vmav<double,1> get_gridweights(const string &type, size_t nrings)
  {
  vmav<double,1> wgt({nrings}, UNINITIALIZED);
  get_gridweights(type, wgt);
  return wgt;
  }


bool downsampling_ok(const cmav<double,1> &theta, size_t lmax,
  bool &npi, bool &spi, size_t &ntheta_out)
  {
  size_t ntheta = theta.shape(0);
  if (ntheta<=500) return false; // not worth thinking about shortcuts
  npi = abs_approx(theta(0), 0., 1e-14);
  spi = abs_approx(theta(ntheta-1), pi, 1e-14);
  size_t nthetafull = 2*ntheta-npi-spi;
  double dtheta = 2*pi/nthetafull;
  for (size_t i=0; i<ntheta; ++i)
    if (!abs_approx(theta(i),(0.5*(1-npi)+i)*dtheta, 1e-14))
      return false;
  size_t npairs = ntheta*(2-(npi==spi))/2;
  ntheta_out = good_size_complex(lmax+1)+1;
  if (2*npairs<1.2*ntheta_out)  // not worth taking the shortcut
    return false;
  return true;
  }

static void mul_cth(const vmav<complex<double>,2> &clm, size_t m, size_t lmax)
  {
  dcmplx oldm0=0, oldm1=0;
  for (size_t l=m; l<=lmax+1; ++l)
    {
    dcmplx r0=0, r1=0;
    if (l>m)
      {
      double fct = sqrt((l+m)*(l-m)/((2.*l+1.)*(2.*l-1.)));
      r0 += oldm0*fct;
      r1 += oldm1*fct;
      }
    if (l<lmax)
      {
      double fct = sqrt((l+1.+m)*(l+1.-m)/((2.*l+3.)*(2.*l+1.)));
      r0 += clm(0,l+1)*fct;
      r1 += clm(1,l+1)*fct;
      }
    oldm0=clm(0,l);
    oldm1=clm(1,l);
    clm(0,l) = r0;
    clm(1,l) = r1;
    }
  }

void spin0to1 (const vmav<complex<double>,2> &alm, size_t lmax, size_t m)
  {
  double em = double(m);
  dcmplx last0=0, last1=0;
  for (size_t l=m; l<=lmax; ++l)
    {
    double el = double(l);
    dcmplx coeff0(0), coeff1(0);
    // contribution from l
    if (l>0)
      {
      auto fct = dcmplx(0., em);
      coeff0 += -fct*alm(1,l);
      coeff1 += -fct*alm(0,l);
      }
    // contribution from l-1
    if (l>m)
      {
      double stdtx = sqrt((el+em)*(el-em)/((2.*el+1.)*(2.*el-1.))) * (-el-1.);
      coeff0 +=  stdtx*last0;
      coeff1 += -stdtx*last1;
      }
    // contribution from l+1;
    if (true) // (l<base_in.Lmax())
      {
      double stdtx = sqrt((el+1.+em)*(el+1.-em)/((2.*el+3.)*(2.*el+1.))) * el;
      coeff0 +=  stdtx*alm(0,l+1);
      coeff1 += -stdtx*alm(1,l+1);
      }
    last0 = alm(0,l);
    last1 = alm(1,l);
    double norm = (l>0) ? 1./sqrt(el*(el+1)) : 0;
    alm(0,l) = norm*coeff0;
    alm(1,l) =-norm*coeff1;
    }
  }
void spin0to2 (const vmav<complex<double>,2> &alm,
  const cmav<double,1> &f2, const vmav<dcmplx,2> &glm, size_t lmax, size_t m)
  {
  constexpr dcmplx img(0.,1.);

  double em = double(m);
  // copy in glm
  for (size_t l=m; l<=lmax+2; ++l)
    {
    // component alm(0) needs sign flipped, no idea why
    glm(0,l) = alm(0,l) + img*alm(1,l);
    glm(1,l) = alm(0,l) - img*alm(1,l);
    }
  for (size_t l=m; l<=lmax; ++l)
    {
    double el=l;
    alm(0,l) = (2.*em*em-el*(el+1))*glm(0,l);
    alm(1,l) = (2.*em*em-el*(el+1))*glm(1,l);
    if (l>m)
      {
      alm(0,l) +=  2.*sqrt((2.*el+1.)/(2.*el-1.)*(el*el-em*em))*em*glm(0,l-1);
      alm(1,l) += -2.*sqrt((2.*el+1.)/(2.*el-1.)*(el*el-em*em))*em*glm(1,l-1);
      }
    }
  mul_cth(glm, m, lmax+2);
  for (size_t l=m; l<=lmax; ++l)
    {
    double el=l;
    alm(0,l) += -2.*em*(el-1.)*glm(0,l);
    alm(1,l) +=  2.*em*(el-1.)*glm(1,l);
    if (l>m)
      {
      alm(0,l) += 2.*sqrt((2.*el+1.)/(2.*el-1.)*(el*el-em*em))*glm(0,l-1);
      alm(1,l) += 2.*sqrt((2.*el+1.)/(2.*el-1.)*(el*el-em*em))*glm(1,l-1);
      }
    }
  mul_cth(glm, m, lmax+3);
  for (size_t l=m; l<=lmax; ++l)
    {
    double el=l;
    alm(0,l) += el*(el-1)*glm(0,l);
    alm(1,l) += el*(el-1)*glm(1,l);
    alm(0,l) *= f2(l);
    alm(1,l) *= f2(l);
    }
  // copy to result
  for (size_t l=m; l<=lmax; ++l)
    {
    // component alm_out(1) needs sign flipped, no idea why
    auto t0 = alm(0,l);
    auto t1 = alm(1,l);
    alm(0,l) = -0.5*(t1+t0);
    alm(1,l) = -0.5*img*(t1-t0);
    }
  }
void spin1to0 (const vmav<complex<double>,2> &alm, size_t lmax, size_t m)
  {
  double em = double(m);
  dcmplx last0=0, last1=0;
  for (size_t l=m; l<=lmax+1; ++l)
    {
    double el = double(l);
    dcmplx coeff0(0), coeff1(0);
    // contribution from l; identical to 0 if m==0 or l==0
    if ((l>0) && (l<=lmax))
      {
      auto fct = dcmplx(0., em)/sqrt(el*(el+1.));
      coeff0 +=  fct*alm(1,l);
      coeff1 += -fct*alm(0,l);
      }
    // contribution from l-1; identical to 0 if l<2
    if ((l>m) && (l>1))
      {
      double stdtx = sqrt((el+em)*(el-em)/((2.*el+1.)*(2.*el-1.))) * (el-1.);
      stdtx /= sqrt(el*(el-1.));
      coeff0 += -stdtx*last0;
      coeff1 += -stdtx*last1;
      }
    // contribution from l+1
    if (l<lmax)
      {
      double stdtx = sqrt((el+1.+em)*(el+1.-em)/((2.*el+3.)*(2.*el+1.))) * (-el-2.);
      stdtx /= sqrt((el+1.)*(el+2.));
      coeff0 += -stdtx*alm(0,l+1);
      coeff1 += -stdtx*alm(1,l+1);
      }
    last0 = alm(0,l);
    last1 = alm(1,l);
    alm(0,l) = -coeff0;
    alm(1,l) = -coeff1;
    }
  }
void spin2to0 (const vmav<complex<double>,2> &alm, const cmav<double,1> &f1,
  const cmav<double,1> &f2, const vmav<dcmplx,2> &glm, size_t lmax, size_t m)
  {
  constexpr dcmplx img(0.,1.);

  double em = double(m);
  // copy in glm
  for (size_t l=m; l<=lmax; ++l)
    {
    // component alm_in(1) needs sign flipped, no idea why
    glm(0,l) = -alm(0,l) - img*alm(1,l);
    glm(1,l) = -alm(0,l) + img*alm(1,l);
    }
  for (size_t l=m; l<=lmax; ++l)
    {
    double el=l;
    alm(0,l) = glm(0,l)*f2(l)*el*(el-1.);
    alm(1,l) = glm(1,l)*f2(l)*el*(el-1.);
    }
  mul_cth(alm, m, lmax);
  for (size_t l=m; l<=lmax; ++l)
    {
    double el=l;
    alm(0,l) += -glm(0,l)*f2(l)*2.*em*(el-1.);
    alm(1,l) +=  glm(1,l)*f2(l)*2.*em*(el-1.);
    if (l<lmax)
      {
      double fct = 2.*sqrt(f1(l)*((el+1.)*(el+1.)-em*em));
      alm(0,l) += glm(0,l+1)*fct;
      alm(1,l) += glm(1,l+1)*fct;
      }
    }
  mul_cth(alm, m, lmax+1);
  for (size_t l=m; l<=lmax; ++l)
    {
    double el=l;
    alm(0,l) += glm(0,l)*f2(l)*(2.*em*em-el*(el+1.));
    alm(1,l) += glm(1,l)*f2(l)*(2.*em*em-el*(el+1.));
    if (l<lmax)
      {
      double fct = 2.*em*sqrt(f1(l)*((el+1.)*(el+1.)-em*em));
      alm(0,l) +=  glm(0,l+1)*fct;
      alm(1,l) += -glm(1,l+1)*fct;
      }
    }
  // copy to result
  for (size_t l=m; l<=lmax+2; ++l)
    {
    // component alm_out(0) needs sign flipped, no idea why
    auto t0 = alm(0,l);
    auto t1 = alm(1,l);
    alm(0,l) = 0.5*(t1+t0);
    alm(1,l) = 0.5*img*(t1-t0);
    }
  }

template<typename T> vmav<complex<T>,3> allocate_leg
  (size_t ncomp, size_t ntheta, size_t nm, bool synth, bool resample_theta, size_t nthreads)
  {
  if ((!resample_theta) && ((!synth)||(nthreads==1)))
    return vmav<complex<T>,3>::build_noncritical
      ({ncomp, ntheta, nm}, PAGE_IN(nthreads));
  return vmav<complex<T>,3>::build_noncritical
    ({ncomp, nm, ntheta}, PAGE_IN(nthreads)).transpose({0,2,1});
  }

template<typename T> void alm2leg(  // associated Legendre transform
  const cmav<complex<T>,2> &alm, // (ncomp, lmidx)
  const vmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol)
  {
  // sanity checks
  auto nrings=theta.shape(0);
  MR_assert(nrings==leg.shape(1), "nrings mismatch");
  auto nm=mval.shape(0);
  MR_assert(nm==mstart.shape(0), "nm mismatch");
  MR_assert(nm==leg.shape(2), "nm mismatch");
  auto nalm=alm.shape(0);
  auto mmax = get_mmax(mval, lmax);
  if (mode==DERIV1)
    {
    spin=1;
    MR_assert(nalm==1, "need one a_lm component");
    MR_assert(leg.shape(0)==2, "need two Legendre components");
    }
  else if (mode==GRAD_ONLY)
    {
    MR_assert(spin>0, "spin must be positive for grad-only SHTs");
    MR_assert(nalm==1, "need one a_lm component");
    MR_assert(leg.shape(0)==2, "need two Legendre components");
    }
  else
    {
    size_t ncomp = (spin==0) ? 1 : 2;
    MR_assert(nalm==ncomp, "incorrect number of a_lm components");
    MR_assert(leg.shape(0)==ncomp, "incorrect number of Legendre components");
    }

  if (even_odd_m(mval))
    {
    bool npi, spi;
    size_t ntheta_tmp;
    if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
      {
      vmav<double,1> theta_tmp({ntheta_tmp}, UNINITIALIZED);
      for (size_t i=0; i<ntheta_tmp; ++i)
        theta_tmp(i) = i*pi/(ntheta_tmp-1);
      auto leg_tmp = (ntheta_tmp<=nrings) ?
        subarray<3>(leg,{{},{0,ntheta_tmp},{}}) :
        allocate_leg<T>(leg.shape(0), ntheta_tmp, leg.shape(2), true, true, nthreads);
      alm2leg(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode);
      resample_theta(leg_tmp, true, true, leg, npi, spi, spin, nthreads, false);
      return;
      }

    if (theta_interpol && (nrings>500) && (nrings>1.5*lmax)) // irregular and worth resampling
      {
      auto ntheta_tmp = good_size_complex(lmax+1)+1;
      vmav<double,1> theta_tmp({ntheta_tmp}, UNINITIALIZED);
      for (size_t i=0; i<ntheta_tmp; ++i)
        theta_tmp(i) = i*pi/(ntheta_tmp-1);
      auto leg_tmp = (ntheta_tmp<=leg.shape(1)) ?
        subarray<3>(leg,{{},{0,ntheta_tmp},{}}) :
        allocate_leg<T>(leg.shape(0), ntheta_tmp, leg.shape(2), true, true, nthreads);
      alm2leg(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode);
      resample_leg_CC_to_irregular(leg_tmp, leg, theta, spin, mval, nthreads);
      return;
      }
    }

  vector<ringdata> rdata_normal, rdata_fast;
  if ((lmax>=500)&&(mode==STANDARD)&&((spin==1)||(spin==2)))
    {
    auto rdata = make_ringdata(theta, lmax, spin);
    double limit = (spin==1) ? 0.0001 : 0.01;
    for (const auto &rd: rdata)
      (abs(rd.sth)>=limit) ? rdata_fast.push_back(rd) : rdata_normal.push_back(rd);
    }
  else
    rdata_normal = make_ringdata(theta, lmax, spin);

  if (!rdata_fast.empty())
    {
    auto norm_l = Ylmgen::get_norm (lmax+spin, 0);
    auto &rdata(rdata_fast);
    // adjust ring weights
    for (size_t ith=0; ith<rdata.size(); ++ith)
      rdata[ith].wgt = 1./ ((spin==1) ? rdata[ith].sth : (rdata[ith].sth*rdata[ith].sth));
    YlmBase base(lmax+spin, mmax, 0);
    vector<double> ringfct(theta.size());
    for (size_t i=0; i<theta.size(); ++i)
      {
      ringfct[i] = 1./(sin(theta(i)));
      if (spin==2) ringfct[i] *= ringfct[i];
      }
    size_t isspin2 = (spin==2) ? 1 : 0;
    vmav<double,1> f1({isspin2*(lmax+1)}, UNINITIALIZED), f2({isspin2*(lmax+1)}, UNINITIALIZED);
    if (spin==2)
      {
      f1(0) = f2(0) = f2(1) = 0;
      for (size_t l=1; l<=lmax; ++l)
        f1(l) = (2.*l+3.)/(2.*l+1.) / (l*(l+1.)*(l+2.)*(l+3.));
      for (size_t l=2; l<=lmax; ++l)
        f2(l) = sqrt(1./((l-1.)*l*(l+1.)*(l+2.)));
      }

    ducc0::execDynamic(nm, nthreads, 1, [&](ducc0::Scheduler &sched)
      {
      Ylmgen gen(base);
      vmav<complex<double>,2> almtmp({nalm,lmax+2+spin}, UNINITIALIZED);
      vmav<complex<double>,2> glm({nalm,isspin2*(lmax+3)}, UNINITIALIZED);
      vmav<complex<double>,2> almtmp0(&almtmp(0,0), {almtmp.shape(1),1}, {1,1});
      vmav<complex<double>,2> almtmp1(&almtmp(1,0), {almtmp.shape(1),1}, {1,1});
      auto leg0 = subarray<3>(leg,{{0,1},{},{}});
      auto leg1 = subarray<3>(leg,{{1,2},{},{}});

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
        auto m=mval(mi);
        auto lmin=max(size_t(0),m);
        for (size_t ialm=0; ialm<nalm; ++ialm)
          {
          for (size_t l=m; l<lmin; ++l)
            almtmp(ialm,l) = 0;
          for (size_t l=lmin; l<=lmax; ++l)
            almtmp(ialm,l) = alm(ialm,mstart(mi)+l*lstride);
          }
        (spin==1) ? spin1to0(almtmp, lmax, m)
                  : spin2to0(almtmp, f1, f2, glm, lmax, m);
// zero alm beyond lmax+spin
        for (size_t ialm=0; ialm<nalm; ++ialm)
          almtmp(ialm,lmax+spin+1) = 0;
        for (size_t ialm=0; ialm<nalm; ++ialm)
          {
          for (size_t l=lmin; l<=lmax+spin; ++l)
            almtmp(ialm,l) *= norm_l[l];
          }
        gen.prepare(m);
        inner_loop_a2m (mode, almtmp0, leg0, rdata, gen, mi);
        inner_loop_a2m (mode, almtmp1, leg1, rdata, gen, mi);
        }
      }); /* end of parallel region */
    }

  if (!rdata_normal.empty())
    {
    auto norm_l = (mode==DERIV1) ? Ylmgen::get_d1norm (lmax) :
                                   Ylmgen::get_norm (lmax, spin);
    auto &rdata(rdata_normal);
    YlmBase base(lmax, mmax, spin);

    ducc0::execDynamic(nm, nthreads, 1, [&](ducc0::Scheduler &sched)
      {
      Ylmgen gen(base);
      vmav<complex<double>,2> almtmp({lmax+2,nalm}, UNINITIALIZED);

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
        auto m=mval(mi);
        auto lmin=max(spin,m);
        for (size_t ialm=0; ialm<nalm; ++ialm)
          {
          for (size_t l=m; l<lmin; ++l)
            almtmp(l,ialm) = 0;
          for (size_t l=lmin; l<=lmax; ++l)
            almtmp(l,ialm) = alm(ialm,mstart(mi)+l*lstride)*T(norm_l[l]);
          almtmp(lmax+1,ialm) = 0;
          }
        gen.prepare(m);
        inner_loop_a2m (mode, almtmp, leg, rdata, gen, mi);
        }
      }); /* end of parallel region */
    }
  }
template void alm2leg(  // associated Legendre transform
  const cmav<complex<float>,2> &alm, // (ncomp, lmidx)
  const vmav<complex<float>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol);
template void alm2leg(  // associated Legendre transform
  const cmav<complex<double>,2> &alm, // (ncomp, lmidx)
  const vmav<complex<double>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol);

template<typename T> void leg2alm_internal(  // associated Legendre transform
  const vmav<complex<T>,2> &alm, // (ncomp, lmidx)
  const vmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol,
  bool leg_can_be_overwritten)
  {
  // sanity checks
  auto nrings=theta.shape(0);
  MR_assert(nrings==leg.shape(1), "nrings mismatch");
  auto nm=mval.shape(0);
  MR_assert(nm==mstart.shape(0), "nm mismatch");
  MR_assert(nm==leg.shape(2), "nm mismatch");
  auto mmax = get_mmax(mval, lmax);
  auto nalm = alm.shape(0);
  if (mode==DERIV1)
    {
    spin=1;
    MR_assert(nalm==1, "need one a_lm component");
    MR_assert(leg.shape(0)==2, "need two Legendre components");
    }
  else if (mode==GRAD_ONLY)
    {
    MR_assert(spin>0, "spin must be positive for grad-only SHTs");
    MR_assert(nalm==1, "need one a_lm component");
    MR_assert(leg.shape(0)==2, "need two Legendre components");
    }
  else
    {
    size_t ncomp = (spin==0) ? 1 : 2;
    MR_assert(nalm==ncomp, "incorrect number of a_lm components");
    MR_assert(leg.shape(0)==ncomp, "incorrect number of Legendre components");
    }

  if (even_odd_m(mval))
    {
    bool npi, spi;
    size_t ntheta_tmp;
    if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
      {
      vmav<double,1> theta_tmp({ntheta_tmp}, UNINITIALIZED);
      for (size_t i=0; i<ntheta_tmp; ++i)
        theta_tmp(i) = i*pi/(ntheta_tmp-1);
      auto leg_tmp = (leg_can_be_overwritten && ntheta_tmp<=leg.shape(1)) ?
        subarray<3>(leg, {{},{0,ntheta_tmp},{}}) :
        allocate_leg<T>(leg.shape(0), ntheta_tmp, leg.shape(2), false, true, nthreads);
      resample_theta(leg, npi, spi, leg_tmp, true, true, spin, nthreads, true);
      leg2alm_internal(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode, false, true);
      return;
      }

    if (theta_interpol && (nrings>500) && (nrings>1.5*lmax)) // irregular and worth resampling
      {
      auto ntheta_tmp = good_size_complex(lmax+1)+1;
      vmav<double,1> theta_tmp({ntheta_tmp}, UNINITIALIZED);
      for (size_t i=0; i<ntheta_tmp; ++i)
        theta_tmp(i) = i*pi/(ntheta_tmp-1);
      auto leg_tmp = (leg_can_be_overwritten && ntheta_tmp<=leg.shape(1)) ?
        subarray<3>(leg, {{},{0,ntheta_tmp},{}}) :
        allocate_leg<T>(leg.shape(0), ntheta_tmp, leg.shape(2), false, true, nthreads);
      resample_leg_irregular_to_CC(leg, leg_tmp, theta, spin, mval, nthreads);
      leg2alm_internal(alm, leg_tmp, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, mode, false, true);
      return;
      }
    }

  vector<ringdata> rdata_normal, rdata_fast;
  if ((lmax>=500)&&(mode==STANDARD)&&((spin==1)||(spin==2)))
    {
    double limit = (spin==1) ? 0.0001 : 0.01;
    auto rdata = make_ringdata(theta, lmax, spin);
    for (const auto &rd: rdata)
      (abs(rd.sth)>=limit) ? rdata_fast.push_back(rd) : rdata_normal.push_back(rd);
    }
  else
    rdata_normal = make_ringdata(theta, lmax, spin);

  if (!rdata_fast.empty())
    {
    auto norm_l = Ylmgen::get_norm (lmax+spin, 0);
    auto &rdata(rdata_fast);
    // adjust ring weights
    for (size_t ith=0; ith<rdata.size(); ++ith)
      rdata[ith].wgt = 1./ ((spin==1) ? rdata[ith].sth : (rdata[ith].sth*rdata[ith].sth));
    YlmBase base(lmax+spin, mmax, 0);
    size_t isspin2 = (spin==2) ? 1 : 0;
    vmav<double,1> f2({isspin2*(lmax+1)}, UNINITIALIZED);
    if (spin==2)
      {
      f2(0) = f2(1) = 0;
      for (size_t l=2; l<=lmax; ++l)
        f2(l) = sqrt(1./((l-1.)*l*(l+1.)*(l+2.)));
      }

    ducc0::execDynamic(nm, nthreads, 1, [&](ducc0::Scheduler &sched)
      {
      Ylmgen gen(base);
      vmav<complex<double>,2> almtmp({2, lmax+2+spin}, UNINITIALIZED);
      vmav<complex<double>,2> glm({nalm,isspin2*(lmax+5)}, UNINITIALIZED);
      vmav<complex<double>,2> almtmp0(&almtmp(0,0), {almtmp.shape(1),1}, {1,1});
      vmav<complex<double>,2> almtmp1(&almtmp(1,0), {almtmp.shape(1),1}, {1,1});
      auto leg0 = subarray<3>(leg,{{0,1},{},{}});
      auto leg1 = subarray<3>(leg,{{1,2},{},{}});

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
        auto m=mval(mi);
        gen.prepare(m);
        for (size_t ialm=0; ialm<nalm; ++ialm)
          for (size_t l=m; l<almtmp.shape(1); ++l)
            almtmp(ialm,l) = 0.;
        inner_loop_m2a (mode, almtmp0, leg0, rdata, gen, mi);
        inner_loop_m2a (mode, almtmp1, leg1, rdata, gen, mi);
        for (size_t ialm=0; ialm<nalm; ++ialm)
          for (size_t l=m; l<=lmax+spin; ++l)
            almtmp(ialm,l) *= norm_l[l];
        for (size_t ialm=0; ialm<nalm; ++ialm)
          almtmp(ialm,lmax+spin+1) = 0.;
        (spin==1) ? spin0to1(almtmp, lmax, m)
                  : spin0to2(almtmp, f2, glm, lmax, m);
        auto lmin=max(spin,m);
        for (size_t l=m; l<lmin; ++l)
          for (size_t ialm=0; ialm<nalm; ++ialm)
            alm(ialm,mstart(mi)+l*lstride) = 0;
        for (size_t l=lmin; l<=lmax; ++l)
          for (size_t ialm=0; ialm<nalm; ++ialm)
            alm(ialm,mstart(mi)+l*lstride) = complex<T>(almtmp(ialm,l));
        }
      }); /* end of parallel region */
    }

  if (!rdata_normal.empty())
    {
    auto norm_l = (mode==DERIV1) ? Ylmgen::get_d1norm (lmax) :
                                   Ylmgen::get_norm (lmax, spin);
    auto &rdata(rdata_normal);
    YlmBase base(lmax, mmax, spin);

    ducc0::execDynamic(nm, nthreads, 1, [&](ducc0::Scheduler &sched)
      {
      Ylmgen gen(base);
      vmav<complex<double>,2> almtmp({lmax+2,nalm}, UNINITIALIZED);

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
        auto m=mval(mi);
        gen.prepare(m);
        for (size_t l=m; l<almtmp.shape(0); ++l)
          for (size_t ialm=0; ialm<nalm; ++ialm)
            almtmp(l,ialm) = 0.;
        inner_loop_m2a (mode, almtmp, leg, rdata, gen, mi);
        auto lmin=max(spin,m);
        if (rdata_fast.empty())
          {
          for (size_t l=m; l<lmin; ++l)
            for (size_t ialm=0; ialm<nalm; ++ialm)
              alm(ialm,mstart(mi)+l*lstride) = 0;
          for (size_t l=lmin; l<=lmax; ++l)
            for (size_t ialm=0; ialm<nalm; ++ialm)
              alm(ialm,mstart(mi)+l*lstride) = complex<T>(almtmp(l,ialm)*norm_l[l]);
          }
        else
          for (size_t l=lmin; l<=lmax; ++l)
            for (size_t ialm=0; ialm<nalm; ++ialm)
              alm(ialm,mstart(mi)+l*lstride) += complex<T>(almtmp(l,ialm)*norm_l[l]);
        }
      }); /* end of parallel region */
    }
  }
template<typename T> void leg2alm(  // associated Legendre transform
  const vmav<complex<T>,2> &alm, // (ncomp, lmidx)
  const cmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol)
  {
  vmav<complex<T>,3> leg2(const_cast<complex<T> *>(leg.data()), leg.shape(), leg.stride());
  leg2alm_internal(alm, leg2, spin, lmax, mval, mstart, lstride, theta, nthreads,
    mode, theta_interpol, false);
  }
template void leg2alm(  // associated Legendre transform
  const vmav<complex<float>,2> &alm, // (ncomp, lmidx)
  const cmav<complex<float>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol);
template void leg2alm(  // associated Legendre transform
  const vmav<complex<double>,2> &alm, // (ncomp, lmidx)
  const cmav<complex<double>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol);

#if 0
cmav<size_t,1> get_ringidx(const cmav<size_t,1> &nphi, const cmav<double,1> &/*phi0*/)
  {
  vmav<size_t,1> res({nphi.shape(0)});
  for (size_t i=0; i<res.shape(0); ++i) res(i)=i;
  stable_sort(res.data(), res.data()+res.shape(0), [&nphi /*, &phi0*/](size_t a, size_t b){
//    if (nphi(a)==nphi(b)) return phi0(a)<phi0(b);
    return nphi(a)>nphi(b);
    });
  return res;
  }
#endif

template<typename T> void leg2map(  // FFT
  const vmav<T,2> &map, // (ncomp, pix)
  const cmav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t ncomp=map.shape(0);
  MR_assert(ncomp==leg.shape(0), "number of components mismatch");
  size_t nrings=leg.shape(1);
  MR_assert(nrings>=1, "need at least one ring");
  MR_assert((nrings==nphi.shape(0)) && (nrings==ringstart.shape(0))
         && (nrings==phi0.shape(0)), "inconsistent number of rings");
  MR_assert(leg.shape(2)>=1, "bad mmax");
  size_t mmax=leg.shape(2)-1;

//  bool well_behaved=true;
//  if (nrings==1) well_behaved=false;
//  size_t dring = (nrings>1) ? ringstart(1)-ringstart(0) : ~size_t(0);
  size_t nphmax=0;
  for (size_t i=0; i<nrings; ++i)
    {
    nphmax=max(nphi(i),nphmax);
//    if (nphi(i) != nphmax) well_behaved=false;
//    if (phi0(i) != 0.) well_behaved=false;
//    if ((i>0) && (ringstart(i)-ringstart(i-1) != dring)) well_behaved=false;
    }
//  if (nphmax<2*mmax+1) well_behaved=false;

#if 0
  if (well_behaved)
    {
    auto xmap(map.template reinterpret<3>({ncomp, nrings, nphmax},
      {map.stride(0), ptrdiff_t(dring*map.stride(1)), pixstride*map.stride(1)}));
    execParallel(nrings, nthreads, [&](size_t lo, size_t hi)
      {
      if (lo==hi) return;
      for (size_t icomp=0; icomp<ncomp; ++icomp)
        {
        for (size_t iring=lo; iring<hi; ++iring)
          {
          double rf = ringfactor(iring);
          xmap(icomp, iring, 0) = leg(icomp, iring, 0).real()*rf;
          for (size_t m=1; m<=mmax; ++m)
            {
            xmap(icomp, iring, 2*m-1) = leg(icomp, iring, m).real()*rf;
            xmap(icomp, iring, 2*m  ) = leg(icomp, iring, m).imag()*rf;
            }
          for (size_t ix=2*mmax+1; ix<nphmax; ++ix) xmap(icomp, iring, ix) = T(0);
          }
        vfmav<T> xmapf=subarray<2>(xmap,{{icomp},{lo,hi},{}});
        r2r_fftpack(xmapf,xmapf,{1},false,false,T(1),1);
        }
      });
    }
  else
#endif
    execDynamic(nrings, nthreads, 8, [&](Scheduler &sched)
      {
      ringhelper helper;
      vmav<double,1> ringtmp({nphmax+2}, UNINITIALIZED);
      while (auto rng=sched.getNext()) for(auto ith=rng.lo; ith<rng.hi; ++ith)
        {
        double rf = ringfactor(ith);
        for (size_t icomp=0; icomp<ncomp; ++icomp)
          {
          auto ltmp = subarray<1>(leg, {{icomp}, {ith}, {}});
          helper.phase2ring (nphi(ith),phi0(ith),ringtmp,mmax,ltmp);
          for (size_t i=0; i<nphi(ith); ++i)
            map(icomp,ringstart(ith)+i*pixstride) = T(ringtmp(i+1)*rf);
          }
        }
      }); /* end of parallel region */
  }

template<typename T> void map2leg(  // FFT
  const cmav<T,2> &map, // (ncomp, pix)
  const vmav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t ncomp=map.shape(0);
  MR_assert(ncomp==leg.shape(0), "number of components mismatch");
  size_t nrings=leg.shape(1);
  MR_assert(nrings>=1, "need at least one ring");
  MR_assert((nrings==nphi.shape(0)) && (nrings==ringstart.shape(0))
         && (nrings==phi0.shape(0)), "inconsistent number of rings");
  MR_assert(leg.shape(2)>=1, "bad mmax");
  size_t mmax=leg.shape(2)-1;

//  bool well_behaved=true;
//  if (nrings==1) well_behaved=false;
//  size_t dring = (nrings>1) ? ringstart(1)-ringstart(0) : ~size_t(0);
  size_t nphmax=0;
  for (size_t i=0; i<nrings; ++i)
    {
    nphmax=max(nphi(i),nphmax);
//    if (nphi(i) != nphmax) well_behaved=false;
//    if (phi0(i) != 0.) well_behaved=false;
//    if ((i>0) && (ringstart(i)-ringstart(i-1) != dring)) well_behaved=false;
    }
//  if (nphmax<2*mmax+1) well_behaved=false;

#if 0
  if (well_behaved)
    {
    auto xmap(map.template reinterpret<3>({ncomp, nrings, nphmax},
      {map.stride(0), ptrdiff_t(dring*map.stride(1)), pixstride*map.stride(1)}));
    size_t blksz=min<size_t>(nrings,64);
    size_t nblocks = (nrings+blksz-1)/blksz;
    execParallel(nblocks, nthreads, [&](size_t lo, size_t hi)
      {
      if (lo==hi) return;
      vmav<T,2> buf({blksz, nphmax}, UNINITIALIZED);
      for (size_t icomp=0; icomp<ncomp; ++icomp)
        for (size_t iblock=lo; iblock<hi; ++iblock)
          {
          size_t r0=iblock*blksz, r1=min(nrings,(iblock+1)*blksz);
          cfmav<T> xmapf=subarray<2>(xmap,{{icomp},{r0,r1},{}});
          vfmav<T> buff=subarray<2>(buf,{{0,r1-r0},{}});
          r2r_fftpack(xmapf,buff,{1},true,true,T(1),1);
          for (size_t iring=r0; iring<r1; ++iring)
            {
            double rf = ringfactor(iring);
            leg(icomp, iring, 0) = buf(iring-r0, 0)*rf;
            for (size_t m=1; m<=mmax; ++m)
              leg(icomp, iring, m) = complex<T>(buf(iring-r0, 2*m-1)*rf,
                                                buf(iring-r0, 2*m)*rf);
            }
          }
      });
    }
  else
#endif
    execDynamic(nrings, nthreads, 4, [&](Scheduler &sched)
      {
      ringhelper helper;
      vmav<double,1> ringtmp({nphmax+2}, UNINITIALIZED);
      while (auto rng=sched.getNext()) for(auto ith=rng.lo; ith<rng.hi; ++ith)
        {
        double rf = ringfactor(ith);
        for (size_t icomp=0; icomp<ncomp; ++icomp)
          {
          for (size_t i=0; i<nphi(ith); ++i)
            ringtmp(i+1) = map(icomp,ringstart(ith)+i*pixstride)*rf;
          auto ltmp = subarray<1>(leg, {{icomp}, {ith}, {}});
          helper.ring2phase (nphi(ith),phi0(ith),ringtmp,mmax,ltmp);
          }
        }
      }); /* end of parallel region */
  }

// NOTE: legi and lego may overlap, with identical start address and strides
template<typename T> void resample_to_prepared_CC(const cmav<complex<T>,3> &legi,
  bool npi, bool spi, const vmav<complex<T>,3> &lego, size_t spin, size_t lmax,
  size_t nthreads)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-npi-spi;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-2;
  bool need_first_resample = !(npi&&spi&&(nrings_in>=2*lmax+2));
  size_t nfull = need_first_resample ? 2*nfull_out : nfull_in;

  vector<complex<T>> shift(npi ? 0 : nrings_in+1);
  if (!npi)
    {
    UnityRoots<T,complex<T>> roots(2*nfull_in);
    for (size_t i=0; i<shift.size(); ++i)
      shift[i] = roots[i];
    }
  auto wgt = get_gridweights("CC", nfull/2+1);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(need_first_resample ? nfull_in : 1),
                 plan_out(nfull_out), plan_full(nfull);

  execDynamic(nm, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({max(nfull,nfull_in)}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), max(plan_out.bufsize(),
      plan_full.bufsize()))}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{rng.lo,MAXIDX}}));
// FIXME: this may benefit from blocking
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          T fct2 = fct * (((j+rng.lo)&1)? T(-1) : T(1));
          for (size_t i=0, im=nfull_in-1+npi; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,j);
            tmp(i) = v1;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct2 * v1;
            else
              tmp(i) = T(0.5)*(tmp(i)+fct2*v1);
            }
          if (need_first_resample)
            {
            plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);

            // shift
            if (!npi)
              for (size_t i=1, im=nfull_in-1; (i<nrings_in+1)&&(i<=im); ++i,--im)
                {
                if (i!=im)
                  tmp(i) *= conj(shift[i]);
                tmp(im) *= shift[i];
                }

            // zero padding to full-resolution CC grid
            if (nfull>nfull_in) // pad
              {
              size_t dist = nfull-nfull_in;
              size_t nmove = nfull_in/2;
              for (size_t i=nfull-1; i+1+nmove>nfull; --i)
                tmp(i) = tmp(i-dist);
              for (size_t i=nfull-nmove-dist; i+nmove<nfull; ++i)
                tmp(i) = 0;
              }
            if (nfull<nfull_in) // truncate
              {
              size_t dist = nfull_in-nfull;
              size_t nmove = nfull/2;
              for (size_t i=nfull_in-nmove; i<nfull_in; ++i)
                tmp(i-dist) = tmp(i);
              }
            plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
            }
          for (size_t i=0, im=nfull; i<=im; ++i, --im)
            {
            tmp(i) *= T(wgt(i));
            if ((i==0) || (i==im)) tmp(i)*=2;
            if ((im<nfull) && (im!=i))
              tmp(im) *= T(wgt(i));
            }
          plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);
          if (nfull_out<nfull) // truncate
            {
            size_t dist = nfull-nfull_out;
            size_t nmove = nfull_out/2;
            for (size_t i=nfull-nmove; i<nfull; ++i)
            tmp(i-dist) = tmp(i);
            }
          plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
          auto norm = T(.5/(nfull_out*((need_first_resample ? nfull_in : 1))));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-i;
            if (im==nfull_out) im=0;
            auto norm2 = norm * (T(1)-T(0.5)*(i==im));
            llego(i,j  ) = norm2 * (tmp(i) + fct2*tmp(im));
            }
          }
        }
      }
    });
  }

// NOTE: legi and lego may overlap, with identical start address and strides
template<typename T> void resample_from_prepared_CC(const cmav<complex<T>,3> &legi,
  const vmav<complex<T>,3> &lego, bool npo, bool spo, size_t spin, size_t lmax,
  size_t nthreads)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-2;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-npo-spo;
  bool need_second_resample = !(npo&&spo&&(nrings_out>=2*lmax+2));
  size_t nfull = need_second_resample ? 2*nfull_in : nfull_out;

  vector<complex<T>> shift(npo ? 0 : nrings_out+1);
  if (!npo)
    {
    UnityRoots<T,complex<T>> roots(2*nfull_out);
    for (size_t i=0; i<shift.size(); ++i)
      shift[i] = roots[i];
    }
  auto wgt = get_gridweights("CC", nfull/2+1);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(nfull_in),
                 plan_out(need_second_resample ? nfull_out : 1), plan_full(nfull);

  execDynamic(nm, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({max(nfull,nfull_out)}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), max(plan_out.bufsize(),
      plan_full.bufsize()))}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{rng.lo,MAXIDX}}));
// FIXME: this may benefit from blocking
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          T fct2 = fct * (((j+rng.lo)&1)? T(-1) : T(1));
          for (size_t i=0, im=nfull_in; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,j);
            tmp(i) = v1;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct2 * v1;
            else
              tmp(i) = T(0.5)*(tmp(i)+fct2*v1);
            }
          plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
          // zero padding to full-resolution CC grid
          if (nfull>nfull_in) // pad
            {
            size_t dist = nfull-nfull_in;
            size_t nmove = nfull_in/2;
            for (size_t i=nfull-1; i+1+nmove>nfull; --i)
              tmp(i) = tmp(i-dist);
            for (size_t i=nfull-nmove-dist; i+nmove<nfull; ++i)
              tmp(i) = 0;
            }
          MR_assert(nfull>=nfull_in, "must not happen");
          plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);
          for (size_t i=0, im=nfull; i<=im; ++i, --im)
            {
            tmp(i) *= T(wgt(i));
            if ((i==0) || (i==im)) tmp(i)*=2;
            if ((im<nfull) && (im!=i))
              tmp(im) *= T(wgt(i));
            }

          if (need_second_resample)
            {
            plan_full.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), false);
            if (nfull_out>nfull) // pad
              {
              size_t dist = nfull_out-nfull;
              size_t nmove = nfull/2;
              for (size_t i=nfull_out-1; i+1+nmove>nfull_out; --i)
                tmp(i) = tmp(i-dist);
              for (size_t i=nfull_out-nmove-dist; i+nmove<nfull_out; ++i)
                tmp(i) = 0;
              }
            if (nfull_out<nfull) // truncate
              {
              size_t dist = nfull-nfull_out;
              size_t nmove = nfull_out/2;
              for (size_t i=nfull-nmove; i<nfull; ++i)
                tmp(i-dist) = tmp(i);
              }
            // shift
            if (!npo)
              for (size_t i=1, im=nfull_out-1; (i<nrings_out+1)&&(i<=im); ++i,--im)
                {
                if (i!=im)
                  tmp(i) *= conj(shift[i]);
                tmp(im) *= shift[i];
                }
            plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), true);
            }
          auto norm = T(.5/(nfull_in*((need_second_resample ? nfull_out : 1))));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-1+npo-i;
            if (im==nfull_out) im=0;
            auto norm2 = norm * (T(1)-T(0.5)*(i==im));
            llego(i,j) = norm2 * (tmp(i) + fct2*tmp(im));
            }
          }
        }
      }
    });
  }

void sanity_checks(
  const mav_info<2> &alm, // (ncomp, *)
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  const mav_info<2> &map, // (ncomp, *)
  const cmav<double,1> &theta, // (nrings)
  const mav_info<1> &phi0, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  size_t spin,
  SHT_mode mode)
  {
  size_t nm = mstart.shape(0);
  MR_assert(nm>0, "mstart too small");
  size_t mmax = nm-1;
  MR_assert(lmax>=mmax, "lmax must be >= mmax");
  size_t nrings = theta.shape(0);
  MR_assert(nrings>0, "need at least one ring");
  MR_assert((phi0.shape(0)==nrings) &&
            (nphi.shape(0)==nrings) &&
            (ringstart.shape(0)==nrings) &&
            (ringfactor.shape(0)==nrings),
    "inconsistency in the number of rings");
  if ((mode==DERIV1) || (mode==GRAD_ONLY))
    {
    MR_assert(spin>0, "DERIV and GRAD_ONLY modes require spin>0");
    MR_assert((alm.shape(0)==1) && (map.shape(0)==2),
      "inconsistent number of components");
    }
  else
    {
    size_t ncomp = 1+(spin>0);
    MR_assert((alm.shape(0)==ncomp) && (map.shape(0)==ncomp),
      "inconsistent number of components");
    }
  }

template<typename T> void synthesis(
  const cmav<complex<T>,2> &alm, // (ncomp, *)
  const vmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol)
  {
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, ringfactor, spin, mode);
  vmav<size_t,1> mval({mstart.shape(0)}, UNINITIALIZED);
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;

  bool npi, spi;
  size_t ntheta_tmp;
  if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
    {
    vmav<double,1> theta_tmp({ntheta_tmp}, UNINITIALIZED);
    for (size_t i=0; i<ntheta_tmp; ++i)
      theta_tmp(i) = i*pi/(ntheta_tmp-1);
    auto leg(allocate_leg<T>(map.shape(0), max(theta.shape(0),ntheta_tmp),
                          mstart.shape(0), true, true, nthreads));
    auto legi(subarray<3>(leg, {{},{0,ntheta_tmp},{}}));
    auto lego(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));
    alm2leg(alm, legi, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads,
      mode, theta_interpol);
    resample_theta(legi, true, true, lego, npi, spi, spin, nthreads, false);
    leg2map(map, lego, nphi, phi0, ringstart, ringfactor, pixstride, nthreads);
    }
  else
    {
    auto leg(allocate_leg<T>(map.shape(0), theta.shape(0), mstart.shape(0), true,
                          false, nthreads));
    alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, mode,
      theta_interpol);
    leg2map(map, leg, nphi, phi0, ringstart, ringfactor, pixstride, nthreads);
    }
  }

void get_ringtheta_2d(const string &type, const vmav<double, 1> &theta)
  {
  auto nrings = theta.shape(0);

  if (type=="GL") // Gauss-Legendre
    {
    ducc0::GL_Integrator integ(nrings);
    auto th = integ.thetas();
    for (size_t m=0; m<nrings; ++m)
      theta(m) = th[nrings-1-m];
    }
  else if (type=="F1") // Fejer 1
    for (size_t m=0; m<(nrings+1)/2; ++m)
      {
      theta(m)=pi*(m+0.5)/nrings;
      theta(nrings-1-m)=pi-theta(m);
      }
  else if (type=="CC") // Clenshaw-Curtis
    for (size_t m=0; m<(nrings+1)/2; ++m)
      {
      theta(m)=pi*m/(nrings-1.);
      theta(nrings-1-m)=pi-theta(m);
      }
  else if (type=="F2") // Fejer 2
    for (size_t m=0; m<nrings; ++m)
      theta(m)=pi*(m+1)/(nrings+1.);
  else if (type=="DH") // Driscoll-Healy
    for (size_t m=0; m<nrings; ++m)
      theta(m) = m*pi/nrings;
  else if (type=="MW") // McEwen-Wiaux
    for (size_t m=0; m<nrings; ++m)
      theta(m)=pi*(2.*m+1.)/(2.*nrings-1.);
  else if (type=="MWflip") // McEwen-Wiaux mirrored
    for (size_t m=0; m<nrings; ++m)
      theta(m)=pi*(2.*m)/(2.*nrings-1.);
  else
    MR_fail("unsupported grid type");
  }

template<typename T> void synthesis_2d(const cmav<complex<T>,2> &alm, const vmav<T,3> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor,
  size_t nthreads, SHT_mode mode)
  {
  auto nphi = cmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0_ = cmav<double,1>::build_uniform({map.shape(1)}, phi0);
  vmav<size_t,1> ringstart({map.shape(1)}, UNINITIALIZED);
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  auto map2(map.template reinterpret<2>({map.shape(0), 1/*placeholder*/},
                                        {map.stride(0), 1}));
  vmav<double,1> theta({map.shape(1)}, UNINITIALIZED);
  get_ringtheta_2d(geometry, theta);
  synthesis(alm, map2, spin, lmax, mstart, lstride, theta, nphi, phi0_,
    ringstart, ringfactor, pixstride, nthreads, mode);
  }
template void synthesis_2d(const cmav<complex<double>,2> &alm,
  const vmav<double,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride, const string &geometry,
  double phi0, const cmav<double,1> &ringfactor, size_t nthreads, SHT_mode mode);
template void synthesis_2d(const cmav<complex<float>,2> &alm,
  const vmav<float,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride, const string &geometry,
  double phi0, const cmav<double,1> &ringfactor, size_t nthreads, SHT_mode mode);

template<typename T> void adjoint_synthesis(
  const vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  SHT_mode mode,
  bool theta_interpol)
  {
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, ringfactor, spin, mode);
  vmav<size_t,1> mval({mstart.shape(0)}, UNINITIALIZED);
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;

  bool npi, spi;
  size_t ntheta_tmp;
  if (downsampling_ok(theta, lmax, npi, spi, ntheta_tmp))
    {
    vmav<double,1> theta_tmp({ntheta_tmp}, UNINITIALIZED);
    for (size_t i=0; i<ntheta_tmp; ++i)
      theta_tmp(i) = i*pi/(ntheta_tmp-1);
    auto leg(allocate_leg<T>(map.shape(0), max(theta.shape(0),ntheta_tmp),
                          mstart.shape(0), false, true, nthreads));
    auto legi(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));
    auto lego(subarray<3>(leg, {{},{0,ntheta_tmp},{}}));
    map2leg(map, legi, nphi, phi0, ringstart, ringfactor, pixstride, nthreads);
    resample_theta(legi, npi, spi, lego, true, true, spin, nthreads, true);
    leg2alm_internal(alm, lego, spin, lmax, mval, mstart, lstride, theta_tmp,
      nthreads, mode, theta_interpol, true);
    }
  else
    {
    auto leg(allocate_leg<T>(map.shape(0), theta.shape(0), mstart.shape(0), false,
                          false, nthreads));
    map2leg(map, leg, nphi, phi0, ringstart, ringfactor, pixstride, nthreads);
    leg2alm_internal(alm, leg, spin, lmax, mval, mstart, lstride, theta,
      nthreads, mode, theta_interpol, true);
    }
  }
template<typename T> tuple<size_t, size_t, double, double> pseudo_analysis(
  const vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  size_t maxiter,
  double epsilon,
  bool theta_interpol,
  bool alm_contains_initial_guess)
  {
  auto op = [&](const cmav<complex<T>,2> &xalm, const vmav<T,2> &xmap)
    {
    synthesis(xalm, xmap, spin, lmax, mstart, lstride, theta, nphi, phi0,
              ringstart, ringfactor, pixstride, nthreads, STANDARD, theta_interpol);
    };
  auto op_adj = [&](const cmav<T,2> &xmap, const vmav<complex<T>,2> &xalm)
    {
    adjoint_synthesis(xalm, xmap, spin, lmax, mstart, lstride, theta, nphi,
                      phi0, ringstart, ringfactor, pixstride, nthreads, STANDARD, theta_interpol);
    };
  auto mapnorm = [&](const cmav<T,2> &xmap)
    {
    double res=0;
    for (size_t icomp=0; icomp<xmap.shape(0); ++icomp)
      for (size_t iring=0; iring<ringstart.shape(0); ++iring)
        for (size_t ipix=0; ipix<nphi(iring); ++ipix)
          {
          auto tmp = xmap(icomp,ringstart(iring)+ipix*pixstride)*ringfactor(iring);
          res += tmp*tmp;
          }
    return sqrt(res);
    };
  auto almnorm = [&](const cmav<complex<T>,2> &xalm)
    {
    double res=0;
    for (size_t icomp=0; icomp<xalm.shape(0); ++icomp)
      for (size_t m=0; m<mstart.shape(0); ++m)
        for (size_t l=m; l<=lmax; ++l)
          {
          auto tmp = xalm(icomp,mstart(m)+l*lstride);
          res += norm(tmp) * ((m==0) ? 1 : 2);
          }
    return sqrt(res);
    };
  // try to estimate ATOL according to Paige & Saunders
  // assuming an absolute error of machine epsilon in every matrix element
  // and a sum of squares of 1 along every row/column
  size_t npix=0;
  mav_apply([&npix](size_t v){npix+=v;}, 1, nphi);
  double atol = 1e-14*sqrt(npix);
  if (!alm_contains_initial_guess)  // start with a zero vector guess
    mav_apply([](auto &v){v=0;}, nthreads, alm);
  auto [dum, istop, itn, normr, normar, normA, condA, normx, normb]
    = lsmr(op, op_adj, almnorm, mapnorm, map, alm,
           0., atol, epsilon, 1e8, maxiter, false, nthreads);
  return make_tuple(istop, itn, normr/normb, normar/(normA*normr));
  }
template tuple<size_t, size_t, double, double> pseudo_analysis(
  const vmav<complex<double>,2> &alm, // (ncomp, *)
  const cmav<double,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  size_t maxiter,
  double epsilon,
  bool theta_interpol,
  bool alm_contains_initial_guess);
template tuple<size_t, size_t, double, double> pseudo_analysis(
  const vmav<complex<float>,2> &alm, // (ncomp, *)
  const cmav<float,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  size_t maxiter,
  double epsilon,
  bool theta_interpol,
  bool alm_contains_initial_guess);

template<typename T> void adjoint_synthesis_2d(const vmav<complex<T>,2> &alm,
  const cmav<T,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads, SHT_mode mode)
  {
  auto nphi = cmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0_ = cmav<double,1>::build_uniform({map.shape(1)}, phi0);
  vmav<size_t,1> ringstart({map.shape(1)}, UNINITIALIZED);
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  auto map2(map.template reinterpret<2>({map.shape(0), 1/*placeholder*/},
                                        {map.stride(0), 1}));
  vmav<double,1> theta({map.shape(1)}, UNINITIALIZED);
  get_ringtheta_2d(geometry, theta);
  adjoint_synthesis(alm, map2, spin, lmax, mstart, lstride, theta, nphi, phi0_, ringstart, ringfactor, pixstride, nthreads, mode);
  }
template void adjoint_synthesis_2d(const vmav<complex<double>,2> &alm,
  const cmav<double,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads, SHT_mode mode);
template void adjoint_synthesis_2d(const vmav<complex<float>,2> &alm,
  const cmav<float,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads, SHT_mode mode);

template<typename T> void analysis_2d(
  const vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const string &geometry,
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t nrings_min = lmax+1;
  if (geometry=="CC")
    nrings_min = lmax+2;
  else if (geometry=="DH")
    nrings_min = 2*lmax+2;
  else if (geometry=="F2")
    nrings_min = 2*lmax+1;
  MR_assert(ringstart.shape(0)>=nrings_min,
    "too few rings for analysis up to requested lmax");

  vmav<size_t,1> mval({mstart.shape(0)}, UNINITIALIZED);
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;
  vmav<double,1> theta({nphi.shape(0)}, UNINITIALIZED);
  get_ringtheta_2d(geometry, theta);
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, ringfactor, spin, STANDARD);
  if ((geometry=="CC")||(geometry=="F1")||(geometry=="MW")||(geometry=="MWflip"))
    {
    bool npi, spi;
    if (geometry=="CC")
      { npi=spi=true; }
    else if (geometry=="F1")
      { npi=spi=false; }
    else if (geometry=="MW")
      { npi=false; spi=true; }
    else
      { npi=true; spi=false; }

    size_t ntheta_leg = good_size_complex(lmax+1)+1;
    auto leg(allocate_leg<T>(map.shape(0), max(theta.shape(0),ntheta_leg),
                          mstart.shape(0), false, true, nthreads));
    auto legi(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));
    auto lego(subarray<3>(leg, {{},{0,ntheta_leg},{}}));
    vmav<double,1> ringfactor2(ringfactor.shape());
    for (size_t i=0; i<nphi.shape(0); ++i)
      ringfactor2(i) = ringfactor(i)/nphi(i);
    map2leg(map, legi, nphi, phi0, ringstart, ringfactor2, pixstride, nthreads);

    resample_to_prepared_CC(legi, npi, spi, lego, spin, lmax, nthreads);
    vmav<double,1> newtheta({ntheta_leg}, UNINITIALIZED);
    for (size_t i=0; i<ntheta_leg; ++i)
      newtheta(i) = (pi*i)/(ntheta_leg-1);
    leg2alm_internal(alm, lego, spin, lmax, mval, mstart, lstride, newtheta, nthreads, STANDARD, false, true);
    return;
    }
  else
    {
    auto wgt = get_gridweights(geometry, theta.shape(0));
    auto leg(allocate_leg<T>(map.shape(0), theta.shape(0), mstart.shape(0),
                          false, false, nthreads));
    vmav<double,1> ringfactor2(ringfactor.shape());
    for (size_t i=0; i<nphi.shape(0); ++i)
      ringfactor2(i) = ringfactor(i)*wgt(i)/nphi(i);
    map2leg(map, leg, nphi, phi0, ringstart, ringfactor2, pixstride, nthreads);
    leg2alm_internal(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, STANDARD, false, true);
    }
  }

template<typename T> void analysis_2d(const vmav<complex<T>,2> &alm,
  const cmav<T,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads)
  {
  auto nphi = cmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0_ = cmav<double,1>::build_uniform({map.shape(1)}, phi0);
  vmav<size_t,1> ringstart({map.shape(1)}, UNINITIALIZED);
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  auto map2(map.template reinterpret<2>({map.shape(0), 1/*placeholder*/},
                                        {map.stride(0), 1}));

  analysis_2d(alm, map2, spin, lmax, mstart, lstride, geometry, nphi, phi0_, ringstart, ringfactor, pixstride, nthreads);
  }
template void analysis_2d(const vmav<complex<double>,2> &alm,
  const cmav<double,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads);
template void analysis_2d(const vmav<complex<float>,2> &alm,
  const cmav<float,3> &map, size_t spin, size_t lmax,
  const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads);

template<typename T> void adjoint_analysis_2d(
  const cmav<complex<T>,2> &alm, // (ncomp, *)
  const vmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const string &geometry,
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  const cmav<double,1> &ringfactor, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads)
  {
  size_t nrings_min = lmax+1;
  if (geometry=="CC")
    nrings_min = lmax+2;
  else if (geometry=="DH")
    nrings_min = 2*lmax+2;
  else if (geometry=="F2")
    nrings_min = 2*lmax+1;
  MR_assert(ringstart.shape(0)>=nrings_min,
    "too few rings for adjoint analysis up to requested lmax");

  vmav<size_t,1> mval({mstart.shape(0)}, UNINITIALIZED);
  for (size_t i=0; i<mstart.shape(0); ++i)
    mval(i) = i;
  vmav<double,1> theta({nphi.shape(0)}, UNINITIALIZED);
  get_ringtheta_2d(geometry, theta);
  sanity_checks(alm, lmax, mstart, map, theta, phi0, nphi, ringstart, ringfactor, spin, STANDARD);
  if ((geometry=="CC")||(geometry=="F1")||(geometry=="MW")||(geometry=="MWflip"))
    {
    bool npo, spo;
    if (geometry=="CC")
      { npo=spo=true; }
    else if (geometry=="F1")
      { npo=spo=false; }
    else if (geometry=="MW")
      { npo=false; spo=true; }
    else
      { npo=true; spo=false; }

    size_t ntheta_leg = good_size_complex(lmax+1)+1;
    auto leg(allocate_leg<T>(map.shape(0), max(ntheta_leg,theta.shape(0)),
                          mstart.shape(0), true, true, nthreads));
    auto legi(subarray<3>(leg, {{},{0,ntheta_leg},{}}));
    auto lego(subarray<3>(leg, {{},{0,theta.shape(0)},{}}));

    vmav<double,1> theta_tmp({ntheta_leg}, UNINITIALIZED);
    for (size_t i=0; i<ntheta_leg; ++i)
      theta_tmp(i) = (pi*i)/(ntheta_leg-1);
    alm2leg(alm, legi, spin, lmax, mval, mstart, lstride, theta_tmp, nthreads, STANDARD);
    resample_from_prepared_CC(legi, lego, npo, spo, spin, lmax, nthreads);
    vmav<double,1> ringfactor2(ringfactor.shape());
    for (size_t i=0; i<nphi.shape(0); ++i)
      ringfactor2(i) = ringfactor(i)/nphi(i);
    leg2map(map, lego, nphi, phi0, ringstart, ringfactor2, pixstride, nthreads);
    return;
    }
  else
    {
    auto wgt = get_gridweights(geometry, theta.shape(0));
    auto leg(allocate_leg<T>(map.shape(0), theta.shape(0), mstart.shape(0),
                          true, false, nthreads));
    alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, STANDARD);
    vmav<double,1> ringfactor2(ringfactor.shape());
    for (size_t i=0; i<nphi.shape(0); ++i)
      ringfactor2(i) = ringfactor(i)*wgt(i)/nphi(i);
    leg2map(map, leg, nphi, phi0, ringstart, ringfactor2, pixstride, nthreads);
    }
  }

template<typename T> void adjoint_analysis_2d(const cmav<complex<T>,2> &alm, const vmav<T,3> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride, const string &geometry, double phi0, const cmav<double,1> &ringfactor, size_t nthreads)
  {
  auto nphi = cmav<size_t,1>::build_uniform({map.shape(1)}, map.shape(2));
  auto phi0_ = cmav<double,1>::build_uniform({map.shape(1)}, phi0);
  vmav<size_t,1> ringstart({map.shape(1)}, UNINITIALIZED);
  auto ringstride = map.stride(1);
  auto pixstride = map.stride(2);
  for (size_t i=0; i<map.shape(1); ++i)
    ringstart(i) = i*ringstride;
  auto map2(map.template reinterpret<2>({map.shape(0), 1/*placeholder*/},
                                        {map.stride(0), 1}));
  vmav<double,1> theta({map.shape(1)}, UNINITIALIZED);
  adjoint_analysis_2d(alm, map2, spin, lmax, mstart, lstride, geometry, nphi, phi0_,
    ringstart, ringfactor, pixstride, nthreads);
  }
template void adjoint_analysis_2d(const cmav<complex<double>,2> &alm, const vmav<double,3> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor,
  size_t nthreads);
template void adjoint_analysis_2d(const cmav<complex<float>,2> &alm, const vmav<float,3> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const string &geometry, double phi0, const cmav<double,1> &ringfactor,
  size_t nthreads);

template<typename T, typename Tloc> void synthesis_general(
  const cmav<complex<T>,2> &alm, const vmav<T,2> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const cmav<Tloc,2> &loc,
  double epsilon, double sigma_min, double sigma_max, size_t nthreads, SHT_mode mode, bool verbose)
  {
  TimerHierarchy timers("synthesis_general");
  timers.push("setup");
  MR_assert(loc.shape(1)==2, "last dimension of loc must have size 2");
  MR_assert(mstart.shape(0)>0, "need at least m=0");
  size_t nalm = (spin==0) ? 1 : ((mode==STANDARD) ? 2 : 1);
  MR_assert(alm.shape(0)==nalm, "number of components mismatch in alm");
  size_t nmaps = (spin==0) ? 1 : 2;
  MR_assert(map.shape(0)==nmaps, "number of components mismatch in map");

  timers.poppush("SphereInterpol setup");
  SphereInterpol<T> inter(lmax, mstart.shape(0)-1, spin, loc.shape(0),
    sigma_min, sigma_max, epsilon, nthreads);
  timers.poppush("build_planes");
  auto planes = inter.build_planes();
  timers.poppush("getPlane");
  inter.getPlane(alm, mstart, lstride, planes, mode, timers);
  auto xtheta = subarray<1>(loc, {{},{0}});
  auto xphi = subarray<1>(loc, {{},{1}});
  timers.poppush("interpol (u2nu)");
  inter.interpol(planes, 0, 0, xtheta, xphi, map, timers);
  timers.pop();
  if (verbose) timers.report(cerr);
  }

template void synthesis_general(
  const cmav<complex<float>,2> &alm, const vmav<float,2> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const cmav<double,2> &loc, double epsilon, double sigma_min, double sigma_max,
  size_t nthreads, SHT_mode mode, bool verbose);
template void synthesis_general(
  const cmav<complex<double>,2> &alm, const vmav<double,2> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const cmav<double,2> &loc, double epsilon, double sigma_min, double sigma_max,
  size_t nthreads, SHT_mode mode, bool verbose);

template<typename T, typename Tloc> void adjoint_synthesis_general(
  const vmav<complex<T>,2> &alm, const cmav<T,2> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride,
  const cmav<Tloc,2> &loc, double epsilon, double sigma_min, double sigma_max,
  size_t nthreads, SHT_mode mode, bool verbose)
  {
  TimerHierarchy timers("adjoint_synthesis_general");
  timers.push("setup");
  MR_assert(loc.shape(1)==2, "last dimension of loc must have size 2");
  size_t nalm = (spin==0) ? 1 : ((mode==STANDARD) ? 2 : 1);
  MR_assert(alm.shape(0)==nalm, "number of components mismatch in alm");
  size_t nmaps = (spin==0) ? 1 : 2;
  MR_assert(map.shape(0)==nmaps, "number of components mismatch in map");
  MR_assert(mstart.shape(0)>0, "need at least m=0");

  timers.poppush("SphereInterpol setup");
  SphereInterpol<T> inter(lmax, mstart.shape(0)-1, spin, loc.shape(0),
    sigma_min, sigma_max, epsilon, nthreads);
  timers.poppush("build_planes");
  auto planes = inter.build_planes();
  mav_apply([](auto &v){v=0;}, nthreads, planes);
  timers.poppush("deinterpol (nu2u)");
  auto xtheta = subarray<1>(loc, {{},{0}});
  auto xphi = subarray<1>(loc, {{},{1}});
  inter.deinterpol(planes, 0, 0, xtheta, xphi, map, timers);
  timers.poppush("updateAlm");
  inter.updateAlm(alm, mstart, lstride, planes, mode, timers);
  timers.pop();
  if (verbose) timers.report(cerr);
  }
template void adjoint_synthesis_general(
  const vmav<complex<float>,2> &alm, const cmav<float,2> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride, const cmav<double,2> &loc,
  double epsilon, double sigma_min, double sigma_max, size_t nthreads, SHT_mode mode, bool verbose);
template void adjoint_synthesis_general(
  const vmav<complex<double>,2> &alm, const cmav<double,2> &map,
  size_t spin, size_t lmax, const cmav<size_t,1> &mstart, ptrdiff_t lstride, const cmav<double,2> &loc,
  double epsilon, double sigma_min, double sigma_max, size_t nthreads, SHT_mode mode, bool verbose);

template<typename T> tuple<size_t, size_t, double, double> pseudo_analysis_general(
  const vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, npix)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart,
  ptrdiff_t lstride,
  const cmav<double,2> &loc, // (npix,2)
  double sigma_min, double sigma_max,
  size_t nthreads,
  size_t maxiter,
  double epsilon,
  bool verbose,
  bool alm_contains_initial_guess)
  {
  auto op = [&](const cmav<complex<T>,2> &xalm, const vmav<T,2> &xmap)
    {
    synthesis_general(xalm, xmap, spin, lmax, mstart, lstride, loc, 1e-1*epsilon,
                      sigma_min, sigma_max, nthreads, STANDARD, verbose);
    };
  auto op_adj = [&](const cmav<T,2> &xmap, const vmav<complex<T>,2> &xalm)
    {
    adjoint_synthesis_general(xalm, xmap, spin, lmax, mstart, lstride, loc, 1e-1*epsilon,
                              sigma_min, sigma_max, nthreads, STANDARD, verbose);
    };
  auto mapnorm = [&](const cmav<T,2> &xmap)
    {
    double res=0;
    for (size_t icomp=0; icomp<xmap.shape(0); ++icomp)
      for (size_t ipix=0; ipix<xmap.shape(1); ++ipix)
          {
          auto tmp = xmap(icomp,ipix);
          res += tmp*tmp;
          }
    return sqrt(res);
    };
  auto almnorm = [&](const cmav<complex<T>,2> &xalm)
    {
    double res=0;
    for (size_t icomp=0; icomp<xalm.shape(0); ++icomp)
      for (size_t m=0; m<mstart.shape(0); ++m)
        for (size_t l=m; l<=lmax; ++l)
          {
          auto tmp = xalm(icomp,mstart(m)+l*lstride);
          res += norm(tmp) * ((m==0) ? 1 : 2);
          }
    return sqrt(res);
    };
  // try to estimate ATOL according to Paige & Saunders
  // assuming an absolute error of machine epsilon in every matrix element
  double atol = 1e-14*sqrt(map.shape(1));
  if (!alm_contains_initial_guess)  // start with a zero vector guess
    mav_apply([](auto &v){v=0;}, nthreads, alm);
  auto [dum, istop, itn, normr, normar, normA, condA, normx, normb]
    = lsmr(op, op_adj, almnorm, mapnorm, map, alm,
           0., atol, epsilon, 1e8, maxiter, false, nthreads);
  return make_tuple(istop, itn, normr/normb, normar/(normA*normr));
  }
template tuple<size_t, size_t, double, double> pseudo_analysis_general(
  const vmav<complex<float>,2> &alm, // (ncomp, *)
  const cmav<float,2> &map, // (ncomp, npix)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart,
  ptrdiff_t lstride,
  const cmav<double,2> &loc, // (npix,2)
  double sigma_min, double sigma_max,
  size_t nthreads,
  size_t maxiter,
  double epsilon,
  bool verbose,
  bool alm_contains_initial_guess);
template tuple<size_t, size_t, double, double> pseudo_analysis_general(
  const vmav<complex<double>,2> &alm, // (ncomp, *)
  const cmav<double,2> &map, // (ncomp, npix)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart,
  ptrdiff_t lstride,
  const cmav<double,2> &loc, // (npix,2)
  double sigma_min, double sigma_max,
  size_t nthreads,
  size_t maxiter,
  double epsilon,
  bool verbose,
  bool alm_contains_initial_guess);

}}
