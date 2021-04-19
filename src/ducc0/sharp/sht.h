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

/*! \file sht.h
 *  Functionality related to spherical harmonic transforms
 *
 *  Copyright (C) 2020-2021 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_SHT_H
#define DUCC0_SHT_H

#include <complex>
#include "ducc0/infra/mav.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/fft1d.h"
#include "ducc0/sharp/sharp.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

static constexpr double sharp_fbig=0x1p+800,sharp_fsmall=0x1p-800;
static constexpr double sharp_ftol=0x1p-60;
static constexpr double sharp_fbighalf=0x1p+400;

enum SHT_mode { MAP2ALM,
                ALM2MAP,
                ALM2MAP_DERIV1
              };

struct ringdata
  {
  size_t mlim, idx, midx;
  double cth, sth;
  };

class YlmBase
  {
  public:
    size_t lmax, mmax, s;
    vector<double> powlimit;
    /* used if s==0 */
    vector<double> mfac;

  protected:
    /* used if s==0 */
    vector<double> root, iroot;

    /* used if s!=0 */
    vector<double> flm1, flm2, inv;

  public:
    vector<double> prefac;
    vector<int> fscale;

  protected:
    inline void normalize (double &val, int &scale, double xfmax)
      {
      while (abs(val)>xfmax) { val*=sharp_fsmall; ++scale; }
      if (val!=0.)
        while (abs(val)<xfmax*sharp_fsmall) { val*=sharp_fbig; --scale; }
      }

  public:
    static vector<double> get_norm(size_t lmax, size_t spin)
      {
      /* sign convention for H=1 (LensPix paper) */
#if 1
       double spinsign = (spin>0) ? -1.0 : 1.0;
#else
       double spinsign = 1.0;
#endif

      if (spin==0)
        return vector<double>(lmax+1,1.);

      vector<double> res(lmax+1);
      spinsign = (spin&1) ? -spinsign : spinsign;
      for (size_t l=0; l<=lmax; ++l)
        res[l] = (l<spin) ? 0. : spinsign*0.5*sqrt((2*l+1)/(4*pi));
      return res;
      }

    /*! Returns a vector with \a lmax+1 entries containing
      normalisation factors that must be applied to Y_lm values computed for
      first derivatives. */
    static vector<double> get_d1norm(size_t lmax)
      {
      vector<double> res(lmax+1);
      for (size_t l=0; l<=lmax; ++l)
        res[l] = (l<1) ? 0. : 0.5*sqrt(l*(l+1.)*(2*l+1.)/(4*pi));
      return res;
      }

    YlmBase(size_t l_max, size_t m_max, size_t spin)
      : lmax(l_max), mmax(m_max), s(spin),
        powlimit(mmax+s+1),
        mfac((s==0) ? (mmax+1) : 0),
        root((s==0) ? (2*lmax+8) : 0),
        iroot((s==0) ? (2*lmax+8) : 0),
        flm1((s==0) ? 0 : (2*lmax+3)),
        flm2((s==0) ? 0 : (2*lmax+3)),
        inv((s==0) ? 0 : (lmax+2)),
        prefac((s==0) ? 0 : (mmax+1)),
        fscale((s==0) ? 0 : (mmax+1))
      {
      MR_assert(l_max>=spin,"incorrect l_max: must be >= spin");
      MR_assert(l_max>=m_max,"incorrect l_max: must be >= m_max");
      powlimit[0]=0.;
      constexpr double expo=-400*ln2;
      for (size_t i=1; i<=m_max+spin; ++i)
        powlimit[i]=exp(expo/i);

      if (s==0)
        {
        mfac[0] = inv_sqrt4pi;
        for (size_t i=1; i<=mmax; ++i)
          mfac[i] = mfac[i-1]*sqrt((2*i+1.)/(2*i));
        for (size_t i=0; i<2*lmax+8; ++i)
          {
          root[i] = sqrt(i);
          iroot[i] = (i==0) ? 0. : 1./root[i];
          }
        }
      else
        {
        inv[0]=0;
        for (size_t i=1; i<lmax+2; ++i) inv[i]=1./i;
        for (size_t i=0; i<2*lmax+3; ++i)
          {
          flm1[i] = sqrt(1./(i+1.));
          flm2[i] = sqrt(i/(i+1.));
          }
        vector<double> fac(2*lmax+1);
        vector<int> facscale(2*lmax+1);
        fac[0]=1; facscale[0]=0;
        for (size_t i=1; i<2*lmax+1; ++i)
          {
          fac[i]=fac[i-1]*sqrt(i);
          facscale[i]=facscale[i-1];
          normalize(fac[i],facscale[i],sharp_fbighalf);
          }
        for (size_t i=0; i<=mmax; ++i)
          {
          size_t mlo_=min(s,i), mhi_=max(s,i);
          double tfac=fac[2*mhi_]/fac[mhi_+mlo_];
          int tscale=facscale[2*mhi_]-facscale[mhi_+mlo_];
          normalize(tfac,tscale,sharp_fbighalf);
          tfac/=fac[mhi_-mlo_];
          tscale-=facscale[mhi_-mlo_];
          normalize(tfac,tscale,sharp_fbighalf);
          prefac[i]=tfac;
          fscale[i]=tscale;
          }
        }
      }
  };

class Ylmgen: public YlmBase
  {
  public:
    struct dbl2 { double a, b; };

    size_t m;

    vector<double> alpha;
    vector<dbl2> coef;

    /* used if s==0 */
    vector<double> eps;

    /* used if s!=0 */
    size_t sinPow, cosPow;
    bool preMinus_p, preMinus_m;

    size_t mlo, mhi;

    Ylmgen(const YlmBase &base)
      : YlmBase(base),
        m(~size_t(0)),
        alpha((s==0) ? (lmax/2+2) : (lmax+3), 0.),
        coef((s==0) ? (lmax/2+2) : (lmax+3), {0.,0.}),
        eps((s==0) ? (lmax+4) : 0),
        mlo(~size_t(0)),
        mhi(~size_t(0))
      {}

    void prepare (size_t m_)
      {
      if (m_==m) return;
      m = m_;

      if (s==0)
        {
        eps[m] = 0.;
        for (size_t l=m+1; l<lmax+4; ++l)
          eps[l] = root[l+m]*root[l-m]*iroot[2*l+1]*iroot[2*l-1];
        alpha[0] = 1./eps[m+1];
        alpha[1] = eps[m+1]/(eps[m+2]*eps[m+3]);
        for (size_t il=1, l=m+2; l<lmax+1; ++il, l+=2)
          alpha[il+1]= ((il&1) ? -1 : 1) / (eps[l+2]*eps[l+3]*alpha[il]);
        for (size_t il=0, l=m; l<lmax+2; ++il, l+=2)
          {
          coef[il].a = ((il&1) ? -1 : 1)*alpha[il]*alpha[il];
          double t1 = eps[l+2], t2 = eps[l+1];
          coef[il].b = -coef[il].a*(t1*t1+t2*t2);
          }
        }
      else
        {
        size_t mlo_=m, mhi_=s;
        if (mhi_<mlo_) swap(mhi_,mlo_);
        bool ms_similar = ((mhi==mhi_) && (mlo==mlo_));

        mlo = mlo_; mhi = mhi_;

        if (!ms_similar)
          {
          alpha[mhi] = 1.;
          coef[mhi].a = coef[mhi].b = 0.;
          for (size_t l=mhi; l<=lmax; ++l)
            {
            double t = flm1[l+m]*flm1[l-m]*flm1[l+s]*flm1[l-s];
            double lt = 2*l+1;
            double l1 = l+1;
            double flp10=l1*lt*t;
            double flp11=m*s*inv[l]*inv[l+1];
            t = flm2[l+m]*flm2[l-m]*flm2[l+s]*flm2[l-s];
            double flp12=t*l1*inv[l];
            if (l>mhi)
              alpha[l+1] = alpha[l-1]*flp12;
            else
              alpha[l+1] = 1.;
            coef[l+1].a = flp10*alpha[l]/alpha[l+1];
            coef[l+1].b = flp11*coef[l+1].a;
            }
          }

        preMinus_p = preMinus_m = false;
        if (mhi==m)
          {
          cosPow = mhi+s; sinPow = mhi-s;
          preMinus_p = preMinus_m = ((mhi-s)&1);
          }
        else
          {
          cosPow = mhi+m; sinPow = mhi-m;
          preMinus_m = ((mhi+m)&1);
          }
        }
      }
  };

struct ringhelper
  {
  using dcmplx = complex<double>;
  double phi0_;
  vector<dcmplx> shiftarr;
  size_t s_shift;
  unique_ptr<pocketfft_r<double>> plan;
  size_t length;
  bool norot;
  ringhelper() : length(0) {}
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
      plan.reset(new pocketfft_r<double>(nph));
      length=nph;
      }
    }
  template<typename T> DUCC0_NOINLINE void phase2ring (size_t nph,
    double phi0, mav<double,1> &data, size_t mmax, const mav<complex<T>,1> &phase)
    {
    update (nph, mmax, phi0);

    if (nph>=2*mmax+1)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          {
          data.v(2*m)=phase(m).real();
          data.v(2*m+1)=phase(m).imag();
          }
      else
        for (size_t m=0; m<=mmax; ++m)
          {
          dcmplx tmp = dcmplx(phase(m))*shiftarr[m];
          data.v(2*m)=tmp.real();
          data.v(2*m+1)=tmp.imag();
          }
      for (size_t m=2*(mmax+1); m<nph+2; ++m)
        data.v(m)=0.;
      }
    else
      {
      data.v(0)=phase(0).real();
      fill(&data.v(1),&data.v(nph+2),0.);

      for (size_t m=1, idx1=1, idx2=nph-1; m<=mmax; ++m,
           idx1=(idx1+1==nph) ? 0 : idx1+1, idx2=(idx2==0) ? nph-1 : idx2-1)
        {
        dcmplx tmp = phase(m);
        if(!norot) tmp*=shiftarr[m];
        if (idx1<(nph+2)/2)
          {
          data.v(2*idx1)+=tmp.real();
          data.v(2*idx1+1)+=tmp.imag();
          }
        if (idx2<(nph+2)/2)
          {
          data.v(2*idx2)+=tmp.real();
          data.v(2*idx2+1)-=tmp.imag();
          }
        }
      }
    data.v(1)=data(0);
    plan->exec(&(data.v(1)), 1., false);
    }
  template<typename T> DUCC0_NOINLINE void ring2phase (size_t nph, double phi0,
    mav<double,1> &data, size_t mmax, mav<complex<T>,1> &phase)
    {
    update (nph, mmax, -phi0);

    plan->exec (&(data.v(1)), 1., true);
    data.v(0)=data(1);
    data.v(1)=data.v(nph+1)=0.;

    if (mmax<=nph/2)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          phase.v(m) = complex<T>(T(data(2*m)), T(data(2*m+1)));
      else
        for (size_t m=0; m<=mmax; ++m)
          phase.v(m) = complex<T>(dcmplx(data(2*m), data(2*m+1)) * shiftarr[m]);
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
        phase.v(m)=complex<T>(val);
        }
      }
    }
  };

DUCC0_NOINLINE size_t get_mlim (size_t lmax, size_t spin, double sth, double cth);

template<typename T> DUCC0_NOINLINE void inner_loop(SHT_mode mode,
  mav<complex<double>,2> &almtmp,
  mav<complex<T>,3> &phase, const vector<ringdata> &rdata,
  Ylmgen &gen, size_t mi);

void get_gridweights(const string &type, mav<double,1> &wgt);
mav<double,1> get_gridweights(const string &type, size_t nrings);

template<typename T> void alm2leg(  // associated Legendre transform
  const mav<complex<T>,2> &alm, // (ncomp, lmidx)
  mav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads,
  SHT_mode mode);

template<typename T> void leg2map(  // FFT
  const mav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  mav<T,2> &map, // (ncomp, pix)
  const mav<size_t,1> &nphi, // (nrings)
  const mav<size_t,1> &offset, // (nrings)
  const mav<double,1> &phi0, // (nrings)
  size_t nthreads);

template<typename T> void map2leg(  // FFT
  const mav<T,2> &map, // (ncomp, pix)
  mav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const mav<size_t,1> &nphi, // (nrings)
  const mav<size_t,1> &offset, // (nrings)
  const mav<double,1> &phi0, // (nrings)
  size_t nthreads);

template<typename T> void leg2alm(  // associated Legendre transform
  const mav<complex<T>,3> &leg, //  (ncomp, nrings, nm)
  mav<complex<T>,2> &alm, // (ncomp, lmidx)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads);

void prep_for_analysis(mav<complex<double>,3> &leg, size_t spin, size_t nthreads);
void prep_for_analysis2(mav<complex<double>,3> &leg, size_t spin, size_t nthreads);
void resample_theta(const mav<complex<double>,2> &legi, bool npi, bool spi,
  mav<complex<double>,2> &lego, bool npo, bool spo, size_t spin, size_t nthreads);

// fully general map synthesis
// conditions:
// - ncomp = 1+(spin>0)
// - all mval together must form a gapless sequence starting from 0
template<typename T> void synthesis(
  const mav<complex<T>,2> &alm, // (ncomp, *)
  size_t lmax,
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  mav<T,2> &map, // (ncomp, *)
  const mav<double,1> &theta, // (nrings)
  const mav<double,1> &phi0, // (nrings)
  const mav<size_t,1> &nphi, // (nrings)
  const mav<size_t,1> &ringstart, // (nrings)
  size_t spin,
  size_t nthreads);
// - check that mval are a gapless sequence starting at 0
// - check that lmax>=mmax
// - (check that mstart has consistent values?)
// - check that theta are in [0;pi]
// - (check that ringstart and nphi are consistent)
// - check that ncomp and spin are consistent
// - is theta grid equidistant and (CC, F!, MW)?
//   - if no, run standard synthesis
//   - check lmax and nrings whether accelerated synthesis makes sense
//   - if yes, run synthesis on grid with minimal nrings and upsample
//   - otherwise run standard synthesis
template<typename T> void adjoint_synthesis(
  mav<complex<T>,2> &alm, // (ncomp, *)
  size_t lmax,
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  const mav<T,2> &map, // (ncomp, *)
  const mav<double,1> &theta, // (nrings)
  const mav<double,1> &phi0, // (nrings)
  const mav<size_t,1> &nphi, // (nrings)
  const mav<size_t,1> &ringstart, // (nrings)
  size_t spin,
  size_t nthreads);

template<typename T> void synthesis(const mav<complex<T>,2> &alm, size_t lmax,
  mav<T,3> &map, size_t spin, const string &geometry, size_t nthreads)
  {
  unique_ptr<sharp_geom_info> ginfo;
  ginfo = sharp_make_2d_geom_info (map.shape(1), map.shape(2), 0.,
    map.stride(2), map.stride(1), geometry);
  MR_assert(((lmax+1)*(lmax+2))/2==alm.shape(1), "bad a_lm size");
  auto ainfo = sharp_make_triangular_alm_info(lmax, lmax, alm.stride(1));
  (spin==0) ?
    sharp_alm2map(alm.cdata(), map.vdata(), *ginfo, *ainfo, 0, nthreads) :
    sharp_alm2map_spin(spin, &alm(0,0), &alm(1,0), &map.v(0,0,0), &map.v(1,0,0),
      *ginfo, *ainfo, 0, nthreads);
  }
template<typename T> void synthesis(const mav<complex<T>,1> &alm, size_t lmax,
  mav<T,2> &map, const string &geometry, size_t nthreads)
  {
  mav<complex<T>,2> alm2(alm.cdata(), {1,alm.shape(0)}, {0,alm.stride(0)});
  mav<T,3> map2(map.vdata(), {1,map.shape(0),map.shape(1)}, {0,map.stride(0),map.stride(1)}, true);
  synthesis (alm2, lmax, map2, 0, geometry, nthreads);
  }
template<typename T> void adjoint_synthesis(mav<complex<T>,2> &alm, size_t lmax,
  const mav<T,3> &map, size_t spin, const string &geometry, size_t nthreads)
  {
  unique_ptr<sharp_geom_info> ginfo;
  ginfo = sharp_make_2d_geom_info (map.shape(1), map.shape(2), 0.,
    map.stride(2), map.stride(1), geometry);
  MR_assert(((lmax+1)*(lmax+2))/2==alm.shape(1), "bad a_lm size");
  auto ainfo = sharp_make_triangular_alm_info(lmax, lmax, alm.stride(1));
  (spin==0) ?
    sharp_alm2map_adjoint(alm.vdata(), map.cdata(), *ginfo, *ainfo, 0, nthreads) :
    sharp_alm2map_spin_adjoint(spin, &alm.v(0,0), &alm.v(1,0), &map(0,0,0), &map(1,0,0),
      *ginfo, *ainfo, 0, nthreads);
  }
template<typename T> void adjoint_synthesis(mav<complex<T>,1> &alm, size_t lmax,
  const mav<T,2> &map, const string &geometry, size_t nthreads)
  {
  mav<complex<T>,2> alm2(alm.vdata(), {1,alm.shape(0)}, {0,alm.stride(0)}, true);
  mav<T,3> map2(map.cdata(), {1,map.shape(0),map.shape(1)}, {0,map.stride(0),map.stride(1)});
  adjoint_synthesis (alm2, lmax, map2, 0, geometry, nthreads);
  }
// template<typename T> void analysis(mav<complex<T>,2> &alm, size_t lmax,
//   size_t mmax, const mav<T,3> &map, size_t spin, const string &geometry,
//   size_t nthreads)
//   {
// // if nphi is too low => error
// // if nrings is too low for geometry => error
// // nrings high enough for analysis with quadrature weights => standard analysis
// // - map2leg
// // - upsample to CC grid with enough rings for quadrature
// // - apply CC weights
// // - resample odd rings to fall on even rings
// // - adjoint synthesis
//   }
}

using detail_sht::SHT_mode;
using detail_sht::ALM2MAP;
using detail_sht::MAP2ALM;
using detail_sht::get_gridweights;
using detail_sht::alm2leg;
using detail_sht::leg2alm;
using detail_sht::map2leg;
using detail_sht::leg2map;
using detail_sht::prep_for_analysis;
//using detail_sht::prep_for_analysis2;
using detail_sht::resample_theta;
using detail_sht::synthesis;
using detail_sht::adjoint_synthesis;

}

#endif
