/*
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef MRUTIL_INTERPOL_NG_H
#define MRUTIL_INTERPOL_NG_H

#include <vector>
#include <complex>
#include <cmath>
#include "mr_util/math/constants.h"
#include "mr_util/math/gl_integrator.h"
#include "mr_util/math/es_kernel.h"
#include "mr_util/infra/mav.h"
#include "mr_util/sharp/sharp.h"
#include "mr_util/sharp/sharp_almhelpers.h"
#include "mr_util/sharp/sharp_geomhelpers.h"
#include "alm.h"
#include "mr_util/math/fft.h"
#include "mr_util/bindings/pybind_utils.h"

namespace mr {

namespace detail_interpol_ng {

using namespace std;

constexpr double ofmin=1.5;

template<typename T> class Interpolator
  {
  protected:
    bool adjoint;
    size_t lmax, kmax, nphi0, ntheta0, nphi, ntheta;
    int nthreads;
    double ofactor;
    size_t supp;
    ES_Kernel kernel;
    size_t ncomp;
    mav<T,4> cube; // the data cube (theta, phi, 2*mbeam+1, TGC)

    void correct(mav<T,2> &arr, int spin)
      {
      double sfct = (spin&1) ? -1 : 1;
      mav<T,2> tmp({nphi,nphi});
      tmp.apply([](T &v){v=0.;});
      auto tmp0=tmp.template subarray<2>({0,0},{nphi0, nphi0});
      fmav<T> ftmp0(tmp0);
      for (size_t i=0; i<ntheta0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) = arr(i,j);
      // extend to second half
      for (size_t i=1, i2=nphi0-1; i+1<ntheta0; ++i,--i2)
        for (size_t j=0,j2=nphi0/2; j<nphi0; ++j,++j2)
          {
          if (j2>=nphi0) j2-=nphi0;
          tmp0.v(i2,j) = sfct*tmp0(i,j2);
          }
      // FFT to frequency domain on minimal grid
      r2r_fftpack(ftmp0,ftmp0,{0,1},true,true,1./(nphi0*nphi0),nthreads);
      // correct amplitude at Nyquist frequency
      for (size_t i=0; i<nphi0; ++i)
        {
        tmp0.v(i,nphi0-1)*=0.5;
        tmp0.v(nphi0-1,i)*=0.5;
        }
      auto fct = kernel.correction_factors(nphi, nphi0/2+1, nthreads);
      for (size_t i=0; i<nphi0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) *= fct[(i+1)/2] * fct[(j+1)/2];
      auto tmp1=tmp.template subarray<2>({0,0},{nphi, nphi0});
      fmav<T> ftmp1(tmp1);
      // zero-padded FFT in theta direction
      r2r_fftpack(ftmp1,ftmp1,{0},false,false,1.,nthreads);
      auto tmp2=tmp.template subarray<2>({0,0},{ntheta, nphi});
      fmav<T> ftmp2(tmp2);
      fmav<T> farr(arr);
      // zero-padded FFT in phi direction
      r2r_fftpack(ftmp2,farr,{1},false,false,1.,nthreads);
      }
    void decorrect(mav<T,2> &arr, int spin)
      {
      double sfct = (spin&1) ? -1 : 1;
      mav<T,2> tmp({nphi,nphi});
      fmav<T> ftmp(tmp);

      for (size_t i=0; i<ntheta; ++i)
        for (size_t j=0; j<nphi; ++j)
          tmp.v(i,j) = arr(i,j);
      // extend to second half
      for (size_t i=1, i2=nphi-1; i+1<ntheta; ++i,--i2)
        for (size_t j=0,j2=nphi/2; j<nphi; ++j,++j2)
          {
          if (j2>=nphi) j2-=nphi;
          tmp.v(i2,j) = sfct*tmp(i,j2);
          }
      r2r_fftpack(ftmp,ftmp,{1},true,true,1.,nthreads);
      auto tmp1=tmp.template subarray<2>({0,0},{nphi, nphi0});
      fmav<T> ftmp1(tmp1);
      r2r_fftpack(ftmp1,ftmp1,{0},true,true,1.,nthreads);
      auto fct = kernel.correction_factors(nphi, nphi0/2+1, nthreads);
      auto tmp0=tmp.template subarray<2>({0,0},{nphi0, nphi0});
      fmav<T> ftmp0(tmp0);
      for (size_t i=0; i<nphi0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) *= fct[(i+1)/2] * fct[(j+1)/2];
      // FFT to (theta, phi) domain on minimal grid
      r2r_fftpack(ftmp0,ftmp0,{0,1},false, false,1./(nphi0*nphi0),nthreads);
      for (size_t j=0; j<nphi0; ++j)
        {
        tmp0.v(0,j)*=0.5;
        tmp0.v(ntheta0-1,j)*=0.5;
        }
      for (size_t i=0; i<ntheta0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          arr.v(i,j) = tmp0(i,j);
      }

    vector<size_t> getIdx(const mav<T,2> &ptg) const
      {
      vector<size_t> idx(ptg.shape(0));
      constexpr size_t cellsize=16;
      size_t nct = ntheta/cellsize+1,
             ncp = nphi/cellsize+1;
      vector<vector<size_t>> mapper(nct*ncp);
      for (size_t i=0; i<ptg.shape(0); ++i)
        {
        size_t itheta=min(nct-1,size_t(ptg(i,0)/pi*nct)),
               iphi=min(ncp-1,size_t(ptg(i,1)/(2*pi)*ncp));
        mapper[itheta*ncp+iphi].push_back(i);
        }
      size_t cnt=0;
      for (const auto &vec: mapper)
        for (auto i:vec)
          idx[cnt++] = i;
      return idx;
      }

  public:
    Interpolator(const vector<Alm<complex<T>>> &slm,
                 const vector<Alm<complex<T>>> &blm,
                 bool separate, double epsilon, int nthreads_)
      : adjoint(false),
        lmax(slm.at(0).Lmax()),
        kmax(blm.at(0).Mmax()),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*ofmin/2.)))),
        ntheta(nphi/2+1),
        nthreads(nthreads_),
        ofactor(double(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, nthreads),
        ncomp(separate ? slm.size() : 1),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1, ncomp})
      {
      MR_assert(slm.size()==blm.size(), "inconsistent slm and blm vectors");
      for (size_t i=0; i<slm.size(); ++i)
        {
        MR_assert(slm[i].Lmax()==lmax, "inconsistent Sky lmax");
        MR_assert(slm[i].Mmax()==lmax, "Sky lmax must be equal to Sky mmax");
        MR_assert(blm[i].Lmax()==lmax, "Sky and beam lmax must be equal");
        MR_assert(blm[i].Mmax()==kmax, "Inconcistent beam mmax");
        }

      MR_assert((supp<=ntheta) && (supp<=nphi), "support too large!");
      Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
      auto ginfo = sharp_make_cc_geom_info(ntheta0,nphi0,0.,cube.stride(1),cube.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      vector<double>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=std::sqrt(4*pi/(2*i+1.));

      for (size_t icomp=0; icomp<ncomp; ++icomp)
        {
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            {
            if (separate)
              a1(l,m) = slm[icomp](l,m)*blm[icomp](l,0).real()*T(lnorm[l]);
            else
              {
              a1(l,m) = 0;
              for (size_t j=0; j<slm.size(); ++j)
                a1(l,m) += slm[j](l,m)*blm[j](l,0).real()*T(lnorm[l]);
              }
            }
        auto m1 = cube.template subarray<2>({supp,supp,0,icomp},{ntheta,nphi,0,0});
        sharp_alm2map(a1.Alms().data(), m1.vdata(), *ginfo, *ainfo, 0, nthreads);
        correct(m1,0);

        for (size_t k=1; k<=kmax; ++k)
          {
          for (size_t m=0; m<=lmax; ++m)
            for (size_t l=m; l<=lmax; ++l)
              {
              if (l<k)
                a1(l,m)=a2(l,m)=0.;
              else
                {
                if (separate)
                  {
                  auto tmp = -2.*blm[icomp](l,k)*T(lnorm[l]);
                  a1(l,m) = slm[icomp](l,m)*tmp.real();
                  a2(l,m) = slm[icomp](l,m)*tmp.imag();
                  }
                else
                  {
                  a1(l,m) = a2(l,m) = 0;
                  for (size_t j=0; j<slm.size(); ++j)
                    {
                    auto tmp = -2.*blm[j](l,k)*T(lnorm[l]);
                    a1(l,m) += slm[j](l,m)*tmp.real();
                    a2(l,m) += slm[j](l,m)*tmp.imag();
                    }
                  }
              }
            auto m1 = cube.template subarray<2>({supp,supp,2*k-1,icomp},{ntheta,nphi,0,0});
            auto m2 = cube.template subarray<2>({supp,supp,2*k  ,icomp},{ntheta,nphi,0,0});
            sharp_alm2map_spin(k, a1.Alms().data(), a2.Alms().data(), m1.vdata(),
              m2.vdata(), *ginfo, *ainfo, 0, nthreads);
            correct(m1,k);
            correct(m2,k);
            }
          }
        }

      // fill border regions
      for (size_t i=0; i<supp; ++i)
        for (size_t j=0, j2=nphi/2; j<nphi; ++j,++j2)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            double fct = (((k+1)/2)&1) ? -1 : 1;
            if (j2>=nphi) j2-=nphi;
            for (size_t l=0; l<cube.shape(3); ++l)
              {
              cube.v(supp-1-i,j2+supp,k,l) = fct*cube(supp+1+i,j+supp,k,l);
              cube.v(supp+ntheta+i,j2+supp,k,l) = fct*cube(supp+ntheta-2-i,j+supp,k,l);
              }
            }
      for (size_t i=0; i<ntheta+2*supp; ++i)
        for (size_t j=0; j<supp; ++j)
          for (size_t k=0; k<cube.shape(2); ++k)
            for (size_t l=0; l<cube.shape(3); ++l)
            {
            cube.v(i,j,k,l) = cube(i,j+nphi,k,l);
            cube.v(i,j+nphi+supp,k,l) = cube(i,j+supp,k,l);
            }
      }

    Interpolator(size_t lmax_, size_t kmax_, size_t ncomp_, double epsilon, int nthreads_)
      : adjoint(true),
        lmax(lmax_),
        kmax(kmax_),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*ofmin/2.)))),
        ntheta(nphi/2+1),
        nthreads(nthreads_),
        ofactor(double(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, nthreads),
        ncomp(ncomp_),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1, ncomp_})
      {
      MR_assert((supp<=ntheta) && (supp<=nphi), "support too large!");
      cube.apply([](T &v){v=0.;});
      }

    void interpol (const mav<T,2> &ptg, mav<T,2> &res) const
      {
      MR_assert(!adjoint, "cannot be called in adjoint mode");
      MR_assert(ptg.shape(0)==res.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      MR_assert(res.shape(1)==ncomp, "# of components mismatch");
      double delta = 2./supp;
      double xdtheta = (ntheta-1)/pi,
             xdphi = nphi/(2*pi);
      auto idx = getIdx(ptg);
      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        vector<T> wt(supp), wp(supp);
        vector<T> psiarr(2*kmax+1);
        vector<T> val(ncomp);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          size_t i=idx[ind];
          double f0=0.5*supp+ptg(i,0)*xdtheta;
          size_t i0 = size_t(f0+1.);
          for (size_t t=0; t<supp; ++t)
            wt[t] = kernel((t+i0-f0)*delta - 1);
          double f1=0.5*supp+ptg(i,1)*xdphi;
          size_t i1 = size_t(f1+1.);
          for (size_t t=0; t<supp; ++t)
            wp[t] = kernel((t+i1-f1)*delta - 1);
          psiarr[0]=1.;
          double psi=ptg(i,2);
          double cpsi=cos(psi), spsi=sin(psi);
          double cnpsi=cpsi, snpsi=spsi;
          for (size_t l=1; l<=kmax; ++l)
            {
            psiarr[2*l-1]=cnpsi;
            psiarr[2*l]=snpsi;
            const double tmp = snpsi*cpsi + cnpsi*spsi;
            cnpsi=cnpsi*cpsi - snpsi*spsi;
            snpsi=tmp;
            }
          for (size_t m=0; m<ncomp; ++m)
            val[m]=0;
          for (size_t j=0; j<supp; ++j)
            for (size_t k=0; k<supp; ++k)
              for (size_t l=0; l<2*kmax+1; ++l)
                for (size_t m=0; m<ncomp; ++m)
                  val[m] += cube(i0+j,i1+k,l,m)*wt[j]*wp[k]*psiarr[l];
          for (size_t m=0; m<ncomp; ++m)
            res.v(i,m) = val[m];
          }
        });
      }

    void deinterpol (const mav<T,2> &ptg, const mav<T,2> &data)
      {
      MR_assert(adjoint, "can only be called in adjoint mode");
      MR_assert(ptg.shape(0)==data.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      MR_assert(data.shape(1)==ncomp, "# of components mismatch");
      double delta = 2./supp;
      double xdtheta = (ntheta-1)/pi,
             xdphi = nphi/(2*pi);
      auto idx = getIdx(ptg);

      constexpr size_t cellsize=16;
      size_t nct = ntheta/cellsize+5,
             ncp = nphi/cellsize+5;
      mav<std::mutex,2> locks({nct,ncp});

      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        size_t b_theta=99999999999999, b_phi=9999999999999999;
        vector<T> wt(supp), wp(supp);
        vector<T> psiarr(2*kmax+1);
        vector<T> val(ncomp);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          size_t i=idx[ind];
          double f0=0.5*supp+ptg(i,0)*xdtheta;
          size_t i0 = size_t(f0+1.);
          for (size_t t=0; t<supp; ++t)
            wt[t] = kernel((t+i0-f0)*delta - 1);
          double f1=0.5*supp+ptg(i,1)*xdphi;
          size_t i1 = size_t(f1+1.);
          for (size_t t=0; t<supp; ++t)
            wp[t] = kernel((t+i1-f1)*delta - 1);
          for (size_t m=0; m<ncomp; ++m)
            val[m] = data(i,m);
          psiarr[0]=1.;
          double psi=ptg(i,2);
          double cpsi=cos(psi), spsi=sin(psi);
          double cnpsi=cpsi, snpsi=spsi;
          for (size_t l=1; l<=kmax; ++l)
            {
            psiarr[2*l-1]=cnpsi;
            psiarr[2*l]=snpsi;
            const double tmp = snpsi*cpsi + cnpsi*spsi;
            cnpsi=cnpsi*cpsi - snpsi*spsi;
            snpsi=tmp;
            }
          size_t b_theta_new = i0/cellsize,
                 b_phi_new = i1/cellsize;
          if ((b_theta_new!=b_theta) || (b_phi_new!=b_phi))
            {
            if (b_theta<locks.shape(0))  // unlock
              {
              locks.v(b_theta,b_phi).unlock();
              locks.v(b_theta,b_phi+1).unlock();
              locks.v(b_theta+1,b_phi).unlock();
              locks.v(b_theta+1,b_phi+1).unlock();
              }
            b_theta = b_theta_new;
            b_phi = b_phi_new;
            locks.v(b_theta,b_phi).lock();
            locks.v(b_theta,b_phi+1).lock();
            locks.v(b_theta+1,b_phi).lock();
            locks.v(b_theta+1,b_phi+1).lock();
            }
          for (size_t j=0; j<supp; ++j)
            for (size_t k=0; k<supp; ++k)
              for (size_t l=0; l<2*kmax+1; ++l)
                for (size_t m=0; m<ncomp; ++m)
                  cube.v(i0+j,i1+k,l,m) += val[m]*wt[j]*wp[k]*psiarr[l];
          }
        if (b_theta<locks.shape(0))  // unlock
          {
          locks.v(b_theta,b_phi).unlock();
          locks.v(b_theta,b_phi+1).unlock();
          locks.v(b_theta+1,b_phi).unlock();
          locks.v(b_theta+1,b_phi+1).unlock();
          }
        });
      }
    void getSlm (const vector<Alm<complex<T>>> &blm, vector<Alm<complex<T>>> &slm)
      {
      MR_assert(adjoint, "can only be called in adjoint mode");
      MR_assert((blm.size()==ncomp) || (ncomp==1), "incorrect number of beam a_lm sets");
      MR_assert((slm.size()==ncomp) || (ncomp==1), "incorrect number of sky a_lm sets");
      Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
      auto ginfo = sharp_make_cc_geom_info(ntheta0,nphi0,0.,cube.stride(1),cube.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      // move stuff from border regions onto the main grid
      for (size_t i=0; i<cube.shape(0); ++i)
        for (size_t j=0; j<supp; ++j)
          for (size_t k=0; k<cube.shape(2); ++k)
            for (size_t l=0; l<cube.shape(3); ++l)
              {
              cube.v(i,j+nphi,k,l) += cube(i,j,k,l);
              cube.v(i,j+supp,k,l) += cube(i,j+nphi+supp,k,l);
              }
      for (size_t i=0; i<supp; ++i)
        for (size_t j=0, j2=nphi/2; j<nphi; ++j,++j2)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            double fct = (((k+1)/2)&1) ? -1 : 1;
            if (j2>=nphi) j2-=nphi;
            for (size_t l=0; l<cube.shape(3); ++l)
              {
              cube.v(supp+1+i,j+supp,k,l) += fct*cube(supp-1-i,j2+supp,k,l);
              cube.v(supp+ntheta-2-i, j+supp,k,l) += fct*cube(supp+ntheta+i,j2+supp,k,l);
              }
            }

      // special treatment for poles
      for (size_t j=0,j2=nphi/2; j<nphi/2; ++j,++j2)
        for (size_t k=0; k<cube.shape(2); ++k)
          for (size_t l=0; l<cube.shape(3); ++l)
            {
            double fct = (((k+1)/2)&1) ? -1 : 1;
            if (j2>=nphi) j2-=nphi;
            double tval = (cube(supp,j+supp,k,l) + fct*cube(supp,j2+supp,k,l));
            cube.v(supp,j+supp,k,l) = tval;
            cube.v(supp,j2+supp,k,l) = fct*tval;
            tval = (cube(supp+ntheta-1,j+supp,k,l) + fct*cube(supp+ntheta-1,j2+supp,k,l));
            cube.v(supp+ntheta-1,j+supp,k,l) = tval;
            cube.v(supp+ntheta-1,j2+supp,k,l) = fct*tval;
            }

      vector<double>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=std::sqrt(4*pi/(2*i+1.));

      for (size_t icomp=0; icomp<ncomp; ++icomp)
        {
        bool separate = ncomp==blm.size();
        {
        auto m1 = cube.template subarray<2>({supp,supp,0,separate?icomp:0},{ntheta,nphi,0,0});
        decorrect(m1,0);
        sharp_alm2map_adjoint(a1.Alms().vdata(), m1.data(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            for (size_t j=0; j<ncomp; ++j)
              slm[j](l,m)=conj(a1(l,m))*blm[j](l,0).real()*T(lnorm[l]);
        }

        for (size_t k=1; k<=kmax; ++k)
          {
          auto m1 = cube.template subarray<2>({supp,supp,2*k-1,separate?icomp:0},{ntheta,nphi,0,0});
          auto m2 = cube.template subarray<2>({supp,supp,2*k  ,separate?icomp:0},{ntheta,nphi,0,0});
          decorrect(m1,k);
          decorrect(m2,k);

          sharp_alm2map_spin_adjoint(k, a1.Alms().vdata(), a2.Alms().vdata(), m1.data(),
            m2.data(), *ginfo, *ainfo, 0, nthreads);
          for (size_t m=0; m<=lmax; ++m)
            for (size_t l=m; l<=lmax; ++l)
              if (l>=k)
                for (size_t j=0; j<ncomp; ++j)
                  {
                  auto tmp = -2.*conj(blm[j](l,k))*T(lnorm[l]);
                  slm[j](l,m) += conj(a1(l,m))*tmp.real();
                  slm[j](l,m) -= conj(a2(l,m))*tmp.imag();
                  }
          }
        }
      }
  };

}

using detail_interpol_ng::Interpolator;

}

#endif
