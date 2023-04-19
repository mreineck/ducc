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

/*
 *  Copyright (C) 2020-2023 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC0_SPHERE_INTERPOL_H
#define DUCC0_SPHERE_INTERPOL_H

#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>
#include <complex>
#include <cmath>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/bucket_sort.h"
#include "ducc0/sht/sht.h"
#include "ducc0/sht/alm.h"
#include "ducc0/fft/fft1d.h"
#include "ducc0/fft/fft.h"
#include "ducc0/math/math_utils.h"
#include "ducc0/sht/sht_utils.h"
#include "ducc0/infra/timers.h"
#include "ducc0/nufft/nufft.h"

namespace ducc0 {

namespace detail_sphereinterpol {

using namespace std;

template<typename T> class SphereInterpol
  {
  protected:
    constexpr static auto vlen = min<size_t>(8, native_simd<T>::size());
    using Tsimd = typename simd_select<T, vlen>::type;

    size_t nthreads;
    size_t lmax, mmax, spin;
    // _s: small grid
    // _b: oversampled grid
    // no suffix: grid with borders
    size_t nphi_s, ntheta_s;
    size_t kernel_index;
    shared_ptr<PolynomialKernel> kernel;
    size_t nphi_b, ntheta_b;
    double dphi, dtheta, xdphi, xdtheta;

    size_t nbphi, nbtheta;
    size_t nphi, ntheta;
    double phi0, theta0;

    auto getKernel(size_t axlen, size_t axlen2) const
      {
      auto axlen_big = max(axlen, axlen2);
      auto axlen_small = min(axlen, axlen2);
      auto fct = kernel->corfunc(axlen_small/2+1, 1./axlen_big, nthreads);
      return fct;
      }

    template<typename Tloc>quick_array<uint32_t> getIdx(const cmav<Tloc,1> &theta, const cmav<Tloc,1> &phi,
      size_t patch_ntheta, size_t patch_nphi, size_t itheta0, size_t iphi0, size_t supp) const
      {
      size_t nptg = theta.shape(0);
      constexpr size_t cellsize=8;
      size_t nct = patch_ntheta/cellsize+1,
             ncp = patch_nphi/cellsize+1;
      double theta0 = (int(itheta0)-int(nbtheta))*dtheta,
             phi0 = (int(iphi0)-int(nbphi))*dphi;
      double theta_lo=theta0, theta_hi=theta_lo+(patch_ntheta+1)*dtheta;
      double phi_lo=phi0, phi_hi=phi_lo+(patch_nphi+1)*dphi;
      MR_assert(uint64_t(nct)*uint64_t(ncp)<(uint64_t(1)<<32),
        "key space too large");

      quick_array<uint32_t> key(nptg);
      execParallel(nptg, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          MR_assert((theta(i)>=theta_lo) && (theta(i)<=theta_hi), "theta out of range: ", theta(i));
          MR_assert((phi(i)>=phi_lo) && (phi(i)<=phi_hi), "phi out of range: ", phi(i));
          auto ftheta = (theta(i)-theta0)*xdtheta-supp*0.5;
          auto itheta = size_t(ftheta+1);
          auto fphi = (phi(i)-phi0)*xdphi-supp*0.5;
          auto iphi = size_t(fphi+1);
          itheta /= cellsize;
          iphi /= cellsize;
          MR_assert(itheta<nct, "bad itheta");
          MR_assert(iphi<ncp, "bad iphi");
          key[i] = itheta*ncp+iphi;
          }
        });
      quick_array<uint32_t> res(key.size());
      bucket_sort2(key, res, ncp*nct, nthreads);
      return res;
      }

    template<size_t supp> class WeightHelper
      {
      public:
        static constexpr size_t vlen = Tsimd::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;
        const SphereInterpol &plan;
        union kbuf {
          T scalar[2*nvec*vlen];
          Tsimd simd[2*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

      private:
        TemplateKernel<supp, Tsimd> tkrn;
        double mytheta0, myphi0;

      public:
        WeightHelper(const SphereInterpol &plan_, const mav_info<3> &info, size_t itheta0, size_t iphi0)
          : plan(plan_),
            tkrn(*plan.kernel),
            mytheta0(plan.theta0+itheta0*plan.dtheta),
            myphi0(plan.phi0+iphi0*plan.dphi),
            wtheta(&buf.scalar[0]),
            wphi(&buf.simd[nvec]),
            jumptheta(info.stride(1))
          {
          MR_assert(info.stride(2)==1, "last axis of cube must be contiguous");
          }
        void prep(double theta, double phi)
          {
          auto ftheta = (theta-mytheta0)*plan.xdtheta-supp*0.5;
          itheta = size_t(ftheta+1);
          ftheta = -1+(itheta-ftheta)*2;
          auto fphi = (phi-myphi0)*plan.xdphi-supp*0.5;
          iphi = size_t(fphi+1);
          fphi = -1+(iphi-fphi)*2;
          tkrn.eval2(T(ftheta), T(fphi), &buf.simd[0]);
          }
        size_t itheta, iphi;
        const T * DUCC0_RESTRICT wtheta;
        const Tsimd * DUCC0_RESTRICT wphi;
        ptrdiff_t jumptheta;
      };

    // prefetching distance
    static constexpr size_t pfdist=2;

    template<size_t supp, typename Tloc> void interpolx(size_t supp_, const cmav<T,3> &cube,
      size_t itheta0, size_t iphi0, const cmav<Tloc,1> &theta, const cmav<Tloc,1> &phi,
      vmav<T,2> &signal) const
      {
      if constexpr (supp>=8)
        if (supp_<=supp/2) return interpolx<supp/2>(supp_, cube, itheta0, iphi0, theta, phi, signal);
      if constexpr (supp>4)
        if (supp_<supp) return interpolx<supp-1>(supp_, cube, itheta0, iphi0, theta, phi, signal);
      MR_assert(supp_==supp, "requested support out of range");

      MR_assert(cube.stride(2)==1, "last axis of cube must be contiguous");
      MR_assert(phi.shape(0)==theta.shape(0), "array shape mismatch");
      MR_assert(signal.shape(1)==theta.shape(0), "array shape mismatch");
      const auto ncomp = cube.shape(0);
      MR_assert(signal.shape(0)==ncomp, "array shape mismatch");
      static constexpr size_t vlen = Tsimd::size();
      static constexpr size_t nvec = (supp+vlen-1)/vlen;
      auto idx = getIdx(theta, phi, cube.shape(1), cube.shape(2), itheta0, iphi0, supp);

      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        WeightHelper<supp> hlp(*this, cube, itheta0, iphi0);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          if (ind+pfdist<rng.hi)
            {
            size_t i=idx[ind+pfdist];
            DUCC0_PREFETCH_R(&theta(i));
            DUCC0_PREFETCH_R(&phi(i));
            for (size_t j=0; j<ncomp; ++j)
              {
              DUCC0_PREFETCH_R(&signal(j,i));
              DUCC0_PREFETCH_W(&signal(j,i));
              }
            }
          size_t i=idx[ind];
          hlp.prep(theta(i), phi(i));

          if (ncomp==2)
            {
            if constexpr(nvec==1)
              {
              const T * DUCC0_RESTRICT ptr0 = &cube(0, hlp.itheta,hlp.iphi);
              const T * DUCC0_RESTRICT ptr1 = &cube(1, hlp.itheta,hlp.iphi);
              Tsimd tres0=0, tres1=0;
              for (size_t itheta=0; itheta<supp; ++itheta, ptr0+=hlp.jumptheta, ptr1+=hlp.jumptheta)
                {
                tres0 += hlp.wtheta[itheta]*Tsimd(ptr0, element_aligned_tag());
                tres1 += hlp.wtheta[itheta]*Tsimd(ptr1, element_aligned_tag());
                }
              signal(0, i) = reduce(tres0*hlp.wphi[0], std::plus<>());
              signal(1, i) = reduce(tres1*hlp.wphi[0], std::plus<>());
              }
            else
              {
              const T * DUCC0_RESTRICT ptr0 = &cube(0, hlp.itheta,hlp.iphi);
              const T * DUCC0_RESTRICT ptr1 = &cube(1, hlp.itheta,hlp.iphi);
              Tsimd tres0=0, tres1=0;
              for (size_t itheta=0; itheta<supp; ++itheta, ptr0+=hlp.jumptheta, ptr1+=hlp.jumptheta)
                {
                Tsimd ttres0=0, ttres1=0;
                for (size_t iphi=0; iphi<nvec; ++iphi)
                  {
                  ttres0 += hlp.wphi[iphi]*Tsimd(ptr0+iphi*vlen,element_aligned_tag());
                  ttres1 += hlp.wphi[iphi]*Tsimd(ptr1+iphi*vlen,element_aligned_tag());
                  }
                tres0 += ttres0*hlp.wtheta[itheta];
                tres1 += ttres1*hlp.wtheta[itheta];
                }
              signal(0, i) = reduce(tres0, std::plus<>());
              signal(1, i) = reduce(tres1, std::plus<>());
              }
            }
          else
            {
            if constexpr(nvec==1)
              for (size_t icomp=0; icomp<ncomp; ++icomp)
                {
                const T * DUCC0_RESTRICT ptr = &cube(icomp, hlp.itheta,hlp.iphi);
                Tsimd tres=0;
                for (size_t itheta=0; itheta<supp; ++itheta, ptr+=hlp.jumptheta)
                  tres += hlp.wtheta[itheta]*Tsimd(ptr, element_aligned_tag());
                signal(icomp, i) = reduce(tres*hlp.wphi[0], std::plus<>());
                }
            else
              for (size_t icomp=0; icomp<ncomp; ++icomp)
                {
                const T * DUCC0_RESTRICT ptr = &cube(icomp, hlp.itheta,hlp.iphi);
                Tsimd tres=0;
                for (size_t itheta=0; itheta<supp; ++itheta, ptr+=hlp.jumptheta)
                  {
                  Tsimd tres2=0;
                  for (size_t iphi=0; iphi<nvec; ++iphi)
                    tres2 += hlp.wphi[iphi]*Tsimd(ptr+iphi*vlen,element_aligned_tag());
                  tres += tres2*hlp.wtheta[itheta];
                  }
                signal(icomp, i) = reduce(tres, std::plus<>());
                }
            }
          }
        });
      }
    template<size_t supp, typename Tloc> void deinterpolx(size_t supp_, vmav<T,3> &cube,
      size_t itheta0, size_t iphi0, const cmav<Tloc,1> &theta, const cmav<Tloc,1> &phi,
      const cmav<T,2> &signal) const
      {
      if constexpr (supp>=8)
        if (supp_<=supp/2) return deinterpolx<supp/2>(supp_, cube, itheta0, iphi0, theta, phi, signal);
      if constexpr (supp>4)
        if (supp_<supp) return deinterpolx<supp-1>(supp_, cube, itheta0, iphi0, theta, phi, signal);
      MR_assert(supp_==supp, "requested support out of range");

      MR_assert(cube.stride(2)==1, "last axis of cube must be contiguous");
      MR_assert(phi.shape(0)==theta.shape(0), "array shape mismatch");
      MR_assert(signal.shape(1)==theta.shape(0), "array shape mismatch");
      const auto ncomp = cube.shape(0);
      MR_assert(signal.shape(0)==ncomp, "array shape mismatch");
      static constexpr size_t vlen = Tsimd::size();
      static constexpr size_t nvec = (supp+vlen-1)/vlen;
      auto idx = getIdx(theta, phi, cube.shape(1), cube.shape(2), itheta0, iphi0, supp);

      constexpr size_t cellsize=16;
      size_t nct = cube.shape(1)/cellsize+10,
             ncp = cube.shape(2)/cellsize+10;
      vmav<Mutex,2> locks({nct,ncp});

      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        size_t b_theta=~(size_t(0)), b_phi=~(size_t(0));
        WeightHelper<supp> hlp(*this, cube, itheta0, iphi0);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          if (ind+pfdist<rng.hi)
            {
            size_t i=idx[ind+pfdist];
            DUCC0_PREFETCH_R(&theta(i));
            DUCC0_PREFETCH_R(&phi(i))
            for (size_t j=0; j<ncomp; ++j)
              DUCC0_PREFETCH_R(&signal(j,i));
            }
          size_t i=idx[ind];
          hlp.prep(theta(i), phi(i));

          size_t b_theta_new = hlp.itheta/cellsize,
                 b_phi_new = hlp.iphi/cellsize;
          if ((b_theta_new!=b_theta) || (b_phi_new!=b_phi))
            {
            if (b_theta<locks.shape(0))  // unlock
              {
              locks(b_theta,b_phi).unlock();
              locks(b_theta,b_phi+1).unlock();
              locks(b_theta+1,b_phi).unlock();
              locks(b_theta+1,b_phi+1).unlock();
              }
            b_theta = b_theta_new;
            b_phi = b_phi_new;
            locks(b_theta,b_phi).lock();
            locks(b_theta,b_phi+1).lock();
            locks(b_theta+1,b_phi).lock();
            locks(b_theta+1,b_phi+1).lock();
            }

          if (ncomp==2)
            {
            if constexpr (nvec==1)
              {
              Tsimd tmp0=signal(0, i)*hlp.wphi[0];
              Tsimd tmp1=signal(1, i)*hlp.wphi[0];
              T * DUCC0_RESTRICT ptr0 = &cube(0,hlp.itheta,hlp.iphi);
              T * DUCC0_RESTRICT ptr1 = &cube(1,hlp.itheta,hlp.iphi);
              for (size_t itheta=0; itheta<supp; ++itheta, ptr0+=hlp.jumptheta, ptr1+=hlp.jumptheta)
                {
                Tsimd var0=Tsimd(ptr0,element_aligned_tag());
                Tsimd var1=Tsimd(ptr1,element_aligned_tag());
                var0 += tmp0*hlp.wtheta[itheta];
                var1 += tmp1*hlp.wtheta[itheta];
                var0.copy_to(ptr0,element_aligned_tag());
                var1.copy_to(ptr1,element_aligned_tag());
                }
              }
            else
              {
              Tsimd tmp0=signal(0, i);
              Tsimd tmp1=signal(1, i);
              T * DUCC0_RESTRICT ptr0 = &cube(0,hlp.itheta,hlp.iphi);
              T * DUCC0_RESTRICT ptr1 = &cube(1,hlp.itheta,hlp.iphi);
              for (size_t itheta=0; itheta<supp; ++itheta, ptr0+=hlp.jumptheta, ptr1+=hlp.jumptheta)
                {
                auto ttmp0=tmp0*hlp.wtheta[itheta];
                auto ttmp1=tmp1*hlp.wtheta[itheta];
                for (size_t iphi=0; iphi<nvec; ++iphi)
                  {
                  Tsimd var0=Tsimd(ptr0+iphi*vlen, element_aligned_tag());
                  Tsimd var1=Tsimd(ptr1+iphi*vlen, element_aligned_tag());
                  var0 += ttmp0*hlp.wphi[iphi];
                  var1 += ttmp1*hlp.wphi[iphi];
                  var0.copy_to(ptr0+iphi*vlen, element_aligned_tag());
                  var1.copy_to(ptr1+iphi*vlen, element_aligned_tag());
                  }
                }
              }
            }
          else
            {
            if constexpr (nvec==1)
              for (size_t icomp=0; icomp<ncomp; ++icomp)
                {
                Tsimd tmp=signal(icomp, i)*hlp.wphi[0];
                T * DUCC0_RESTRICT ptr = &cube(icomp,hlp.itheta,hlp.iphi);
                for (size_t itheta=0; itheta<supp; ++itheta, ptr+=hlp.jumptheta)
                  {
                  Tsimd var=Tsimd(ptr,element_aligned_tag());
                  var += tmp*hlp.wtheta[itheta];
                  var.copy_to(ptr,element_aligned_tag());
                  }
                }
            else
              for (size_t icomp=0; icomp<ncomp; ++icomp)
                {
                Tsimd tmp=signal(icomp, i);
                T * DUCC0_RESTRICT ptr = &cube(icomp,hlp.itheta,hlp.iphi);
                for (size_t itheta=0; itheta<supp; ++itheta, ptr+=hlp.jumptheta)
                  {
                  auto ttmp=tmp*hlp.wtheta[itheta];
                  for (size_t iphi=0; iphi<nvec; ++iphi)
                    {
                    Tsimd var=Tsimd(ptr+iphi*vlen, element_aligned_tag());
                    var += ttmp*hlp.wphi[iphi];
                    var.copy_to(ptr+iphi*vlen, element_aligned_tag());
                    }
                  }
                }
            }
          }
        if (b_theta<locks.shape(0))  // unlock
          {
          locks(b_theta,b_phi).unlock();
          locks(b_theta,b_phi+1).unlock();
          locks(b_theta+1,b_phi).unlock();
          locks(b_theta+1,b_phi+1).unlock();
          }
        });
      }

  public:
    SphereInterpol(size_t lmax_, size_t mmax_, size_t spin_, size_t npoints,
      double sigma_min, double sigma_max, double epsilon, size_t nthreads_)
      : nthreads(adjust_nthreads(nthreads_)),
        lmax(lmax_),
        mmax(mmax_),
        spin(spin_),
        nphi_s(2*good_size_real(mmax+1)),
        ntheta_s(good_size_real(lmax+1)+1),
        kernel_index(findNufftKernel<T,T>(epsilon, sigma_min, sigma_max,
          {(2*ntheta_s-2), nphi_s}, npoints, true, nthreads)),
        kernel(selectKernel(kernel_index)),
        nphi_b(std::max<size_t>(20,2*good_size_real(size_t((2*mmax+1)*ducc0::getKernel(kernel_index).ofactor/2.)))),
        ntheta_b(std::max<size_t>(21,good_size_real(size_t((lmax+1)*ducc0::getKernel(kernel_index).ofactor))+1)),
        dphi(2*pi/nphi_b),
        dtheta(pi/(ntheta_b-1)),
        xdphi(1./dphi),
        xdtheta(1./dtheta),
        nbphi((kernel->support()+1)/2),
        nbtheta((kernel->support()+1)/2),
        nphi(nphi_b+2*nbphi+vlen),
        ntheta(ntheta_b+2*nbtheta),
        phi0(nbphi*(-dphi)),
        theta0(nbtheta*(-dtheta))
      {
      auto supp = kernel->support();
      MR_assert((supp<=ntheta) && (supp<=nphi_b), "kernel support too large!");
      }

    size_t Lmax() const { return lmax; }
    size_t Mmax() const { return mmax; }
    size_t Spin() const { return spin; }
    size_t Ntheta() const { return ntheta; }
    size_t Nphi() const { return nphi; }

    vector<size_t> getPatchInfo(double theta_lo, double theta_hi, double phi_lo, double phi_hi) const
      {
      vector<size_t> res(4);
      auto tmp = (theta_lo-theta0)*xdtheta-nbtheta;
      res[0] = min(size_t(max(0., tmp)), ntheta);
      tmp = (theta_hi-theta0)*xdtheta+nbtheta+1.;
      res[1] = min(size_t(max(0., tmp)), ntheta);
      tmp = (phi_lo-phi0)*xdphi-nbphi;
      res[2] = min(size_t(max(0., tmp)), nphi);
      tmp = (phi_hi-phi0)*xdphi+nbphi+1.+vlen;
      res[3] = min(size_t(max(0., tmp)), nphi);
      return res;
      }

    void getPlane(const cmav<complex<T>,2> &valm, const cmav<size_t,1> &mstart, ptrdiff_t lstride, vmav<T,3> &planes, SHT_mode mode, TimerHierarchy &timers) const
      {
      timers.push("alm2leg");
      size_t nplanes=1+(spin>0);
      MR_assert(planes.conformable({nplanes, Ntheta(), Nphi()}), "bad planes shape");

      auto subplanes=subarray<3>(planes,{{}, {nbtheta, nbtheta+ntheta_s}, {nbphi, nbphi+nphi_s}});

      MR_assert(planes.stride(2)==1, "last axis must have stride 1");
      MR_assert((planes.stride(1)&1)==0, "stride must be even");
      MR_assert((planes.stride(0)&1)==0, "stride must be even");
      MR_assert(2*(mmax+1)<=nphi_b, "aargh");
      vmav<complex<T>,3> leg_s(reinterpret_cast<complex<T> *>(&planes(0,nbtheta,nbphi-1)),
        {nplanes, ntheta_s, mmax+1}, {subplanes.stride(0)/2, subplanes.stride(1)/2, 1});
      vmav<complex<T>,3> leg_b(reinterpret_cast<complex<T> *>(&planes(0,nbtheta,nbphi-1)),
        {nplanes, ntheta_b, mmax+1}, {subplanes.stride(0)/2, subplanes.stride(1)/2, 1});
      vmav<double,1> theta({ntheta_s}, UNINITIALIZED);
      for (size_t i=0; i<ntheta_s; ++i)
        theta(i) = (i*pi)/(ntheta_s-1);
      
      vmav<size_t,1> mval({mmax+1}, UNINITIALIZED);
      for (size_t i=0; i<=mmax; ++i)
        mval(i) = i;
      alm2leg(valm, leg_s, spin, lmax, mval, mstart, lstride, theta, nthreads, mode);
      timers.poppush("theta resampling and deconvolution");
      auto kernel = getKernel(2*ntheta_s-2, 2*ntheta_b-2);
      ducc0::detail_sht::resample_and_convolve_theta<T>
        (leg_s, true, true, leg_b, true, true, kernel, spin, nthreads, false);
      timers.poppush("phi FFT and dconvolution");
      // fix phi
      size_t nj=2*mmax+1;
      auto phikrn = getKernel(nphi_s, nphi_b);
      vmav<T,1> phikrn2({nj});
      for (size_t j=0; j<nj; ++j)
        phikrn2(j) = T(phikrn[(j+1)/2]);
      pocketfft_r<T> rplan(nphi_b);
      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        auto arr = subarray<2>(planes, {{iplane},{nbtheta, nbtheta+ntheta_b}, {nbphi, nbphi+nphi_b}});
        execParallel(ntheta_b, nthreads, [&](size_t lo, size_t hi)
          {
          vmav<T,1> buf({rplan.bufsize()});
          for (size_t i=lo; i<hi; ++i)
            {
            // make halfcomplex
            planes(iplane, nbtheta+i, nbphi) = planes(iplane, nbtheta+i, nbphi-1);
            for (size_t j=0; j<nj; ++j)
              arr(i,j) *= phikrn2(j);
            for (size_t j=nj; j<nphi_b; ++j)
              arr(i,j) = T(0);
            rplan.exec_copyback(&arr(i,0), buf.data(), T(1), false);
            }
          });
        }

      timers.poppush("dealing with borders");
      // fill border regions
      T fct = (spin&1) ? -1 : 1;
      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        for (size_t i=0; i<nbtheta; ++i)
          for (size_t j=0, j2=nphi_b/2; j<nphi_b; ++j,++j2)
            {
            if (j2>=nphi_b) j2-=nphi_b;
            planes(iplane,nbtheta-1-i,j2+nbphi) = fct*planes(iplane,nbtheta+1+i,j+nbphi);
            planes(iplane,nbtheta+ntheta_b+i,j2+nbphi) = fct*planes(iplane,nbtheta+ntheta_b-2-i,j+nbphi);
            }
        for (size_t i=0; i<ntheta; ++i)
          {
          for (size_t j=0; j<nbphi; ++j)
            {
            planes(iplane,i,j) = planes(iplane,i,j+nphi_b);
            planes(iplane,i,j+nphi_b+nbphi) = planes(iplane,i,j+nbphi);
            }
          // SIMD buffer
          for (size_t j=0; j<vlen; ++j)
            planes(iplane, i, nphi-vlen+j) = T(0);
          }
        }
      timers.pop();
      }

    void getPlane(const cmav<complex<T>,1> &alm, vmav<T,3> &planes) const
      {
      cmav<complex<T>,2> valm(&alm(0), {1,alm.shape(0)}, {0,alm.stride(0)});
      getPlane(valm, planes);
      }

    template<typename Tloc> void interpol(const cmav<T,3> &cube, size_t itheta0,
      size_t iphi0, const cmav<Tloc,1> &theta, const cmav<Tloc,1> &phi,
      vmav<T,2> &signal) const
      {
      constexpr size_t maxsupp = is_same<T, double>::value ? 16 : 8;
      interpolx<maxsupp>(kernel->support(), cube, itheta0, iphi0, theta, phi, signal);
      }

    template<typename Tloc> void deinterpol(vmav<T,3> &cube, size_t itheta0,
      size_t iphi0, const cmav<Tloc,1> &theta, const cmav<Tloc,1> &phi,
      const cmav<T,2> &signal) const
      {
      constexpr size_t maxsupp = is_same<T, double>::value ? 16 : 8;
      deinterpolx<maxsupp>(kernel->support(), cube, itheta0, iphi0, theta, phi, signal);
      }

    void updateAlm(vmav<complex<T>,2> &valm, const cmav<size_t,1> &mstart, ptrdiff_t lstride, vmav<T,3> &planes, SHT_mode mode, TimerHierarchy &timers) const
      {
      size_t nplanes=1+(spin>0);
      MR_assert(planes.conformable({nplanes, Ntheta(), Nphi()}), "bad planes shape");

      timers.push("dealing with borders");
      // move stuff from border regions onto the main grid
      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        for (size_t i=0; i<ntheta; ++i)
          for (size_t j=0; j<nbphi; ++j)
            {
            planes(iplane,i,j+nphi_b) += planes(iplane,i,j);
            planes(iplane,i,j+nbphi) += planes(iplane,i,j+nphi_b+nbphi);
            }

        for (size_t i=0; i<nbtheta; ++i)
          for (size_t j=0, j2=nphi_b/2; j<nphi_b; ++j,++j2)
            {
            T fct = (spin&1) ? -1 : 1;
            if (j2>=nphi_b) j2-=nphi_b;
            planes(iplane,nbtheta+1+i,j+nbphi) += fct*planes(iplane,nbtheta-1-i,j2+nbphi);
            planes(iplane,nbtheta+ntheta_b-2-i, j+nbphi) += fct*planes(iplane,nbtheta+ntheta_b+i,j2+nbphi);
            }
        }

      timers.poppush("phi FFT and deconvolution");

      // fix phi
      size_t nj=2*mmax+1;
      auto phikrn = getKernel(nphi_b, nphi_s);
      vmav<T,1> phikrn2({nj});
      for (size_t j=0; j<nj; ++j)
        phikrn2(j) = T(phikrn[(j+1)/2]);
      pocketfft_r<T> rplan(nphi_b);
      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        auto arr = subarray<2>(planes, {{iplane}, {nbtheta, nbtheta+ntheta_b}, {nbphi,nbphi+nphi_b}});
        execParallel(ntheta_b, nthreads, [&](size_t lo, size_t hi)
          {
          vmav<T,1> buf({rplan.bufsize()});
          for (size_t i=lo; i<hi; ++i)
            {
            rplan.exec_copyback(&arr(i,0), buf.data(), T(1), true);
            for (size_t j=0; j<nj; ++j)
              arr(i,j) *= phikrn2(j);
            // back from halfcomplex
            planes(iplane, nbtheta+i, nbphi-1) = planes(iplane, nbtheta+i, nbphi);
            planes(iplane, nbtheta+i, nbphi) = T(0);
            }
          });
        }
      timers.poppush("theta resampling and deconvolution");
      auto subplanes=subarray<3>(planes, {{0, nplanes}, {nbtheta, nbtheta+ntheta_s}, {nbphi,nbphi+nphi_s}});

      MR_assert(planes.stride(2)==1, "last axis must have stride 1");
      MR_assert((planes.stride(1)&1)==0, "stride must be even");
      MR_assert((planes.stride(0)&1)==0, "stride must be even");
      MR_assert(2*(mmax+1)<=nphi_b, "aargh");
      vmav<complex<T>,3> leg_s(reinterpret_cast<complex<T> *>(&planes(0,nbtheta,nbphi-1)),
        {nplanes, ntheta_s, mmax+1}, {subplanes.stride(0)/2, subplanes.stride(1)/2, 1});
      vmav<complex<T>,3> leg_b(reinterpret_cast<complex<T> *>(&planes(0,nbtheta,nbphi-1)),
        {nplanes, ntheta_b, mmax+1}, {subplanes.stride(0)/2, subplanes.stride(1)/2, 1});
      vmav<double,1> theta({ntheta_s}, UNINITIALIZED);
      for (size_t i=0; i<ntheta_s; ++i)
        theta(i) = (i*pi)/(ntheta_s-1);
      
      vmav<size_t,1> mval({mmax+1}, UNINITIALIZED);
      for (size_t i=0; i<=mmax; ++i)
        mval(i) = i;

      auto kernel = getKernel(2*ntheta_b-2, 2*ntheta_s-2);
      ducc0::detail_sht::resample_and_convolve_theta<T>
        (leg_b, true, true, leg_s, true, true, kernel, spin, nthreads, true);
      timers.poppush("leg2alm");
      leg2alm(valm, leg_s, spin, lmax, mval, mstart, lstride, theta, nthreads, mode);
      timers.pop();
      }

    void updateAlm(vmav<complex<T>,1> &alm, vmav<T,3> &planes, SHT_mode mode) const
      {
      auto valm(alm.prepend_1());
      updateAlm(valm, planes, mode);
      }

    vmav<T,3> build_planes() const
      {
      size_t nplanes=1+(spin>0);
      auto planes_ = vmav<T,4>::build_noncritical({nplanes, Ntheta(), (Nphi()+1)/2, 2}, UNINITIALIZED);
      vmav<T,3> planes = planes_.template reinterpret<3>(
        {nplanes, Ntheta(), Nphi()}, {planes_.stride(0), planes_.stride(1), 1});
      return planes;
      }
  };

}

using detail_sphereinterpol::SphereInterpol;

}

#endif
