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
 *  Copyright (C) 2020-2021 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC0_TOTALCONVOLVE_H
#define DUCC0_TOTALCONVOLVE_H

#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>
#include <complex>
#include <cmath>
#include <mutex>
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

namespace ducc0 {

namespace detail_totalconvolve {

using namespace std;

template<typename T> class ConvolverPlan
  {
  protected:
    constexpr static auto vlen = min<size_t>(8, native_simd<T>::size());
    using Tsimd = typename simd_select<T, vlen>::type;

    size_t nthreads;
    size_t lmax, kmax;
    // _s: small grid
    // _b: oversampled grid
    // no suffix: grid with borders
    size_t nphi_s, ntheta_s, npsi_s, nphi_b, ntheta_b, npsi_b;
    double dphi, dtheta, dpsi, xdphi, xdtheta, xdpsi;

    shared_ptr<HornerKernel> kernel;
    size_t nbphi, nbtheta;
    size_t nphi, ntheta;
    double phi0, theta0;

    cmav<T,1> getKernel(size_t axlen, size_t axlen2) const
      {
      auto axlen_big = max(axlen, axlen2);
      auto axlen_small = min(axlen, axlen2);
      auto fct = kernel->corfunc(axlen_small/2+1, 1./axlen_big, nthreads);
      vmav<T,1> k2({axlen});
      mav_apply([](T &v){v=T(0);}, 1, k2);
      {
      k2(0) = T(fct[0])/axlen_small;
      size_t i=1;
      for (; 2*i<axlen_small; ++i)
        k2(2*i-1) = T(fct[i])/axlen_small;
      if (2*i==axlen_small)
        k2(2*i-1) = T(0.5)*T(fct[i])/axlen_small;
      }
      pocketfft_r<T> plan(axlen);
      plan.exec(k2.data(), T(1), false, nthreads);
      return k2;
      }

    void correct(vmav<T,2> &arr, int spin) const
      {
      T sfct = (spin&1) ? -1 : 1;
      vmav<T,2> tmp({nphi_b,nphi_s});
      // copy and extend to second half
      for (size_t j=0; j<nphi_s; ++j)
        tmp(0,j) = arr(0,j);
      for (size_t i=1, i2=nphi_s-1; i+1<ntheta_s; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp(i,j2) = arr(i,j2);
          tmp(i2,j) = sfct*tmp(i,j2);
          }
      for (size_t j=0; j<nphi_s; ++j)
        tmp(ntheta_s-1,j) = arr(ntheta_s-1,j);
      auto fct = kernel->corfunc(nphi_s/2+1, 1./nphi_b, nthreads);
      vector<T> k2(fct.size());
      for (size_t i=0; i<fct.size(); ++i) k2[i] = T(fct[i]/nphi_s);
      vfmav<T> ftmp(tmp);
      cfmav<T> ftmp0(subarray<2>(tmp, {{0, nphi_s}, {0, nphi_s}}));
      auto kern = getKernel(nphi_s, nphi_b);
      convolve_axis(ftmp0, ftmp, 0, kern, nthreads);
      cfmav<T> ftmp2(subarray<2>(tmp, {{0, ntheta_b}, {0, nphi_s}}));
      vfmav<T> farr(arr);
      convolve_axis(ftmp2, farr, 1, kern, nthreads);
      }
    void decorrect(vmav<T,2> &arr, int spin) const
      {
      T sfct = (spin&1) ? -1 : 1;
      vmav<T,2> tmp({nphi_b,nphi_s});
      auto fct = kernel->corfunc(nphi_s/2+1, 1./nphi_b, nthreads);
      vector<T> k2(fct.size());
      for (size_t i=0; i<fct.size(); ++i) k2[i] = T(fct[i]/nphi_s);
      cfmav<T> farr(arr);
      vfmav<T> ftmp2(subarray<2>(tmp, {{0, ntheta_b}, {0, nphi_s}}));
      auto kern = getKernel(nphi_b, nphi_s);
      convolve_axis(farr, ftmp2, 1, kern, nthreads);
      // extend to second half
      for (size_t i=1, i2=nphi_b-1; i+1<ntheta_b; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp(i2,j) = sfct*tmp(i,j2);
          }
      cfmav<T> ftmp(tmp);
      vfmav<T> ftmp0(subarray<2>(tmp, {{0, nphi_s}, {0, nphi_s}}));
      convolve_axis(ftmp, ftmp0, 0, kern, nthreads);
      for (size_t j=0; j<nphi_s; ++j)
        arr(0,j) = T(0.5)*tmp(0,j);
      for (size_t i=1; i+1<ntheta_s; ++i)
        for (size_t j=0; j<nphi_s; ++j)
          arr(i,j) = tmp(i,j);
      for (size_t j=0; j<nphi_s; ++j)
        arr(ntheta_s-1,j) = T(0.5)*tmp(ntheta_s-1,j);
      }

    aligned_array<uint32_t> getIdx(const cmav<T,1> &theta, const cmav<T,1> &phi, const cmav<T,1> &psi,
      size_t patch_ntheta, size_t patch_nphi, size_t itheta0, size_t iphi0, size_t supp) const
      {
      size_t nptg = theta.shape(0);
      constexpr size_t cellsize=8;
      size_t nct = patch_ntheta/cellsize+1,
             ncp = patch_nphi/cellsize+1,
             ncpsi = npsi_b/cellsize+1;
      double theta0 = (int(itheta0)-int(nbtheta))*dtheta,
             phi0 = (int(iphi0)-int(nbphi))*dphi;
      double theta_lo=theta0, theta_hi=theta_lo+(patch_ntheta+1)*dtheta;
      double phi_lo=phi0, phi_hi=phi_lo+(patch_nphi+1)*dphi;
      MR_assert(uint64_t(nct)*uint64_t(ncp)*uint64_t(ncpsi)<(uint64_t(1)<<32),
        "key space too large");

      aligned_array<uint32_t> key(nptg);
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
          auto fpsi = psi(i)*xdpsi;
          fpsi = fmodulo(fpsi, double(npsi_b));
          size_t ipsi = size_t(fpsi);
          ipsi /= cellsize;
          itheta /= cellsize;
          iphi /= cellsize;
          MR_assert(itheta<nct, "bad itheta");
          MR_assert(iphi<ncp, "bad iphi");
          key[i] = (itheta*ncp+iphi)*ncpsi+ipsi;
          }
        });
      aligned_array<uint32_t> res(key.size());
      bucket_sort(&key[0], &res[0], key.size(), ncp*nct*ncpsi, nthreads);
      return res;
      }

    template<size_t supp> class WeightHelper
      {
      public:
        static constexpr size_t vlen = Tsimd::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;
        const ConvolverPlan &plan;
        union kbuf {
          T scalar[3*nvec*vlen];
          Tsimd simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

      private:
        TemplateKernel<supp, Tsimd> tkrn;
        double mytheta0, myphi0;

      public:
        WeightHelper(const ConvolverPlan &plan_, const mav_info<3> &info, size_t itheta0, size_t iphi0)
          : plan(plan_),
            tkrn(*plan.kernel),
            mytheta0(plan.theta0+itheta0*plan.dtheta),
            myphi0(plan.phi0+iphi0*plan.dphi),
            wpsi(&buf.scalar[0]),
            wtheta(&buf.scalar[nvec*vlen]),
            wphi(&buf.simd[2*nvec]),
            jumptheta(info.stride(1))
          {
          MR_assert(info.stride(2)==1, "last axis of cube must be contiguous");
          }
        void prep(double theta, double phi, double psi)
          {
          auto ftheta = (theta-mytheta0)*plan.xdtheta-supp*0.5;
          itheta = size_t(ftheta+1);
          ftheta = -1+(itheta-ftheta)*2;
          auto fphi = (phi-myphi0)*plan.xdphi-supp*0.5;
          iphi = size_t(fphi+1);
          fphi = -1+(iphi-fphi)*2;
          auto fpsi = psi*plan.xdpsi-supp*0.5;
          fpsi = fmodulo(fpsi, double(plan.npsi_b));
          ipsi = size_t(fpsi+1);
          fpsi = -1+(ipsi-fpsi)*2;
          if (ipsi>=plan.npsi_b) ipsi-=plan.npsi_b;
          tkrn.eval3(T(fpsi), T(ftheta), T(fphi), &buf.simd[0]);
          }
        size_t itheta, iphi, ipsi;
        const T * DUCC0_RESTRICT wpsi;
        const T * DUCC0_RESTRICT wtheta;
        const Tsimd * DUCC0_RESTRICT wphi;
        ptrdiff_t jumptheta;
      };

    // prefetching distance
    static constexpr size_t pfdist=2;

    template<size_t supp> void interpolx(size_t supp_, const cmav<T,3> &cube,
      size_t itheta0, size_t iphi0, const cmav<T,1> &theta, const cmav<T,1> &phi,
      const cmav<T,1> &psi, vmav<T,1> &signal) const
      {
      if constexpr (supp>=8)
        if (supp_<=supp/2) return interpolx<supp/2>(supp_, cube, itheta0, iphi0, theta, phi, psi, signal);
      if constexpr (supp>4)
        if (supp_<supp) return interpolx<supp-1>(supp_, cube, itheta0, iphi0, theta, phi, psi, signal);
      MR_assert(supp_==supp, "requested support ou of range");

      MR_assert(cube.stride(2)==1, "last axis of cube must be contiguous");
      MR_assert(phi.shape(0)==theta.shape(0), "array shape mismatch");
      MR_assert(psi.shape(0)==theta.shape(0), "array shape mismatch");
      MR_assert(signal.shape(0)==theta.shape(0), "array shape mismatch");
      static constexpr size_t vlen = Tsimd::size();
      static constexpr size_t nvec = (supp+vlen-1)/vlen;
      MR_assert(cube.shape(0)==npsi_b, "bad psi dimension");
      auto idx = getIdx(theta, phi, psi, cube.shape(1), cube.shape(2), itheta0, iphi0, supp);

      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        WeightHelper<supp> hlp(*this, cube, itheta0, iphi0);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          if (ind+pfdist<rng.hi)
            {
            size_t i=idx[ind+pfdist];
            DUCC0_PREFETCH_R(&theta(i));
            DUCC0_PREFETCH_R(&phi(i))
            DUCC0_PREFETCH_R(&psi(i));
            DUCC0_PREFETCH_R(&signal(i));
            DUCC0_PREFETCH_W(&signal(i));
            }
          size_t i=idx[ind];
          hlp.prep(theta(i), phi(i), psi(i));
          auto ipsi = hlp.ipsi;
          const T * DUCC0_RESTRICT ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
          Tsimd res=0;
          if constexpr(nvec==1)
            {
            for (size_t ipsic=0; ipsic<supp; ++ipsic)
              {
              const T * DUCC0_RESTRICT ptr2 = ptr;
              Tsimd tres=0;
              for (size_t itheta=0; itheta<supp; ++itheta, ptr2+=hlp.jumptheta)
                tres += hlp.wtheta[itheta]*Tsimd(ptr2, element_aligned_tag());
              res += tres*hlp.wpsi[ipsic];
              if (++ipsi>=npsi_b) ipsi=0;
              ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
              }
            res *= hlp.wphi[0];
            }
          else
            {
            for (size_t ipsic=0; ipsic<supp; ++ipsic)
              {
              const T * DUCC0_RESTRICT ptr2 = ptr;
              Tsimd tres=0;
              for (size_t itheta=0; itheta<supp; ++itheta, ptr2+=hlp.jumptheta)
                for (size_t iphi=0; iphi<nvec; ++iphi)
                  tres += hlp.wtheta[itheta]*hlp.wphi[iphi]*Tsimd(ptr2+iphi*vlen,element_aligned_tag());
              res += tres*hlp.wpsi[ipsic];
              if (++ipsi>=npsi_b) ipsi=0;
              ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
              }
            }
          signal(i) = reduce(res, std::plus<>());
          }
        });
      }
    template<size_t supp> void deinterpolx(size_t supp_, vmav<T,3> &cube,
      size_t itheta0, size_t iphi0, const cmav<T,1> &theta, const cmav<T,1> &phi,
      const cmav<T,1> &psi, const cmav<T,1> &signal) const
      {
      if constexpr (supp>=8)
        if (supp_<=supp/2) return deinterpolx<supp/2>(supp_, cube, itheta0, iphi0, theta, phi, psi, signal);
      if constexpr (supp>4)
        if (supp_<supp) return deinterpolx<supp-1>(supp_, cube, itheta0, iphi0, theta, phi, psi, signal);
      MR_assert(supp_==supp, "requested support ou of range");

      MR_assert(cube.stride(2)==1, "last axis of cube must be contiguous");
      MR_assert(phi.shape(0)==theta.shape(0), "array shape mismatch");
      MR_assert(psi.shape(0)==theta.shape(0), "array shape mismatch");
      MR_assert(signal.shape(0)==theta.shape(0), "array shape mismatch");
      static constexpr size_t vlen = Tsimd::size();
      static constexpr size_t nvec = (supp+vlen-1)/vlen;
      MR_assert(cube.shape(0)==npsi_b, "bad psi dimension");
      auto idx = getIdx(theta, phi, psi, cube.shape(1), cube.shape(2), itheta0, iphi0, supp);

      constexpr size_t cellsize=16;
      size_t nct = cube.shape(1)/cellsize+10,
             ncp = cube.shape(2)/cellsize+10;
      vmav<std::mutex,2> locks({nct,ncp});

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
            DUCC0_PREFETCH_R(&psi(i));
            DUCC0_PREFETCH_R(&signal(i));
            }
          size_t i=idx[ind];
          hlp.prep(theta(i), phi(i), psi(i));
          auto ipsi = hlp.ipsi;
          T * DUCC0_RESTRICT ptr = &cube(ipsi,hlp.itheta,hlp.iphi);

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

            {
            Tsimd tmp=signal(i);
            if constexpr (nvec==1)
              {
              tmp *= hlp.wphi[0];
              for (size_t ipsic=0; ipsic<supp; ++ipsic)
                {
                auto ttmp=tmp*hlp.wpsi[ipsic];
                T * DUCC0_RESTRICT ptr2 = ptr;
                for (size_t itheta=0; itheta<supp; ++itheta, ptr2+=hlp.jumptheta)
                  {
                  Tsimd var=Tsimd(ptr2,element_aligned_tag());
                  var += ttmp*hlp.wtheta[itheta];
                  var.copy_to(ptr2,element_aligned_tag());
                  }
                if (++ipsi>=npsi_b) ipsi=0;
                ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
                }
              }
            else
              {
              for (size_t ipsic=0; ipsic<supp; ++ipsic)
                {
                auto ttmp=tmp*hlp.wpsi[ipsic];
                T * DUCC0_RESTRICT ptr2 = ptr;
                for (size_t itheta=0; itheta<supp; ++itheta)
                  {
                  auto tttmp=ttmp*hlp.wtheta[itheta];
                  for (size_t iphi=0; iphi<nvec; ++iphi)
                    {
                    Tsimd var=Tsimd(ptr2+iphi*vlen, element_aligned_tag());
                    var += tttmp*hlp.wphi[iphi];
                    var.copy_to(ptr2+iphi*vlen, element_aligned_tag());
                    }
                  ptr2 += hlp.jumptheta;
                  }
                if (++ipsi>=npsi_b) ipsi=0;
                ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
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
    double realsigma() const
      {
      return min(double(npsi_b)/(2*kmax+1),
                 min(double(nphi_b)/(2*lmax+1), double(ntheta_b)/(lmax+1)));
      }

  public:
    ConvolverPlan(size_t lmax_, size_t kmax_, double sigma, double epsilon,
      size_t nthreads_)
      : nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        lmax(lmax_),
        kmax(kmax_),
        nphi_s(2*good_size_real(lmax+1)),
        ntheta_s(nphi_s/2+1),
        npsi_s(kmax*2+1),
        nphi_b(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*sigma/2.)))),
        ntheta_b(nphi_b/2+1),
        npsi_b(size_t(npsi_s*sigma+0.99999)),
        dphi(2*pi/nphi_b),
        dtheta(pi/(ntheta_b-1)),
        dpsi(2*pi/npsi_b),
        xdphi(1./dphi),
        xdtheta(1./dtheta),
        xdpsi(1./dpsi),
        kernel(selectKernel<T>(realsigma(), epsilon/3.)),
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
    size_t Kmax() const { return kmax; }
    size_t Ntheta() const { return ntheta; }
    size_t Nphi() const { return nphi; }
    size_t Npsi() const { return npsi_b; }

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

    void getPlane(const cmav<complex<T>,2> &vslm, const cmav<complex<T>,2> &vblm,
      size_t mbeam, vmav<T,3> &planes) const
      {
      size_t nplanes=1+(mbeam>0);
      auto ncomp = vslm.shape(0);
      MR_assert(ncomp>0, "need at least one component");
      MR_assert(vblm.shape(0)==ncomp, "inconsistent slm and blm vectors");
      Alm_Base islm(lmax, lmax), iblm(lmax, kmax);
      MR_assert(islm.Num_Alms()==vslm.shape(1), "bad array dimension");
      MR_assert(iblm.Num_Alms()==vblm.shape(1), "bad array dimension");
      MR_assert(planes.conformable({nplanes, Ntheta(), Nphi()}), "bad planes shape");
      MR_assert(mbeam <= kmax, "mbeam too high");

      vector<T> lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=T(std::sqrt(4*pi/(2*i+1.)));

      Alm_Base base(lmax, lmax);
      vmav<complex<T>,2> aarr({nplanes,base.Num_Alms()});
      for (size_t m=0; m<=lmax; ++m)
        for (size_t l=m; l<=lmax; ++l)
          {
          aarr(0, base.index(l,m))=0.;
          if (mbeam>0)
            aarr(1, base.index(l,m))=0.;
          if (l>=mbeam)
            {
            auto norm = (mbeam>0) ? -lnorm[l] : lnorm[l];
            for (size_t i=0; i<ncomp; ++i)
              {
              auto tmp = vblm(i,iblm.index(l,mbeam))*norm;
              aarr(0,base.index(l,m)) += vslm(i,islm.index(l,m))*tmp.real();
              if (mbeam>0)
                aarr(1,base.index(l,m)) += vslm(i,islm.index(l,m))*tmp.imag();
              }
            }
          }
      auto subplanes=subarray<3>(planes,{{0, nplanes}, {nbtheta, nbtheta+ntheta_s}, {nbphi, nbphi+nphi_s}});
      synthesis_2d(aarr, subplanes, mbeam, lmax, lmax, "CC", nthreads);
      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        auto m = subarray<2>(planes, {{iplane},{nbtheta, nbtheta+ntheta_b}, {nbphi, nbphi+nphi_b}});
        correct(m,mbeam);
        }

      // fill border regions
      T fct = (mbeam&1) ? -1 : 1;
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
      }

    void getPlane(const cmav<complex<T>,1> &slm, const cmav<complex<T>,1> &blm,
      size_t mbeam, vmav<T,3> &planes) const
      {
      cmav<complex<T>,2> vslm(&slm(0), {1,slm.shape(0)}, {0,slm.stride(0)});
      cmav<complex<T>,2> vblm(&blm(0), {1,blm.shape(0)}, {0,blm.stride(0)});
      getPlane(vslm, vblm, mbeam, planes);
      }

    void interpol(const cmav<T,3> &cube, size_t itheta0,
      size_t iphi0, const cmav<T,1> &theta, const cmav<T,1> &phi,
      const cmav<T,1> &psi, vmav<T,1> &signal) const
      {
      constexpr size_t maxsupp = is_same<T, double>::value ? 16 : 8;
      interpolx<maxsupp>(kernel->support(), cube, itheta0, iphi0, theta, phi, psi, signal);
      }

    void deinterpol(vmav<T,3> &cube, size_t itheta0,
      size_t iphi0, const cmav<T,1> &theta, const cmav<T,1> &phi,
      const cmav<T,1> &psi, const cmav<T,1> &signal) const
      {
      constexpr size_t maxsupp = is_same<T, double>::value ? 16 : 8;
      deinterpolx<maxsupp>(kernel->support(), cube, itheta0, iphi0, theta, phi, psi, signal);
      }

    void updateSlm(vmav<complex<T>,2> &vslm, const cmav<complex<T>,2> &vblm,
      size_t mbeam, vmav<T,3> &planes) const
      {
      size_t nplanes=1+(mbeam>0);
      auto ncomp = vslm.shape(0);
      MR_assert(ncomp>0, "need at least one component");
      MR_assert(vblm.shape(0)==ncomp, "inconsistent slm and blm vectors");
      Alm_Base islm(lmax, lmax), iblm(lmax, kmax);
      MR_assert(islm.Num_Alms()==vslm.shape(1), "bad array dimension");
      MR_assert(iblm.Num_Alms()==vblm.shape(1), "bad array dimension");
      MR_assert(planes.conformable({nplanes, Ntheta(), Nphi()}), "bad planes shape");
      MR_assert(mbeam <= kmax, "mbeam too high");

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
            T fct = (mbeam&1) ? -1 : 1;
            if (j2>=nphi_b) j2-=nphi_b;
            planes(iplane,nbtheta+1+i,j+nbphi) += fct*planes(iplane,nbtheta-1-i,j2+nbphi);
            planes(iplane,nbtheta+ntheta_b-2-i, j+nbphi) += fct*planes(iplane,nbtheta+ntheta_b+i,j2+nbphi);
            }

        // special treatment for poles
        for (size_t j=0,j2=nphi_b/2; j<nphi_b/2; ++j,++j2)
          {
          T fct = (mbeam&1) ? -1 : 1;
          if (j2>=nphi_b) j2-=nphi_b;
          T tval = planes(iplane,nbtheta,j+nbphi) + fct*planes(iplane,nbtheta,j2+nbphi);
          planes(iplane,nbtheta,j+nbphi) = tval;
          planes(iplane,nbtheta,j2+nbphi) = fct*tval;
          tval = planes(iplane,nbtheta+ntheta_b-1,j+nbphi) + fct*planes(iplane,nbtheta+ntheta_b-1,j2+nbphi);
          planes(iplane,nbtheta+ntheta_b-1,j+nbphi) = tval;
          planes(iplane,nbtheta+ntheta_b-1,j2+nbphi) = fct*tval;
          }
        }

      vector<T>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=T(std::sqrt(4*pi/(2*i+1.)));

      Alm_Base base(lmax,lmax);

      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        auto m = subarray<2>(planes, {{iplane}, {nbtheta, nbtheta+ntheta_b}, {nbphi,nbphi+nphi_b}});
        decorrect(m,mbeam);
        }
      auto subplanes=subarray<3>(planes, {{0, nplanes}, {nbtheta, nbtheta+ntheta_s}, {nbphi,nbphi+nphi_s}});

      vmav<complex<T>,2> aarr({nplanes, base.Num_Alms()});
      adjoint_synthesis_2d(aarr, subplanes, mbeam, lmax, lmax, "CC", nthreads);
      for (size_t m=0; m<=lmax; ++m)
        for (size_t l=m; l<=lmax; ++l)
          if (l>=mbeam)
            for (size_t i=0; i<ncomp; ++i)
              {
              auto tmp = vblm(i,iblm.index(l,mbeam))*lnorm[l] * T((mbeam==0) ? 1 : (-2));
              vslm(i,islm.index(l,m)) += aarr(0,base.index(l,m))*tmp.real();
              if (mbeam>0)
                vslm(i,islm.index(l,m)) += aarr(1,base.index(l,m))*tmp.imag();
              }
      }

    void updateSlm(vmav<complex<T>,1> &slm, const cmav<complex<T>,1> &blm,
      size_t mbeam, vmav<T,3> &planes) const
      {
      vmav<complex<T>,2> vslm(&slm(0), {1,slm.shape(0)}, {0,slm.stride(0)});
      cmav<complex<T>,2> vblm(&blm(0), {1,blm.shape(0)}, {0,blm.stride(0)});
      updateSlm(vslm, vblm, mbeam, planes);
      }

    void prepPsi(vmav<T,3> &subcube) const
      {
      MR_assert(subcube.shape(0)==npsi_b, "bad psi dimension");
      auto newpart = subarray<3>(subcube, {{npsi_s, MAXIDX}, {}, {}});
      mav_apply([](T &v){v=T(0);}, nthreads, newpart);
      auto fct = kernel->corfunc(npsi_s/2+1, 1./npsi_b, nthreads);
      for (size_t k=0; k<npsi_s; ++k)
        {
        auto factor = T(fct[(k+1)/2]);
        for (size_t i=0; i<subcube.shape(1); ++i)
          for (size_t j=0; j<subcube.shape(2); ++j)
            subcube(k,i,j) *= factor;
        }
      vfmav<T> fsubcube(subcube);
      r2r_fftpack(fsubcube, fsubcube, {0}, false, true, T(1), nthreads);
      }

    void deprepPsi(vmav<T,3> &subcube) const
      {
      MR_assert(subcube.shape(0)==npsi_b, "bad psi dimension");
      vfmav<T> fsubcube(subcube);
      r2r_fftpack(fsubcube, fsubcube, {0}, true, false, T(1), nthreads);
      auto fct = kernel->corfunc(npsi_s/2+1, 1./npsi_b, nthreads);
      for (size_t k=0; k<npsi_s; ++k)
        {
        auto factor = T(fct[(k+1)/2]);
        for (size_t i=0; i<subcube.shape(1); ++i)
          for (size_t j=0; j<subcube.shape(2); ++j)
            subcube(k,i,j) *= factor;
        }
      }
  };

}

using detail_totalconvolve::ConvolverPlan;


}

#endif
