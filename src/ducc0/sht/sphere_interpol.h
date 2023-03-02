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

namespace ducc0 {

namespace detail_sphereinterpol {

using namespace std;

/*! Selects the most efficient combination of gridding kernel and oversampled
    grid size for the provided problem parameters. */
template<typename Tcalc> auto findNufftParameters(double epsilon,
  double sigma_min, double sigma_max, const vector<size_t> &dims,
  size_t npoints, size_t nthreads)
  {
  constexpr static auto vlen = min<size_t>(8, native_simd<Tcalc>::size());
  auto ndim = dims.size();
  auto idx = getAvailableKernels<Tcalc>(epsilon, ndim, sigma_min, sigma_max);
  double mincost = 1e300;
  constexpr double nref_fft=2048;
  constexpr double costref_fft=0.0693;
  vector<size_t> bigdims(ndim, 0);
  size_t minidx=~(size_t(0));
  for (size_t i=0; i<idx.size(); ++i)
    {
    const auto &krn(getKernel(idx[i]));
    auto supp = krn.W;
    auto nvec = (supp+vlen-1)/vlen;
    auto ofactor = krn.ofactor;
    vector<size_t> lbigdims(ndim,0);
    double gridsize=1;
    for (size_t idim=0; idim<ndim; ++idim)
      {
      lbigdims[idim] = 2*good_size_complex(size_t(dims[idim]*ofactor*0.5)+1);
      lbigdims[idim] = max<size_t>(lbigdims[idim], 16);
      gridsize *= lbigdims[idim];
      }
    double logterm = log(gridsize)/log(nref_fft*nref_fft);
    double fftcost = gridsize/(nref_fft*nref_fft)*logterm*costref_fft;
    size_t kernelpoints = nvec*vlen;
    for (size_t idim=0; idim+1<ndim; ++idim)
      kernelpoints*=supp;
    double gridcost = 2.2e-10*npoints*(kernelpoints + (ndim*nvec*(supp+3)*vlen));
    // FIXME: heuristics could be improved
    gridcost /= nthreads;  // assume perfect scaling for now
    constexpr double max_fft_scaling = 6;
    constexpr double scaling_power=2;
    auto sigmoid = [](double x, double m, double s)
      {
      auto x2 = x-1;
      auto m2 = m-1;
      return 1.+x2/pow((1.+pow(x2/m2,s)),1./s);
      };
    fftcost /= sigmoid(nthreads, max_fft_scaling, scaling_power);
    double cost = fftcost+gridcost;
    if (cost<mincost)
      {
      mincost=cost;
      bigdims=lbigdims;
      minidx = idx[i];
      }
    }
  return minidx;
  }

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

    cmav<T,1> getKernel(size_t axlen, size_t axlen2) const
      {
      auto axlen_big = max(axlen, axlen2);
      auto axlen_small = min(axlen, axlen2);
      auto fct = kernel->corfunc(axlen_small/2+1, 1./axlen_big, nthreads);
      vmav<T,1> k2({axlen}, UNINITIALIZED);
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
      vmav<T,2> tmp({2*ntheta_b-2,nphi_s}, UNINITIALIZED);
      // copy and extend to second half
      for (size_t j=0; j<nphi_s; ++j)
        tmp(0,j) = arr(0,j);
      for (size_t i=1, i2=2*ntheta_s-2-1; i+1<ntheta_s; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp(i,j2) = arr(i,j2);
          tmp(i2,j) = sfct*tmp(i,j2);
          }
      for (size_t j=0; j<nphi_s; ++j)
        tmp(ntheta_s-1,j) = arr(ntheta_s-1,j);

      {
      vfmav<T> ftmp(tmp);
      cfmav<T> ftmp0(subarray<2>(tmp, {{0, (2*ntheta_s-2)}, {}}));
      auto kern = getKernel(2*ntheta_s-2, 2*ntheta_b-2);
      convolve_axis(ftmp0, ftmp, 0, kern, nthreads);
      }
      {
      cfmav<T> ftmp2(subarray<2>(tmp, {{0, ntheta_b}, {0, nphi_s}}));
      vfmav<T> farr(arr);
      auto kern = getKernel(nphi_s, nphi_b);
      convolve_axis(ftmp2, farr, 1, kern, nthreads);
      }
      }
    void decorrect(vmav<T,2> &arr, int spin) const
      {
      T sfct = (spin&1) ? -1 : 1;
      vmav<T,2> tmp({2*ntheta_b-2,nphi_s}, UNINITIALIZED);
      cfmav<T> farr(arr);
      vfmav<T> ftmp2(subarray<2>(tmp, {{0, ntheta_b}, {0, nphi_s}}));
      {
      auto kern = getKernel(nphi_b, nphi_s);
      convolve_axis(farr, ftmp2, 1, kern, nthreads);
      }
      // extend to second half
      for (size_t i=1, i2=2*ntheta_b-2-1; i+1<ntheta_b; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp(i2,j) = sfct*tmp(i,j2);
          }
      cfmav<T> ftmp(tmp);
      vfmav<T> ftmp0(subarray<2>(tmp, {{0, 2*ntheta_s-2}, {0, nphi_s}}));
      {
      auto kern = getKernel(2*ntheta_b-2, 2*ntheta_s-2);
      convolve_axis(ftmp, ftmp0, 0, kern, nthreads);
      }
      for (size_t j=0; j<nphi_s; ++j)
        arr(0,j) = T(0.5)*tmp(0,j);
      for (size_t i=1; i+1<ntheta_s; ++i)
        for (size_t j=0; j<nphi_s; ++j)
          arr(i,j) = tmp(i,j);
      for (size_t j=0; j<nphi_s; ++j)
        arr(ntheta_s-1,j) = T(0.5)*tmp(ntheta_s-1,j);
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
          if constexpr(nvec==1)
            {
            for (size_t icomp=0; icomp<ncomp; ++icomp)
              {
              const T * DUCC0_RESTRICT ptr = &cube(icomp, hlp.itheta,hlp.iphi);
              const T * DUCC0_RESTRICT ptr2 = ptr;
              Tsimd tres=0;
              for (size_t itheta=0; itheta<supp; ++itheta, ptr2+=hlp.jumptheta)
                tres += hlp.wtheta[itheta]*Tsimd(ptr2, element_aligned_tag());
              signal(icomp, i) = reduce(tres*hlp.wphi[0], std::plus<>());
              }
            }
          else
            {
            for (size_t icomp=0; icomp<ncomp; ++icomp)
              {
              const T * DUCC0_RESTRICT ptr = &cube(icomp, hlp.itheta,hlp.iphi);
              const T * DUCC0_RESTRICT ptr2 = ptr;
              Tsimd tres=0;
              for (size_t itheta=0; itheta<supp; ++itheta, ptr2+=hlp.jumptheta)
                for (size_t iphi=0; iphi<nvec; ++iphi)
                  tres += hlp.wtheta[itheta]*hlp.wphi[iphi]*Tsimd(ptr2+iphi*vlen,element_aligned_tag());
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

            {
            if constexpr (nvec==1)
              {
              for (size_t icomp=0; icomp<ncomp; ++icomp)
                {
                Tsimd tmp=signal(icomp, i);
                tmp *= hlp.wphi[0];
                T * DUCC0_RESTRICT ptr = &cube(icomp,hlp.itheta,hlp.iphi);
                T * DUCC0_RESTRICT ptr2 = ptr;
                for (size_t itheta=0; itheta<supp; ++itheta, ptr2+=hlp.jumptheta)
                  {
                  Tsimd var=Tsimd(ptr2,element_aligned_tag());
                  var += tmp*hlp.wtheta[itheta];
                  var.copy_to(ptr2,element_aligned_tag());
                  }
                }
              }
            else
              {
              for (size_t icomp=0; icomp<ncomp; ++icomp)
                {
                Tsimd tmp=signal(icomp, i);
                T * DUCC0_RESTRICT ptr = &cube(icomp,hlp.itheta,hlp.iphi);
                T * DUCC0_RESTRICT ptr2 = ptr;
                for (size_t itheta=0; itheta<supp; ++itheta)
                  {
                  auto ttmp=tmp*hlp.wtheta[itheta];
                  for (size_t iphi=0; iphi<nvec; ++iphi)
                    {
                    Tsimd var=Tsimd(ptr2+iphi*vlen, element_aligned_tag());
                    var += ttmp*hlp.wphi[iphi];
                    var.copy_to(ptr2+iphi*vlen, element_aligned_tag());
                    }
                  ptr2 += hlp.jumptheta;
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
        kernel_index(findNufftParameters<T>(epsilon, sigma_min, sigma_max,
          {(2*ntheta_s-2), nphi_s}, npoints, nthreads)),
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

    void getPlane(const cmav<complex<T>,2> &vslm, vmav<T,3> &planes) const
      {
      size_t nplanes=1+(spin>0);
      auto ncomp = vslm.shape(0);
      MR_assert(ncomp==nplanes, "number of components mismatch");
      Alm_Base islm(lmax, mmax);
      MR_assert(islm.Num_Alms()==vslm.shape(1), "bad array dimension");
      MR_assert(planes.conformable({nplanes, Ntheta(), Nphi()}), "bad planes shape");

      auto subplanes=subarray<3>(planes,{{}, {nbtheta, nbtheta+ntheta_s}, {nbphi, nbphi+nphi_s}});
      synthesis_2d(vslm, subplanes, spin, lmax, mmax, "CC", nthreads);
      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        auto m = subarray<2>(planes, {{iplane},{nbtheta, nbtheta+ntheta_b}, {nbphi, nbphi+nphi_b}});
        correct(m,spin);
        }

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
      }

    void getPlane(const cmav<complex<T>,1> &slm, vmav<T,3> &planes) const
      {
      cmav<complex<T>,2> vslm(&slm(0), {1,slm.shape(0)}, {0,slm.stride(0)});
      getPlane(vslm, planes);
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

    void updateSlm(vmav<complex<T>,2> &vslm, vmav<T,3> &planes) const
      {
      size_t nplanes=1+(spin>0);
      auto ncomp = vslm.shape(0);
      MR_assert(ncomp>0, "need at least one component");
      Alm_Base islm(lmax, mmax);
      MR_assert(islm.Num_Alms()==vslm.shape(1), "bad array dimension");
      MR_assert(planes.conformable({nplanes, Ntheta(), Nphi()}), "bad planes shape");

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

        // special treatment for poles
        for (size_t j=0,j2=nphi_b/2; j<nphi_b/2; ++j,++j2)
          {
          T fct = (spin&1) ? -1 : 1;
          if (j2>=nphi_b) j2-=nphi_b;
          T tval = planes(iplane,nbtheta,j+nbphi) + fct*planes(iplane,nbtheta,j2+nbphi);
          planes(iplane,nbtheta,j+nbphi) = tval;
          planes(iplane,nbtheta,j2+nbphi) = fct*tval;
          tval = planes(iplane,nbtheta+ntheta_b-1,j+nbphi) + fct*planes(iplane,nbtheta+ntheta_b-1,j2+nbphi);
          planes(iplane,nbtheta+ntheta_b-1,j+nbphi) = tval;
          planes(iplane,nbtheta+ntheta_b-1,j2+nbphi) = fct*tval;
          }
        }

      for (size_t iplane=0; iplane<nplanes; ++iplane)
        {
        auto m = subarray<2>(planes, {{iplane}, {nbtheta, nbtheta+ntheta_b}, {nbphi,nbphi+nphi_b}});
        decorrect(m,spin);
        }
      auto subplanes=subarray<3>(planes, {{0, nplanes}, {nbtheta, nbtheta+ntheta_s}, {nbphi,nbphi+nphi_s}});

      adjoint_synthesis_2d(vslm, subplanes, spin, lmax, mmax, "CC", nthreads);
      }

    void updateSlm(vmav<complex<T>,1> &slm, vmav<T,3> &planes) const
      {
      vmav<complex<T>,2> vslm(slm.data(), {1,slm.shape(0)}, {0,slm.stride(0)});
      updateSlm(vslm, planes);
      }
  };

}

using detail_sphereinterpol::SphereInterpol;

}

#endif
