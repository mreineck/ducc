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

#include <vector>
#include <complex>
#include <cmath>
#include "ducc0/math/constants.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/bucket_sort.h"
#include "ducc0/sharp/sharp.h"
#include "ducc0/sharp/sharp_almhelpers.h"
#include "ducc0/sharp/sharp_geomhelpers.h"
#include "python/alm.h"
#include "ducc0/math/fft.h"
#include "ducc0/math/math_utils.h"

namespace ducc0 {

namespace detail_fft {

using std::vector;

template<typename T, typename T0> aligned_array<T> alloc_tmp_conv
  (const fmav_info &info, size_t axis, size_t len)
  {
  auto othersize = info.size()/info.shape(axis);
  constexpr auto vlen = native_simd<T0>::size();
  return aligned_array<T>(len*std::min(vlen, othersize));
  }

template<typename Tplan, typename T, typename T0, typename Exec>
DUCC0_NOINLINE void general_convolve(const fmav<T> &in, fmav<T> &out,
  const size_t axis, const vector<T0> &kernel, size_t nthreads,
  const Exec &exec)
  {
  std::unique_ptr<Tplan> plan1, plan2;

  size_t l_in=in.shape(axis), l_out=out.shape(axis);
  size_t l_min=std::min(l_in, l_out), l_max=std::max(l_in, l_out);
  MR_assert(kernel.size()==l_min/2+1, "bad kernel size");
  plan1 = std::make_unique<Tplan>(l_in);
  plan2 = std::make_unique<Tplan>(l_out);

  execParallel(
    util::thread_count(nthreads, in, axis, native_simd<T0>::size()),
    [&](Scheduler &sched) {
      constexpr auto vlen = native_simd<T0>::size();
      auto storage = alloc_tmp_conv<T,T0>(in, axis, l_max);
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef DUCC0_NO_SIMD
      if constexpr (vlen>1)
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          auto tdatav = reinterpret_cast<add_vec_t<T, vlen> *>(storage.data());
          exec(it, in, out, tdatav, *plan1, *plan2, kernel);
          }
      if constexpr (simd_exists<T,vlen/2>)
        if (it.remaining()>=vlen/2)
          {
          it.advance(vlen/2);
          auto tdatav = reinterpret_cast<add_vec_t<T, vlen/2> *>(storage.data());
          exec(it, in, out, tdatav, *plan1, *plan2, kernel);
          }
      if constexpr (simd_exists<T,vlen/4>)
        if (it.remaining()>=vlen/4)
          {
          it.advance(vlen/4);
          auto tdatav = reinterpret_cast<add_vec_t<T, vlen/4> *>(storage.data());
          exec(it, in, out, tdatav, *plan1, *plan2, kernel);
          }
#endif
      while (it.remaining()>0)
        {
        it.advance(1);
        auto buf = reinterpret_cast<T *>(storage.data());
        exec(it, in, out, buf, *plan1, *plan2, kernel);
        }
    });  // end of parallel region
  }

struct ExecConvR1
  {
  template <typename T0, typename T, typename Titer> void operator() (
    const Titer &it, const fmav<T0> &in, fmav<T0> &out,
    T * buf, const pocketfft_r<T0> &plan1, const pocketfft_r<T0> &plan2,
    const vector<T0> &kernel) const
    {
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out);
    copy_input(it, in, buf);
    plan1.exec(buf, T0(1), true);
    for (size_t i=0; i<l_min; ++i) buf[i]*=kernel[(i+1)/2];
    for (size_t i=l_in; i<l_out; ++i) buf[i] = T(0);
    plan2.exec(buf, T0(1), false);
    copy_output(it, buf, out);
    }
  };

template<typename T> void convolve_1d(const fmav<T> &in,
  fmav<T> &out, size_t axis, const vector<T> &kernel, size_t nthreads=1)
  {
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.cdata()==out.cdata())
    MR_assert(in.stride()==out.stride(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  MR_assert(!((in.shape(axis)&1) || (out.shape(axis)&1)),
    "input and output axis lengths must be even");
  if (in.size()==0) return;
  general_convolve<pocketfft_r<T>>(in, out, axis, kernel, nthreads,
    ExecConvR1());
  }

}

using detail_fft::convolve_1d;

namespace detail_totalconvolve {

using namespace std;

template<typename T> class ConvolverPlan
  {
  protected:
    constexpr static auto vlen = min<size_t>(8, native_simd<T>::size());
    using Tsimd = simd<T, vlen>;

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

    void correct(mav<T,2> &arr, int spin) const
      {
      T sfct = (spin&1) ? -1 : 1;
      mav<T,2> tmp({nphi_b,nphi_s});
      // copy and extend to second half
      for (size_t j=0; j<nphi_s; ++j)
        tmp.v(0,j) = arr(0,j);
      for (size_t i=1, i2=nphi_s-1; i+1<ntheta_s; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp.v(i,j2) = arr(i,j2);
          tmp.v(i2,j) = sfct*tmp(i,j2);
          }
      for (size_t j=0; j<nphi_s; ++j)
        tmp.v(ntheta_s-1,j) = arr(ntheta_s-1,j);
      auto fct = kernel->corfunc(nphi_s/2+1, 1./nphi_b, nthreads);
      vector<T> k2(fct.size());
      for (size_t i=0; i<fct.size(); ++i) k2[i] = T(fct[i]/nphi_s);
      fmav<T> ftmp(tmp);
      fmav<T> ftmp0(subarray<2>(tmp, {0,0},{nphi_s, nphi_s}));
      convolve_1d(ftmp0, ftmp, 0, k2, nthreads);
      fmav<T> ftmp2(subarray<2>(tmp, {0,0},{ntheta_b, nphi_s}));
      fmav<T> farr(arr);
      convolve_1d(ftmp2, farr, 1, k2, nthreads);
      }
    void decorrect(mav<T,2> &arr, int spin) const
      {
      T sfct = (spin&1) ? -1 : 1;
      mav<T,2> tmp({nphi_b,nphi_s});
      auto fct = kernel->corfunc(nphi_s/2+1, 1./nphi_b, nthreads);
      vector<T> k2(fct.size());
      for (size_t i=0; i<fct.size(); ++i) k2[i] = T(fct[i]/nphi_s);
      fmav<T> farr(arr);
      fmav<T> ftmp2(subarray<2>(tmp, {0,0},{ntheta_b, nphi_s}));
      convolve_1d(farr, ftmp2, 1, k2, nthreads);
      // extend to second half
      for (size_t i=1, i2=nphi_b-1; i+1<ntheta_b; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp.v(i2,j) = sfct*tmp(i,j2);
          }
      fmav<T> ftmp(tmp);
      fmav<T> ftmp0(subarray<2>(tmp, {0,0},{nphi_s, nphi_s}));
      convolve_1d(ftmp, ftmp0, 0, k2, nthreads);
      for (size_t j=0; j<nphi_s; ++j)
        arr.v(0,j) = T(0.5)*tmp(0,j);
      for (size_t i=1; i+1<ntheta_s; ++i)
        for (size_t j=0; j<nphi_s; ++j)
          arr.v(i,j) = tmp(i,j);
      for (size_t j=0; j<nphi_s; ++j)
        arr.v(ntheta_s-1,j) = T(0.5)*tmp(ntheta_s-1,j);
      }

    aligned_array<uint32_t> getIdx(const mav<T,1> &theta, const mav<T,1> &phi, const mav<T,1> &psi,
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
      MR_assert(nct*ncp*ncpsi<(size_t(1)<<32), "key space too large");

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

    template<size_t supp> void interpolx(const mav<T,3> &cube,
      size_t itheta0, size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, mav<T,1> &signal) const
      {
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
            DUCC0_PREFETCH_R(&signal.v(i));
            DUCC0_PREFETCH_W(&signal.v(i));
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
                tres += hlp.wtheta[itheta]*Tsimd::loadu(ptr2);
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
                  tres += hlp.wtheta[itheta]*hlp.wphi[iphi]*Tsimd::loadu(ptr2+iphi*vlen);
              res += tres*hlp.wpsi[ipsic];
              if (++ipsi>=npsi_b) ipsi=0;
              ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
              }
            }
          signal.v(i) = reduce(res, std::plus<>());
          }
        });
      }
    template<size_t supp> void deinterpolx(mav<T,3> &cube,
      size_t itheta0, size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, const mav<T,1> &signal) const
      {
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
      mav<std::mutex,2> locks({nct,ncp});

      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        size_t b_theta=99999999999999, b_phi=9999999999999999;
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
          T * DUCC0_RESTRICT ptr = &cube.v(ipsi,hlp.itheta,hlp.iphi);

          size_t b_theta_new = hlp.itheta/cellsize,
                 b_phi_new = hlp.iphi/cellsize;
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
                  Tsimd var=Tsimd::loadu(ptr2);
                  var += ttmp*hlp.wtheta[itheta];
                  var.storeu(ptr2);
                  }
                if (++ipsi>=npsi_b) ipsi=0;
                ptr = &cube.v(ipsi,hlp.itheta,hlp.iphi);
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
                    Tsimd var=Tsimd::loadu(ptr2+iphi*vlen);
                    var += tttmp*hlp.wphi[iphi];
                    var.storeu(ptr2+iphi*vlen);
                    }
                  ptr2 += hlp.jumptheta;
                  }
                if (++ipsi>=npsi_b) ipsi=0;
                ptr = &cube.v(ipsi,hlp.itheta,hlp.iphi);
                }
              }
            }
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

    void getPlane(const mav<complex<T>,2> &vslm, const mav<complex<T>,2> &vblm,
      size_t mbeam, mav<T,2> &re, mav<T,2> &im) const
      {
      auto ncomp = vslm.shape(1);
      MR_assert(ncomp>0, "need at least one component");
      MR_assert(vblm.shape(1)==ncomp, "inconsistent slm and blm vectors");
      Alm_Base islm(lmax, lmax), iblm(lmax, kmax);
      MR_assert(islm.Num_Alms()==vslm.shape(0), "bad array dimenion");
      MR_assert(iblm.Num_Alms()==vblm.shape(0), "bad array dimenion");
      MR_assert(re.conformable({Ntheta(), Nphi()}), "bad re shape");
      if (mbeam>0)
        {
        MR_assert(re.shape()==im.shape(), "re and im must have identical shape");
        MR_assert(re.stride()==im.stride(), "re and im must have identical strides");
        }
      MR_assert(mbeam <= kmax, "mbeam too high");

      auto ginfo = sharp_make_cc_geom_info(ntheta_s,nphi_s,0.,re.stride(1),re.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      vector<T> lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=T(std::sqrt(4*pi/(2*i+1.)));

      if (mbeam==0)
        {
        Alm<complex<T>> a1(lmax, lmax);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            {
            a1(l,m) = vslm(islm.index(l,m),0)*vblm(iblm.index(l,0),0).real()*lnorm[l];
            for (size_t i=1; i<ncomp; ++i)
              a1(l,m) += vslm(islm.index(l,m),i)*vblm(iblm.index(l,0),i).real()*lnorm[l];
            }
        auto m1 = subarray<2>(re, {nbtheta,nbphi},{ntheta_b,nphi_b});
        sharp_alm2map(a1.Alms().cdata(), m1.vdata(), *ginfo, *ainfo, 0, nthreads);
        correct(m1,0);
        }
      else
        {
        Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            {
            a1(l,m)=a2(l,m)=0.;
            if (l>=mbeam)
              for (size_t i=0; i<ncomp; ++i)
                {
                auto tmp = vblm(iblm.index(l,mbeam),i)*(-lnorm[l]);
                a1(l,m) += vslm(islm.index(l,m),i)*tmp.real();
                a2(l,m) += vslm(islm.index(l,m),i)*tmp.imag();
                }
            }
        auto m1 = subarray<2>(re, {nbtheta,nbphi},{ntheta_b,nphi_b});
        auto m2 = subarray<2>(im, {nbtheta,nbphi},{ntheta_b,nphi_b});
        sharp_alm2map_spin(mbeam, a1.Alms().cdata(), a2.Alms().cdata(),
          m1.vdata(), m2.vdata(), *ginfo, *ainfo, 0, nthreads);
        correct(m1,mbeam);
        correct(m2,mbeam);
        }
      // fill border regions
      T fct = (mbeam&1) ? -1 : 1;
      for (size_t i=0; i<nbtheta; ++i)
        for (size_t j=0, j2=nphi_b/2; j<nphi_b; ++j,++j2)
          {
          if (j2>=nphi_b) j2-=nphi_b;
          for (size_t l=0; l<re.shape(1); ++l)
            {
            re.v(nbtheta-1-i,j2+nbphi) = fct*re(nbtheta+1+i,j+nbphi);
            re.v(nbtheta+ntheta_b+i,j2+nbphi) = fct*re(nbtheta+ntheta_b-2-i,j+nbphi);
            }
          if (mbeam>0)
            {
            im.v(nbtheta-1-i,j2+nbphi) = fct*im(nbtheta+1+i,j+nbphi);
            im.v(nbtheta+ntheta_b+i,j2+nbphi) = fct*im(nbtheta+ntheta_b-2-i,j+nbphi);
            }
          }
      for (size_t i=0; i<ntheta; ++i)
        {
        for (size_t j=0; j<nbphi; ++j)
          {
          re.v(i,j) = re(i,j+nphi_b);
          re.v(i,j+nphi_b+nbphi) = re(i,j+nbphi);
          if (mbeam>0)
            {
            im.v(i,j) = im(i,j+nphi_b);
            im.v(i,j+nphi_b+nbphi) = im(i,j+nbphi);
            }
          }
        // SIMD buffer
        for (size_t j=0; j<vlen; ++j)
          {
          re.v(i, nphi-vlen+j) = T(0);
          if (mbeam>0)
            im.v(i, nphi-vlen+j) = T(0);
          }
        }
      }
    void getPlane(const mav<complex<T>,1> &slm, const mav<complex<T>,1> &blm,
      size_t mbeam, mav<T,2> &re, mav<T,2> &im) const
      {
      mav<complex<T>,2> vslm(&slm(0), {slm.shape(0),1}, {slm.stride(0),0});
      mav<complex<T>,2> vblm(&blm(0), {blm.shape(0),1}, {blm.stride(0),0});
      getPlane(vslm, vblm, mbeam, re, im);
      }

    void interpol(const mav<T,3> &cube, size_t itheta0,
      size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, mav<T,1> &signal) const
      {
      if constexpr(is_same<T,double>::value)
        switch(kernel->support())
          {
          case  9: interpolx< 9>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 10: interpolx<10>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 11: interpolx<11>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 12: interpolx<12>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 13: interpolx<13>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 14: interpolx<14>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 15: interpolx<15>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 16: interpolx<16>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          }
      switch(kernel->support())
        {
        case 4: interpolx<4>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 5: interpolx<5>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 6: interpolx<6>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 7: interpolx<7>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 8: interpolx<8>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        default: MR_fail("must not happen");
        }
      }

    void deinterpol(mav<T,3> &cube, size_t itheta0,
      size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, const mav<T,1> &signal) const
      {
      if constexpr(is_same<T,double>::value)
        switch(kernel->support())
          {
          case  9: deinterpolx< 9>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 10: deinterpolx<10>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 11: deinterpolx<11>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 12: deinterpolx<12>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 13: deinterpolx<13>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 14: deinterpolx<14>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 15: deinterpolx<15>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          case 16: deinterpolx<16>(cube, itheta0, iphi0, theta, phi, psi, signal); return;
          }
      switch(kernel->support())
        {
        case 4: deinterpolx<4>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 5: deinterpolx<5>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 6: deinterpolx<6>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 7: deinterpolx<7>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 8: deinterpolx<8>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        default: MR_fail("must not happen");
        }
      }

     void updateSlm(mav<complex<T>,2> &vslm, const mav<complex<T>,2> &vblm,
      size_t mbeam, mav<T,2> &re, mav<T,2> &im) const
      {
      auto ncomp = vslm.shape(1);
      MR_assert(ncomp>0, "need at least one component");
      MR_assert(vblm.shape(1)==ncomp, "inconsistent slm and blm vectors");
      Alm_Base islm(lmax, lmax), iblm(lmax, kmax);
      MR_assert(islm.Num_Alms()==vslm.shape(0), "bad array dimenion");
      MR_assert(iblm.Num_Alms()==vblm.shape(0), "bad array dimenion");
      MR_assert(re.conformable({Ntheta(), Nphi()}), "bad re shape");
      if (mbeam>0)
        {
        MR_assert(re.shape()==im.shape(), "re and im must have identical shape");
        MR_assert(re.stride()==im.stride(), "re and im must have identical strides");
        }
      MR_assert(mbeam <= kmax, "mbeam too high");

      auto ginfo = sharp_make_cc_geom_info(ntheta_s,nphi_s,0.,re.stride(1),re.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      // move stuff from border regions onto the main grid
      for (size_t i=0; i<ntheta; ++i)
        for (size_t j=0; j<nbphi; ++j)
          {
          re.v(i,j+nphi_b) += re(i,j);
          re.v(i,j+nbphi) += re(i,j+nphi_b+nbphi);
          if (mbeam>0)
            {
            im.v(i,j+nphi_b) += im(i,j);
            im.v(i,j+nbphi) += im(i,j+nphi_b+nbphi);
            }
          }

      for (size_t i=0; i<nbtheta; ++i)
        for (size_t j=0, j2=nphi_b/2; j<nphi_b; ++j,++j2)
          {
          T fct = (mbeam&1) ? -1 : 1;
          if (j2>=nphi_b) j2-=nphi_b;
          re.v(nbtheta+1+i,j+nbphi) += fct*re(nbtheta-1-i,j2+nbphi);
          re.v(nbtheta+ntheta_b-2-i, j+nbphi) += fct*re(nbtheta+ntheta_b+i,j2+nbphi);
          if (mbeam>0)
            {
            im.v(nbtheta+1+i,j+nbphi) += fct*im(nbtheta-1-i,j2+nbphi);
            im.v(nbtheta+ntheta_b-2-i, j+nbphi) += fct*im(nbtheta+ntheta_b+i,j2+nbphi);
            }
          }

      // special treatment for poles
      for (size_t j=0,j2=nphi_b/2; j<nphi_b/2; ++j,++j2)
        {
        T fct = (mbeam&1) ? -1 : 1;
        if (j2>=nphi_b) j2-=nphi_b;
        T tval = (re(nbtheta,j+nbphi) + fct*re(nbtheta,j2+nbphi));
        re.v(nbtheta,j+nbphi) = tval;
        re.v(nbtheta,j2+nbphi) = fct*tval;
        tval = (re(nbtheta+ntheta_b-1,j+nbphi) + fct*re(nbtheta+ntheta_b-1,j2+nbphi));
        re.v(nbtheta+ntheta_b-1,j+nbphi) = tval;
        re.v(nbtheta+ntheta_b-1,j2+nbphi) = fct*tval;
        if (mbeam>0)
          {
          tval = (im(nbtheta,j+nbphi) + fct*im(nbtheta,j2+nbphi));
          im.v(nbtheta,j+nbphi) = tval;
          im.v(nbtheta,j2+nbphi) = fct*tval;
          tval = (im(nbtheta+ntheta_b-1,j+nbphi) + fct*im(nbtheta+ntheta_b-1,j2+nbphi));
          im.v(nbtheta+ntheta_b-1,j+nbphi) = tval;
          im.v(nbtheta+ntheta_b-1,j2+nbphi) = fct*tval;
          }
        }

      vector<T>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=T(std::sqrt(4*pi/(2*i+1.)));

      if (mbeam==0)
        {
        Alm<complex<T>> a1(lmax, lmax);
        auto m1 = subarray<2>(re, {nbtheta,nbphi},{ntheta_b,nphi_b});
        decorrect(m1,0);
        sharp_alm2map_adjoint(a1.Alms().vdata(), m1.cdata(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            for (size_t i=0; i<ncomp; ++i)
              vslm.v(islm.index(l,m),i) += conj(a1(l,m))*vblm(iblm.index(l,0),i).real()*lnorm[l];
        }
      else
        {
        Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
        auto m1 = subarray<2>(re, {nbtheta,nbphi},{ntheta_b,nphi_b});
        auto m2 = subarray<2>(im, {nbtheta,nbphi},{ntheta_b,nphi_b});
        decorrect(m1,mbeam);
        decorrect(m2,mbeam);

        sharp_alm2map_spin_adjoint(mbeam, a1.Alms().vdata(), a2.Alms().vdata(), m1.cdata(),
          m2.cdata(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            if (l>=mbeam)
              for (size_t i=0; i<ncomp; ++i)
                {
                auto tmp = vblm(iblm.index(l,mbeam),i)*(-2*lnorm[l]);
                vslm.v(islm.index(l,m),i) += conj(a1(l,m))*tmp.real();
                vslm.v(islm.index(l,m),i) += conj(a2(l,m))*tmp.imag();
                }
        }
      }
    void updateSlm(mav<complex<T>,1> &slm, const mav<complex<T>,1> &blm,
      size_t mbeam, mav<T,2> &re, mav<T,2> &im) const
      {
      mav<complex<T>,2> vslm(&slm.v(0), {slm.shape(0),1}, {slm.stride(0),0}, true);
      mav<complex<T>,2> vblm(&blm(0), {blm.shape(0),1}, {blm.stride(0),0});
      updateSlm(vslm, vblm, mbeam, re, im);
      }

    void prepPsi(mav<T,3> &subcube) const
      {
      MR_assert(subcube.shape(0)==npsi_b, "bad psi dimension");
      auto newpart = subarray<3>(subcube, {npsi_s,0,0},{MAXIDX,MAXIDX,MAXIDX});
      newpart.fill(T(0));
      auto fct = kernel->corfunc(npsi_s/2+1, 1./npsi_b, nthreads);
      for (size_t k=0; k<npsi_s; ++k)
        {
        auto factor = T(fct[(k+1)/2]);
        for (size_t i=0; i<subcube.shape(1); ++i)
          for (size_t j=0; j<subcube.shape(2); ++j)
            subcube.v(k,i,j) *= factor;
        }
      fmav<T> fsubcube(subcube);
      r2r_fftpack(fsubcube, fsubcube, {0}, false, true, T(1), nthreads);
      }

    void deprepPsi(mav<T,3> &subcube) const
      {
      MR_assert(subcube.shape(0)==npsi_b, "bad psi dimension");
      fmav<T> fsubcube(subcube);
      r2r_fftpack(fsubcube, fsubcube, {0}, true, false, T(1), nthreads);
      auto fct = kernel->corfunc(npsi_s/2+1, 1./npsi_b, nthreads);
      for (size_t k=0; k<npsi_s; ++k)
        {
        auto factor = T(fct[(k+1)/2]);
        for (size_t i=0; i<subcube.shape(1); ++i)
          for (size_t j=0; j<subcube.shape(2); ++j)
            subcube.v(k,i,j) *= factor;
        }
      }
  };

}

using detail_totalconvolve::ConvolverPlan;


}

#endif
