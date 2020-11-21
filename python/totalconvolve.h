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
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC0_INTERPOL_NG_H
#define DUCC0_INTERPOL_NG_H

#define SIMD_INTERPOL
#define SPECIAL_CASING

#include <vector>
#include <complex>
#include <cmath>
#include "ducc0/math/constants.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/sharp/sharp.h"
#include "ducc0/sharp/sharp_almhelpers.h"
#include "ducc0/sharp/sharp_geomhelpers.h"
#include "python/alm.h"
#include "ducc0/math/fft.h"

namespace ducc0 {

namespace detail_fft {

using std::vector;

template<typename T, typename T0> aligned_array<T> alloc_tmp_conv
  (const fmav_info &info, size_t axis, size_t len)
  {
  auto othersize = info.size()/info.shape(axis);
  constexpr auto vlen = native_simd<T0>::size();
  auto tmpsize = len*((othersize>=vlen) ? vlen : 1);
  return aligned_array<T>(tmpsize);
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
  if (in.data()==out.data())
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
    size_t nborder;
    size_t lmax, kmax;
    // _s: small grid
    // _b: oversampled grid
    // no suffix: grid with borders
    size_t nphi_s, ntheta_s, npsi_s, nphi_b, ntheta_b, npsi_b, nphi, ntheta;
    T dphi, dtheta, dpsi, xdphi, xdtheta, xdpsi, phi0, theta0;

    shared_ptr<HornerKernel> kernel;

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
      fmav<T> ftmp0(tmp.template subarray<2>({0,0},{nphi_s, nphi_s}));
      convolve_1d(ftmp0, ftmp, 0, k2, nthreads);
      fmav<T> ftmp2(tmp.template subarray<2>({0,0},{ntheta_b, nphi_s}));
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
      fmav<T> ftmp2(tmp.template subarray<2>({0,0},{ntheta_b, nphi_s}));
      convolve_1d(farr, ftmp2, 1, k2, nthreads);
      // extend to second half
      for (size_t i=1, i2=nphi_b-1; i+1<ntheta_b; ++i,--i2)
        for (size_t j=0,j2=nphi_s/2; j<nphi_s; ++j,++j2)
          {
          if (j2>=nphi_s) j2-=nphi_s;
          tmp.v(i2,j) = sfct*tmp(i,j2);
          }
      fmav<T> ftmp(tmp);
      fmav<T> ftmp0(tmp.template subarray<2>({0,0},{nphi_s, nphi_s}));
      convolve_1d(ftmp, ftmp0, 0, k2, nthreads);
      for (size_t j=0; j<nphi_s; ++j)
        arr.v(0,j) = T(0.5)*tmp(0,j);
      for (size_t i=1; i+1<ntheta_s; ++i)
        for (size_t j=0; j<nphi_s; ++j)
          arr.v(i,j) = tmp(i,j);
      for (size_t j=0; j<nphi_s; ++j)
        arr.v(ntheta_s-1,j) = T(0.5)*tmp(ntheta_s-1,j);
      }

    vector<size_t> getIdx(const mav<T,1> &theta, const mav<T,1> &phi,
      size_t patch_ntheta, size_t patch_nphi, size_t itheta0, size_t iphi0, size_t supp) const
      {
      constexpr size_t cellsize=16;
      size_t nct = patch_ntheta/cellsize+1,
             ncp = patch_nphi/cellsize+1;
      double theta0 = (int(itheta0)-int(nborder))*dtheta,
             phi0 = (int(iphi0)-int(nborder))*dphi;
      vector<vector<size_t>> mapper(nct*ncp);
      MR_assert(theta.conformable(phi), "theta/phi size mismatch");
// FIXME: parallelize?
      for (size_t i=0; i<theta.shape(0); ++i)
        {
        auto ftheta = (theta(i)-theta0)*xdtheta-supp/T(2);
        auto itheta = size_t(ftheta+1);
        auto fphi = (phi(i)-phi0)*xdphi-supp/T(2);
        auto iphi = size_t(fphi+1);
        itheta /= cellsize;
        iphi /= cellsize;
//if (itheta>=nct) cout << theta0 << " " << dtheta << " " << theta(i) << " " << itheta << " " << patch_ntheta << " " << nct << endl;
//if (iphi>=ncp) cout << iphi << endl;
        MR_assert(itheta<nct, "bad itheta");
        MR_assert(iphi<ncp, "bad iphi");
//        size_t itheta=min(nct-1,size_t((theta(i)-theta0)/pi*nct)),
//               iphi=min(ncp-1,size_t((phi(i)-phi0)/(2*pi)*ncp));
        mapper[itheta*ncp+iphi].push_back(i);
        }
      vector<size_t> idx(theta.shape(0));
      size_t cnt=0;
      for (auto &vec: mapper)
        {
        for (auto i:vec)
          idx[cnt++] = i;
        vector<size_t>().swap(vec); // cleanup
        }
      return idx;
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
          };
        kbuf buf;

      private:
        TemplateKernel<supp, Tsimd> tkrn;
        T mytheta0, myphi0;

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
        void prep(T theta, T phi, T psi)
          {
          T ftheta = (theta-mytheta0)*plan.xdtheta-supp/T(2);
          itheta = size_t(ftheta+1);
          ftheta = -1+(itheta-ftheta)*2;
          T fphi = (phi-myphi0)*plan.xdphi-supp/T(2);
          iphi = size_t(fphi+1);
          fphi = -1+(iphi-fphi)*2;
          T fpsi = psi*plan.xdpsi-supp/T(2)+plan.npsi_b;
          ipsi = size_t(fpsi+1);
          fpsi = -1+(ipsi-fpsi)*2;
          if (ipsi>=plan.npsi_b) ipsi-=plan.npsi_b;
if (ipsi>=plan.npsi_b) cout << "aargh " << ipsi << endl;
          tkrn.eval3(fpsi, ftheta, fphi, &buf.simd[0]);
          }
        size_t itheta, iphi, ipsi;
        const T * DUCC0_RESTRICT wpsi;
        const T * DUCC0_RESTRICT wtheta;
        const Tsimd * DUCC0_RESTRICT wphi;
        ptrdiff_t jumptheta;
      };

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
      auto idx = getIdx(theta, phi, cube.shape(1), cube.shape(2), itheta0, iphi0, supp);
      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        WeightHelper<supp> hlp(*this, cube, itheta0, iphi0);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          size_t i=idx[ind];
          hlp.prep(theta(i), phi(i), psi(i));
          auto ipsi = hlp.ipsi;
          const T * DUCC0_RESTRICT ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
            {
            Tsimd res=0;
            for (size_t ipsic=0; ipsic<supp; ++ipsic)
              {
              const T * DUCC0_RESTRICT ptr2 = ptr;
              for (size_t itheta=0; itheta<supp; ++itheta)
                {
                auto twgt=hlp.wpsi[ipsic]*hlp.wtheta[itheta];
                for (size_t iphi=0; iphi<nvec; ++iphi)
                  res += twgt*hlp.wphi[iphi]*Tsimd::loadu(ptr2+iphi*vlen);
                ptr2 += hlp.jumptheta;
                }
              if (++ipsi>=npsi_b) ipsi=0;
              ptr = &cube(ipsi,hlp.itheta,hlp.iphi);
              }
            signal.v(i) = reduce(res, std::plus<>());
            }
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
      auto idx = getIdx(theta, phi, cube.shape(1), cube.shape(2), itheta0, iphi0, supp);

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
        if (b_theta<locks.shape(0))  // unlock
          {
          locks.v(b_theta,b_phi).unlock();
          locks.v(b_theta,b_phi+1).unlock();
          locks.v(b_theta+1,b_phi).unlock();
          locks.v(b_theta+1,b_phi+1).unlock();
          }
        });
      }
    T realsigma() const
      {
      return min(T(npsi_b)/(2*kmax+1),
                 min(T(nphi_b)/(2*lmax+1), T(ntheta_b)/(lmax+1)));
      }

  public:
    ConvolverPlan(size_t lmax_, size_t kmax_, double sigma, double epsilon,
      size_t nthreads_)
      : nthreads(nthreads_),
        nborder(8),
        lmax(lmax_),
        kmax(kmax_),
        nphi_s(2*good_size_real(lmax+1)),
        ntheta_s(nphi_s/2+1),
        npsi_s(kmax*2+1),
        nphi_b(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*sigma/2.)))),
        ntheta_b(nphi_b/2+1),
        npsi_b(size_t(npsi_s*sigma+0.99999)),
        nphi(nphi_b+2*nborder),
        ntheta(ntheta_b+2*nborder),
        dphi(T(2*pi/nphi_b)),
        dtheta(T(pi/(ntheta_b-1))),
        dpsi(T(2*pi/npsi_b)),
        xdphi(T(1)/dphi),
        xdtheta(T(1)/dtheta),
        xdpsi(T(1)/dpsi),
        phi0(nborder*(-dphi)),
        theta0(nborder*(-dtheta)),
        kernel(selectKernel(realsigma(), 0.5*epsilon))
      {
      auto supp = kernel->support();
      MR_assert(supp<=8, "kernel support too large");
      MR_assert((supp<=ntheta) && (supp<=nphi_b), "kernel support too large!");
      }

    size_t Ntheta() const { return ntheta; }
    size_t Nphi() const { return nphi; }
    size_t Npsi() const { return npsi_b; }

    vector<size_t> getPatchInfo(T theta_lo, T theta_hi, T phi_lo, T phi_hi) const
      {
      vector<size_t> res(4);
      auto tmp = (theta_lo-theta0)*xdtheta-nborder;
      res[0] = min(size_t(max(T(0), tmp)), ntheta);
      tmp = (theta_hi-theta0)*xdtheta+nborder+T(1);
      res[1] = min(size_t(max(T(0), tmp)), ntheta);
      tmp = (phi_lo-phi0)*xdphi-nborder;
      res[2] = min(size_t(max(T(0), tmp)), nphi);
      tmp = (phi_hi-phi0)*xdphi+nborder+T(1);
      res[3] = min(size_t(max(T(0), tmp)), nphi);
      return res;
      }

    void getPlane(const Alm<complex<T>> &slm, const Alm<complex<T>> &blm,
      size_t mbeam, mav<T,2> &re, mav<T,2> &im) const
      {
      MR_assert(slm.Lmax()==lmax, "inconsistent Sky lmax");
      MR_assert(slm.Mmax()==lmax, "Sky lmax must be equal to Sky mmax");
      MR_assert(blm.Lmax()==lmax, "Sky and beam lmax must be equal");
      MR_assert(re.conformable({Ntheta(), Nphi()}), "bad re shape");
      if (mbeam>0)
        {
        MR_assert(re.shape()==im.shape(), "re and im must have identical shape");
        MR_assert(re.stride()==im.stride(), "re and im must have identical strides");
        }
      MR_assert(mbeam <= blm.Mmax(), "mbeam too high");

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
            a1(l,m) = slm(l,m)*blm(l,0).real()*lnorm[l];
        auto m1 = re.template subarray<2>({nborder,nborder},{ntheta_b,nphi_b});
        sharp_alm2map(a1.Alms().data(), m1.vdata(), *ginfo, *ainfo, 0, nthreads);
        correct(m1,0);
        }
      else
        {
        Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            {
            if (l<mbeam)
              a1(l,m)=a2(l,m)=0.;
            else
              {
              auto tmp = blm(l,mbeam)*(-lnorm[l]);
              a1(l,m) = slm(l,m)*tmp.real();
              a2(l,m) = slm(l,m)*tmp.imag();
              }
            }
        auto m1 = re.template subarray<2>({nborder,nborder},{ntheta_b,nphi_b});
        auto m2 = im.template subarray<2>({nborder,nborder},{ntheta_b,nphi_b});
        sharp_alm2map_spin(mbeam, a1.Alms().data(), a2.Alms().data(),
          m1.vdata(), m2.vdata(), *ginfo, *ainfo, 0, nthreads);
        correct(m1,mbeam);
        correct(m2,mbeam);
        }
      // fill border regions
      T fct = (mbeam&1) ? -1 : 1;
      for (size_t i=0; i<nborder; ++i)
        for (size_t j=0, j2=nphi_b/2; j<nphi_b; ++j,++j2)
          {
          if (j2>=nphi_b) j2-=nphi_b;
          for (size_t l=0; l<re.shape(1); ++l)
            {
            re.v(nborder-1-i,j2+nborder) = fct*re(nborder+1+i,j+nborder);
            re.v(nborder+ntheta_b+i,j2+nborder) = fct*re(nborder+ntheta_b-2-i,j+nborder);
            }
          if (mbeam>0)
            {
            im.v(nborder-1-i,j2+nborder) = fct*im(nborder+1+i,j+nborder);
            im.v(nborder+ntheta_b+i,j2+nborder) = fct*im(nborder+ntheta_b-2-i,j+nborder);
            }
          }
      for (size_t i=0; i<ntheta_b+2*nborder; ++i)
        for (size_t j=0; j<nborder; ++j)
          {
          re.v(i,j) = re(i,j+nphi_b);
          re.v(i,j+nphi_b+nborder) = re(i,j+nborder);
          if (mbeam>0)
            {
            im.v(i,j) = im(i,j+nphi_b);
            im.v(i,j+nphi_b+nborder) = im(i,j+nborder);
            }
          }
      }

    void interpol(const mav<T,3> &cube, size_t itheta0,
      size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, mav<T,1> &signal) const
      {
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

    void updateSlm(Alm<complex<T>> &slm, const Alm<complex<T>> &blm,
      size_t mbeam, mav<T,2> &re, mav<T,2> &im) const
      {
      MR_assert(slm.Lmax()==lmax, "inconsistent Sky lmax");
      MR_assert(slm.Mmax()==lmax, "Sky lmax must be equal to Sky mmax");
      MR_assert(blm.Lmax()==lmax, "Sky and beam lmax must be equal");
      MR_assert(re.conformable({Ntheta(), Nphi()}), "bad re shape");
      if (mbeam>0)
        {
        MR_assert(re.shape()==im.shape(), "re and im must have identical shape");
        MR_assert(re.stride()==im.stride(), "re and im must have identical strides");
        }
      MR_assert(mbeam <= blm.Mmax(), "mbeam too high");

      auto ginfo = sharp_make_cc_geom_info(ntheta_s,nphi_s,0.,re.stride(1),re.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      // move stuff from border regions onto the main grid
      for (size_t i=0; i<ntheta_b+2*nborder; ++i)
        for (size_t j=0; j<nborder; ++j)
          {
          re.v(i,j+nphi_b) += re(i,j);
          re.v(i,j+nborder) += re(i,j+nphi_b+nborder);
          if (mbeam>0)
            {
            im.v(i,j+nphi_b) += im(i,j);
            im.v(i,j+nborder) += im(i,j+nphi_b+nborder);
            }
          }

      for (size_t i=0; i<nborder; ++i)
        for (size_t j=0, j2=nphi_b/2; j<nphi_b; ++j,++j2)
          {
          T fct = (mbeam&1) ? -1 : 1;
          if (j2>=nphi_b) j2-=nphi_b;
          re.v(nborder+1+i,j+nborder) += fct*re(nborder-1-i,j2+nborder);
          re.v(nborder+ntheta_b-2-i, j+nborder) += fct*re(nborder+ntheta_b+i,j2+nborder);
          if (mbeam>0)
            {
            im.v(nborder+1+i,j+nborder) += fct*im(nborder-1-i,j2+nborder);
            im.v(nborder+ntheta_b-2-i, j+nborder) += fct*im(nborder+ntheta_b+i,j2+nborder);
            }
          }

      // special treatment for poles
      for (size_t j=0,j2=nphi_b/2; j<nphi_b/2; ++j,++j2)
        {
        T fct = (mbeam&1) ? -1 : 1;
        if (j2>=nphi_b) j2-=nphi_b;
        T tval = (re(nborder,j+nborder) + fct*re(nborder,j2+nborder));
        re.v(nborder,j+nborder) = tval;
        re.v(nborder,j2+nborder) = fct*tval;
        tval = (re(nborder+ntheta_b-1,j+nborder) + fct*re(nborder+ntheta_b-1,j2+nborder));
        re.v(nborder+ntheta_b-1,j+nborder) = tval;
        re.v(nborder+ntheta_b-1,j2+nborder) = fct*tval;
        if (mbeam>0)
          {
          tval = (im(nborder,j+nborder) + fct*im(nborder,j2+nborder));
          im.v(nborder,j+nborder) = tval;
          im.v(nborder,j2+nborder) = fct*tval;
          tval = (im(nborder+ntheta_b-1,j+nborder) + fct*im(nborder+ntheta_b-1,j2+nborder));
          im.v(nborder+ntheta_b-1,j+nborder) = tval;
          im.v(nborder+ntheta_b-1,j2+nborder) = fct*tval;
          }
        }

      vector<T>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=T(std::sqrt(4*pi/(2*i+1.)));

      if (mbeam==0)
        {
        Alm<complex<T>> a1(lmax, lmax);
        auto m1 = re.template subarray<2>({nborder,nborder},{ntheta_b,nphi_b});
        decorrect(m1,0);
        sharp_alm2map_adjoint(a1.Alms().vdata(), m1.data(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
              slm(l,m) += conj(a1(l,m))*blm(l,0).real()*lnorm[l];
        }
      else
        {
        Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
        auto m1 = re.template subarray<2>({nborder,nborder},{ntheta_b,nphi_b});
        auto m2 = im.template subarray<2>({nborder,nborder},{ntheta_b,nphi_b});
        decorrect(m1,mbeam);
        decorrect(m2,mbeam);

        sharp_alm2map_spin_adjoint(mbeam, a1.Alms().vdata(), a2.Alms().vdata(), m1.data(),
          m2.data(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            if (l>=mbeam)
              {
              auto tmp = blm(l,mbeam)*(-lnorm[l]);
              slm(l,m) += conj(a1(l,m))*tmp.real();
              slm(l,m) += conj(a2(l,m))*tmp.imag();
              }
        }
      }

    void prepPsi(mav<T,3> &subcube) const
      {
      MR_assert(subcube.shape(0)==npsi_b, "bad psi dimension");
      auto newpart = subcube.template subarray<3>({npsi_s,0,0},{npsi_b-npsi_s,subcube.shape(1),subcube.shape(2)});
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
