#ifndef DUCC0_TOTALCONVOLVE2_H
#define DUCC0_TOTALCONVOLVE2_H

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

namespace detail_totalconvolve2 {

using namespace std;

template<typename T> class ConvolverPlan
  {
  protected:
    constexpr static auto vlen = min<size_t>(8, native_simd<T>::size());
    using Tsimd = simd<T, vlen>;

    size_t nthreads;
    size_t nborder;
    size_t lmax;
    // _s: small grid
    // _b: oversampled grid
    // no suffix: grid with borders
    size_t nphi_s, ntheta_s, nphi_b, ntheta_b, nphi, ntheta;
    T dphi, dtheta, xdphi, xdtheta, phi0, theta0;

    shared_ptr<HornerKernel<Tsimd>> kernel;

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

    vector<size_t> getIdx(const mav<T,1> &theta, const mav<T,1> &phi,
      size_t patch_ntheta, size_t patch_nphi, size_t itheta0, size_t iphi0) const
      {
      constexpr size_t cellsize=16;
      size_t nct = (patch_ntheta-2*nborder)/cellsize+1,
             ncp = (patch_nphi-2*nborder)/cellsize+1;
      double theta0 = (itheta0-nborder)*dtheta,
             phi0 = (iphi0-nborder)*dphi;
      vector<vector<size_t>> mapper(nct*ncp);
      MR_assert(theta.conformable(phi), "theta/phi size mismatch");
// FIXME: parallelize?
      for (size_t i=0; i<theta.shape(0); ++i)
        {
        size_t itheta=min(nct-1,size_t((theta(i)-theta0)/pi*nct)),
               iphi=min(ncp-1,size_t((phi(i)-phi0)/(2*pi)*ncp));
//        MR_assert((itheta<nct)&&(iphi<ncp), "oops");
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
          T scalar[2*nvec*vlen];
          Tsimd simd[2*nvec];
          };
        kbuf buf;

      private:
        TemplateKernel<supp, Tsimd> tkrn;
        size_t beammmax;
        T mytheta0, myphi0;

      public:
        WeightHelper(const ConvolverPlan &plan_, const mav_info<3> &info, size_t itheta0, size_t iphi0)
          : plan(plan_),
            tkrn(*plan.kernel),
            beammmax(info.shape(0)/2),
            mytheta0(plan.theta0+itheta0*plan.dtheta),
            myphi0(plan.phi0+iphi0*plan.dphi),
            wtheta(&buf.scalar[0]),
            wphi(&buf.simd[nvec]),
            wpsi(info.shape(0)),
            jumptheta(info.stride(1)),
            jumppsi(info.stride(0))
          {
          MR_assert(info.stride(2)==1, "last axis of cube must be contiguous");
          MR_assert(info.shape(0)&1, "number of psi planes must be odd");
          wpsi[0]=1.;
          }
        DUCC0_NOINLINE void prep(T theta, T phi, T psi)
          {
          T ftheta = (theta-mytheta0)*plan.xdtheta-supp/T(2);
          itheta = size_t(ftheta+1);
          ftheta = -1+(itheta-ftheta)*2;
          T fphi = (phi-myphi0)*plan.xdphi-supp/T(2);
          iphi = size_t(fphi+1);
          fphi = -1+(iphi-fphi)*2;
          tkrn.eval2(ftheta, fphi, &buf.simd[0]);
#if 1
          auto cpsi=cos(double(psi));
          auto spsi=sin(double(psi));
          auto cnpsi=cpsi;
          auto snpsi=spsi;
          for (size_t l=1; l<=beammmax; ++l)
            {
            wpsi[2*l-1]=T(cnpsi);
            wpsi[2*l]=T(snpsi);
            const double tmp = snpsi*cpsi + cnpsi*spsi;
            cnpsi=cnpsi*cpsi - snpsi*spsi;
            snpsi=tmp;
            }
#else
          for (size_t i=1; i<=beammmax; ++i)
            {
            wpsi[2*i-1] = cos(i*psi);
            wpsi[2*i] = sin(i*psi);
            }
#endif
//cout << itheta << " " << iphi << endl;
          }
        size_t itheta, iphi;
        const T * DUCC0_RESTRICT wtheta;
        const Tsimd * DUCC0_RESTRICT wphi;
        vector<T> wpsi;
        ptrdiff_t jumptheta, jumppsi;
      };

    template<size_t supp> void interpol2(const mav<T,3> &cube,
      size_t itheta0, size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, mav<T,1> &signal) const
      {
      MR_assert(cube.stride(2)==1, "last axis of cube must be contiguous");
      MR_assert(phi.shape(0)==theta.shape(0), "aray shape mismatch");
      MR_assert(psi.shape(0)==theta.shape(0), "aray shape mismatch");
      MR_assert(signal.shape(0)==theta.shape(0), "aray shape mismatch");
      static constexpr size_t vlen = Tsimd::size();
      static constexpr size_t nvec = (supp+vlen-1)/vlen;
      size_t npsi = cube.shape(0);
      auto idx = getIdx(theta, phi, cube.shape(1), cube.shape(2), itheta0, iphi0);
      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        WeightHelper<supp> hlp(*this, cube, itheta0, iphi0);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          size_t i=idx[ind];
          hlp.prep(theta(i), phi(i), psi(i));
          const T * DUCC0_RESTRICT ptr = &cube(0,hlp.itheta,hlp.iphi);
          if constexpr (nvec==1)
            {
            Tsimd res=0;
            for (size_t ipsi=0; ipsi<npsi; ++ipsi)
              {
              const T * DUCC0_RESTRICT ptr2 = ptr;
              Tsimd tres=0;
              for (size_t itheta=0; itheta<supp; ++itheta)
                {
                tres += Tsimd::loadu(ptr2)*hlp.wtheta[itheta];
                ptr2 += hlp.jumptheta;
                }
              res += tres*hlp.wpsi[ipsi];
              ptr += hlp.jumppsi;
              }
            signal.v(i) = reduce(res*hlp.wphi[0], std::plus<>());
            }
          else if constexpr (nvec==2)
            {
            Tsimd res0=0, res1=0;
            for (size_t ipsi=0; ipsi<npsi; ++ipsi)
              {
              const T * DUCC0_RESTRICT ptr2 = ptr;
              Tsimd tres0=0, tres1=0;
              for (size_t itheta=0; itheta<supp; ++itheta)
                {
                tres0 += Tsimd::loadu(ptr2)*hlp.wtheta[itheta];
                tres1 += Tsimd::loadu(ptr2+vlen)*hlp.wtheta[itheta];
                ptr2 += hlp.jumptheta;
                }
              res0 += tres0*hlp.wpsi[ipsi];
              res1 += tres1*hlp.wpsi[ipsi];
              ptr += hlp.jumppsi;
              }
            signal.v(i) = reduce(res0*hlp.wphi[0]+res1*hlp.wphi[1], std::plus<>());
            }
          else
            {
            Tsimd res=0;
            for (size_t ipsi=0; ipsi<npsi; ++ipsi)
              {
              const T * DUCC0_RESTRICT ptr2 = ptr;
              for (size_t itheta=0; itheta<supp; ++itheta)
                {
                auto twgt=hlp.wpsi[ipsi]*hlp.wtheta[itheta];
                for (size_t iphi=0; iphi<nvec; ++iphi)
                  res += twgt*hlp.wphi[iphi]*Tsimd::loadu(ptr2+iphi*vlen);
                ptr2 += hlp.jumptheta;
                }
              ptr += hlp.jumppsi;
              }
            signal.v(i) = reduce(res, std::plus<>());
            }
          }
        });
      }

  public:
    ConvolverPlan(size_t lmax_, double sigma, double epsilon,
      size_t nthreads_)
      : nthreads(nthreads_),
        nborder(8),
        lmax(lmax_),
        nphi_s(2*good_size_real(lmax+1)),
        ntheta_s(nphi_s/2+1),
        nphi_b(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*sigma/2.)))),
        ntheta_b(nphi_b/2+1),
        nphi(nphi_b+2*nborder),
        ntheta(ntheta_b+2*nborder),
        dphi(T(2*pi/nphi_b)),
        dtheta(T(pi/(ntheta_b-1))),
        xdphi(T(1)/dphi),
        xdtheta(T(1)/dtheta),
        phi0(nborder*(-dphi)),
        theta0(nborder*(-dtheta)),
        kernel(selectKernel<Tsimd>(T(nphi_b)/(2*lmax+1), 0.5*epsilon))
      {
//      static_assert(is_same<T, float>::value, "only accepting floats for the moment");
      auto supp = kernel->support();
      MR_assert(supp<=8, "kernel support too large");
      MR_assert((supp<=ntheta) && (supp<=nphi_b), "kernel support too large!");
      }

    size_t Ntheta() const { return ntheta; }
    size_t Nphi() const { return nphi; }
//     double Theta0() const { return -nborder*Dtheta(); }
//     double Phi0() const { return -nborder*Dphi(); }
//     double Dtheta() const { return pi/(ntheta-1); }
//     double Dphi() const { return 2*pi/nphi; }
//     double Theta(size_t itheta) const { return Theta0()+itheta*Dtheta(); }
//     double Phi(size_t iphi) const { return Phi0()+iphi*Dphi(); }

//    size_t Nborder() const { return nborder; }

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
              auto tmp = blm(l,mbeam)*(-2*lnorm[l]);
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
        case 4: interpol2<4>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 5: interpol2<5>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 6: interpol2<6>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 7: interpol2<7>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        case 8: interpol2<8>(cube, itheta0, iphi0, theta, phi, psi, signal); break;
        default: MR_fail("must not happen");
        }
      }

    void deinterpol(mav<T,3> &cube, size_t itheta0,
      size_t iphi0, const mav<T,1> &theta, const mav<T,1> &phi,
      const mav<T,1> &psi, const mav<T,1> &signal) const;

    void extractFromPlane(Alm<T> &slm, const Alm<T> &blm,
      size_t mbeam, const mav<T,2> &re, const mav<T,2> &im) const;
  };

}

using detail_totalconvolve2::ConvolverPlan;

}

#endif
