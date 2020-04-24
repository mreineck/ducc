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

#if 0
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
MRUTIL_NOINLINE void general_convolve(const fmav<T> &in, fmav<T> &out,
  const size_t axis, const vector<T0> &kernel, size_t nthreads,
  const Exec &exec, const bool allow_inplace=true)
  {
  std::shared_ptr<Tplan> plan1, plan2;

  size_t l_in=in.shape(axis), l_out=out.shape(axis);
  size_t l_min=std::min(l_in, l_out), l_max=std::max(l_in, l_out);
  MR_assert(kernel.size()==l_min/2+1, "bad kernel size");
  plan1 = get_plan<Tplan>(l_in);
  plan2 = get_plan<Tplan>(l_out);

  execParallel(
    util::thread_count(nthreads, in, axis, native_simd<T0>::size()),
    [&](Scheduler &sched) {
      constexpr auto vlen = native_simd<T0>::size();
      auto storage = alloc_tmp_conv<T,T0>(in, axis, l_max); //FIXME!
      multi_iter<vlen> it(in, out, axis, sched.num_threads(), sched.thread_num());
#ifndef MRUTIL_NO_SIMD
      if (vlen>1)
        while (it.remaining()>=vlen)
          {
          it.advance(vlen);
          auto tdatav = reinterpret_cast<add_vec_t<T> *>(storage.data());
          exec(it, in, out, tdatav, *plan1, *plan2, kernel);
          }
#endif
      while (it.remaining()>0)
        {
        it.advance(1);
        auto buf = allow_inplace && it.stride_out() == 1 ?
          &out.vraw(it.oofs(0)) : reinterpret_cast<T *>(storage.data());
        exec(it, in, out, buf, *plan1, *plan2, kernel);
        }
    });  // end of parallel region
  }

struct ExecConvR1
  {
  template <typename T0, typename T, size_t vlen> void operator() (
    const multi_iter<vlen> &it, const fmav<T0> &in, fmav<T0> &out,
    T * buf, const pocketfft_r<T0> &plan1, const pocketfft_r<T0> &plan2,
    const vector<T0> &kernel) const
    {
    size_t l_in = plan1.length(),
           l_out = plan2.length(),
           l_min = std::min(l_in, l_out);
    copy_input(it, in, buf);
    plan1.exec(buf, T0(1), true);
    buf[0] *= kernel[0];
    for (size_t i=1; i<l_min; ++i)
      { buf[2*i-1]*=kernel[i]; buf[2*i] *=kernel[i]; }
    for (size_t i=l_in; i<l_out; ++i) buf[i] = T(0);
    plan2.exec(buf, T0(1), false);
    copy_output(it, buf, out);
    }
  };

template<typename T> void convolve_1d(const fmav<T> &in,
  fmav<T> &out, size_t axis, const vector<T> &kernel, size_t nthreads=1)
  {
//  util::sanity_check_onetype(in, out, in.data()==out.data(), axes);
  MR_assert(axis<in.ndim(), "bad axis number");
  MR_assert(in.ndim()==out.ndim(), "dimensionality mismatch");
  if (in.data()==out.data())
    MR_assert(in.strides()==out.strides(), "strides mismatch");
  for (size_t i=0; i<in.ndim(); ++i)
    if (i!=axis)
      MR_assert(in.shape(i)==out.shape(i), "shape mismatch");
  if (in.size()==0) return;
  general_convolve<pocketfft_r<T>>(in, out, axis, kernel, nthreads,
    ExecConvR1());
  }

}
#endif
namespace detail_interpol_ng {

using namespace std;

template<typename T> class Interpolator
  {
  protected:
    bool adjoint;
    size_t lmax, kmax, nphi0, ntheta0, nphi, ntheta;
    int nthreads;
    T ofactor;
    size_t supp;
    ES_Kernel kernel;
    size_t ncomp;
    mav<T,4> cube; // the data cube (theta, phi, 2*mbeam+1, TGC)

    void correct(mav<T,2> &arr, int spin)
      {
      T sfct = (spin&1) ? -1 : 1;
      mav<T,2> tmp({nphi,nphi});
      tmp.apply([](T &v){v=0.;});
      auto tmp0=tmp.template subarray<2>({0,0},{nphi0, nphi0});
      fmav<T> ftmp0(tmp0);
      for (size_t i=0; i<ntheta0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) = arr(i,j);
      // extend to second half
// FIXME: merge with loop above to avoid edundant memory reads.
      for (size_t i=1, i2=nphi0-1; i+1<ntheta0; ++i,--i2)
        for (size_t j=0,j2=nphi0/2; j<nphi0; ++j,++j2)
          {
          if (j2>=nphi0) j2-=nphi0;
          tmp0.v(i2,j) = sfct*tmp0(i,j2);
          }
      // FFT to frequency domain on minimal grid
// one bad FFT axis
      r2r_fftpack(ftmp0,ftmp0,{0,1},true,true,T(1./(nphi0*nphi0)),nthreads);

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
// one bad FFT axis
      r2r_fftpack(ftmp1,ftmp1,{0},false,false,T(1),nthreads);

      auto tmp2=tmp.template subarray<2>({0,0},{ntheta, nphi});
      fmav<T> ftmp2(tmp2);
      fmav<T> farr(arr);
      // zero-padded FFT in phi direction
      r2r_fftpack(ftmp2,farr,{1},false,false,T(1),nthreads);
      }
    void decorrect(mav<T,2> &arr, int spin)
      {
      T sfct = (spin&1) ? -1 : 1;
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
      r2r_fftpack(ftmp,ftmp,{1},true,true,T(1),nthreads);
      auto tmp1=tmp.template subarray<2>({0,0},{nphi, nphi0});
      fmav<T> ftmp1(tmp1);
      r2r_fftpack(ftmp1,ftmp1,{0},true,true,T(1),nthreads);
      auto fct = kernel.correction_factors(nphi, nphi0/2+1, nthreads);
      auto tmp0=tmp.template subarray<2>({0,0},{nphi0, nphi0});
      fmav<T> ftmp0(tmp0);
      for (size_t i=0; i<nphi0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) *= fct[(i+1)/2] * fct[(j+1)/2];
      // FFT to (theta, phi) domain on minimal grid
      r2r_fftpack(ftmp0,ftmp0,{0,1},false, false,T(1./(nphi0*nphi0)),nthreads);
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
//        MR_assert((itheta<nct)&&(iphi<ncp), "oops");
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
                 bool separate, T epsilon, T ofmin, int nthreads_)
      : adjoint(false),
        lmax(slm.at(0).Lmax()),
        kmax(blm.at(0).Mmax()),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*ofmin/2.)))),
        ntheta(nphi/2+1),
        nthreads(nthreads_),
        ofactor(T(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, nthreads),
        ncomp(separate ? slm.size() : 1),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1, ncomp})
      {
      MR_assert((ncomp==1)||(ncomp==3), "currently only 1 or 3 components allowed");
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

      vector<T>lnorm(lmax+1);
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
                  auto tmp = blm[icomp](l,k)*T(-2*lnorm[l]);
                  a1(l,m) = slm[icomp](l,m)*tmp.real();
                  a2(l,m) = slm[icomp](l,m)*tmp.imag();
                  }
                else
                  {
                  a1(l,m) = a2(l,m) = 0;
                  for (size_t j=0; j<slm.size(); ++j)
                    {
                    auto tmp = blm[j](l,k)*T(-2*lnorm[l]);
                    a1(l,m) += slm[j](l,m)*tmp.real();
                    a2(l,m) += slm[j](l,m)*tmp.imag();
                    }
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

      // fill border regions
      for (size_t i=0; i<supp; ++i)
        for (size_t j=0, j2=nphi/2; j<nphi; ++j,++j2)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            T fct = (((k+1)/2)&1) ? -1 : 1;
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

    Interpolator(size_t lmax_, size_t kmax_, size_t ncomp_, T epsilon, T ofmin, int nthreads_)
      : adjoint(true),
        lmax(lmax_),
        kmax(kmax_),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(std::max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*ofmin/2.)))),
        ntheta(nphi/2+1),
        nthreads(nthreads_),
        ofactor(T(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, nthreads),
        ncomp(ncomp_),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1, ncomp_})
      {
      MR_assert((ncomp==1)||(ncomp==3), "currently only 1 or 3 components allowed");
      MR_assert((supp<=ntheta) && (supp<=nphi), "support too large!");
      cube.apply([](T &v){v=0.;});
      }

    void interpol (const mav<T,2> &ptg, mav<T,2> &res) const
      {
      MR_assert(!adjoint, "cannot be called in adjoint mode");
      MR_assert(ptg.shape(0)==res.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      MR_assert(res.shape(1)==ncomp, "# of components mismatch");
      T delta = T(2)/supp;
      T xdtheta = T((ntheta-1)/pi),
        xdphi = T(nphi/(2*pi));
      auto idx = getIdx(ptg);
      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        vector<T> wt(supp), wp(supp);
        vector<T> psiarr(2*kmax+1);
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          size_t i=idx[ind];
          T f0=T(0.5*supp+ptg(i,0)*xdtheta);
          size_t i0 = size_t(f0+T(1));
          for (size_t t=0; t<supp; ++t)
            wt[t] = kernel((t+i0-f0)*delta - 1);
          T f1=T(0.5)*supp+ptg(i,1)*xdphi;
          size_t i1 = size_t(f1+1.);
          for (size_t t=0; t<supp; ++t)
            wp[t] = kernel((t+i1-f1)*delta - 1);
          psiarr[0]=1.;
          double psi=ptg(i,2);
          double cpsi=cos(psi), spsi=sin(psi);
          double cnpsi=cpsi, snpsi=spsi;
          for (size_t l=1; l<=kmax; ++l)
            {
            psiarr[2*l-1]=T(cnpsi);
            psiarr[2*l]=T(snpsi);
            const double tmp = snpsi*cpsi + cnpsi*spsi;
            cnpsi=cnpsi*cpsi - snpsi*spsi;
            snpsi=tmp;
            }
          if (ncomp==1)
            {
            T vv=0;
            for (size_t j=0; j<supp; ++j)
              for (size_t k=0; k<supp; ++k)
                for (size_t l=0; l<2*kmax+1; ++l)
                  vv += cube(i0+j,i1+k,l,0)*wt[j]*wp[k]*psiarr[l];
            res.v(i,0) = vv;
            }
          else // ncomp==3
            {
            T v0=0., v1=0., v2=0.;
            for (size_t j=0; j<supp; ++j)
              for (size_t k=0; k<supp; ++k)
                for (size_t l=0; l<2*kmax+1; ++l)
                  {
                  auto tmp = wt[j]*wp[k]*psiarr[l];
                  v0 += cube(i0+j,i1+k,l,0)*tmp;
                  v1 += cube(i0+j,i1+k,l,1)*tmp;
                  v2 += cube(i0+j,i1+k,l,2)*tmp;
                  }
            res.v(i,0) = v0;
            res.v(i,1) = v1;
            res.v(i,2) = v2;
            }
          }
        });
      }

    size_t support() const
      { return supp; }

    void deinterpol (const mav<T,2> &ptg, const mav<T,2> &data)
      {
      MR_assert(adjoint, "can only be called in adjoint mode");
      MR_assert(ptg.shape(0)==data.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      MR_assert(data.shape(1)==ncomp, "# of components mismatch");
      T delta = T(2)/supp;
      T xdtheta = T((ntheta-1)/pi),
        xdphi = T(nphi/(2*pi));
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
        while (auto rng=sched.getNext()) for(auto ind=rng.lo; ind<rng.hi; ++ind)
          {
          size_t i=idx[ind];
          T f0=0.5*supp+ptg(i,0)*xdtheta;
          size_t i0 = size_t(f0+1.);
          for (size_t t=0; t<supp; ++t)
            wt[t] = kernel((t+i0-f0)*delta - 1);
          T f1=0.5*supp+ptg(i,1)*xdphi;
          size_t i1 = size_t(f1+1.);
          for (size_t t=0; t<supp; ++t)
            wp[t] = kernel((t+i1-f1)*delta - 1);
          psiarr[0]=1.;
          double psi=ptg(i,2);
          double cpsi=cos(psi), spsi=sin(psi);
          double cnpsi=cpsi, snpsi=spsi;
          for (size_t l=1; l<=kmax; ++l)
            {
            psiarr[2*l-1]=T(cnpsi);
            psiarr[2*l]=T(snpsi);
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
          if (ncomp==1)
            {
            T val = data(i,0);
            for (size_t j=0; j<supp; ++j)
              for (size_t k=0; k<supp; ++k)
                for (size_t l=0; l<2*kmax+1; ++l)
                  cube.v(i0+j,i1+k,l,0) += val*wt[j]*wp[k]*psiarr[l];
            }
          else // ncomp==3
            {
            T v0=data(i,0), v1=data(i,1), v2=data(i,2);
            for (size_t j=0; j<supp; ++j)
              for (size_t k=0; k<supp; ++k)
                {
                T t0 = wt[j]*wp[k];
                for (size_t l=0; l<2*kmax+1; ++l)
                  {
                  T tmp = t0*psiarr[l];
                  cube.v(i0+j,i1+k,l,0) += v0*tmp;
                  cube.v(i0+j,i1+k,l,1) += v1*tmp;
                  cube.v(i0+j,i1+k,l,2) += v2*tmp;
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
            T fct = (((k+1)/2)&1) ? -1 : 1;
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
            T fct = (((k+1)/2)&1) ? -1 : 1;
            if (j2>=nphi) j2-=nphi;
            T tval = (cube(supp,j+supp,k,l) + fct*cube(supp,j2+supp,k,l));
            cube.v(supp,j+supp,k,l) = tval;
            cube.v(supp,j2+supp,k,l) = fct*tval;
            tval = (cube(supp+ntheta-1,j+supp,k,l) + fct*cube(supp+ntheta-1,j2+supp,k,l));
            cube.v(supp+ntheta-1,j+supp,k,l) = tval;
            cube.v(supp+ntheta-1,j2+supp,k,l) = fct*tval;
            }

      vector<T>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=std::sqrt(4*pi/(2*i+1.));

      for (size_t j=0; j<blm.size(); ++j)
        slm[j].SetToZero();

      for (size_t icomp=0; icomp<ncomp; ++icomp)
        {
        bool separate = ncomp>1;
        {
        auto m1 = cube.template subarray<2>({supp,supp,0,icomp},{ntheta,nphi,0,0});
        decorrect(m1,0);
        sharp_alm2map_adjoint(a1.Alms().vdata(), m1.data(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            if (separate)
              slm[icomp](l,m) += conj(a1(l,m))*blm[icomp](l,0).real()*T(lnorm[l]);
            else
              for (size_t j=0; j<blm.size(); ++j)
                slm[j](l,m) += conj(a1(l,m))*blm[j](l,0).real()*T(lnorm[l]);
        }
        for (size_t k=1; k<=kmax; ++k)
          {
          auto m1 = cube.template subarray<2>({supp,supp,2*k-1,icomp},{ntheta,nphi,0,0});
          auto m2 = cube.template subarray<2>({supp,supp,2*k  ,icomp},{ntheta,nphi,0,0});
          decorrect(m1,k);
          decorrect(m2,k);

          sharp_alm2map_spin_adjoint(k, a1.Alms().vdata(), a2.Alms().vdata(), m1.data(),
            m2.data(), *ginfo, *ainfo, 0, nthreads);
          for (size_t m=0; m<=lmax; ++m)
            for (size_t l=m; l<=lmax; ++l)
              if (l>=k)
                {
                if (separate)
                  {
                  auto tmp = conj(blm[icomp](l,k))*T(-2*lnorm[l]);
                  slm[icomp](l,m) += conj(a1(l,m))*tmp.real();
                  slm[icomp](l,m) -= conj(a2(l,m))*tmp.imag();
                  }
                else
                  for (size_t j=0; j<blm.size(); ++j)
                    {
                    auto tmp = conj(blm[j](l,k))*T(-2*lnorm[l]);
                    slm[j](l,m) += conj(a1(l,m))*tmp.real();
                    slm[j](l,m) -= conj(a2(l,m))*tmp.imag();
                    }
                }
          }
        }
      }
  };

double epsilon_guess(size_t support, double ofactor)
  { return std::sqrt(12.)*std::exp(-(support*ofactor)); }

}

using detail_interpol_ng::Interpolator;
using detail_interpol_ng::epsilon_guess;

}

#endif
