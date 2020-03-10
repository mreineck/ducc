/*
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>
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

using namespace std;
using namespace mr;

namespace py = pybind11;

namespace {

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
    mav<T,3> cube; // the data cube (theta, phi, 2*mbeam+1[, IQU])

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
    Interpolator(const Alm<complex<T>> &slmT, const Alm<complex<T>> &blmT,
      double epsilon, int nthreads_)
      : adjoint(false),
        lmax(slmT.Lmax()),
        kmax(blmT.Mmax()),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*ofmin/2.)))),
        ntheta(nphi/2+1),
        nthreads(nthreads_),
        ofactor(double(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, nthreads),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1})
      {
      MR_assert((supp<=ntheta) && (supp<=nphi), "support too large!");
      MR_assert(slmT.Mmax()==lmax, "Sky lmax must be equal to Sky mmax");
      MR_assert(blmT.Lmax()==lmax, "Sky and beam lmax must be equal");
      Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
      auto ginfo = sharp_make_cc_geom_info(ntheta0,nphi0,0.,cube.stride(1),cube.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      vector<double>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=sqrt(4*pi/(2*i+1.));

      {
      for (size_t m=0; m<=lmax; ++m)
        for (size_t l=m; l<=lmax; ++l)
          a1(l,m) = slmT(l,m)*blmT(l,0).real()*T(lnorm[l]);
      auto m1 = cube.template subarray<2>({supp,supp,0},{ntheta,nphi,0});
      sharp_alm2map(a1.Alms().data(), m1.vdata(), *ginfo, *ainfo, 0, nthreads);
      correct(m1,0);
      }
      for (size_t k=1; k<=kmax; ++k)
        {
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            {
            if (l<k)
              a1(l,m)=a2(l,m)=0.;
            else
              {
              auto tmp = -2.*blmT(l,k)*T(lnorm[l]);
              a1(l,m) = slmT(l,m)*tmp.real();
              a2(l,m) = slmT(l,m)*tmp.imag();
              }
            }
        auto m1 = cube.template subarray<2>({supp,supp,2*k-1},{ntheta,nphi,0});
        auto m2 = cube.template subarray<2>({supp,supp,2*k  },{ntheta,nphi,0});
        sharp_alm2map_spin(k, a1.Alms().data(), a2.Alms().data(), m1.vdata(),
          m2.vdata(), *ginfo, *ainfo, 0, nthreads);
        correct(m1,k);
        correct(m2,k);
        }
      // fill border regions
      for (size_t i=0; i<supp; ++i)
        for (size_t j=0, j2=nphi/2; j<nphi; ++j,++j2)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            double fct = (((k+1)/2)&1) ? -1 : 1;
            if (j2>=nphi) j2-=nphi;
            cube.v(supp-1-i,j2+supp,k) = fct*cube(supp+1+i,j+supp,k);
            cube.v(supp+ntheta+i,j2+supp,k) = fct*cube(supp+ntheta-2-i, j+supp,k);
            }
      for (size_t i=0; i<ntheta+2*supp; ++i)
        for (size_t j=0; j<supp; ++j)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            cube.v(i,j,k) = cube(i,j+nphi,k);
            cube.v(i,j+nphi+supp,k) = cube(i,j+supp,k);
            }
      }

    Interpolator(size_t lmax_, size_t kmax_, double epsilon, int nthreads_)
      : adjoint(true),
        lmax(lmax_),
        kmax(kmax_),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(max<size_t>(20,2*good_size_real(size_t((2*lmax+1)*ofmin/2.)))),
        ntheta(nphi/2+1),
        nthreads(nthreads_),
        ofactor(double(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, nthreads),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1})
      {
      MR_assert((supp<=ntheta) && (supp<=nphi), "support too large!");
      cube.apply([](T &v){v=0.;});
      }

    void interpolx (const mav<T,2> &ptg, mav<T,1> &res) const
      {
      MR_assert(!adjoint, "cannot be called in adjoint mode");
      MR_assert(ptg.shape(0)==res.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      double delta = 2./supp;
      double xdtheta = (ntheta-1)/pi,
             xdphi = nphi/(2*pi);
      auto idx = getIdx(ptg);
      execStatic(idx.size(), nthreads, 0, [&](Scheduler &sched)
        {
        vector<T> wt(supp), wp(supp);
        vector<T> psiarr(2*kmax+1);
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
          double val=0;
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
          for (size_t j=0; j<supp; ++j)
            for (size_t k=0; k<supp; ++k)
              for (size_t l=0; l<2*kmax+1; ++l)
                val += cube(i0+j,i1+k,l)*wt[j]*wp[k]*psiarr[l];
          res.v(i) = val;
          }
        });
      }

    void deinterpolx (const mav<T,2> &ptg, const mav<T,1> &data)
      {
      MR_assert(adjoint, "can only be called in adjoint mode");
      MR_assert(ptg.shape(0)==data.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      double delta = 2./supp;
      double xdtheta = (ntheta-1)/pi,
             xdphi = nphi/(2*pi);
      auto idx = getIdx(ptg);
      execStatic(idx.size(), 1, 0, [&](Scheduler &sched) // not parallel yet
        {
        vector<T> wt(supp), wp(supp);
        vector<T> psiarr(2*kmax+1);
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
          double val=data(i);
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
          for (size_t j=0; j<supp; ++j)
            for (size_t k=0; k<supp; ++k)
              for (size_t l=0; l<2*kmax+1; ++l)
                cube.v(i0+j,i1+k,l) += val*wt[j]*wp[k]*psiarr[l];
          }
        });
      }
    void getSlmx (const Alm<complex<T>> &blmT, Alm<complex<T>> &slmT)
      {
      MR_assert(adjoint, "can only be called in adjoint mode");
      Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
      auto ginfo = sharp_make_cc_geom_info(ntheta0,nphi0,0.,cube.stride(1),cube.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      // move stuff from border regions onto the main grid
      for (size_t i=0; i<cube.shape(0); ++i)
        for (size_t j=0; j<supp; ++j)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            cube.v(i,j+nphi,k) += cube(i,j,k);
            cube.v(i,j+supp,k) += cube(i,j+nphi+supp,k);
            }
      for (size_t i=0; i<supp; ++i)
        for (size_t j=0, j2=nphi/2; j<nphi; ++j,++j2)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            double fct = (((k+1)/2)&1) ? -1 : 1;
            if (j2>=nphi) j2-=nphi;
            cube.v(supp+1+i,j+supp,k) += fct*cube(supp-1-i,j2+supp,k);
            cube.v(supp+ntheta-2-i, j+supp,k) += fct*cube(supp+ntheta+i,j2+supp,k);
            }

      // special treatment for poles
      for (size_t j=0,j2=nphi/2; j<nphi/2; ++j,++j2)
        for (size_t k=0; k<cube.shape(2); ++k)
          {
          double fct = (((k+1)/2)&1) ? -1 : 1;
          if (j2>=nphi) j2-=nphi;
          double tval = (cube(supp,j+supp,k) + fct*cube(supp,j2+supp,k));
          cube.v(supp,j+supp,k) = tval;
          cube.v(supp,j2+supp,k) = fct*tval;
          tval = (cube(supp+ntheta-1,j+supp,k) + fct*cube(supp+ntheta-1,j2+supp,k));
          cube.v(supp+ntheta-1,j+supp,k) = tval;
          cube.v(supp+ntheta-1,j2+supp,k) = fct*tval;
          }

      vector<double>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=sqrt(4*pi/(2*i+1.));

      {
      auto m1 = cube.template subarray<2>({supp,supp,0},{ntheta,nphi,0});
      decorrect(m1,0);
      sharp_alm2map_adjoint(a1.Alms().vdata(), m1.data(), *ginfo, *ainfo, 0, nthreads);
      for (size_t m=0; m<=lmax; ++m)
        for (size_t l=m; l<=lmax; ++l)
          slmT(l,m)=conj(a1(l,m))*blmT(l,0).real()*T(lnorm[l]);
      }

      for (size_t k=1; k<=kmax; ++k)
        {
        auto m1 = cube.template subarray<2>({supp,supp,2*k-1},{ntheta,nphi,0});
        auto m2 = cube.template subarray<2>({supp,supp,2*k  },{ntheta,nphi,0});
        decorrect(m1,k);
        decorrect(m2,k);

        sharp_alm2map_spin_adjoint(k, a1.Alms().vdata(), a2.Alms().vdata(), m1.data(),
          m2.data(), *ginfo, *ainfo, 0, nthreads);
        for (size_t m=0; m<=lmax; ++m)
          for (size_t l=m; l<=lmax; ++l)
            {
            if (l>=k)
              {
              auto tmp = -2.*conj(blmT(l,k))*T(lnorm[l]);
              slmT(l,m) += conj(a1(l,m))*tmp.real();
              slmT(l,m) -= conj(a2(l,m))*tmp.imag();
              }
            }
        }
      }
  };

template<typename T> class PyInterpolator: public Interpolator<T>
  {
  protected:
    using Interpolator<T>::lmax;
    using Interpolator<T>::kmax;
    using Interpolator<T>::interpolx;
    using Interpolator<T>::deinterpolx;
    using Interpolator<T>::getSlmx;

  public:
    PyInterpolator(const py::array &slmT, const py::array &blmT,
      int64_t lmax, int64_t kmax, double epsilon, int nthreads=0)
      : Interpolator<T>(Alm<complex<T>>(to_mav<complex<T>,1>(slmT), lmax, lmax),
                        Alm<complex<T>>(to_mav<complex<T>,1>(blmT), lmax, kmax),
                        epsilon, nthreads) {}
    PyInterpolator(int64_t lmax, int64_t kmax, double epsilon, int nthreads=0)
      : Interpolator<T>(lmax, kmax, epsilon, nthreads) {}
    py::array interpol(const py::array &ptg) const
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto res = make_Pyarr<double>({ptg2.shape(0)});
      auto res2 = to_mav<double,1>(res,true);
      interpolx(ptg2, res2);
      return res;
      }

    void deinterpol(const py::array &ptg, const py::array &data)
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto data2 = to_mav<T,1>(data);
      deinterpolx(ptg2, data2);
      }
    py::array getSlm(const py::array &blmT_)
      {
      auto res = make_Pyarr<complex<T>>({Alm_Base::Num_Alms(lmax, lmax)});
      Alm<complex<T>> blmT(to_mav<complex<T>,1>(blmT_, false), lmax, kmax);
      auto slmT_=to_mav<complex<T>,1>(res, true);
      Alm<complex<T>> slmT(slmT_, lmax, lmax);
      getSlmx(blmT, slmT);
      return res;
      }
  };

#if 1
template<typename T> py::array pyrotate_alm(const py::array &alm_, int64_t lmax,
  double psi, double theta, double phi)
  {
  auto a1 = to_mav<complex<T>,1>(alm_);
  auto alm = make_Pyarr<complex<T>>({a1.shape(0)});
  auto a2 = to_mav<complex<T>,1>(alm,true);
  for (size_t i=0; i<a1.shape(0); ++i) a2.v(i)=a1(i);
  auto blah = Alm<complex<T>>(a2,lmax,lmax);
  rotate_alm(blah, psi, theta, phi);
  return alm;
  }
#endif

} // unnamed namespace

PYBIND11_MODULE(interpol_ng, m)
  {
  using namespace pybind11::literals;

  py::class_<PyInterpolator<double>> (m, "PyInterpolator")
    .def(py::init<const py::array &, const py::array &, int64_t, int64_t, double, int>(),
      "sky"_a, "beam"_a, "lmax"_a, "kmax"_a, "epsilon"_a, "nthreads"_a)
    .def(py::init<int64_t, int64_t, double, int>(),
      "lmax"_a, "kmax"_a, "epsilon"_a, "nthreads"_a)
    .def ("interpol", &PyInterpolator<double>::interpol, "ptg"_a)
    .def ("deinterpol", &PyInterpolator<double>::deinterpol, "ptg"_a, "data"_a)
    .def ("getSlm", &PyInterpolator<double>::getSlm, "blmT"_a);
#if 1
  m.def("rotate_alm", &pyrotate_alm<double>, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a);
#endif
  }
