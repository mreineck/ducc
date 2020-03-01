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

template<typename T> class Interpolator
  {
  protected:
    size_t lmax, kmax, nphi0, ntheta0, nphi, ntheta;
    double ofactor;
    size_t supp;
    ES_Kernel kernel;
    mav<T,3> cube; // the data cube (theta, phi, 2*mbeam+1[, IQU])

    void correct(mav<T,2> &arr)
      {
      mav<T,2> tmp({nphi,nphi});
      auto tmp0=tmp.template subarray<2>({0,0},{nphi0, nphi0});
      fmav<T> ftmp0(tmp0);
      for (size_t i=0; i<ntheta0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) = arr(i,j);
      // extend to second half
      for (size_t i=1, i2=2*ntheta0-3; i+1<ntheta0; ++i,--i2)
        for (size_t j=0,j2=nphi0/2; j<nphi0; ++j,++j2)
          {
          if (j2>=nphi0) j2-=nphi0;
          tmp0.v(i2,j) = arr(i,j2);
          }
      // FFT to frequency domain on minimal grid
      r2r_fftpack(ftmp0,ftmp0,{0,1},true,true,1./(nphi0*nphi0),0);
      auto fct = kernel.correction_factors(nphi, nphi0, 0);
      for (size_t i=0; i<nphi0; ++i)
        for (size_t j=0; j<nphi0; ++j)
          tmp0.v(i,j) *= fct[(i+1)/2] * fct[(j+1)/2];
      auto tmp1=tmp.template subarray<2>({0,0},{nphi, nphi0});
      fmav<T> ftmp1(tmp1);
      // zero-padded FFT in theta direction
      r2r_fftpack(ftmp1,ftmp1,{0},false,false,1.,0);
      auto tmp2=tmp.template subarray<2>({0,0},{ntheta, nphi});
      fmav<T> ftmp2(tmp2);
      fmav<T> farr(arr);
      // zero-padded FFT in phi direction
      r2r_fftpack(ftmp2,farr,{1},false,false,1.,0);
      }

  public:
    Interpolator(const Alm<complex<T>> &slmT, const Alm<complex<T>> &blmT,
      double epsilon)
      : lmax(slmT.Lmax()),
        kmax(blmT.Mmax()),
        nphi0(2*good_size_real(lmax+1)),
        ntheta0(nphi0/2+1),
        nphi(2*good_size_real(2*lmax+1)),
        ntheta(nphi/2+1),
        ofactor(double(nphi)/(2*lmax+1)),
        supp(ES_Kernel::get_supp(epsilon, ofactor)),
        kernel(supp, ofactor, 1),
        cube({ntheta+2*supp, nphi+2*supp, 2*kmax+1})
      {
      MR_assert((supp<=ntheta) && (supp<=nphi), "support too large!");
      MR_assert(slmT.Mmax()==lmax, "Sky lmax must be equal to Sky mmax");
      MR_assert(blmT.Lmax()==lmax, "Sky and beam lmax must be equal");
      Alm<complex<T>> a1(lmax, lmax), a2(lmax,lmax);
      auto ginfo = sharp_make_cc_geom_info(ntheta0,nphi0,0,cube.stride(1),cube.stride(0));
      auto ainfo = sharp_make_triangular_alm_info(lmax,lmax,1);

      vector<double>lnorm(lmax+1);
      for (size_t i=0; i<=lmax; ++i)
        lnorm[i]=sqrt(4*pi/(2*i+1.));

      for (size_t k=0; k<=kmax; ++k)
        {
        double spinsign = (k==0) ? 1. : -1.;
        for (size_t m=0; m<=lmax; ++m)
          {
          T mfac=T((m&1) ? -1.:1.);
          for (size_t l=m; l<=lmax; ++l)
            {
            if (l<k)
              a1(l,m)=a2(l,m)=0.;
            else
              {
              complex<T> v1=slmT(l,m)*blmT(l,k),
                         v2=conj(slmT(l,m))*blmT(l,k)*mfac;
              a1(l,m) = (v1+conj(v2)*mfac)*T(0.5*spinsign*lnorm[l]);
              if (k>0)
                {
                complex<T> tmp = (v1-conj(v2)*mfac)*T(-spinsign*0.5*lnorm[l]);
                a2(l,m) = complex<T>(-tmp.imag(), tmp.real());
                }
              }
            }
          }
        size_t kidx1 = (k==0) ? 0 : 2*k-1,
               kidx2 = (k==0) ? 0 : 2*k;
        auto quadrant=k%4;
        if (quadrant&1)
          swap(kidx1, kidx2);
        auto m1 = cube.template subarray<2>({supp,supp,kidx1},{ntheta,nphi,0});
        auto m2 = cube.template subarray<2>({supp,supp,kidx2},{ntheta,nphi,0});
        if (k==0)
          sharp_alm2map(a1.Alms().data(), m1.vdata(), *ginfo, *ainfo, 0, nullptr, nullptr);
        else
          sharp_alm2map_spin(k, a1.Alms().data(), a2.Alms().data(), m1.vdata(), m2.vdata(), *ginfo, *ainfo, 0, nullptr, nullptr);
        correct(m1);
        if (k!=0) correct(m2);

        if ((quadrant==1)||(quadrant==2)) m1.apply([](T &v){v=-v;});
        if ((k>0) &&((quadrant==0)||(quadrant==1))) m2.apply([](T &v){v=-v;});
        }
      // fill border regions
      for (size_t i=0; i<supp; ++i)
        for (size_t j=0, j2=nphi/2; j<nphi; ++j,++j2)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            if (j2>=nphi) j2-=nphi;
            cube.v(supp-1-i,j2+supp,k) = cube(supp+1+i,j+supp,k);
            cube.v(supp+ntheta+i,j2+supp,k) = cube(supp+ntheta-2-i, j+supp,k);
            }
      for (size_t i=0; i<ntheta+2*supp; ++i)
        for (size_t j=0; j<supp; ++j)
          for (size_t k=0; k<cube.shape(2); ++k)
            {
            cube.v(i,j,k) = cube(i,j+nphi,k);
            cube.v(i,j+nphi+supp,k) = cube(i,j+supp,k);
            }
      }

    void interpolx (const mav<T,2> &ptg, mav<T,1> &res) const
      {
      MR_assert(ptg.shape(0)==res.shape(0), "dimension mismatch");
      MR_assert(ptg.shape(1)==3, "second dimension must have length 3");
      vector<T> wt(supp), wp(supp);
      vector<T> psiarr(2*kmax+1);
      double delta = 2./supp;
      double xdtheta = (ntheta-1)/pi,
             xdphi = nphi/(2*pi);
      for (size_t i=0; i<ptg.shape(0); ++i)
        {
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
        double cpsi=cos(ptg(i,2)), spsi=sin(ptg(i,2));
        double cnpsi=cpsi, snpsi=spsi;
        for (size_t l=1; l<=kmax; ++l)
          {
          psiarr[2*l-1]=cnpsi;
          psiarr[2*l]=-snpsi;
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
      }
  };

template<typename T> class PyInterpolator: public Interpolator<T>
  {
  public:
    PyInterpolator(const py::array &slmT, const py::array &blmT, int64_t lmax, int64_t kmax, double epsilon)
      : Interpolator<T>(Alm<complex<T>>(to_mav<complex<T>,1>(slmT), lmax, lmax),
                        Alm<complex<T>>(to_mav<complex<T>,1>(blmT), lmax, kmax),
                        epsilon) {}
    using Interpolator<T>::interpolx;
    py::array interpol(const py::array &ptg)
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto res = make_Pyarr<double>({ptg2.shape(0)});
      auto res2 = to_mav<double,1>(res,true);
      interpolx(ptg2, res2);
      return res;
      }
  };

} // unnamed namespace

PYBIND11_MODULE(interpol_ng, m)
  {
  using namespace pybind11::literals;

  py::class_<PyInterpolator<double>> (m, "PyInterpolator")
    .def(py::init<const py::array &, const py::array &, int64_t, int64_t, double>(),
      "sky"_a, "beam"_a, "lmax"_a, "kmax"_a, "epsilon"_a)
    .def ("interpol", &PyInterpolator<double>::interpol, "ptg"_a);
  }
