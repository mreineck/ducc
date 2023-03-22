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

/*! \file sht_utils.cc
 *  Utility functions related to spherical harmonic transforms
 *
 *  Copyright (C) 2020-2023 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_SHT_UTILS_H
#define DUCC0_SHT_UTILS_H

#include <complex>
#include <vector>
#include "ducc0/infra/mav.h"
#include "ducc0/math/constants.h"
#include "ducc0/fft/fft1d.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

template<typename T> void resample_theta(const cmav<complex<T>,3> &legi, bool npi, bool spi,
  vmav<complex<T>,3> &lego, bool npo, bool spo, size_t spin, size_t nthreads, bool adjoint)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  if ((npi==npo)&&(spi==spo)&&(legi.shape(1)==lego.shape(1)))  // shortcut
    {
    mav_apply([](complex<T> &a, complex<T> b) {a=b;}, nthreads, lego, legi);
    return;
    }
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-npi-spi;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-npo-spo;
  auto dthi = T(2*pi/nfull_in);
  auto dtho = T(2*pi/nfull_out);
  auto shift = T(0.5*(dtho*(1-npo)-dthi*(1-npi)));
  size_t nfull = max(nfull_in, nfull_out);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(nfull_in), plan_out(nfull_out);
  MultiExp<T,complex<T>> phase(adjoint ? -shift : shift, (shift==0.) ? 1 : nrings_in+2);
  execDynamic((nm+1)/2, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({nfull}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), plan_out.bufsize())}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{2*rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{2*rng.lo,MAXIDX}}));
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          for (size_t i=0, im=nfull_in-1+npi; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,2*j);
            complex<T> v2 = ((2*j+1)<llegi.shape(1)) ? llegi(i,2*j+1) : 0;
            tmp(i) = v1 + v2;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct * (v1-v2);
            else
              tmp(i) = (adjoint ? T(1) : T(0.5)) * (tmp(i) + fct*(v1-v2)); // sic!
            }
          plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), !adjoint);
          if (shift!=0)
            for (size_t i=1, im=nfull_in-1; (i<nrings_in+1)&&(i<=im); ++i,--im)
              {
              if (i!=im)
                tmp(i) *= phase[i];
              tmp(im) *= conj(phase[i]);
              }

          // zero padding/truncation
          if (nfull_out>nfull_in) // pad
            {
            size_t dist = nfull_out-nfull_in;
            size_t nmove = nfull_in/2;
            for (size_t i=nfull_out-1; i>nfull_out-1-nmove; --i)
              tmp(i) = tmp(i-dist);
            for (size_t i=nfull_out-nmove-dist; i<nfull_out-nmove; ++i)
              tmp(i) = 0;
            }
          if (nfull_out<nfull_in) // truncate
            {
            size_t dist = nfull_in-nfull_out;
            size_t nmove = nfull_out/2;
            for (size_t i=nfull_in-nmove; i<nfull_in; ++i)
              tmp(i-dist) = tmp(i);
            }
          plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), adjoint);
          auto norm = T(1./(2*(adjoint ? nfull_out : nfull_in)));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-1+npo-i;
            if (im==nfull_out) im=0;
            T fct2 = (adjoint && (im==i)) ? T(0.5) : 1;
            complex<T> v1 = fct2*tmp(i);
            complex<T> v2 = fct2*fct*tmp(im);
            llego(i,2*j) = norm * (v1 + v2);
            if ((2*j+1)<llego.shape(1))
              llego(i,2*j+1) = norm * (v1 - v2);
            }
          }
        }
      }
    });
  }

template<typename T> void resample_and_convolve_theta(const cmav<complex<T>,3> &legi, bool npi, bool spi,
  vmav<complex<T>,3> &lego, bool npo, bool spo, const vector<double> &kernel, size_t spin, size_t nthreads, bool adjoint)
  {
  constexpr size_t chunksize=64;
  MR_assert(legi.shape(0)==lego.shape(0), "number of components mismatch");
  auto nm = legi.shape(2);
  MR_assert(lego.shape(2)==nm, "dimension mismatch");
  if ((npi==npo)&&(spi==spo)&&(legi.shape(1)==lego.shape(1)))  // shortcut
    {
    mav_apply([](complex<T> &a, complex<T> b) {a=b;}, nthreads, lego, legi);
    return;
    }
  size_t nrings_in = legi.shape(1);
  size_t nfull_in = 2*nrings_in-npi-spi;
  size_t nrings_out = lego.shape(1);
  size_t nfull_out = 2*nrings_out-npo-spo;
  auto dthi = T(2*pi/nfull_in);
  auto dtho = T(2*pi/nfull_out);
  auto shift = T(0.5*(dtho*(1-npo)-dthi*(1-npi)));
  size_t nfull = max(nfull_in, nfull_out);
  T fct = ((spin&1)==0) ? 1 : -1;
  pocketfft_c<T> plan_in(nfull_in), plan_out(nfull_out);
  MultiExp<T,complex<T>> phase(adjoint ? -shift : shift, (shift==0.) ? 1 : nrings_in+2);
  size_t nsmall=min(nfull_in,nfull_out);
  execDynamic((nm+1)/2, nthreads, chunksize, [&](Scheduler &sched)
    {
    vmav<complex<T>,1> tmp({nfull}, UNINITIALIZED);
    vmav<complex<T>,1> buf({max(plan_in.bufsize(), plan_out.bufsize())}, UNINITIALIZED);
    while (auto rng=sched.getNext())
      {
      for (size_t n=0; n<legi.shape(0); ++n)
        {
        auto llegi(subarray<2>(legi, {{n},{},{2*rng.lo,MAXIDX}}));
        auto llego(subarray<2>(lego, {{n},{},{2*rng.lo,MAXIDX}}));
        for (size_t j=0; j+rng.lo<rng.hi; ++j)
          {
          // fill dark side
          for (size_t i=0, im=nfull_in-1+npi; (i<nrings_in)&&(i<=im); ++i,--im)
            {
            complex<T> v1 = llegi(i,2*j);
            complex<T> v2 = ((2*j+1)<llegi.shape(1)) ? llegi(i,2*j+1) : 0;
            tmp(i) = v1 + v2;
            if ((im<nfull_in) && (i!=im))
              tmp(im) = fct * (v1-v2);
            else
              tmp(i) = (adjoint ? T(1) : T(0.5)) * (tmp(i) + fct*(v1-v2)); // sic!
            }
          plan_in.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), !adjoint);
          tmp(0) *= T(kernel[0]);
          if (shift!=0)
            for (size_t i=1, im=nfull_in-1; (i<nrings_in+1)&&(i<=nsmall-i); ++i,--im)
              {
              if (i!=im)
                tmp(i) *= phase[i]*T(kernel[i]);
              tmp(im) *= conj(phase[i])*T(kernel[i]);
              }
          else
            for (size_t i=1, im=nfull_in-1; (i<nrings_in+1)&&(i<=nsmall-i); ++i,--im)
              {
              if (i!=im)
                tmp(i) *= T(kernel[i]);
              tmp(im) *= T(kernel[i]);
              }

          // zero padding/truncation
          if (nfull_out>nfull_in) // pad
            {
            size_t dist = nfull_out-nfull_in;
            size_t nmove = nfull_in/2;
            for (size_t i=nfull_out-1; i>nfull_out-1-nmove; --i)
              tmp(i) = tmp(i-dist);
            for (size_t i=nfull_out-nmove-dist; i<nfull_out-nmove; ++i)
              tmp(i) = 0;
            }
          if (nfull_out<nfull_in) // truncate
            {
            size_t dist = nfull_in-nfull_out;
            size_t nmove = nfull_out/2;
            for (size_t i=nfull_in-nmove; i<nfull_in; ++i)
              tmp(i-dist) = tmp(i);
            }
          plan_out.exec_copyback((Cmplx<T> *)tmp.data(), (Cmplx<T> *)buf.data(), T(1), adjoint);
          auto norm = T(1./(2*(adjoint ? nfull_out : nfull_in)));
          for (size_t i=0; i<nrings_out; ++i)
            {
            size_t im = nfull_out-1+npo-i;
            if (im==nfull_out) im=0;
            T fct2 = (adjoint && (im==i)) ? T(0.5) : 1;
            complex<T> v1 = fct2*tmp(i);
            complex<T> v2 = fct2*fct*tmp(im);
            llego(i,2*j) = norm * (v1 + v2);
            if ((2*j+1)<llego.shape(1))
              llego(i,2*j+1) = norm * (v1 - v2);
            }
          }
        }
      }
    });
  }

}

}

#endif
