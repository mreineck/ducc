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

/*! \file sht.h
 *  Functionality related to spherical harmonic transforms
 *
 *  \copyright Copyright (C) 2020-2021 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_SHT_H
#define DUCC0_SHT_H

#include <cmath>
#include <cstddef>
#include <string>
#include <complex>
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

enum SHT_mode { MAP2ALM,
                ALM2MAP,
                ALM2MAP_DERIV1
              };

void get_gridweights(const string &type, vmav<double,1> &wgt);
vmav<double,1> get_gridweights(const string &type, size_t nrings);

template<typename T> void alm2leg(  // associated Legendre transform
  const cmav<complex<T>,2> &alm, // (ncomp, lmidx)
  vmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads,
  SHT_mode mode=ALM2MAP);
template<typename T> void leg2alm(  // associated Legendre transform
  vmav<complex<T>,2> &alm, // (ncomp, lmidx)
  const cmav<complex<T>,3> &leg, // (ncomp, nrings, nm)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mval, // (nm)
  const cmav<size_t,1> &mstart, // (nm)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  size_t nthreads);

template<typename T> void map2leg(  // FFT
  const cmav<T,2> &map, // (ncomp, pix)
  vmav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads);
template<typename T> void leg2map(  // FFT
  vmav<T,2> &map, // (ncomp, pix)
  const cmav<complex<T>,3> &leg, // (ncomp, nrings, mmax+1)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads);

template<typename T> void synthesis(
  const cmav<complex<T>,2> &alm, // (ncomp, *)
  vmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads,
  SHT_mode mode=ALM2MAP);

template<typename T> void adjoint_synthesis(
  vmav<complex<T>,2> &alm, // (ncomp, *)
  const cmav<T,2> &map, // (ncomp, *)
  size_t spin,
  size_t lmax,
  const cmav<size_t,1> &mstart, // (mmax+1)
  ptrdiff_t lstride,
  const cmav<double,1> &theta, // (nrings)
  const cmav<size_t,1> &nphi, // (nrings)
  const cmav<double,1> &phi0, // (nrings)
  const cmav<size_t,1> &ringstart, // (nrings)
  ptrdiff_t pixstride,
  size_t nthreads);

template<typename T> void synthesis_2d(const cmav<complex<T>,2> &alm, vmav<T,3> &map,
  size_t spin, size_t lmax, size_t mmax, const string &geometry, size_t nthreads, SHT_mode mode=ALM2MAP);

template<typename T> void adjoint_synthesis_2d(vmav<complex<T>,2> &alm,
  const cmav<T,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);

template<typename T> void analysis_2d(vmav<complex<T>,2> &alm,
  const cmav<T,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);

template<typename T> void adjoint_analysis_2d(const cmav<complex<T>,2> &alm,
  vmav<T,3> &map, size_t spin, size_t lmax, size_t mmax,
  const string &geometry, size_t nthreads);

}

using detail_sht::SHT_mode;
using detail_sht::ALM2MAP;
using detail_sht::ALM2MAP_DERIV1;
using detail_sht::MAP2ALM;
using detail_sht::get_gridweights;
using detail_sht::alm2leg;
using detail_sht::leg2alm;
using detail_sht::map2leg;
using detail_sht::leg2map;
using detail_sht::synthesis;
using detail_sht::adjoint_synthesis;
using detail_sht::synthesis_2d;
using detail_sht::adjoint_synthesis_2d;
using detail_sht::analysis_2d;
using detail_sht::adjoint_analysis_2d;

}

#endif
