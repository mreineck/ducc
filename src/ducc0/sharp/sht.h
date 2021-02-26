#ifndef DUCC0_SHT_H
#define DUCC0_SHT_H

#include <complex>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

template<typename T> void alm2leg(  // associated Legendre transform
  const mav<complex<T>,2> &alm, // (lmidx, ncomp)
  mav<complex<T>,3> &leg, // (nrings, nm, ncomp)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads);

template<typename T> void leg2alm(  // associated Legendre transform
  mav<complex<T>,2> &alm, // (lmidx, ncomp)
  const mav<complex<T>,3> &leg, // (nrings, nm, ncomp)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads);

}

using detail_sht::alm2leg;
using detail_sht::leg2alm;

}

#endif
