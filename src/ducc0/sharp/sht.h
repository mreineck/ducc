#ifndef DUCC0_SHT_H
#define DUCC0_SHT_H

#include <complex>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_sht {

using namespace std;

enum SHT_mode { MAP2ALM,
                ALM2MAP,
                ALM2MAP_DERIV1
              };

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
  const mav<complex<T>,3> &leg, // (lmidx, ncomp)
  mav<complex<T>,2> &alm, // (nrings, nm, ncomp)
  const mav<double,1> &theta, // (nrings)
  const mav<size_t,1> &mval, // (nm)
  const mav<size_t,1> &mstart, // (nm)
  size_t lmax,
  size_t spin,
  size_t nthreads);

}

using detail_sht::SHT_mode;
using detail_sht::ALM2MAP;
using detail_sht::MAP2ALM;
using detail_sht::alm2leg;
using detail_sht::leg2alm;

}

#endif
