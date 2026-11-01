/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file constants.h
 *  Mathematical constants.
 */

#ifndef DUCC0_CONSTANTS_H
#define DUCC0_CONSTANTS_H

#include <numbers>

namespace DUCC0_NAMESPACE {

/*! \defgroup mathconstgroup Mathematical constants */
/*! \{ */

constexpr double pi=std::numbers::pi_v<double>;
constexpr double twopi=2*pi;
constexpr double inv_twopi=0.5*std::numbers::inv_pi_v<double>;
constexpr double fourpi=4*pi;
constexpr double halfpi=0.5*pi;
constexpr double inv_halfpi=2./pi;
constexpr double inv_sqrt4pi = 0.5*std::numbers::inv_sqrtpi_v<double>;
constexpr double ln2 = std::numbers::ln2_v<double>;

#if 0
//! Ratio between FWHM and sigma of a Gaussian (\f$\sqrt{8\ln2}\f$).
constexpr double sigma2fwhm=2.3548200450309493; // sqrt(8*log(2.))
//! Ratio between sigma and FWHM of a Gaussian (\f$1/\sqrt{8\ln2}\f$).
constexpr double fwhm2sigma=1/sigma2fwhm;
#endif

/*! \} */

}

#endif
