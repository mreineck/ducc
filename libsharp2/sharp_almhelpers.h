/*
 *  This file is part of libsharp2.
 *
 *  libsharp2 is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp2 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp2; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* libsharp2 is being developed at the Max-Planck-Institut fuer Astrophysik */

/*! \file sharp_almhelpers.h
 *  SHARP helper function for the creation of a_lm data structures
 *
 *  Copyright (C) 2008-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef SHARP2_ALMHELPERS_H
#define SHARP2_ALMHELPERS_H


#include <memory>
#include "libsharp2/sharp.h"

/*! Initialises an a_lm data structure according to the scheme used by
    Healpix_cxx.
    \ingroup almgroup */
std::unique_ptr<sharp_alm_info> sharp_make_triangular_alm_info (int lmax, int mmax, int stride);

/*! Initialises an a_lm data structure according to the scheme used by
    Fortran Healpix
    \ingroup almgroup */
std::unique_ptr<sharp_alm_info> sharp_make_rectangular_alm_info (int lmax, int mmax, int stride);

#endif
