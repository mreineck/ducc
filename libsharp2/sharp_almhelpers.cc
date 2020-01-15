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

/*! \file sharp_almhelpers.c
 *  Spherical transform library
 *
 *  Copyright (C) 2008-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include "libsharp2/sharp_almhelpers.h"

using namespace std;

unique_ptr<sharp_standard_alm_info> sharp_make_triangular_alm_info (int lmax, int mmax, int stride)
  {
  vector<ptrdiff_t> mvstart(mmax+1);
  ptrdiff_t tval = 2*lmax+1;
  for (ptrdiff_t m=0; m<=mmax; ++m)
    mvstart[m] = stride*((m*(tval-m))>>1);
  return make_unique<sharp_standard_alm_info>(lmax, mmax, stride, mvstart.data());
  }

unique_ptr<sharp_standard_alm_info> sharp_make_rectangular_alm_info (int lmax, int mmax, int stride)
  {
  vector<ptrdiff_t> mvstart(mmax+1);
  for (ptrdiff_t m=0; m<=mmax; ++m)
    mvstart[m] = stride*m*(lmax+1);
  return make_unique<sharp_standard_alm_info>(lmax, mmax, stride, mvstart.data());
  }
