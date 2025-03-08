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

/* Copyright (C) 2025 Max-Planck-Society
   Author: Martin Reinecke */

#include "ducc0/nufft/spreadinterp_impl.h"

namespace ducc0 {
namespace detail_nufft {

#define BOILERPLATE \
template class Spreadinterp2<Tcalc,Tacc,Tcoord,Tidx>; \
template void Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::spread( \
      const cmav<complex<Tpoints>,1> &points, const vfmav<complex<Tgrid>> &grid); \
template void Spreadinterp2<Tcalc,Tacc,Tcoord,Tidx>::interp( \
      const cfmav<complex<Tgrid>> &grid, const vmav<complex<Tpoints>,1> &points); \
template void Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::spread( \
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points, \
      const vfmav<complex<Tgrid>> &grid); \
template void Spreadinterp2<Tcalc,Tacc,Tcoord,Tidx>::interp( \
      const cfmav<complex<Tgrid>> &grid, const cmav<Tcoord,2> &coords, \
      const vmav<complex<Tpoints>,1> &points);

#define Tidx uint32_t

#define Tcalc float
#define Tacc float
#define Tcoord float

#define Tpoints float
#define Tgrid float

BOILERPLATE

#undef Tcoord
#define Tcoord double

BOILERPLATE

#undef Tgrid
#undef Tpoints

#undef Tcoord
#undef Tacc
#undef Tcalc

}}
