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

/* Copyright (C) 2019-2025 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_NUFFT_SPREADINTERP_H
#define DUCC0_NUFFT_SPREADINTERP_H

#include <complex>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_nufft {

using namespace std;

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx, size_t ndim> class Spreadinterp;

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> class Spreadinterp2
  {
  private:
    unique_ptr<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 1>> si1;
    unique_ptr<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 2>> si2;
    unique_ptr<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 3>> si3;

  public:
    Spreadinterp2(size_t npoints,
      const vector<size_t> &over_shape, size_t kidx,
      size_t nthreads,
      const vector<double> &periodicity,
      const vector<double> &corigin=vector<double>());
    Spreadinterp2(const cmav<Tcoord,2> &coords,
      const vector<size_t> &over_shape, size_t kidx,
      size_t nthreads, const vector<double> &periodicity,
      const vector<double> &corigin=vector<double>());
    ~Spreadinterp2();
    template<typename Tpoints, typename Tgrid> void spread(
      const cmav<complex<Tpoints>,1> &points, const vfmav<complex<Tgrid>> &grid);
    template<typename Tpoints, typename Tgrid> void interp(
      const cfmav<complex<Tgrid>> &grid, const vmav<complex<Tpoints>,1> &points);
    template<typename Tpoints, typename Tgrid> void spread(
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vfmav<complex<Tgrid>> &grid);
    template<typename Tpoints, typename Tgrid> void interp(
      const cfmav<complex<Tgrid>> &grid, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points);
  };

}}

#endif
