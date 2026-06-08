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

#ifndef DUCC0_WGRIDDER_H
#define DUCC0_WGRIDDER_H

#include <complex>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_gridder {

using namespace std;

template<typename Tcalc, typename Tacc, typename Tms, typename Tms_in=cmav<complex<Tms>,2>, typename Timg> void ms2dirty(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const Tms_in &ms,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
  double sigma_max, double center_x, double center_y, bool allow_nshift);

template<typename Tcalc, typename Tacc, typename Tms, typename Tms_in=cmav<complex<Tms>,1>, typename Timg>
  void ms2dirty_bda(
    const cmav<double,2> &uvw,                        // (nrows,3)
    const cmav<size_t,1> &freqlist_id,                // (nrows),
    const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
    const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
    const Tms_in &ms,                                 // concatenated array of visibilities, shape (sum_i(freqlist_nfreq[freqlist_id[i]))
    const cmav<Tms,1> &wgt_,                          // same shape as above
    const cmav<uint8_t,1> &mask_,                     // same shape as above
    double pixsize_x, double pixsize_y, double epsilon,
    bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
    bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
    double sigma_max, double center_x, double center_y, bool allow_nshift);

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void dirty2ms(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<Timg,2> &dirty,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,2> &ms,
  size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift);

template<typename Tcalc, typename Tacc, typename Tms, typename Timg>
  void dirty2ms_bda(
    const cmav<double,2> &uvw,
    const cmav<size_t,1> &freqlist_id,                // (nrows),
    const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
    const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
    const cmav<Timg,2> &dirty,
    const cmav<Tms,1> &wgt_, const cmav<uint8_t,1> &mask_, double pixsize_x, double pixsize_y,
    double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,1> &ms,
    size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
    double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift);

tuple<size_t, size_t, size_t, size_t, double, double>
 get_facet_data(size_t npix_x, size_t npix_y, size_t nfx, size_t nfy, size_t ifx, size_t ify,
  double pixsize_x, double pixsize_y, double center_x, double center_y);

template<typename Tcalc, typename Tacc, typename Tms, typename Timg, typename Tms_in=cmav<complex<Tms>,2>> void ms2dirty_tuning(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const Tms_in &ms,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
  double sigma_max, double center_x, double center_y);

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void dirty2ms_tuning(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<Timg,2> &dirty,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,2> &ms,
  size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  double sigma_min, double sigma_max, double center_x, double center_y);

} // namespace detail_gridder

// public names
using detail_gridder::ms2dirty;
using detail_gridder::dirty2ms;
using detail_gridder::ms2dirty_bda;
using detail_gridder::dirty2ms_bda;
using detail_gridder::ms2dirty_tuning;
using detail_gridder::dirty2ms_tuning;

} // namespace ducc0

#endif
