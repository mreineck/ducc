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

/*! \file sharp_internal.h
 *  Internally used functionality for the spherical transform library.
 *
 *  Copyright (C) 2006-2019 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#ifndef SHARP2_INTERNAL_H
#define SHARP2_INTERNAL_H

#include <complex>
#include "libsharp2/sharp.h"
#include "libsharp2/sharp_ylmgen.h"

using std::complex;

struct sharp_job
  {
  sharp_jobtype type;
  size_t spin;
  size_t nmaps, nalm;
  size_t flags;
  void **map;
  void **alm;
  ptrdiff_t s_m, s_th; // strides in m and theta direction
  complex<double> *phase;
  vector<double> norm_l;
  complex<double> *almtmp;
  const sharp_geom_info *ginfo;
  const sharp_alm_info *ainfo;
  double time;
  unsigned long long opcnt;
  void build_common (sharp_jobtype type,
    size_t spin, void *alm, void *map, const sharp_geom_info &geom_info,
    const sharp_alm_info &alm_info, size_t flags);
  void alloc_phase (size_t nm, size_t ntheta, std::vector<complex<double>> &data);
  void alloc_almtmp (size_t lmax, std::vector<complex<double>> &data);
  void init_output();
  void alm2almtmp (size_t lmax, size_t mi);
  void almtmp2alm (size_t lmax, size_t mi);
  void ring2ringtmp (const sharp_geom_info::Tring &ri, std::vector<double> &ringtmp,
    ptrdiff_t rstride);
  void ringtmp2ring (const sharp_geom_info::Tring &ri, const std::vector<double> &ringtmp, ptrdiff_t rstride);
  void map2phase (size_t mmax, size_t llim, size_t ulim);
  void phase2map (size_t mmax, size_t llim, size_t ulim);
  void execute();
  };

void inner_loop (sharp_job &job, const int *ispair,const double *cth,
  const double *sth, size_t llim, size_t ulim, sharp_Ylmgen &gen, size_t mi,
  const size_t *mlim);

size_t sharp_max_nvec(size_t spin);

#endif
