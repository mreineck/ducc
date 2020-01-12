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
  int spin;
  int nmaps, nalm;
  int flags;
  void **map;
  void **alm;
  int s_m, s_th; // strides in m and theta direction
  complex<double> *phase;
  vector<double> norm_l;
  complex<double> *almtmp;
  const sharp_geom_info *ginfo;
  const sharp_alm_info *ainfo;
  double time;
  unsigned long long opcnt;
  void build_common (sharp_jobtype type,
    int spin, void *alm, void *map, const sharp_geom_info &geom_info,
    const sharp_alm_info &alm_info, int flags);
  void alloc_phase (int nm, int ntheta, std::vector<complex<double>> &data);
  void alloc_almtmp (int lmax, std::vector<complex<double>> &data);
  void init_output();
  void alm2almtmp (int lmax, int mi);
  void almtmp2alm (int lmax, int mi);
  void ring2ringtmp (const sharp_geom_info::Tring &ri, std::vector<double> &ringtmp,
    int rstride);
  void ringtmp2ring (const sharp_geom_info::Tring &ri, const std::vector<double> &ringtmp, int rstride);
  void map2phase (int mmax, int llim, int ulim);
  void phase2map (int mmax, int llim, int ulim);
  void execute();
  };

void inner_loop (sharp_job &job, const int *ispair,const double *cth,
  const double *sth, int llim, int ulim, sharp_Ylmgen &gen, int mi,
  const int *mlim);

int sharp_max_nvec(int spin);

#endif
