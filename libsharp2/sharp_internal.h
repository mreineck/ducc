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

typedef struct
  {
  sharp_jobtype type;
  int spin;
  int nmaps, nalm;
  int flags;
  void **map;
  void **alm;
  int s_m, s_th; // strides in m and theta direction
  complex<double> *phase;
  double *norm_l;
  complex<double> *almtmp;
  const sharp_geom_info *ginfo;
  const sharp_alm_info *ainfo;
  double time;
  unsigned long long opcnt;
  } sharp_job;

void inner_loop (sharp_job *job, const int *ispair,const double *cth,
  const double *sth, int llim, int ulim, sharp_Ylmgen_C *gen, int mi,
  const int *mlim);

int sharp_max_nvec(int spin);

#endif
