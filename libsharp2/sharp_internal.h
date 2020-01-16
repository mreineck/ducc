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
#include "mr_util/error_handling.h"

using std::complex;

class sharp_protojob
  {
  public:
    sharp_jobtype type;
    size_t spin;
    size_t flags;
    ptrdiff_t s_m, s_th; // strides in m and theta direction
    complex<double> *phase;
    vector<double> norm_l;
    complex<double> *almtmp;
    const sharp_geom_info &ginfo;
    const sharp_alm_info &ainfo;
    double time;
    unsigned long long opcnt;

    sharp_protojob(sharp_jobtype type_, size_t spin_, const sharp_geom_info &ginfo_,
      const sharp_alm_info &ainfo_, size_t flags_)
      : type(type_), spin(spin_), flags(flags_), ginfo(ginfo_), ainfo(ainfo_),
        time(0.), opcnt(0)
      {
      if (type==SHARP_ALM2MAP_DERIV1) spin_=1;
      if (type==SHARP_MAP2ALM) flags|=SHARP_USE_WEIGHTS;
      if (type==SHARP_Yt) type=SHARP_MAP2ALM;
      if (type==SHARP_WY) { type=SHARP_ALM2MAP; flags|=SHARP_USE_WEIGHTS; }

      MR_assert(spin<=ainfo.lmax(), "bad spin");
      }
    void alloc_phase (size_t nm, size_t ntheta, std::vector<complex<double>> &data);
    void alloc_almtmp (size_t lmax, std::vector<complex<double>> &data);
    size_t nmaps() const { return 1+(spin>0); }
    size_t nalm() const { return (type==SHARP_ALM2MAP_DERIV1) ? 1 : (1+(spin>0)); }
  };

template<typename T> class sharp_job: public sharp_protojob
  {
  private:
    std::vector<std::complex<T> *> alm;
    std::vector<T *> map;

    void init_output();
    void alm2almtmp (size_t mi);
    void almtmp2alm (size_t mi);
    void ring2ringtmp (size_t iring, std::vector<double> &ringtmp,
      ptrdiff_t rstride);
    void ringtmp2ring (size_t iring, const std::vector<double> &ringtmp, ptrdiff_t rstride);
    void map2phase (size_t mmax, size_t llim, size_t ulim);
    void phase2map (size_t mmax, size_t llim, size_t ulim);

  public:
    sharp_job(sharp_jobtype type,
      size_t spin, const std::vector<std::complex<T> *> &alm_,
      const std::vector<T *> &map, const sharp_geom_info &geom_info,
      const sharp_alm_info &alm_info, size_t flags);

      void execute();
  };

void inner_loop (sharp_protojob &job, const std::vector<bool> &ispair,
  const std::vector<double> &cth, const std::vector<double> &sth, size_t llim,
  size_t ulim, sharp_Ylmgen &gen, size_t mi, const std::vector<size_t> &mlim);

size_t sharp_max_nvec(size_t spin);

#endif
