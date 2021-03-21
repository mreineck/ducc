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
 *  Copyright (C) 2006-2020 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#ifndef SHARP2_INTERNAL_H
#define SHARP2_INTERNAL_H

#include <complex>
#include <vector>
#include "ducc0/sharp/sharp.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_sharp {

using std::complex;

class sharp_job
  {
  private:
    std::vector<std::any> alm;
    std::vector<std::any> map;

    void init_output();
    void alm2almtmp (size_t mi, mav<complex<double>,2> &almtmp);
    void almtmp2alm (size_t mi, mav<complex<double>,2> &almtmp);
    void ring2ringtmp (size_t iring, mav<double,2> &ringtmp);
    void ringtmp2ring (size_t iring, const mav<double,2> &ringtmp);
    void map2phase (size_t mmax, size_t llim, size_t ulim, mav<complex<double>,3> &phase);
    void phase2map (size_t mmax, size_t llim, size_t ulim, const mav<complex<double>,3> &phase);

  public:
    sharp_jobtype type;
    size_t spin;
    size_t flags;
    std::vector<double> norm_l;
    const sharp_geom_info &ginfo;
    const sharp_alm_info &ainfo;
    int nthreads;

    sharp_job(sharp_jobtype type,
      size_t spin, const std::vector<std::any> &alm_,
      const std::vector<std::any> &map, const sharp_geom_info &geom_info,
      const sharp_alm_info &alm_info, size_t flags, int nthreads_);

    size_t nmaps() const { return 1+(spin>0); }
    size_t nalm() const { return (type==SHARP_ALM2MAP_DERIV1) ? 1 : (1+(spin>0)); }

    void execute();
  };

}

}

#endif
