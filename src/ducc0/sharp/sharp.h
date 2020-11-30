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

/*! \file sharp.h
 *  Portable interface for the spherical transform library.
 *
 *  Copyright (C) 2012-2020 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#ifndef SHARP_SHARP_H
#define SHARP_SHARP_H

#include <complex>
#include <cstddef>
#include <vector>
#include <memory>
#include <any>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_sharp {

class sharp_geom_info
  {
  public:
    virtual ~sharp_geom_info() {}
    virtual size_t nrings() const = 0;
    virtual size_t npairs() const = 0;
    struct Tpair
      {
      size_t r1, r2;
      };
    virtual size_t nph(size_t iring) const = 0;
    virtual size_t nphmax() const = 0;
    virtual double theta(size_t iring) const = 0;
    virtual double cth(size_t iring) const = 0;
    virtual double sth(size_t iring) const = 0;
    virtual double phi0(size_t iring) const = 0;
    virtual Tpair pair(size_t ipair) const = 0;

    virtual void clear_map(const std::any &map) const = 0;
    virtual void get_ring(bool weighted, size_t iring, const std::any &map, mav<double,1> &ringtmp) const = 0;
    virtual void add_ring(bool weighted, size_t iring, const mav<double,1> &ringtmp, const std::any &map) const = 0;
  };

/*! \defgroup almgroup Helpers for dealing with a_lm */
/*! \{ */

class sharp_alm_info
  {
  public:
    virtual ~sharp_alm_info() {}
    virtual size_t lmax() const = 0;
    virtual size_t mmax() const = 0;
    virtual size_t nm() const = 0;
    virtual size_t mval(size_t i) const = 0;
    virtual void clear_alm(const std::any &alm) const = 0;
    virtual void get_alm(size_t mi, const std::any &alm, mav<std::complex<double>,1> &almtmp) const = 0;
    virtual void add_alm(size_t mi, const mav<std::complex<double>,1> &almtmp, const std::any &alm) const = 0;
  };

/*! \} */

/*! \defgroup geominfogroup Functions for dealing with geometry information */
/*! \{ */

/*! \} */

/*! \defgroup lowlevelgroup Low-level libsharp2 SHT interface */
/*! \{ */

/*! Enumeration of SHARP job types. */
enum sharp_jobtype { SHARP_YtW=0,               /*!< analysis */
               SHARP_MAP2ALM=SHARP_YtW,   /*!< analysis */
               SHARP_Y=1,                 /*!< synthesis */
               SHARP_ALM2MAP=SHARP_Y,     /*!< synthesis */
               SHARP_Yt=2,                /*!< adjoint synthesis */
               SHARP_WY=3,                /*!< adjoint analysis */
               SHARP_ALM2MAP_DERIV1=4     /*!< synthesis of first derivatives */
             };

/*! Job flags */
enum sharp_jobflags {
               SHARP_ADD             = 1<<5,
               /*!< results are added to the output arrays, instead of
                    overwriting them */
               SHARP_USE_WEIGHTS     = 1<<20,    /* internal use only */
             };

void sharp_execute (sharp_jobtype type, size_t spin, const std::vector<std::any> &alm,
  const std::vector<std::any> &map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr);

template<typename T> void sharp_alm2map(const std::complex<T> *alm, T *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr)
  {
  sharp_execute(SHARP_Y, 0, {alm}, {map}, geom_info, alm_info, flags, nthreads, time, opcnt);
  }
template<typename T> void sharp_alm2map_adjoint(std::complex<T> *alm, const T *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr)
  {
  sharp_execute(SHARP_Yt, 0, {alm}, {map}, geom_info, alm_info, flags, nthreads, time, opcnt);
  }
template<typename T> void sharp_alm2map_spin(size_t spin, const std::complex<T> *alm1, const std::complex<T> *alm2, T *map1, T *map2,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr)
  {
  sharp_execute(SHARP_Y, spin, {alm1, alm2}, {map1, map2}, geom_info, alm_info, flags, nthreads, time, opcnt);
  }
template<typename T> void sharp_alm2map_spin_adjoint(size_t spin, std::complex<T> *alm1, std::complex<T> *alm2, const T *map1, const T *map2,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr)
  {
  sharp_execute(SHARP_Yt, spin, {alm1, alm2}, {map1, map2}, geom_info, alm_info, flags, nthreads, time, opcnt);
  }
template<typename T> void sharp_map2alm(std::complex<T> *alm, const T *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr)
  {
  sharp_execute(SHARP_Yt, 0, {alm}, {map}, geom_info, alm_info, flags, nthreads, time, opcnt);
  }
template<typename T> void sharp_map2alm_spin(size_t spin, std::complex<T> *alm1, std::complex<T> *alm2, const T *map1, const T *map2,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1, double *time=nullptr, uint64_t *opcnt=nullptr)
  {
  sharp_execute(SHARP_Yt, spin, {alm1, alm2}, {map1, map2}, geom_info, alm_info, flags, nthreads, time, opcnt);
  }

void sharp_set_chunksize_min(size_t new_chunksize_min);
void sharp_set_nchunks_max(size_t new_nchunks_max);

/*! \} */

size_t sharp_get_mlim (size_t lmax, size_t spin, double sth, double cth);
size_t sharp_veclen(void);
const char *sharp_architecture(void);

}

using detail_sharp::sharp_geom_info;
using detail_sharp::sharp_alm_info;
using detail_sharp::SHARP_ADD;
using detail_sharp::SHARP_USE_WEIGHTS;
using detail_sharp::SHARP_YtW;
using detail_sharp::SHARP_MAP2ALM;
using detail_sharp::SHARP_Y;
using detail_sharp::SHARP_ALM2MAP;
using detail_sharp::SHARP_Yt;
using detail_sharp::SHARP_WY;
using detail_sharp::SHARP_ALM2MAP_DERIV1;
using detail_sharp::sharp_architecture;
using detail_sharp::sharp_veclen;
using detail_sharp::sharp_get_mlim;

}

#endif
