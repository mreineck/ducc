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
    virtual double weight(size_t iring) const = 0;
    virtual ptrdiff_t ofs(size_t iring) const = 0;
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
  size_t flags, int nthreads=1);

template<typename T> void sharp_alm2map(const std::complex<T> *alm, T *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1)
  {
  sharp_execute(SHARP_Y, 0, {alm}, {map}, geom_info, alm_info, flags, nthreads);
  }
template<typename T> void sharp_alm2map_adjoint(std::complex<T> *alm, const T *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1)
  {
  sharp_execute(SHARP_Yt, 0, {alm}, {map}, geom_info, alm_info, flags, nthreads);
  }
template<typename T> void sharp_alm2map_spin(size_t spin, const std::complex<T> *alm1, const std::complex<T> *alm2, T *map1, T *map2,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1)
  {
  sharp_execute(SHARP_Y, spin, {alm1, alm2}, {map1, map2}, geom_info, alm_info, flags, nthreads);
  }
template<typename T> void sharp_alm2map_spin_adjoint(size_t spin, std::complex<T> *alm1, std::complex<T> *alm2, const T *map1, const T *map2,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1)
  {
  sharp_execute(SHARP_Yt, spin, {alm1, alm2}, {map1, map2}, geom_info, alm_info, flags, nthreads);
  }
template<typename T> void sharp_map2alm(std::complex<T> *alm, const T *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1)
  {
  sharp_execute(SHARP_Yt, 0, {alm}, {map}, geom_info, alm_info, flags, nthreads);
  }
template<typename T> void sharp_map2alm_spin(size_t spin, std::complex<T> *alm1, std::complex<T> *alm2, const T *map1, const T *map2,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads=1)
  {
  sharp_execute(SHARP_Yt, spin, {alm1, alm2}, {map1, map2}, geom_info, alm_info, flags, nthreads);
  }

void sharp_set_chunksize_min(size_t new_chunksize_min);
void sharp_set_nchunks_max(size_t new_nchunks_max);

/*! \} */

/*! \internal
    Helper type for index calculation in a_lm arrays. */
class sharp_standard_alm_info: public sharp_alm_info
  {
  private:
    /*! Maximum \a l index of the array */
    size_t lmax_;
    /*! Array with \a nm entries containing the individual m values */
    std::vector<size_t> mval_;
    /*! Array with \a nm entries containing the (hypothetical) indices of
        the coefficients with quantum numbers 0,\a mval[i] */
    std::vector<ptrdiff_t> mvstart;
    /*! Stride between a_lm and a_(l+1),m */
    ptrdiff_t stride;
    template<typename T> void tclear (T *alm) const;
    template<typename T> void tget (size_t mi, const T *alm, mav<std::complex<double>,1> &almtmp) const;
    template<typename T> void tadd (size_t mi, const mav<std::complex<double>,1> &almtmp, T *alm) const;

  public:
  /*! Creates an a_lm data structure from the following parameters:
      \param lmax maximum \a l quantum number (>=0)
      \param mmax maximum \a m quantum number (0<= \a mmax <= \a lmax)
      \param stride the stride between entries with identical \a m, and \a l
        differing by 1.
      \param mstart the index of the (hypothetical) coefficient with the
        quantum numbers 0,\a m. Must have \a mmax+1 entries.
   */
    sharp_standard_alm_info(size_t lmax__, size_t mmax_, ptrdiff_t stride_, const ptrdiff_t *mstart);

  /*! Creates an a_lm data structure which from the following parameters:
      \param lmax maximum \a l quantum number (\a >=0)
      \param nm number of different \a m (\a 0<=nm<=lmax+1)
      \param stride the stride between entries with identical \a m, and \a l
        differing by 1.
      \param mval array with \a nm entries containing the individual m values
      \param mvstart array with \a nm entries containing the (hypothetical)
        indices of the coefficients with the quantum numbers 0,\a mval[i]
   */
    sharp_standard_alm_info (size_t lmax__, size_t nm__, ptrdiff_t stride_, const size_t *mval__,
      const ptrdiff_t *mvstart_);
  /*! Returns the index of the coefficient with quantum numbers \a l,
      \a mval_[mi].
      \note for a \a sharp_alm_info generated by sharp_make_alm_info() this is
      the index for the coefficient with the quantum numbers \a l, \a mi. */
    ptrdiff_t index (size_t l, size_t mi);

    virtual ~sharp_standard_alm_info() {}
    virtual size_t lmax() const { return lmax_; }
    virtual size_t mmax() const;
    virtual size_t nm() const { return mval_.size(); }
    virtual size_t mval(size_t i) const { return mval_[i]; }
    virtual void clear_alm(const std::any &alm) const;
    virtual void get_alm(size_t mi, const std::any &alm, mav<std::complex<double>,1> &almtmp) const;
    virtual void add_alm(size_t mi, const mav<std::complex<double>,1> &almtmp, const std::any &alm) const;
  };

/*! Initialises an a_lm data structure according to the scheme used by
    Healpix_cxx.
    \ingroup almgroup */
std::unique_ptr<sharp_standard_alm_info> sharp_make_triangular_alm_info (size_t lmax, size_t mmax, ptrdiff_t stride);

class sharp_standard_geom_info: public sharp_geom_info
  {
  private:
    struct Tring
      {
      double theta, phi0, weight, cth, sth;
      ptrdiff_t ofs;
      size_t nph;
      };
    std::vector<Tring> ring;
    std::vector<Tpair> pair_;
    ptrdiff_t stride;
    size_t nphmax_;
    template<typename T> void tclear (T *map) const;
    template<typename T> void tget (bool weighted, size_t iring, const T *map, mav<double,1> &ringtmp) const;
    template<typename T> void tadd (bool weighted, size_t iring, const mav<double,1> &ringtmp, T *map) const;

  public:
/*! Creates a geometry information from a set of ring descriptions.
    All arrays passed to this function must have \a nrings elements.
    \param nrings the number of rings in the map
    \param nph the number of pixels in each ring
    \param ofs the index of the first pixel in each ring in the map array
    \param stride the stride between consecutive pixels
    \param phi0 the azimuth (in radians) of the first pixel in each ring
    \param theta the colatitude (in radians) of each ring
    \param wgt the pixel weight to be used for the ring in map2alm
      and adjoint map2alm transforms.
      Pass nullptr to use 1.0 as weight for all rings.
 */
    sharp_standard_geom_info(size_t nrings, const size_t *nph_, const ptrdiff_t *ofs,
      ptrdiff_t stride_, const double *phi0_, const double *theta_, const double *wgt);
    virtual size_t nrings() const { return ring.size(); }
    virtual size_t npairs() const { return pair_.size(); }
    virtual size_t nph(size_t iring) const { return ring[iring].nph; }
    virtual size_t nphmax() const { return nphmax_; }
    virtual double theta(size_t iring) const { return ring[iring].theta; }
    virtual double cth(size_t iring) const { return ring[iring].cth; }
    virtual double sth(size_t iring) const { return ring[iring].sth; }
    virtual double phi0(size_t iring) const { return ring[iring].phi0; }
    virtual ptrdiff_t ofs(size_t iring) const { return ring[iring].ofs; }
    virtual double weight(size_t iring) const { return ring[iring].weight; }
    virtual Tpair pair(size_t ipair) const { return pair_[ipair]; }
    virtual void clear_map(const std::any &map) const;
    virtual void get_ring(bool weighted, size_t iring, const std::any &map, mav<double,1> &ringtmp) const;
    virtual void add_ring(bool weighted, size_t iring, const mav<double,1> &ringtmp, const std::any &map) const;
  };

/*! Creates a geometry information describing a HEALPix map with an
    Nside parameter \a nside. \a weight contains the relative ring
    weights and must have \a 2*nside entries. The rings array contains
    the indices of the rings, with 1 being the first ring at the north
    pole; if nullptr then we take them to be sequential. Pass 4 * nside - 1
    as nrings and nullptr to rings to get the full HEALPix grid.
    \note if \a weight is a null pointer, all weights are assumed to be 1.
    \note if \a rings is a null pointer, take all rings
    \ingroup geominfogroup */
std::unique_ptr<sharp_geom_info> sharp_make_subset_healpix_geom_info (size_t nside, ptrdiff_t stride, size_t nrings,
  const size_t *rings=nullptr, const double *weight=nullptr);

/*! Creates a geometry information describing a HEALPix map with an
    Nside parameter \a nside. \a weight contains the relative ring
    weights and must have \a 2*nside entries.
    \note if \a weight is a null pointer, all weights are assumed to be 1.
    \ingroup geominfogroup */
std::unique_ptr<sharp_geom_info> sharp_make_weighted_healpix_geom_info (size_t nside, ptrdiff_t stride,
  const double *weight=nullptr)
  { return sharp_make_subset_healpix_geom_info(nside, stride, 4*nside-1, nullptr, weight); }


/*! Creates a geometry information describing a HEALPix map with an
    Nside parameter \a nside.
    \ingroup geominfogroup */
static inline std::unique_ptr<sharp_geom_info> sharp_make_healpix_geom_info (size_t nside, ptrdiff_t stride)
  { return sharp_make_weighted_healpix_geom_info (nside, stride, nullptr); }

/*! Creates a geometry information describing a map with \a nrings
    iso-latitude rings and \a nphi pixels per ring. The azimuth of the first
    pixel in each ring is \a phi0 (in radians). The index difference between
    two adjacent pixels in an iso-latitude ring is \a stride_lon, the index
    difference between the two start pixels in consecutive iso-latitude rings
    is \a stride_lat. If \a with_weight is true, ring weights for analysis are
    computed.
    The ring colatitudes depend on \type:
     - "GL" : rings are located at Gauss-Legendre quadrature nodes
     - "CC" : rings are placed according to the Clenshaw-Curtis quadrature rule,
              i.e. theta_i = i*pi/(nrings-1)
     - "F1" : rings are placed according to Fejer's first rule,
              i.e. theta_i = (i+0.5)*(pi/nrings)
     - "F2" : rings are placed according to Fejer's second rule,
              i.e. theta_i = i*pi/(nrings+1)
     - "DH" : rings are placed according to the Driscoll-Healy scheme,
              i.e. theta_i = i*pi/nrings
     - "MW" : rings are placed according to the McEwen-Wiaux scheme,
              i.e. theta_i = (i+0.5)*2*pi/(2*nrings-1) */
std::unique_ptr<sharp_geom_info> sharp_make_2d_geom_info
  (size_t nrings, size_t ppring, double phi0, ptrdiff_t stride_lon,
  ptrdiff_t stride_lat, const std::string &type, bool with_weight=true);

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
using detail_sharp::sharp_set_chunksize_min;
using detail_sharp::sharp_set_nchunks_max;
using detail_sharp::sharp_standard_alm_info;
using detail_sharp::sharp_make_triangular_alm_info;
using detail_sharp::sharp_standard_geom_info;
using detail_sharp::sharp_make_subset_healpix_geom_info;
using detail_sharp::sharp_make_weighted_healpix_geom_info;
using detail_sharp::sharp_make_healpix_geom_info;
using detail_sharp::sharp_make_2d_geom_info;

}

#endif
