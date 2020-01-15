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

#include <cstddef>
#include <vector>
#include <memory>

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

    virtual void clear_map(double *map) const = 0;
    virtual void clear_map(float *map) const = 0;
    virtual void get_ring(bool weighted, size_t iring, const double *map, double *ringtmp) const = 0;
    virtual void get_ring(bool weighted, size_t iring, const float *map, double *ringtmp) const = 0;
    virtual void add_ring(bool weighted, size_t iring, const double *ringtmp, double *map) const = 0;
    virtual void add_ring(bool weighted, size_t iring, const double *ringtmp, float *map) const = 0;
  };

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
    virtual Tpair pair(size_t ipair) const { return pair_[ipair]; }
    virtual void clear_map(double *map) const;
    virtual void clear_map(float *map) const;
    virtual void get_ring(bool weighted, size_t iring, const double *map, double *ringtmp) const;
    virtual void get_ring(bool weighted, size_t iring, const float *map, double *ringtmp) const;
    virtual void add_ring(bool weighted, size_t iring, const double *ringtmp, double *map) const;
    virtual void add_ring(bool weighted, size_t iring, const double *ringtmp, float *map) const;
  };

/*! \defgroup almgroup Helpers for dealing with a_lm */
/*! \{ */

/*! \internal
    Helper type for index calculation in a_lm arrays. */
struct sharp_alm_info
  {
  /*! Maximum \a l index of the array */
  size_t lmax;
  /*! Number of different \a m values in this object */
  size_t nm;
  /*! Array with \a nm entries containing the individual m values */
  std::vector<size_t> mval;
  /*! Array with \a nm entries containing the (hypothetical) indices of
      the coefficients with quantum numbers 0,\a mval[i] */
  std::vector<ptrdiff_t> mvstart;
  /*! Stride between a_lm and a_(l+1),m */
  ptrdiff_t stride;

/*! Creates an a_lm data structure from the following parameters:
    \param lmax maximum \a l quantum number (>=0)
    \param mmax maximum \a m quantum number (0<= \a mmax <= \a lmax)
    \param stride the stride between entries with identical \a m, and \a l
      differing by 1.
    \param mstart the index of the (hypothetical) coefficient with the
      quantum numbers 0,\a m. Must have \a mmax+1 entries.
 */
  sharp_alm_info(size_t lmax_, size_t mmax, ptrdiff_t stride_, const ptrdiff_t *mstart);

/*! Creates an a_lm data structure which from the following parameters:
    \param lmax maximum \a l quantum number (\a >=0)
    \param nm number of different \a m (\a 0<=nm<=lmax+1)
    \param stride the stride between entries with identical \a m, and \a l
      differing by 1.
    \param mval array with \a nm entries containing the individual m values
    \param mvstart array with \a nm entries containing the (hypothetical)
      indices of the coefficients with the quantum numbers 0,\a mval[i]
 */
  sharp_alm_info (size_t lmax_, size_t nm_, ptrdiff_t stride_, const size_t *mval_,
    const ptrdiff_t *mvstart);
/*! Returns the index of the coefficient with quantum numbers \a l,
    \a mval_[mi].
    \note for a \a sharp_alm_info generated by sharp_make_alm_info() this is
    the index for the coefficient with the quantum numbers \a l, \a mi. */
  ptrdiff_t index (int l, int mi);
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
enum sharp_jobflags { SHARP_DP              = 1<<4,
               /*!< map and a_lm are in double precision */
               SHARP_ADD             = 1<<5,
               /*!< results are added to the output arrays, instead of
                    overwriting them */
               SHARP_USE_WEIGHTS     = 1<<20,    /* internal use only */
             };

/*! Performs a libsharp2 SHT job. The interface deliberately does not use
  the C99 "complex" data type, in order to be callable from C89 and C++.
  \param type the type of SHT
  \param spin the spin of the quantities to be transformed
  \param alm contains pointers to the a_lm coefficients. If \a spin==0,
    alm[0] points to the a_lm of the SHT. If \a spin>0, alm[0] and alm[1]
    point to the two a_lm sets of the SHT. The exact data type of \a alm
    depends on whether the SHARP_DP flag is set.
  \param map contains pointers to the maps. If \a spin==0,
    map[0] points to the map of the SHT. If \a spin>0, or \a type is
    SHARP_ALM2MAP_DERIV1, map[0] and map[1] point to the two maps of the SHT.
    The exact data type of \a map depends on whether the SHARP_DP flag is set.
  \param geom_info A \c sharp_geom_info object compatible with the provided
    \a map arrays.
  \param alm_info A \c sharp_alm_info object compatible with the provided
    \a alm arrays. All \c m values from 0 to some \c mmax<=lmax must be present
    exactly once.
  \param flags See sharp_jobflags. In particular, if SHARP_DP is set, then
    \a alm is expected to have the type "complex double **" and \a map is
    expected to have the type "double **"; otherwise, the expected
    types are "complex float **" and "float **", respectively.
  \param time If not nullptr, the wall clock time required for this SHT
    (in seconds) will be written here.
  \param opcnt If not nullptr, a conservative estimate of the total floating point
    operation count for this SHT will be written here. */
void sharp_execute (sharp_jobtype type, int spin, void *alm, void *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  int flags, double *time, unsigned long long *opcnt);

void sharp_set_chunksize_min(int new_chunksize_min);
void sharp_set_nchunks_max(int new_nchunks_max);

/*! \} */

size_t sharp_get_mlim (size_t lmax, size_t spin, double sth, double cth);
size_t sharp_veclen(void);
const char *sharp_architecture(void);

#endif
