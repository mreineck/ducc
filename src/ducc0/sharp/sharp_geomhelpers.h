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

/*! \file sharp_geomhelpers.h
 *  SHARP helper function for the creation of grid geometries
 *
 *  Copyright (C) 2006-2020 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef SHARP2_GEOMHELPERS_H
#define SHARP2_GEOMHELPERS_H

#include <memory>
#include <string>
#include "ducc0/sharp/sharp.h"

namespace ducc0 {

namespace detail_sharp {

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

using detail_sharp::sharp_standard_geom_info;
using detail_sharp::sharp_make_subset_healpix_geom_info;
using detail_sharp::sharp_make_weighted_healpix_geom_info;
using detail_sharp::sharp_make_healpix_geom_info;
using detail_sharp::sharp_make_2d_geom_info;

}

#endif
