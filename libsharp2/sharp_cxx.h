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

/*! \file sharp_cxx.h
 *  Spherical transform library
 *
 *  Copyright (C) 2012-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef SHARP2_CXX_H
#define SHARP2_CXX_H

#include <complex>
#include "libsharp2/sharp.h"
#include "libsharp2/sharp_geomhelpers.h"
#include "libsharp2/sharp_almhelpers.h"

class sharp_base
  {
  protected:
    sharp_alm_info *ainfo;
    sharp_geom_info *ginfo;

  public:
    sharp_base()
      : ainfo(0), ginfo(0) {}
    ~sharp_base()
      {
      if (ginfo) sharp_destroy_geom_info(ginfo);
      if (ainfo) sharp_destroy_alm_info(ainfo);
      }

    void set_general_geometry (int nrings, const int *nph, const ptrdiff_t *ofs,
      const int *stride, const double *phi0, const double *theta,
      const double *wgt)
      {
      if (ginfo) sharp_destroy_geom_info(ginfo);
      sharp_make_geom_info (nrings, nph, ofs, stride, phi0, theta, wgt, &ginfo);
      }

    void set_ECP_geometry (int nrings, int nphi)
      {
      if (ginfo) sharp_destroy_geom_info(ginfo);
      sharp_make_ecp_geom_info (nrings, nphi, 0., 1, nphi, &ginfo);
      }

    void set_Gauss_geometry (int nrings, int nphi)
      {
      if (ginfo) sharp_destroy_geom_info(ginfo);
      sharp_make_gauss_geom_info (nrings, nphi, 0., 1, nphi, &ginfo);
      }

    void set_Healpix_geometry (int nside)
      {
      if (ginfo) sharp_destroy_geom_info(ginfo);
      sharp_make_healpix_geom_info (nside, 1, &ginfo);
      }

    void set_weighted_Healpix_geometry (int nside, const double *weight)
      {
      if (ginfo) sharp_destroy_geom_info(ginfo);
      sharp_make_weighted_healpix_geom_info (nside, 1, weight, &ginfo);
      }

    void set_triangular_alm_info (int lmax, int mmax)
      {
      if (ainfo) sharp_destroy_alm_info(ainfo);
      sharp_make_triangular_alm_info (lmax, mmax, 1, &ainfo);
      }
  };

template<typename T> struct cxxjobhelper__ {};

template<> struct cxxjobhelper__<double>
  { enum {val=SHARP_DP}; };

template<> struct cxxjobhelper__<float>
  { enum {val=0}; };


template<typename T> class sharp_cxxjob: public sharp_base
  {
  private:
    static void *conv (T *ptr)
      { return reinterpret_cast<void *>(ptr); }
    static void *conv (std::complex<T> *ptr)
      { return reinterpret_cast<void *>(ptr); }
    static void *conv (const T *ptr)
      { return const_cast<void *>(reinterpret_cast<const void *>(ptr)); }
    static void *conv (const std::complex<T> *ptr)
      { return const_cast<void *>(reinterpret_cast<const void *>(ptr)); }

  public:
    void alm2map (const T *alm, T *map, bool add) const
      {
      void *aptr=conv(alm), *mptr=conv(map);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_ALM2MAP, 0, &aptr, &mptr, ginfo, ainfo, flags, 0, 0);
      }
    void alm2map (const std::complex<T> *alm, T *map, bool add) const
      {
      void *aptr=conv(alm), *mptr=conv(map);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_ALM2MAP, 0, &aptr, &mptr, ginfo, ainfo, flags, 0, 0);
      }
    void alm2map_spin (const T *alm1, const T *alm2,
      T *map1, T *map2, int spin, bool add) const
      {
      void *aptr[2], *mptr[2];
      aptr[0]=conv(alm1); aptr[1]=conv(alm2);
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_ALM2MAP,spin,aptr,mptr,ginfo,ainfo,flags, 0, 0);
      }
    void alm2map_spin (const std::complex<T> *alm1, const std::complex<T> *alm2,
      T *map1, T *map2, int spin, bool add) const
      {
      void *aptr[2], *mptr[2];
      aptr[0]=conv(alm1); aptr[1]=conv(alm2);
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_ALM2MAP, spin, aptr, mptr, ginfo, ainfo, flags,0,0);
      }
    void alm2map_der1 (const T *alm, T *map1, T *map2, bool add) const
      {
      void *aptr=conv(alm), *mptr[2];
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_ALM2MAP_DERIV1,1,&aptr,mptr,ginfo,ainfo,flags,0,0);
      }
    void alm2map_der1 (const std::complex<T> *alm, T *map1, T *map2, bool add)
      const
      {
      void *aptr=conv(alm), *mptr[2];
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_ALM2MAP_DERIV1,1,&aptr,mptr,ginfo,ainfo,flags,0,0);
      }
    void alm2map_adjoint (const T *map, T *alm, bool add) const
      {
      void *aptr=conv(alm), *mptr=conv(map);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_Yt,0,&aptr,&mptr,ginfo,ainfo,flags,0,0);
      }
    void alm2map_adjoint (const T *map, std::complex<T> *alm, bool add) const
      {
      void *aptr=conv(alm), *mptr=conv(map);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_Yt,0,&aptr,&mptr,ginfo,ainfo,flags,0,0);
      }
    void alm2map_spin_adjoint (const T *map1, const T *map2, T *alm1, T *alm2,
      int spin, bool add) const
      {
      void *aptr[2], *mptr[2];
      aptr[0]=conv(alm1); aptr[1]=conv(alm2);
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_Yt,spin,aptr,mptr,ginfo,ainfo,flags,0,0);
      }
    void alm2map_spin_adjoint (const T *map1, const T *map2,
      std::complex<T> *alm1, std::complex<T> *alm2, int spin, bool add) const
      {
      alm2map_spin_adjoint (map1, map2, reinterpret_cast<T *>(alm1),
        reinterpret_cast<T *>(alm2), spin, add);
      }
    void map2alm (const T *map, T *alm, bool add) const
      {
      void *aptr=conv(alm), *mptr=conv(map);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_MAP2ALM,0,&aptr,&mptr,ginfo,ainfo,flags,0,0);
      }
    void map2alm (const T *map, std::complex<T> *alm, bool add) const
      {
      void *aptr=conv(alm), *mptr=conv(map);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_MAP2ALM,0,&aptr,&mptr,ginfo,ainfo,flags,0,0);
      }
    void map2alm_spin (const T *map1, const T *map2, T *alm1, T *alm2,
      int spin, bool add) const
      {
      void *aptr[2], *mptr[2];
      aptr[0]=conv(alm1); aptr[1]=conv(alm2);
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_MAP2ALM,spin,aptr,mptr,ginfo,ainfo,flags,0,0);
      }
    void map2alm_spin (const T *map1, const T *map2, std::complex<T> *alm1,
      std::complex<T> *alm2, int spin, bool add) const
      {
      void *aptr[2], *mptr[2];
      aptr[0]=conv(alm1); aptr[1]=conv(alm2);
      mptr[0]=conv(map1); mptr[1]=conv(map2);
      int flags=cxxjobhelper__<T>::val | (add ? SHARP_ADD : 0);
      sharp_execute (SHARP_MAP2ALM,spin,aptr,mptr,ginfo,ainfo,flags,0,0);
      }
  };

#endif
