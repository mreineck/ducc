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

/*! \file sharp_geomhelpers.c
 *  Spherical transform library
 *
 *  Copyright (C) 2006-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <cmath>
#include <vector>
#include "libsharp2/sharp_geomhelpers.h"
#include "mr_util/gl_integrator.h"
#include "mr_util/fft.h"
#include "mr_util/error_handling.h"
#include "mr_util/math_utils.h"

using namespace std;
using namespace mr;

sharp_standard_geom_info::sharp_standard_geom_info(size_t nrings, const size_t *nph, const ptrdiff_t *ofs,
  ptrdiff_t stride_, const double *phi0, const double *theta, const double *wgt)
  : ring(nrings), stride(stride_)
  {
  size_t pos=0;

  nphmax_=0;

  for (size_t m=0; m<nrings; ++m)
    {
    ring[m].theta = theta[m];
    ring[m].cth = cos(theta[m]);
    ring[m].sth = sin(theta[m]);
    ring[m].weight = (wgt != nullptr) ? wgt[m] : 1.;
    ring[m].phi0 = phi0[m];
    ring[m].ofs = ofs[m];
    ring[m].nph = nph[m];
    if (nphmax_<nph[m]) nphmax_=nph[m];
    }
  sort(ring.begin(), ring.end(),[](const Tring &a, const Tring &b)
    { return (a.sth<b.sth); });
  while (pos<nrings)
    {
    pair_.push_back(Tpair());
    pair_.back().r1=pos;
    if ((pos<nrings-1) && approx(ring[pos].cth,-ring[pos+1].cth,1e-12))
      {
      if (ring[pos].cth>0)  // make sure northern ring is in r1
        pair_.back().r2=pos+1;
      else
        {
        pair_.back().r1=pos+1;
        pair_.back().r2=pos;
        }
      ++pos;
      }
    else
      pair_.back().r2=size_t(~0);
    ++pos;
    }

  sort(pair_.begin(), pair_.end(), [this] (const Tpair &a, const Tpair &b)
    {
    if (ring[a.r1].nph==ring[b.r1].nph)
    return (ring[a.r1].phi0 < ring[b.r1].phi0) ? true :
      ((ring[a.r1].phi0 > ring[b.r1].phi0) ? false :
        (ring[a.r1].cth>ring[b.r1].cth));
    return ring[a.r1].nph<ring[b.r1].nph;
    });
  }
template<typename T> bool can_cast(any val)
  { return val.type()==typeid(T); }
template<typename T> void sharp_standard_geom_info::tclear(T *map) const
  {
  for (const auto &r: ring)
    {
    if (stride==1)
      memset(&map[r.ofs],0,r.nph*sizeof(T));
    else
      for (size_t i=0;i<r.nph;++i)
        map[r.ofs+i*stride]=T(0);
    }
  }

void sharp_standard_geom_info::clear_map (any map) const
  {
  if (can_cast<double *>(map)) tclear(any_cast<double *>(map));
  else if (can_cast<float *>(map)) tclear(any_cast<float *>(map));
  else MR_fail("bad map data type");
  }

template<typename T> void sharp_standard_geom_info::tadd(bool weighted, size_t iring, const double *ringtmp, T *map) const
  {
  T *MRUTIL_RESTRICT p1=&map[ring[iring].ofs];
  double wgt = weighted ? ring[iring].weight : 1.;
  for (size_t m=0; m<ring[iring].nph; ++m)
    p1[m*stride] += T(ringtmp[m]*wgt);
  }
//virtual
void sharp_standard_geom_info::add_ring(bool weighted, size_t iring, const double *ringtmp, any map) const
  {
  if (can_cast<double *>(map)) tadd(weighted, iring, ringtmp, any_cast<double *>(map));
  else if (can_cast<float *>(map)) tadd(weighted, iring, ringtmp, any_cast<float *>(map));
  else MR_fail("bad map data type");
  }
template<typename T> void sharp_standard_geom_info::tget(bool weighted, size_t iring, const T *map, double *ringtmp) const
  {
  const T *MRUTIL_RESTRICT p1=&map[ring[iring].ofs];
  double wgt = weighted ? ring[iring].weight : 1.;
  for (size_t m=0; m<ring[iring].nph; ++m)
    ringtmp[m] = p1[m*stride]*wgt;
  }
//virtual
void sharp_standard_geom_info::get_ring(bool weighted, size_t iring, any map, double *ringtmp) const
  {
  if (can_cast<const double *>(map)) tget(weighted, iring, any_cast<const double *>(map), ringtmp);
  else if (can_cast<double *>(map)) tget(weighted, iring, any_cast<double *>(map), ringtmp);
  else if (can_cast<const float *>(map)) tget(weighted, iring, any_cast<const float *>(map), ringtmp);
  else if (can_cast<float *>(map)) tget(weighted, iring, any_cast<float *>(map), ringtmp);
  else MR_assert(false,"bad map data type",map.type().name());
  }

unique_ptr<sharp_geom_info> sharp_make_subset_healpix_geom_info (size_t nside, ptrdiff_t stride, size_t nrings,
  const size_t *rings, const double *weight)
  {
  const double pi=3.141592653589793238462643383279502884197;
  ptrdiff_t npix=ptrdiff_t(nside*nside*12);
  ptrdiff_t ncap=2*ptrdiff_t(nside*(nside-1));

  vector<double> theta(nrings), weight_(nrings), phi0(nrings);
  vector<size_t> nph(nrings);
  vector<ptrdiff_t> ofs(nrings);
  ptrdiff_t curofs=0, checkofs; /* checkofs used for assertion introduced when adding rings arg */
  for (size_t m=0; m<nrings; ++m)
    {
    auto ring = (rings==nullptr)? (m+1) : rings[m];
    size_t northring = (ring>2*nside) ? 4*nside-ring : ring;
    if (northring < nside)
      {
      theta[m] = 2*asin(northring/(sqrt(6.)*nside));
      nph[m] = 4*northring;
      phi0[m] = pi/nph[m];
      checkofs = 2*northring*(northring-1)*stride;
      }
    else
      {
      double fact1 = (8.*nside)/npix;
      double costheta = (2*nside-northring)*fact1;
      theta[m] = acos(costheta);
      nph[m] = 4*nside;
      if ((northring-nside) & 1)
        phi0[m] = 0;
      else
        phi0[m] = pi/nph[m];
      checkofs = (ncap + (northring-nside)*nph[m])*stride;
      ofs[m] = curofs;
      }
    if (northring != ring) /* southern hemisphere */
      {
      theta[m] = pi-theta[m];
      checkofs = (npix - nph[m])*stride - checkofs;
      ofs[m] = curofs;
      }
    weight_[m]=4.*pi/npix*((weight==nullptr) ? 1. : weight[northring-1]);
    if (rings==nullptr) {
      MR_assert(curofs==checkofs, "Bug in computing ofs[m]");
    }
    ofs[m] = curofs;
    curofs+=nph[m];
    }

  return unique_ptr<sharp_geom_info>(new sharp_standard_geom_info(nrings, nph.data(), ofs.data(), stride, phi0.data(), theta.data(), weight_.data()));
  }

unique_ptr<sharp_geom_info> sharp_make_weighted_healpix_geom_info (size_t nside, ptrdiff_t stride,
  const double *weight)
  {
  return sharp_make_subset_healpix_geom_info(nside, stride, 4*nside-1, nullptr, weight);
  }

unique_ptr<sharp_geom_info> sharp_make_gauss_geom_info (size_t nrings, size_t nphi, double phi0,
  ptrdiff_t stride_lon, ptrdiff_t stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<size_t> nph(nrings);
  vector<double> phi0_(nrings);
  vector<ptrdiff_t> ofs(nrings);

  mr::GL_Integrator integ(nrings);
  auto theta = integ.coords();
  auto weight = integ.weights();
  for (size_t m=0; m<nrings; ++m)
    {
    theta[m] = acos(-theta[m]);
    nph[m]=nphi;
    phi0_[m]=phi0;
    ofs[m]=ptrdiff_t(m*stride_lat);
    weight[m]*=2*pi/nphi;
    }

  return unique_ptr<sharp_geom_info>(new sharp_standard_geom_info(nrings, nph.data(), ofs.data(), stride_lon, phi0_.data(), theta.data(), weight.data()));
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
unique_ptr<sharp_geom_info> sharp_make_fejer1_geom_info (size_t nrings, size_t ppring, double phi0,
  ptrdiff_t stride_lon, ptrdiff_t stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), weight(nrings), phi0_(nrings);
  vector<size_t> nph(nrings);
  vector<ptrdiff_t> ofs(nrings);

  weight[0]=2.;
  for (size_t k=1; k<=(nrings-1)/2; ++k)
    {
    weight[2*k-1]=2./(1.-4.*k*k)*cos((k*pi)/nrings);
    weight[2*k  ]=2./(1.-4.*k*k)*sin((k*pi)/nrings);
    }
  if ((nrings&1)==0) weight[nrings-1]=0.;
  auto tmp = fmav(weight.data(),{nrings});
  mr::r2r_fftpack(tmp, tmp, {0}, false, false, 1.);

  for (size_t m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*(m+0.5)/nrings;
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=ptrdiff_t(m*stride_lat);
    ofs[nrings-1-m]=ptrdiff_t((nrings-1-m)*stride_lat);
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(nrings*nph[m]);
    }

  return unique_ptr<sharp_geom_info>(new sharp_standard_geom_info(nrings, nph.data(), ofs.data(), stride_lon, phi0_.data(), theta.data(), weight.data()));
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
unique_ptr<sharp_geom_info> sharp_make_cc_geom_info (size_t nrings, size_t ppring, double phi0,
  ptrdiff_t stride_lon, ptrdiff_t stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), weight(nrings,0.), phi0_(nrings);
  vector<size_t> nph(nrings);
  vector<ptrdiff_t> ofs(nrings);

  size_t n=nrings-1;
  double dw=-1./(n*n-1.+(n&1));
  weight[0]=2.+dw;
  for (size_t k=1; k<=(n/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k) + dw;
  weight[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
  auto tmp = fmav(weight.data(),{n});
  mr::r2r_fftpack(tmp, tmp, {0}, false, false, 1.);
  weight[n]=weight[0];

  for (size_t m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*m/(nrings-1.);
    if (theta[m]<1e-15) theta[m]=1e-15;
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=ptrdiff_t(m*stride_lat);
    ofs[nrings-1-m]=ptrdiff_t((nrings-1-m)*stride_lat);
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(n*nph[m]);
    }

  return unique_ptr<sharp_geom_info>(new sharp_standard_geom_info(nrings, nph.data(), ofs.data(), stride_lon, phi0_.data(), theta.data(), weight.data()));
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
unique_ptr<sharp_geom_info> sharp_make_fejer2_geom_info (size_t nrings, size_t ppring, double phi0,
  ptrdiff_t stride_lon, ptrdiff_t stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), weight(nrings+1, 0.), phi0_(nrings);
  vector<size_t> nph(nrings);
  vector<ptrdiff_t> ofs(nrings);

  size_t n=nrings+1;
  weight[0]=2.;
  for (size_t k=1; k<=(n/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k);
  weight[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1.;
  auto tmp = fmav(weight.data(),{n});
  mr::r2r_fftpack(tmp, tmp, {0}, false, false, 1.);
  for (size_t m=0; m<nrings; ++m)
    weight[m]=weight[m+1];

  for (size_t m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*(m+1)/(nrings+1.);
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=ptrdiff_t(m*stride_lat);
    ofs[nrings-1-m]=ptrdiff_t((nrings-1-m)*stride_lat);
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(n*nph[m]);
    }

  return unique_ptr<sharp_geom_info>(new sharp_standard_geom_info(nrings, nph.data(), ofs.data(), stride_lon, phi0_.data(), theta.data(), weight.data()));
  }

unique_ptr<sharp_geom_info> sharp_make_mw_geom_info (size_t nrings, size_t ppring, double phi0,
  ptrdiff_t stride_lon, ptrdiff_t stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), phi0_(nrings);
  vector<size_t> nph(nrings);
  vector<ptrdiff_t> ofs(nrings);

  for (size_t m=0; m<nrings; ++m)
    {
    theta[m]=pi*(2.*m+1.)/(2.*nrings-1.);
    if (theta[m]>pi-1e-15) theta[m]=pi-1e-15;
    nph[m]=ppring;
    phi0_[m]=phi0;
    ofs[m]=ptrdiff_t(m*stride_lat);
    }

  return unique_ptr<sharp_geom_info>(new sharp_standard_geom_info(nrings, nph.data(), ofs.data(), stride_lon, phi0_.data(), theta.data(), nullptr));
  }
