/*
 *  This file is part of DUCC.
 *
 *  DUCC is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  DUCC is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with DUCC; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  DUCC is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2017-2021 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

#include "ducc0/sht/sht.h"
#include "ducc0/sht/sharp.h"
#include "ducc0/sht/alm.h"
#include "ducc0/infra/string_utils.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_sht {

using namespace std;

namespace py = pybind11;

auto None = py::none();

template<typename T> py::array Py2_rotate_alm(const py::array &alm_, int64_t lmax,
  double psi, double theta, double phi, size_t nthreads)
  {
  auto a1 = to_mav<complex<T>,1>(alm_);
  auto alm = make_Pyarr<complex<T>>({a1.shape(0)});
  auto a2 = to_mav<complex<T>,1>(alm,true);
  for (size_t i=0; i<a1.shape(0); ++i) a2.v(i)=a1(i);
  Alm_Base base(lmax,lmax);
  rotate_alm(base, a2, psi, theta, phi, nthreads);
  return move(alm);
  }
py::array Py_rotate_alm(const py::array &alm, int64_t lmax,
  double psi, double theta, double phi, size_t nthreads)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_rotate_alm<float>(alm, lmax, psi, theta, phi, nthreads);
  if (isPyarr<complex<double>>(alm))
    return Py2_rotate_alm<double>(alm, lmax, psi, theta, phi, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }

void getmstuff(size_t lmax, const py::object &mval_, const py::object &mstart_,
  mav<size_t,1> &mval, mav<size_t,1> &mstart)
  {
  MR_assert(mval_.is_none()==mstart_.is_none(), "mval and mstart must be supplied together");
  if (mval_.is_none())
    {
    mav<size_t,1> tmv({lmax+1});
    mval.assign(tmv);
    mav<size_t,1> tms({lmax+1});
    mstart.assign(tms);
    for (size_t m=0, idx=0; m<=lmax; ++m, idx+=lmax+1-m)
      {
      mval.v(m) = m;
      mstart.v(m) = idx;
      }
    }
  else
    {
    auto tmval = to_mav<int64_t,1>(mval_,false);
    auto tmstart = to_mav<int64_t,1>(mstart_,false);
    size_t nm = tmval.shape(0);
    MR_assert(nm==tmstart.shape(0), "size mismatch between mval and mstart");
    mav<size_t,1> tmv({nm});
    mval.assign(tmv);
    mav<size_t,1> tms({nm});
    mstart.assign(tms);
    for (size_t i=0; i<nm; ++i)
      {
      auto m = tmval(i);
      MR_assert((m>=0) && (m<=int64_t(lmax)), "bad m value");
      mval.v(i) = size_t(m);
      mstart.v(i) = size_t(tmstart(i));
      }
    }
  }

py::array Py_get_gridweights(const string &type, size_t nrings)
  {
  auto wgt_ = make_Pyarr<double>({nrings});
  auto wgt = to_mav<double,1>(wgt_, true);
  get_gridweights(type, wgt);
  return wgt_;
  }

size_t min_almdim(size_t lmax, const mav<size_t,1> &mval, const mav<size_t,1> &mstart, ptrdiff_t lstride)
  {
  size_t res=0;
  for (size_t i=0; i<mval.shape(0); ++i)
    {
    auto ifirst = ptrdiff_t(mstart(i)) + ptrdiff_t(mval(i))*lstride;
    MR_assert(ifirst>=0, "impossible a_lm memory layout");
    auto ilast = ptrdiff_t(mstart(i)) + ptrdiff_t(lmax)*lstride;
    MR_assert(ilast>=0, "impossible a_lm memory layout");
    res = max(res, size_t(max(ifirst, ilast)));
    }
  return res+1;
  }
size_t min_mapdim(const mav<size_t,1> &nphi, const mav<size_t,1> &ringstart, ptrdiff_t pixstride)
  {
  size_t res=0;
  for (size_t i=0; i<nphi.shape(0); ++i)
    {
    auto ilast = ptrdiff_t(ringstart(i)) + ptrdiff_t(nphi(i)-1)*pixstride;
    MR_assert(ilast>=0, "impossible map memory layout");
    res = max(res, max(ringstart(i), size_t(ilast)));
    }
  return res+1;
  }

template<typename T> py::array Py2_alm2leg(const py::array &alm_, size_t spin, size_t lmax, const py::object &mval_, const py::object &mstart_, ptrdiff_t lstride, const py::array &theta_, size_t nthreads, py::object &leg__)
  {
  auto alm = to_mav<complex<T>,2>(alm_, false);
  auto theta = to_mav<double,1>(theta_, false);
  mav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  MR_assert(alm.shape(1)>=min_almdim(lmax, mval, mstart, lstride), "bad a_lm array size");
  auto leg_ = get_optional_Pyarr<complex<T>>(leg__, {alm.shape(0),theta.shape(0),mval.shape(0)});
  auto leg = to_mav<complex<T>,3>(leg_, true);
  alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, ALM2MAP);
  return leg_;
  }
py::array Py_alm2leg(const py::array &alm, size_t lmax, const py::array &theta, size_t spin, const py::object &mval, const py::object &mstart, ptrdiff_t lstride, size_t nthreads, py::object &leg)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_alm2leg<float>(alm, spin, lmax, mval, mstart, lstride, theta, nthreads, leg);
  if (isPyarr<complex<double>>(alm))
    return Py2_alm2leg<double>(alm, spin, lmax, mval, mstart, lstride, theta, nthreads, leg);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_leg2alm(const py::array &leg_, const py::array &theta_, size_t spin, size_t lmax, const py::object &mval_, const py::object &mstart_, ptrdiff_t lstride, size_t nthreads, py::object &alm__)
  {
  auto leg = to_mav<complex<T>,3>(leg_, false);
  auto theta = to_mav<double,1>(theta_, false);
  MR_assert(leg.shape(1)==theta.shape(0), "bad leg array size");
  mav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  auto alm_ = get_optional_Pyarr_minshape<complex<T>>(alm__, {leg.shape(0),min_almdim(lmax, mval, mstart, lstride)});
  auto alm = to_mav<complex<T>,2>(alm_, true);
  MR_assert(alm.shape(0)==leg.shape(0), "bad number of components in a_lm array");
  leg2alm(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
  return alm_;
  }
py::array Py_leg2alm(const py::array &leg, size_t lmax, const py::array &theta, size_t spin, const py::object &mval, const py::object &mstart, ptrdiff_t lstride, size_t nthreads, py::object &alm)
  {
  if (isPyarr<complex<float>>(leg))
    return Py2_leg2alm<float>(leg, theta, spin, lmax, mval, mstart, lstride, nthreads, alm);
  if (isPyarr<complex<double>>(leg))
    return Py2_leg2alm<double>(leg, theta, spin, lmax, mval, mstart, lstride, nthreads, alm);
  MR_fail("type matching failed: 'leg' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_map2leg(const py::array &map_, const py::array &nphi_, const py::array &phi0_, const py::array &ringstart_, size_t mmax, ptrdiff_t pixstride, size_t nthreads, py::object &leg__)
  {
  auto map = to_mav<T,2>(map_, false);
  auto nphi = to_mav<size_t,1>(nphi_, false);
  auto phi0 = to_mav<double,1>(phi0_, false);
  auto ringstart = to_mav<size_t,1>(ringstart_, false);
  MR_assert(map.shape(1)>=min_mapdim(nphi, ringstart, pixstride), "bad map array size");
  auto leg_ = get_optional_Pyarr<complex<T>>(leg__, {map.shape(0),nphi.shape(0),mmax+1});
  auto leg = to_mav<complex<T>,3>(leg_, true);
  map2leg(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
  return leg_;
  }
py::array Py_map2leg(const py::array &map, const py::array &nphi, const py::array &phi0, const py::array &ringstart, size_t mmax, ptrdiff_t pixstride, size_t nthreads, py::object &leg)
  {
  if (isPyarr<float>(map))
    return Py2_map2leg<float>(map, nphi, phi0, ringstart, mmax, pixstride, nthreads, leg);
  if (isPyarr<double>(map))
    return Py2_map2leg<double>(map, nphi, phi0, ringstart, mmax, pixstride, nthreads, leg);
  MR_fail("type matching failed: 'map' has neither type 'f4' nor 'f8'");
  }
template<typename T> py::array Py2_leg2map(const py::array &leg_, const py::array &nphi_, const py::array &phi0_, const py::array &ringstart_, ptrdiff_t pixstride, size_t nthreads, py::object &map__)
  {
  auto leg = to_mav<complex<T>,3>(leg_, false);
  auto nphi = to_mav<size_t,1>(nphi_, false);
  auto phi0 = to_mav<double,1>(phi0_, false);
  auto ringstart = to_mav<size_t,1>(ringstart_, false);
  auto map_ = get_optional_Pyarr_minshape<T>(map__, {leg.shape(0),min_mapdim(nphi, ringstart, pixstride)});
  auto map = to_mav<T,2>(map_, true);
  MR_assert(map.shape(0)==leg.shape(0), "bad number of components in map array");
  leg2map(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
  return map_;
  }
py::array Py_leg2map(const py::array &leg, const py::array &nphi, const py::array &phi0, const py::array &ringstart, ptrdiff_t pixstride, size_t nthreads, py::object &map)
  {
  if (isPyarr<complex<float>>(leg))
    return Py2_leg2map<float>(leg, nphi, phi0, ringstart, pixstride, nthreads, map);
  if (isPyarr<complex<double>>(leg))
    return Py2_leg2map<double>(leg, nphi, phi0, ringstart, pixstride, nthreads, map);
  MR_fail("type matching failed: 'leg' has neither type 'c8' nor 'c16'");
  }

template<typename T>py::array Py2_prep_for_analysis(py::array &leg_, size_t spin, size_t nthreads)
  {
  auto leg = to_mav<complex<T>,3>(leg_, true);
  prep_for_analysis(leg, spin, nthreads);
  return leg_;
  }
py::array Py_prep_for_analysis(py::array &leg, size_t spin, size_t nthreads)
  {
  if (isPyarr<complex<float>>(leg))
    return Py2_prep_for_analysis<float>(leg, spin, nthreads);
  if (isPyarr<complex<double>>(leg))
    return Py2_prep_for_analysis<double>(leg, spin, nthreads);
  MR_fail("type matching failed: 'leg' has neither type 'c8' nor 'c16'");
  }

template<typename T> void Py2_resample_theta(const py::array &legi_, bool npi, bool spi,
  py::array &lego_, bool npo, bool spo, size_t spin, size_t nthreads)
  {
  auto legi = to_mav<complex<T>,2>(legi_, false);
  auto lego = to_mav<complex<T>,2>(lego_, true);
  resample_theta(legi, npi, spi, lego, npo, spo, spin, nthreads);
  }
void Py_resample_theta(const py::array &legi, bool npi, bool spi,
  py::array &lego, bool npo, bool spo, size_t spin, size_t nthreads)
  {
  if (isPyarr<complex<float>>(legi))
    return Py2_resample_theta<float>(legi, npi, spi, lego, npo, spo, spin, nthreads);
  if (isPyarr<complex<double>>(legi))
    return Py2_resample_theta<double>(legi, npi, spi, lego, npo, spo, spin, nthreads);
  MR_fail("type matching failed: 'legi' has neither type 'c8' nor 'c16'");
  }

template<typename T> py::array Py2_synthesis(const py::array &alm_,
  py::object &map__, size_t spin, size_t lmax,
  const py::array &mstart_, ptrdiff_t lstride, 
  const py::array &theta_, 
  const py::array &nphi_,
  const py::array &phi0_, const py::array &ringstart_,
  ptrdiff_t pixstride, size_t nthreads)
  {
  auto alm = to_mav<complex<T>,2>(alm_, false);
  auto mstart = to_mav<size_t,1>(mstart_, false);
  auto theta = to_mav<double,1>(theta_, false);
  auto phi0 = to_mav<double,1>(phi0_, false);
  auto nphi = to_mav<size_t,1>(nphi_, false);
  auto ringstart = to_mav<size_t,1>(ringstart_, false);
  auto map_ = get_optional_Pyarr_minshape<T>(map__, {alm.shape(0), min_mapdim(nphi, ringstart, pixstride)});
  auto map = to_mav<T,2>(map_, true);
  MR_assert(map.shape(0)==alm.shape(0), "bad number of components in map array");
  synthesis(alm, map, spin, lmax, mstart, lstride, theta, nphi, phi0, ringstart, pixstride, nthreads);
  return map_;
  }
py::array Py_synthesis(const py::array &alm, const py::array &theta,
  size_t lmax, const py::array &mstart, 
  const py::array &nphi,
  const py::array &phi0, const py::array &ringstart, size_t spin, ptrdiff_t lstride, ptrdiff_t pixstride,
  size_t nthreads, py::object &map)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_synthesis<float>(alm, map, spin, lmax, mstart, lstride, theta, 
      nphi, phi0, ringstart, pixstride, nthreads);
  else if (isPyarr<complex<double>>(alm))
    return Py2_synthesis<double>(alm, map, spin, lmax, mstart, lstride, theta, 
      nphi, phi0, ringstart, pixstride, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_adjoint_synthesis(py::object &alm__,
  size_t lmax, const py::array &mstart_, ptrdiff_t lstride, 
  const py::array &map_, const py::array &theta_, const py::array &phi0_,
  const py::array &nphi_, const py::array &ringstart_, size_t spin, ptrdiff_t pixstride,
  size_t nthreads)
  {
  auto mstart = to_mav<size_t,1>(mstart_, false);
  auto map = to_mav<T,2>(map_, false);
  auto theta = to_mav<double,1>(theta_, false);
  auto phi0 = to_mav<double,1>(phi0_, false);
  auto nphi = to_mav<size_t,1>(nphi_, false);
  auto ringstart = to_mav<size_t,1>(ringstart_, false);
  mav<size_t,1> mval(mstart.shape());
  for (size_t i=0; i<mval.shape(0); ++i)
    mval.v(i) = i;
  auto alm_ = get_optional_Pyarr_minshape<complex<T>>(alm__, {map.shape(0),min_almdim(lmax, mval, mstart, lstride)});
  auto alm = to_mav<complex<T>,2>(alm_, true);
  MR_assert(alm.shape(0)==map.shape(0), "bad number of components in a_lm array");
  adjoint_synthesis(alm, map, spin, lmax, mstart, lstride, theta, nphi, phi0, ringstart, pixstride, nthreads);
  return alm_;
  }
py::array Py_adjoint_synthesis(const py::array &map, const py::array &theta,
 size_t lmax,
  const py::array &mstart,
  const py::array &nphi,
  const py::array &phi0, const py::array &ringstart, size_t spin, ptrdiff_t lstride, ptrdiff_t pixstride,
  size_t nthreads,
  py::object &alm)
  {
  if (isPyarr<float>(map))
    return Py2_adjoint_synthesis<float>(alm, lmax, mstart, lstride, map, theta,
      phi0, nphi, ringstart, spin, pixstride, nthreads);
  else if (isPyarr<double>(map))
    return Py2_adjoint_synthesis<double>(alm, lmax, mstart, lstride, map, theta,
      phi0, nphi, ringstart, spin, pixstride, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }


using a_d = py::array_t<double>;
using a_d_c = py::array_t<double, py::array::c_style | py::array::forcecast>;
using a_c_c = py::array_t<complex<double>,
  py::array::c_style | py::array::forcecast>;

template<typename T> class Py_sharpjob
  {
  private:
    unique_ptr<sharp_geom_info> ginfo;
    unique_ptr<sharp_alm_info> ainfo;
    int64_t lmax_, mmax_, npix_;
    int nthreads;

  public:
    Py_sharpjob () : lmax_(0), mmax_(0), npix_(0), nthreads(1) {}

    string repr() const
      {
      return "<sharpjob_d: lmax=" + dataToString(lmax_) +
        ", mmax=" + dataToString(mmax_) + ", npix=", dataToString(npix_) +".>";
      }

    void set_nthreads(int64_t nthreads_)
      { nthreads = int(nthreads_); }
    void set_gauss_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert((nrings>0)&&(nphi>0),"bad grid dimensions");
      npix_=nrings*nphi;
      ginfo = sharp_make_2d_geom_info (nrings, nphi, 0., 1, nphi, "GL");
      }
    void set_healpix_geometry(int64_t nside)
      {
      MR_assert(nside>0,"bad Nside value");
      npix_=12*nside*nside;
      ginfo = sharp_make_healpix_geom_info (nside, 1);
      }
    void set_fejer1_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_2d_geom_info (nrings, nphi, 0., 1, nphi, "F1");
      }
    void set_fejer2_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_2d_geom_info (nrings, nphi, 0., 1, nphi, "F2");
      }
    void set_cc_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_2d_geom_info (nrings, nphi, 0., 1, nphi, "CC");
      }
    void set_dh_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>1,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_2d_geom_info (nrings, nphi, 0., 1, nphi, "DH");
      }
    void set_mw_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_2d_geom_info (nrings, nphi, 0., 1, nphi, "MW", false);
      }
    void set_triangular_alm_info (int64_t lmax, int64_t mmax)
      {
      MR_assert(mmax>=0,"negative mmax");
      MR_assert(mmax<=lmax,"mmax must not be larger than lmax");
      lmax_=lmax; mmax_=mmax;
      ainfo = sharp_make_triangular_alm_info(lmax,mmax,1);
      }

    int64_t n_alm() const
      { return ((mmax_+1)*(mmax_+2))/2 + (mmax_+1)*(lmax_-mmax_); }

    a_d_c alm2map (const a_c_c &alm) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (alm.size()==n_alm(),
        "incorrect size of a_lm array");
      a_d_c map(npix_);
      auto mr=map.mutable_unchecked<1>();
      auto ar=alm.unchecked<1>();
      sharp_alm2map(&ar[0], &mr[0], *ginfo, *ainfo, 0, nthreads);
      return map;
      }
    a_c_c alm2map_adjoint (const a_d_c &map) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (map.size()==npix_,"incorrect size of map array");
      a_c_c alm(n_alm());
      auto mr=map.unchecked<1>();
      auto ar=alm.mutable_unchecked<1>();
      sharp_map2alm(&ar[0], &mr[0], *ginfo, *ainfo, 0, nthreads);
      return alm;
      }
    a_c_c map2alm (const a_d_c &map) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (map.size()==npix_,"incorrect size of map array");
      a_c_c alm(n_alm());
      auto mr=map.unchecked<1>();
      auto ar=alm.mutable_unchecked<1>();
      sharp_map2alm(&ar[0], &mr[0], *ginfo, *ainfo, SHARP_USE_WEIGHTS, nthreads);
      return alm;
      }
    a_d_c alm2map_spin (const a_c_c &alm, int64_t spin) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      auto ar=alm.unchecked<2>();
      MR_assert((ar.shape(0)==2)&&(ar.shape(1)==n_alm()),
        "incorrect size of a_lm array");
      a_d_c map(vector<size_t>{2,size_t(npix_)});
      auto mr=map.mutable_unchecked<2>();
      sharp_alm2map_spin(spin, &ar(0,0), &ar(1,0), &mr(0,0), &mr(1,0), *ginfo, *ainfo, 0, nthreads);
      return map;
      }
    a_c_c map2alm_spin (const a_d_c &map, int64_t spin) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      auto mr=map.unchecked<2>();
      MR_assert ((mr.shape(0)==2)&&(mr.shape(1)==npix_),
        "incorrect size of map array");
      a_c_c alm(vector<size_t>{2,size_t(n_alm())});
      auto ar=alm.mutable_unchecked<2>();
      sharp_map2alm_spin(spin, &ar(0,0), &ar(1,0), &mr(0,0), &mr(1,0), *ginfo, *ainfo, SHARP_USE_WEIGHTS, nthreads);
      return alm;
      }
  };

constexpr const char *sht_DS = R"""(
Python interface for spherical harmonic transforms and manipulation of
spherical harmonic coefficients.

Error conditions are reported by raising exceptions.
)""";
constexpr const char *sht_experimental_DS = R"""(
Experimental interface to the SHT functionality.

Notes
-----

The functionality in this module is not considered to have a stable interface
and also may be moved to other modules in the future. If you use it, be prepared
to adjust your code at some point ion the future!
)""";

constexpr const char *rotate_alm_DS = R"""(
Rotates a set of spherical harmonic coefficients according to the given Euler angles.

Parameters
----------
alm: numpy.ndarray(((lmax+1)*(lmax=2)/2,), dtype=numpy complex64 or numpy.complex128)
    the spherical harmonic coefficients, in the order
    (0,0), (1,0), (2,0), ... (lmax,0), (1,1), (2,1), ..., (lmax, lmax)
lmax : int >= 0
    Maximum multipole order l of the data set.
psi : float
    First rotation angle about the z-axis. All angles are in radians,
    the rotations are active and the referential system is assumed to be
    right handed.
theta : float
    Second rotation angl about the original (unrotated) y-axis
phi : float
    Third rotation angle about the original (unrotated) z-axis.
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray(same shape and dtype as alm)
)""";

constexpr const char *alm2leg_DS = R"""(
Transforms a set of spherical harmonic coefficients to Legendre coefficients
dependent on theta and m.

Parameters
----------
alm: numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    ncomp must be 1 if spin is 0, else 2.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the parameters `lmax`, 'mval`, `mstart`, and `lstride`.
leg: None or numpy.ndarray((ncomp, nrings, nm), same dtype as `alm`)
    output array containing the Legendre coefficients
    if `None`, a new suitable array is allocated
spin: int >= 0
    the spin to use for the transform
    if spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive)
mval: numpy.ndarray((nm,), dtype = numpy.uint64)
    the m moments for which the transform should be carried out
    entries must be unique and <= lmax
mstart: numpy.ndarray((nm,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with l=0, m=mstart[mi] would be stored, for mi in mval
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
theta: numpy.ndarray((nrings,), dtype=numpy.float64)
    the colatitudes of the map rings
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, nrings, nm), same dtype as `alm`)
    the Legendre coefficients. If `leg` was supplied, this will be the same object.
)""";

constexpr const char *leg2alm_DS = R"""(
Transforms a set of Legendre coefficients to spherical harmonic coefficients

Parameters
----------
leg: numpy.ndarray((ncomp, nrings, nm), dtype=numpy.complex64 or numpy.complex128)
    ncomp must be 1 if spin is 0, else 2
alm: None or numpy.ndarray((ncomp, x), same dtype as `leg`)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the parameters `lmax`, 'mval`, `mstart`, and `lstride`.
    if `None`, a new suitable array is allocated
spin: int >= 0
    the spin to use for the transform
    if spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive)
mval: numpy.ndarray((nm,), dtype = numpy.uint64)
    the m moments for which the transform should be carried out
    entries must be unique and <= lmax
mstart: numpy.ndarray((nm,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with l=0, m=mstart[mi] would be stored, for mi in mval
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
theta: numpy.ndarray((nrings,), dtype=numpy.float64)
    the colatitudes of the map rings
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, *), same dtype as `leg`)
    the Legendre coefficients.
    if `alm` was supplied, this will be the same object
    If newly allocated, the smallest possible second dimensions will be chosen.
)""";

constexpr const char *map2leg_DS = R"""(
Transforms a sky map or several sky maps to Legendre coefficients
dependent on theta and m.

Parameters
----------
map: numpy.ndarray((ncomp, x), dtype=numpy.float32 or numpy.float64)
    the map pixel data.
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
leg: None or numpy.ndarray((ncomp, nrings, mmax+1), dtype=numpy.complex of same accuracy as `map`)
    output array containing the Legendre coefficients
    if `None`, a new suitable array is allocated
nphi: numpy.ndarray((nrings,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((nrings,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
ringstart: numpy.ndarray((nrings,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
mmax: int
    the maximum m moment to compute in this transform. If `leg`
    is provided, `mmax` must be equal to `leg.shape[2]=1`.
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, nrings, nm), dtype=numpy.complex of same accuracy as `map`)
    the Legendre coefficients
    if `leg` was supplied, this will be the same object

Notes
-----
In contrast to `leg2alm` and `alm2leg` the `m` values are assumed to form a
range from 0 to mmax, inclusively. 
)""";

constexpr const char *leg2map_DS = R"""(
Transforms one or more sets of Legendre coefficients to sky maps.

Parameters
----------
leg: numpy.ndarray((ncomp, nrings, mmax+1), numppy.complex64 or numpy.complex128)
    input array containing the Legendre coefficients
map: None or numpy.ndarray((ncomp, x), dtype=numpy.float of same accuracy as `leg`
    the map pixel data.
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
    if `None`, a new suitable array is allocated
nphi: numpy.ndarray((nrings,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((nrings,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
ringstart: numpy.ndarray((nrings,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, x), dtype=numpy.float of same accuracy as `leg`
    the map pixel data.
    If `map` was supplied, this will be the same object
    If newly allocated, the smallest possible second dimensions will be chosen.

Notes
-----
In contrast to `leg2alm` and `alm2leg` the `m` values are assumed to form a
range from 0 to mmax, inclusively. 
)""";

constexpr const char *sharpjob_d_DS = R"""(
Interface class to some of libsharp2's functionality.
)""";

void add_sht(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("sht");
  m.doc() = sht_DS;
  auto m2 = m.def_submodule("experimental");
  m2.doc() = sht_experimental_DS;

//  m2.def("synthesis", &Py_synthesis, "type"_a, "alm"_a, "map"_a, "lmax"_a, "mmax"_a, "spin"_a);
//  m2.def("synthesis", &Py_synthesis, "alm"_a, "map"_a, "lmax"_a, "mmax"_a, "spin"_a, "theta"_a, "nphi"_a, "phi0"_a, "offset"_a);
  m2.def("synthesis", &Py_synthesis, py::kw_only(), "alm"_a, "theta"_a, "lmax"_a, "mstart"_a, "nphi"_a, "phi0"_a, "ringstart"_a, "spin"_a=0, "lstride"_a=1, "pixstride"_a=1, "nthreads"_a=1, "map"_a=None);
  m2.def("adjoint_synthesis", &Py_adjoint_synthesis, py::kw_only(), "map"_a, "theta"_a, "lmax"_a, "mstart"_a, "nphi"_a, "phi0"_a, "ringstart"_a, "spin"_a=0, "lstride"_a=1, "pixstride"_a=1, "nthreads"_a=1, "alm"_a=None);

  m2.def("get_gridweights", &Py_get_gridweights, "type"_a, "nrings"_a);
  m2.def("alm2leg", &Py_alm2leg, alm2leg_DS, py::kw_only(), "alm"_a, "lmax"_a, "theta"_a, "spin"_a=0, "mval"_a=None, "mstart"_a=None, "lstride"_a=1, "nthreads"_a=1, "leg"_a=None);
  m2.def("leg2alm", &Py_leg2alm, leg2alm_DS, py::kw_only(), "leg"_a, "lmax"_a, "theta"_a, "spin"_a=0, "mval"_a=None, "mstart"_a=None, "lstride"_a=1, "nthreads"_a=1, "alm"_a=None);
  m2.def("map2leg", &Py_map2leg, map2leg_DS, py::kw_only(), "map"_a, "nphi"_a, "phi0"_a, "ringstart"_a, "mmax"_a, "pixstride"_a=1, "nthreads"_a=1, "leg"_a=None);
  m2.def("leg2map", &Py_leg2map, leg2map_DS, py::kw_only(), "leg"_a, "nphi"_a, "phi0"_a, "ringstart"_a, "pixstride"_a=1, "nthreads"_a=1, "map"_a=None);
  m2.def("prep_for_analysis", &Py_prep_for_analysis, "leg"_a, "spin"_a, "nthreads"_a=1);
  m2.def("resample_theta", &Py_resample_theta, "legi"_a, "npi"_a, "spi"_a, "lego"_a, "npo"_a, "spo"_a, "spin"_a, "nthreads"_a);
  m.def("rotate_alm", &Py_rotate_alm, rotate_alm_DS, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a, "nthreads"_a=1);

  py::class_<Py_sharpjob<double>> (m, "sharpjob_d", py::module_local(),sharpjob_d_DS)
    .def(py::init<>())
    .def("set_nthreads", &Py_sharpjob<double>::set_nthreads, "nthreads"_a)
    .def("set_gauss_geometry", &Py_sharpjob<double>::set_gauss_geometry,
      "nrings"_a,"nphi"_a)
    .def("set_healpix_geometry", &Py_sharpjob<double>::set_healpix_geometry,
      "nside"_a)
    .def("set_fejer1_geometry", &Py_sharpjob<double>::set_fejer1_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_fejer2_geometry", &Py_sharpjob<double>::set_fejer2_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_cc_geometry", &Py_sharpjob<double>::set_cc_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_dh_geometry", &Py_sharpjob<double>::set_dh_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_mw_geometry", &Py_sharpjob<double>::set_mw_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_triangular_alm_info",
      &Py_sharpjob<double>::set_triangular_alm_info, "lmax"_a, "mmax"_a)
    .def("n_alm", &Py_sharpjob<double>::n_alm)
    .def("alm2map", &Py_sharpjob<double>::alm2map,"alm"_a)
    .def("alm2map_adjoint", &Py_sharpjob<double>::alm2map_adjoint,"map"_a)
    .def("map2alm", &Py_sharpjob<double>::map2alm,"map"_a)
    .def("alm2map_spin", &Py_sharpjob<double>::alm2map_spin,"alm"_a,"spin"_a)
    .def("map2alm_spin", &Py_sharpjob<double>::map2alm_spin,"map"_a,"spin"_a)
    .def("__repr__", &Py_sharpjob<double>::repr);
  }

}

using detail_pymodule_sht::add_sht;

}

