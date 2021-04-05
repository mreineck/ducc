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
 *  Copyright (C) 2017-2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

#include "ducc0/sharp/sht.h"
#include "ducc0/sharp/sharp.h"
#include "ducc0/sharp/alm.h"
#include "ducc0/infra/string_utils.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_sht {

using namespace std;

namespace py = pybind11;

auto None = py::none();

template<typename T> py::array pyrotate_alm(const py::array &alm_, int64_t lmax,
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

#if 1

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

py::array Pyget_gridweights(const string &type, size_t nrings)
  {
  auto wgt_ = make_Pyarr<double>({nrings});
  auto wgt = to_mav<double,1>(wgt_, true);
  get_gridweights(type, wgt);
  return wgt_;
  }

template<typename T> py::array alm2leg2(const py::array &alm_, const py::array &theta_, size_t lmax, size_t spin, const py::object &mval_, const py::object &mstart_, size_t nthreads, py::object &out_)
  {
  auto alm = to_mav<complex<double>,2>(alm_, false);
  auto theta = to_mav<double,1>(theta_, false);
  size_t nalm = ((lmax+1)*(lmax+2))/2;
  MR_assert(alm.shape(1)==nalm, "bad a_lm array size");
  MR_assert(mval_.is_none()==mstart_.is_none(), "mval and mstart must be supplied together");
  mav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  auto leg_ = get_optional_Pyarr<complex<T>>(out_, {alm.shape(0),theta.shape(0),mval.shape(0)});
  auto leg = to_mav<complex<double>,3>(leg_, true);
  alm2leg(alm, leg, theta, mval, mstart, lmax, spin, nthreads, ALM2MAP);
  return leg_;
  }
py::array Pyalm2leg(const py::array &alm, const py::array &theta, size_t lmax, size_t spin, const py::object &mval, const py::object &mstart, size_t nthreads, py::object &out)
  {
  if (isPyarr<complex<float>>(alm))
    return alm2leg2<float>(alm, theta, lmax, spin, mval, mstart, nthreads, out);
  if (isPyarr<complex<double>>(alm))
    return alm2leg2<double>(alm, theta, lmax, spin, mval, mstart, nthreads, out);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array leg2alm2(const py::array &leg_, const py::array &theta_, size_t lmax, size_t spin, const py::object &mval_, const py::object &mstart_, size_t nthreads, py::object &out_)
  {
  auto leg = to_mav<complex<double>,3>(leg_, false);
  auto theta = to_mav<double,1>(theta_, false);
  size_t nalm = ((lmax+1)*(lmax+2))/2;
  MR_assert(leg.shape(1)==theta.shape(0), "bad leg array size");
  auto alm_ = get_optional_Pyarr<complex<T>>(out_, {leg.shape(0),nalm});
  auto alm = to_mav<complex<double>,2>(alm_, true);
  mav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  leg2alm(leg, alm, theta, mval, mstart, lmax, spin, nthreads);
  return alm_;
  }
py::array Pyleg2alm(const py::array &leg, const py::array &theta, size_t lmax, size_t spin, const py::object &mval, const py::object &mstart, size_t nthreads, py::object &out)
  {
  if (isPyarr<complex<float>>(leg))
    return leg2alm2<float>(leg, theta, lmax, spin, mval, mstart, nthreads, out);
  if (isPyarr<complex<double>>(leg))
    return leg2alm2<double>(leg, theta, lmax, spin, mval, mstart, nthreads, out);
  MR_fail("type matching failed: 'leg' has neither type 'c8' nor 'c16'");
  }

py::array Pyprep_for_analysis(py::array &leg_, size_t spin, size_t nthreads)
  {
  auto leg = to_mav<complex<double>,3>(leg_, true);
  prep_for_analysis(leg, spin, nthreads);
  return leg_;
  }
py::array Pyprep_for_analysis2(py::array &leg_, size_t spin, size_t nthreads)
  {
  auto leg = to_mav<complex<double>,3>(leg_, true);
  prep_for_analysis2(leg, spin, nthreads);
  return leg_;
  }
void Pyresample_theta(const py::array &legi_, bool npi, bool spi,
  py::array &lego_, bool npo, bool spo, size_t spin, size_t nthreads)
  {
  auto legi = to_mav<complex<double>,2>(legi_, false);
  auto lego = to_mav<complex<double>,2>(lego_, true);
  resample_theta(legi, npi, spi, lego, npo, spo, spin, nthreads);
  }

#endif

using a_d = py::array_t<double>;
using a_d_c = py::array_t<double, py::array::c_style | py::array::forcecast>;
using a_c_c = py::array_t<complex<double>,
  py::array::c_style | py::array::forcecast>;

template<typename T> class py_sharpjob
  {
  private:
    unique_ptr<sharp_geom_info> ginfo;
    unique_ptr<sharp_alm_info> ainfo;
    int64_t lmax_, mmax_, npix_;
    int nthreads;

  public:
    py_sharpjob () : lmax_(0), mmax_(0), npix_(0), nthreads(1) {}

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

const char *sht_DS = R"""(
Python interface for some of the libsharp functionality

Error conditions are reported by raising exceptions.
)""";

void add_sht(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("sht");
  m.doc() = sht_DS;

#if 1
//  m.def("synthesis", &Pysynthesis, "type"_a, "alm"_a, "map"_a, "lmax"_a, "mmax"_a, "spin"_a);
//  m.def("synthesis", &Pysynthesis, "alm"_a, "map"_a, "lmax"_a, "mmax"_a, "spin"_a, "theta"_a, "nphi"_a, "phi0"_a, "offset"_a);
  m.def("get_gridweights", &Pyget_gridweights, "type"_a, "nrings"_a);
  m.def("alm2leg", &Pyalm2leg, "alm"_a, "theta"_a, "lmax"_a, "spin"_a, "mval"_a=None, "mstart"_a=None, "nthreads"_a=1, "out"_a=None);
  m.def("leg2alm", &Pyleg2alm, "leg"_a, "theta"_a, "lmax"_a, "spin"_a, "mval"_a=None, "mstart"_a=None, "nthreads"_a=1, "out"_a=None);
  m.def("prep_for_analysis", &Pyprep_for_analysis, "leg"_a, "spin"_a, "nthreads"_a=1);
  m.def("prep_for_analysis2", &Pyprep_for_analysis2, "leg"_a, "spin"_a, "nthreads"_a=1);
  m.def("resample_theta", &Pyresample_theta, "legi"_a, "npi"_a, "spi"_a, "lego"_a, "npo"_a, "spo"_a, "spin"_a, "nthreads"_a);
  m.def("rotate_alm", &pyrotate_alm<double>, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a, "nthreads"_a=1);

#endif

  py::class_<py_sharpjob<double>> (m, "sharpjob_d", py::module_local())
    .def(py::init<>())
    .def("set_nthreads", &py_sharpjob<double>::set_nthreads, "nthreads"_a)
    .def("set_gauss_geometry", &py_sharpjob<double>::set_gauss_geometry,
      "nrings"_a,"nphi"_a)
    .def("set_healpix_geometry", &py_sharpjob<double>::set_healpix_geometry,
      "nside"_a)
    .def("set_fejer1_geometry", &py_sharpjob<double>::set_fejer1_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_fejer2_geometry", &py_sharpjob<double>::set_fejer2_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_cc_geometry", &py_sharpjob<double>::set_cc_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_dh_geometry", &py_sharpjob<double>::set_dh_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_mw_geometry", &py_sharpjob<double>::set_mw_geometry,
      "nrings"_a, "nphi"_a)
    .def("set_triangular_alm_info",
      &py_sharpjob<double>::set_triangular_alm_info, "lmax"_a, "mmax"_a)
    .def("n_alm", &py_sharpjob<double>::n_alm)
    .def("alm2map", &py_sharpjob<double>::alm2map,"alm"_a)
    .def("alm2map_adjoint", &py_sharpjob<double>::alm2map_adjoint,"map"_a)
    .def("map2alm", &py_sharpjob<double>::map2alm,"map"_a)
    .def("alm2map_spin", &py_sharpjob<double>::alm2map_spin,"alm"_a,"spin"_a)
    .def("map2alm_spin", &py_sharpjob<double>::map2alm_spin,"map"_a,"spin"_a)
    .def("__repr__", &py_sharpjob<double>::repr);
  }

}

using detail_pymodule_sht::add_sht;

}

