/*
 *  This file is part of pyHealpix.
 *
 *  pyHealpix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  pyHealpix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with pyHealpix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.sourceforge.net
 */

/*
 *  pyHealpix is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2017-2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

#include "mr_util/sharp/sharp.h"
#include "mr_util/sharp/sharp_geomhelpers.h"
#include "mr_util/sharp/sharp_almhelpers.h"
#include "mr_util/infra/string_utils.h"
#include "mr_util/infra/error_handling.h"
#include "mr_util/infra/mav.h"
#include "mr_util/math/fft.h"
#include "mr_util/math/constants.h"
#include "mr_util/bindings/pybind_utils.h"

namespace mr {

namespace detail_pymodule_sht {

using namespace std;

namespace py = pybind11;

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
      ginfo = sharp_make_gauss_geom_info (nrings, nphi, 0., 1, nphi);
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
      ginfo = sharp_make_fejer1_geom_info (nrings, nphi, 0., 1, nphi);
      }
    void set_fejer2_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_fejer2_geom_info (nrings, nphi, 0., 1, nphi);
      }
    void set_cc_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_cc_geom_info (nrings, nphi, 0., 1, nphi);
      }
    void set_dh_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>1,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_dh_geom_info (nrings, nphi, 0., 1, nphi);
      }
    void set_mw_geometry(int64_t nrings, int64_t nphi)
      {
      MR_assert(nrings>0,"bad nrings value");
      MR_assert(nphi>0,"bad nphi value");
      npix_=nrings*nphi;
      ginfo = sharp_make_mw_geom_info (nrings, nphi, 0., 1, nphi);
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

void upsample_to_cc(const mav<double,2> &in, bool has_np, bool has_sp,
  mav<double,2> &out)
  {
  size_t ntheta_in = in.shape(0),
         ntheta_out = out.shape(0),
         nphi = in.shape(1);
  MR_assert(out.shape(1)==nphi, "phi dimensions must be equal");
  MR_assert((nphi&1)==0, "nphi must be even");
  size_t nrings_in = 2*ntheta_in-has_np-has_sp;
  size_t nrings_out = 2*ntheta_out-2;
  MR_assert(nrings_out>=nrings_in, "number of rings must increase");
  constexpr size_t delta=128;
  for (size_t js=0; js<nphi; js+=delta)
    {
    size_t je = min(js+delta, nphi);
    mav<double,2> tmp({nrings_out,je-js});
    fmav<double> ftmp(tmp);
    mav<double,2> tmp2(tmp.vdata(),{nrings_in, je-js}, true);
    fmav<double> ftmp2(tmp2);
    // enhance to "double sphere"
    if (has_np)
      for (size_t j=js; j<je; ++j)
        tmp2.v(0,j-js) = in(0,j);
    if (has_sp)
      for (size_t j=js; j<je; ++j)
        tmp2.v(ntheta_in-1,j-js) = in(ntheta_in-1,j);
    for (size_t i=has_np, i2=nrings_in-1; i+has_sp<ntheta_in; ++i,--i2)
      for (size_t j=js,j2=js+nphi/2; j<je; ++j,++j2)
        {
        if (j2>=nphi) j2-=nphi;
        tmp2.v(i,j-js) = in(i,j);
        tmp2.v(i2,j-js) = in(i,j2);
        }
    // FFT in theta direction
    r2r_fftpack(ftmp2,ftmp2,{0},true,true,1./nrings_in,0);
    if (!has_np)  // shift
      {
      double ang = -pi/nrings_in;
      for (size_t i=1; i<ntheta_in; ++i)
        {
        complex<double> rot(cos(i*ang),sin(i*ang));
        for (size_t j=js; j<je; ++j)
          {
          complex<double> ctmp(tmp2(2*i-1,j-js),tmp2(2*i,j-js));
          ctmp *= rot;
          tmp2.v(2*i-1,j-js) = ctmp.real();
          tmp2.v(2*i  ,j-js) = ctmp.imag();
          }
        }
      }
    // zero-padding
    for (size_t i=nrings_in; i<nrings_out; ++i)
      for (size_t j=js; j<je; ++j)
        tmp.v(i,j-js) = 0;
    // FFT back
    r2r_fftpack(ftmp,ftmp,{0},false,false,1.,0);
    // copy to output map
    for (size_t i=0; i<ntheta_out; ++i)
      for (size_t j=js; j<je; ++j)
        out.v(i,j) = tmp(i,j-js);
    }
  }

py::array py_upsample_to_cc(const py::array &in, size_t nrings_out, bool has_np,
  bool has_sp, py::object &out_)
  {
  auto in2 = to_mav<double,2>(in);
  auto out = get_optional_Pyarr<double>(out_, {nrings_out,size_t(in.shape(1))});
  auto out2 = to_mav<double,2>(out,true);
    MR_assert(out2.writable(),"x1");
  upsample_to_cc(in2, has_np, has_sp, out2);
  return move(out);
  }

void add_sht(py::module &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("sht");
  m.doc() = sht_DS;

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
  m.def("upsample_to_cc",&py_upsample_to_cc, "in"_a, "nrings_out"_a,
    "has_np"_a, "has_sp"_a, "out"_a=py::none());
  }

}

using detail_pymodule_sht::add_sht;

}

