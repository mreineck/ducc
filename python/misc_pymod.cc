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
 *  Copyright (C) 2020-2021 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

#include "ducc0/infra/mav.h"
#include "ducc0/infra/transpose.h"
#include "ducc0/math/fft.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_misc {

using namespace std;
namespace py = pybind11;

py::array Py_GL_weights(size_t nlat, size_t nlon)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_mav<double,1>(res, true);
  GL_Integrator integ(nlat);
  auto wgt = integ.weights();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2.v(i) = wgt[i]*twopi/nlon;
  return move(res);
  }

py::array Py_GL_thetas(size_t nlat)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_mav<double,1>(res, true);
  GL_Integrator integ(nlat);
  auto x = integ.coords();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2.v(i) = acos(-x[i]);
  return move(res);
  }

template<typename T> py::array Py2_transpose(const py::array &in, py::array &out)
  {
  auto in2 = to_fmav<T>(in, false);
  auto out2 = to_fmav<T>(out, true);
  transpose(in2, out2, [](const T &in, T &out){out=in;});
  return out;
  }

py::array Py_transpose(const py::array &in, py::array &out)
  {
  if (isPyarr<float>(in))
    return Py2_transpose<float>(in, out);
  if (isPyarr<double>(in))
    return Py2_transpose<double>(in, out);
  if (isPyarr<complex<float>>(in))
    return Py2_transpose<complex<float>>(in, out);
  if (isPyarr<complex<double>>(in))
    return Py2_transpose<complex<double>>(in, out);
  if (isPyarr<int>(in))
    return Py2_transpose<int>(in, out);
  if (isPyarr<long>(in))
    return Py2_transpose<long>(in, out);
  MR_fail("unsupported datatype");
  }


/*! A numeric filter which produces noise with the power spectrum

    P(f)=(1/fsample)^2*(f^2+fknee^2)/(f^2+fmin^2)

    when fed with Gaussian random numbers of sigma=1.
    \author Stephane Plaszczynski (plaszczy@lal.in2p3.fr) */
class oof2filter
  {
  private:
    double x1, y1, c0, c1, d0;

  public:
    oof2filter (double fmin, double fknee, double fsample)
      : x1(0), y1(0)
      {
      double w0 = pi*fmin/fsample, w1=pi*fknee/fsample;
      c0 = (1+w1)/(1+w0);
      c1 =-(1-w1)/(1+w0);
      d0 = (1-w0)/(1+w0);
      }

    void reset()
      { x1=y1=0; }

    double operator()(double x2)
      {
      double y2 = c0*x2 + c1*x1 + d0*y1;
      x1 = x2;
      y1 = y2;
      return y2;
      }
  };


/*! A numeric filter, based on superposition of 1/f^2 filters.
    see : {Keshner,PROC-IEE,vol-70 (1982)}
    that approximates the power spectrum

    P(f)=(1/fsamp)^2[(f^2+fknee^2)/(f^2+fmin^2)]^(alpha/2)

    for 0<alpha<2, when fed with Gaussian random numbers of sigma=1.

    Errors should be below 1% for any alpha.

    \author Stephane Plaszczynski (plaszczy@lal.in2p3.fr) */
class oofafilter
  {
  private:
    vector<oof2filter> filter;

  public:
    oofafilter (double alpha, double fmin, double fknee, double fsample)
      {
      double lw0 = log10(twopi*fmin), lw1 = log10(twopi*fknee);

      int Nproc = max(1,int(2*(lw1-lw0)));
      double dp = (lw1-lw0)/Nproc;
      double p0 = lw0 + dp*0.5*(1+0.5*alpha);
      for (int i=0; i<Nproc; ++i)
        {
        double p_i = p0+i*dp;
        double z_i = p_i - 0.5*dp*alpha;

        filter.push_back
          (oof2filter(pow(10.,p_i)/twopi,pow(10.,z_i)/twopi,fsample));
        }
      }

    double operator()(double x2)
      {
      for (unsigned int i=0; i<filter.size(); ++i)
        x2 = filter[i](x2);
      return x2;
      }

    void reset()
      {
      for (unsigned int i=0; i<filter.size(); ++i)
        filter[i].reset();
      }
  };


class OofaNoise
  {
  private:
    oofafilter filter;
    double sigma;

  public:
    OofaNoise(double sigmawhite_, double f_knee_, double f_min_,
      double f_samp_, double slope_)
      : filter(slope_, f_min_, f_knee_, f_samp_), sigma(sigmawhite_)
      {}

    void filterGaussian(mav<double,1> &data)
      {
      for (size_t i=0; i<data.shape(0); ++i)
        data.v(i) = sigma*filter(data(i));
      }

    void reset()
      {
      filter.reset();
      }
  };

class Py_OofaNoise
  {
  private:
    OofaNoise gen;

  public:
    Py_OofaNoise(double sigmawhite, double f_knee, double f_min,
      double f_samp, double slope)
      : gen(sigmawhite, f_min, f_knee, f_samp, slope) {}

    py::array filterGaussian(const py::array &rnd_)
      {
      auto rnd = to_mav<double,1>(rnd_, false);
      auto res_ = make_Pyarr<double>({rnd.shape(0)});
      auto res = to_mav<double,1>(res_, true);
      res.apply(rnd, [](double &out, double in) {out=in;});
      gen.filterGaussian(res);
      return res_;
      }
  };

constexpr const char *Py_OofaNoise_DS = R"""(
Class for computing noise with a power spectrum that has a given slope between
a minimum frequency f_min and a knee frequency f_knee, and is white outside
this region.

Original implementation by Stephane Plaszczynski;
for details see https://arxiv.org/abs/astro-ph/0510081.
)""";

constexpr const char *Py_OofaNoise_init_DS = R"""(
OofaNoise constructor

Parameters
----------
sigmawhite : float
    sigma of the white noise part of the produced spectrum above f_knee;
    units are arbitrary
f_knee : float
    knee frequency in Hz. Above this frequency, the spectrum will be white.
f_min : float
    minimum frequency in Hz. Below this frequency, the spectrum will become
    white again. Must be lower than f_knee.
f_samp : float
    sampling frequency in Hz at which the noise samples should be generated.
slope : float
    the slope of the spectrum between f_min and f_knee. Must be in [0; -2]
)""";

constexpr const char *Py_OofaNoise_filterGaussian_DS = R"""(
Apply the noise filter to input Gaussian noise

Parameters
----------
rnd : numpy.ndarray((nsamples,), dtype=numpy.float64)
    input Gaussian random numbers with mean=0 and sigma=1

Returns
-------
numpy.ndarray((nsamples,), dtype=numpy.float64):
    the filtered noise samples with the requested spectral shape.

Notes
-----
Subsequent calls to this method will continue the same noise stream; i.e. it
is possible to generate a very long noise time stream chunk by chunk.
To generate multiple independent noise streams, use different `OofaNoise`
objects (and supply them with independent Gaussian noise streams)! 
)""";

constexpr const char *misc_DS = R"""(
Various unsorted utilities

Notes
-----

The functionality in this module is not considered to have a stable interface
and also may be moved to other modules in the future. If you use it, be prepared
to adjust your code at some point ion the future!
)""";

void add_misc(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("misc");
  m.doc() = misc_DS;

  m.def("GL_weights",&Py_GL_weights, "nlat"_a, "nlon"_a);
  m.def("GL_thetas",&Py_GL_thetas, "nlat"_a);

  m.def("transpose",&Py_transpose, "in"_a, "out"_a);

  py::class_<Py_OofaNoise> (m, "OofaNoise", Py_OofaNoise_DS, py::module_local())
    .def(py::init<double, double, double, double, double>(), Py_OofaNoise_init_DS,
      "sigmawhite"_a, "f_knee"_a, "f_min"_a, "f_samp"_a, "slope"_a)
    .def ("filterGaussian", &Py_OofaNoise::filterGaussian,
      Py_OofaNoise_filterGaussian_DS, "rnd"_a);
  }

}

using detail_pymodule_misc::add_misc;

}

