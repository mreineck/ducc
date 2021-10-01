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
#include <cmath>
#include <complex>

#include "ducc0/infra/mav.h"
#include "ducc0/infra/transpose.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_misc {

using namespace std;
namespace py = pybind11;


constexpr const char *Py_vdot_DS = R"""(
Compute the scalar product of two arrays or scalars., i.e. sum_i(conj(a_i)*b_i)
over all array elements.

Parameters
----------
a : scalar or numpy.ndarray of any shape; dtype must be a float or complex type
b : scalar or numpy.ndarray of the same shape as `a`; dtype must be a float or complex type

Returns
-------
float or complex
    the scalar product.
    If the result can be represented by a float value, it will be returned as
    float, otherwise as complex.

Notes
-----
The accumulation is performed in long double precision for good accuracy.
)""";

template<typename T1, typename T2> py::object Py3_vdot(const py::array &a_, const py::array &b_)
  {
  const auto a = to_cfmav<T1>(a_);
  const auto b = to_cfmav<T2>(b_);
  using Tacc = long double;
  complex<Tacc> acc=0;
  {
  py::gil_scoped_release release;
  mav_apply([&acc](const T1 &v1, const T2 &v2)
    {
    complex<Tacc> cv1(v1), cv2(v2);
    acc += conj(cv1) * cv2;
    }, 1, a, b);
  }
  return (acc.imag()==0) ? py::cast(acc.real()) : py::cast(acc);
  }
template<typename T1> py::object Py2_vdot(const py::array &a, const py::array &b)
  {
  if (isPyarr<float>(b))
    return Py3_vdot<T1,float>(a,b);
  if (isPyarr<double>(b))
    return Py3_vdot<T1,double>(a,b);
  if (isPyarr<long double>(b))
    return Py3_vdot<T1,long double>(a,b);
  if (isPyarr<complex<float>>(b))
    return Py3_vdot<T1,complex<float>>(a,b);
  if (isPyarr<complex<double>>(b))
    return Py3_vdot<T1,complex<double>>(a,b);
  if (isPyarr<complex<long double>>(b))
    return Py3_vdot<T1,complex<long double>>(a,b);
  MR_fail("type matching failed");
  }
py::object Py_vdot(const py::object &a, const py::object &b)
  {
  if ((!isPyarr(a)) || (py::array(a).ndim()==0)) // scalars
    {
    auto xa = a.cast<complex<long double>>(),
         xb = b.cast<complex<long double>>();
    auto res = conj(xa)*xb;
    return (res.imag()==0) ? py::cast(res.real()) : py::cast(res);
    }
  if (isPyarr<float>(a))
    return Py2_vdot<float>(a,b);
  if (isPyarr<double>(a))
    return Py2_vdot<double>(a,b);
  if (isPyarr<long double>(a))
    return Py2_vdot<long double>(a,b);
  if (isPyarr<complex<float>>(a))
    return Py2_vdot<complex<float>>(a,b);
  if (isPyarr<complex<double>>(a))
    return Py2_vdot<complex<double>>(a,b);
  if (isPyarr<complex<long double>>(a))
    return Py2_vdot<complex<long double>>(a,b);
  MR_fail("type matching failed");
  }

constexpr const char *Py_l2error_DS = R"""(
Compute the L2 error between two arrays or scalars.
More specifically, compute
``sqrt(sum_i(|a_i - b_i|^2) / max(sum_i(|a_i|^2), sum_i(|b_i|^2)))``,
where i goes over all array elements.

Parameters
----------
a : scalar or numpy.ndarray of any shape; dtype must be a float or complex type
b : scalar or numpy.ndarray of the same shape as `a`; dtype must be a float or complex type

Returns
-------
float
    the L2 error between the two objects.

Notes
-----
The accumulations are performed in long double precision for good accuracy.
)""";
template<typename T1, typename T2> double Py3_l2error(const py::array &a_, const py::array &b_)
  {
  const auto a = to_cfmav<T1>(a_);
  const auto b = to_cfmav<T2>(b_);
  using Tacc = long double;
  Tacc acc1=0, acc2=0, acc3=0;
  {
  py::gil_scoped_release release;
  mav_apply([&acc1, &acc2, &acc3](const T1 &v1, const T2 &v2)
    {
    complex<Tacc> cv1(v1), cv2(v2);
    acc1 += norm(cv1);
    acc2 += norm(cv2);
    acc3 += norm(cv1-cv2);
    }, 1, a, b);
  }
  return double(sqrt(acc3/max(acc1,acc2)));
  }
template<typename T1> double Py2_l2error(const py::array &a, const py::array &b)
  {
  if (isPyarr<float>(b))
    return Py3_l2error<float,T1>(b,a);
  if (isPyarr<double>(b))
    return Py3_l2error<double,T1>(b,a);
  if (isPyarr<long double>(b))
    return Py3_l2error<long double,T1>(b,a);
  if (isPyarr<complex<float>>(b))
    return Py3_l2error<T1,complex<float>>(a,b);
  if (isPyarr<complex<double>>(b))
    return Py3_l2error<T1,complex<double>>(a,b);
  if (isPyarr<complex<long double>>(b))
    return Py3_l2error<T1,complex<long double>>(a,b);
  MR_fail("type matching failed");
  }
double Py_l2error(const py::object &a, const py::object &b)
  {
  if ((!isPyarr(a)) || (py::array(a).ndim()==0)) // scalars
    {
    auto xa = a.cast<complex<long double>>(),
         xb = b.cast<complex<long double>>();
    auto res = abs(xa-xb)/max(abs(xa), abs(xb));
    return double(res);
    }
  if (isPyarr<float>(a))
    return Py2_l2error<float>(a,b);
  if (isPyarr<double>(a))
    return Py2_l2error<double>(a,b);
  if (isPyarr<long double>(a))
    return Py2_l2error<long double>(a,b);
  if (isPyarr<complex<float>>(a))
    return Py2_l2error<complex<float>>(a,b);
  if (isPyarr<complex<double>>(a))
    return Py2_l2error<complex<double>>(a,b);
  if (isPyarr<complex<long double>>(a))
    return Py2_l2error<complex<long double>>(a,b);
  MR_fail("type matching failed");
  }

py::array Py_GL_weights(size_t nlat, size_t nlon)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_vmav<double,1>(res);
  GL_Integrator integ(nlat);
  auto wgt = integ.weights();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2(i) = wgt[i]*twopi/nlon;
  return move(res);
  }

py::array Py_GL_thetas(size_t nlat)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_vmav<double,1>(res);
  {
  py::gil_scoped_release release;

  GL_Integrator integ(nlat);
  auto x = integ.coords();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2(i) = acos(-x[i]);
  }
  return move(res);
  }

template<typename T> py::array Py2_transpose(const py::array &in, py::array &out)
  {
  auto in2 = to_cfmav<T>(in);
  auto out2 = to_vfmav<T>(out);
  {
  py::gil_scoped_release release;
  transpose(in2, out2, [](const T &in, T &out){out=in;});
  }
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

constexpr const char *Py_make_noncritical_DS = R"""(
Returns a copy of the input array with a memory layout that avoids critical
strides.

As an example, the array generated by `np.zeros((1024,1024))` has a critical
stride of 8192 bytes for axis 0. Accessing the array elements in the order
[0, 0], [1, 0], [2,0] etc. is very slow in this case, but exactly this kind of
access is required for, e.g., computing the FFT over axis 0 of the array.
If this kind of critical stride is detected, `make_noncritical()` embeds the
array into a slightly larger array, in this case with the shape [1024,1027],
which improves access times significantly.

This routine considers every stride which is a multiple of 4096 bytes as
critical, which should be a decent heuristic.

Parameters
----------
in : numpy.ndarray (float or integer dtype)
    the input array

Returns
-------
numpy.ndarray (same dtype and content as `in`)
    A copy of the array with noncritical strides
)""";
template<typename T> py::array Py2_make_noncritical(const py::array &in)
  {
  auto in2 = to_cfmav<T>(in);
  auto out = make_noncritical_Pyarr<T>(in2.shape());
  auto out2 = to_vfmav<T>(out);
  mav_apply([](T &v1, const T &v2) { v1=v2; }, 1, out2, in2);
  return out;
  }

py::array Py_make_noncritical(const py::array &in)
  {
  if (isPyarr<float>(in))
    return Py2_make_noncritical<float>(in);
  if (isPyarr<double>(in))
    return Py2_make_noncritical<double>(in);
  if (isPyarr<long double>(in))
    return Py2_make_noncritical<long double>(in);
  if (isPyarr<complex<float>>(in))
    return Py2_make_noncritical<complex<float>>(in);
  if (isPyarr<complex<double>>(in))
    return Py2_make_noncritical<complex<double>>(in);
  if (isPyarr<complex<long double>>(in))
    return Py2_make_noncritical<complex<long double>>(in);
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

    void filterGaussian(vmav<double,1> &data)
      {
      for (size_t i=0; i<data.shape(0); ++i)
        data(i) = sigma*filter(data(i));
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
      auto rnd = to_cmav<double,1>(rnd_);
      auto res_ = make_Pyarr<double>({rnd.shape(0)});
      auto res = to_vmav<double,1>(res_);
      {
      py::gil_scoped_release release;

      mav_apply([](double &out, double in) {out=in;}, 1, res, rnd);
      gen.filterGaussian(res);
      }
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

  m.def("vdot", &Py_vdot, Py_vdot_DS, "a"_a, "b"_a);
  m.def("l2error", &Py_l2error, Py_l2error_DS, "a"_a, "b"_a);

  m.def("GL_weights",&Py_GL_weights, "nlat"_a, "nlon"_a);
  m.def("GL_thetas",&Py_GL_thetas, "nlat"_a);

  m.def("transpose",&Py_transpose, "in"_a, "out"_a);

  m.def("make_noncritical",&Py_make_noncritical,Py_make_noncritical_DS,"in"_a);

  py::class_<Py_OofaNoise> (m, "OofaNoise", Py_OofaNoise_DS, py::module_local())
    .def(py::init<double, double, double, double, double>(), Py_OofaNoise_init_DS,
      "sigmawhite"_a, "f_knee"_a, "f_min"_a, "f_samp"_a, "slope"_a)
    .def ("filterGaussian", &Py_OofaNoise::filterGaussian,
      Py_OofaNoise_filterGaussian_DS, "rnd"_a);
  }

}

using detail_pymodule_misc::add_misc;

}

