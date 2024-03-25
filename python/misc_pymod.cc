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
 *  Copyright (C) 2020-2024 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <complex>

#include "ducc0/infra/mav.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/wigner3j.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_misc {

using namespace std;
namespace py = pybind11;
auto None = py::none();

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
  {
  py::gil_scoped_release release;
  GL_Integrator integ(nlat);
  auto wgt = integ.weights();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2(i) = wgt[i]*twopi/nlon;
  }
  return res;
  }

py::array Py_GL_thetas(size_t nlat)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_vmav<double,1>(res);
  {
  py::gil_scoped_release release;

  GL_Integrator integ(nlat);
  auto th = integ.thetas();
  for (size_t i=0; i<nlat; ++i)
    res2(i) = th[nlat-1-i];
  }
  return res;
  }

template<typename T> py::array Py2_transpose(const py::array &in,
  py::array &out, size_t nthreads)
  {
  auto in2 = to_cfmav<T>(in);
  auto out2 = to_vfmav<T>(out);
  {
  py::gil_scoped_release release;
  mav_apply([](const T &in, T &out) {out=in;}, nthreads, in2, out2);
  }
  return out;
  }

py::array Py_transpose(const py::array &in, py::array &out, size_t nthreads=1)
  {
  if (isPyarr<float>(in))
    return Py2_transpose<float>(in, out, nthreads);
  if (isPyarr<double>(in))
    return Py2_transpose<double>(in, out, nthreads);
  if (isPyarr<complex<float>>(in))
    return Py2_transpose<complex<float>>(in, out, nthreads);
  if (isPyarr<complex<double>>(in))
    return Py2_transpose<complex<double>>(in, out, nthreads);
  if (isPyarr<int>(in))
    return Py2_transpose<int>(in, out, nthreads);
  if (isPyarr<long>(in))
    return Py2_transpose<long>(in, out, nthreads);
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

constexpr const char *Py_empty_noncritical_DS = R"""(
Creates an uninitialized array of the requested shape and data type,
with a memory layout that avoids critical strides.

As an example, the array generated by `np.zeros((1024,1024))` has a critical
stride of 8192 bytes for axis 0. Accessing the array elements in the order
[0, 0], [1, 0], [2,0] etc. is very slow in this case, but exactly this kind of
access is required for, e.g., computing the FFT over axis 0 of the array.
If this kind of critical stride is detected, `empty_noncritical()` embeds the
array into a slightly larger array, in this case with the shape [1024,1027],
which improves access times significantly.

This routine considers every stride which is a multiple of 4096 bytes as
critical, which should be a decent heuristic.

Parameters
----------
shape : sequence of int
    the array shape
dtype :
    the requested data type

Returns
-------
numpy.ndarray (shape, dtype=dtype)
    An uninitialized numpy array with the requested properties
)""";
py::array Py_empty_noncritical(const vector<size_t> &shape,
  const py::object &dtype_)
  {
  auto dtype = normalizeDtype(dtype_);
  if (isDtype<float>(dtype))
    return make_noncritical_Pyarr<float>(shape);
  if (isDtype<double>(dtype))
    return make_noncritical_Pyarr<double>(shape);
  if (isDtype<long double>(dtype))
    return make_noncritical_Pyarr<long double>(shape);
  if (isDtype<complex<float>>(dtype))
    return make_noncritical_Pyarr<complex<float>>(shape);
  if (isDtype<complex<double>>(dtype))
    return make_noncritical_Pyarr<complex<double>>(shape);
  if (isDtype<complex<long double>>(dtype))
    return make_noncritical_Pyarr<complex<long double>>(shape);
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
      y1 = c0*x2 + c1*x1 + d0*y1;
      x1 = x2;
      return y1;
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

    void filterGaussian(const vmav<double,1> &data)
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

template<typename T> T esknew (T v, T beta, T e0)
  {
  auto tmp = (1-v)*(1+v);
  auto tmp2 = tmp>=0;
  return tmp2*exp(beta*(pow(tmp*tmp2, e0)-1));
  }

py::array get_kernel(double beta, double e0, size_t W, size_t npoints)
  {
  auto res_ = make_Pyarr<double>({npoints});
  auto res = to_vmav<double,1>(res_);
  for (size_t i=0; i<npoints; ++i)
    {
    res(i) = esknew((i+0.5)/npoints, W*beta, e0);
    }
  return res_;
  }
py::array get_correction(double beta, double e0, size_t W, size_t npoints, double dx)
  {
  auto res_ = make_Pyarr<double>({npoints});
  auto res = to_vmav<double,1>(res_);
  beta*=W;
  auto lam = [beta,e0](double v){return esknew(v, beta, e0);};
  ducc0::detail_gridding_kernel::GLFullCorrection corr (W, lam);
  auto vec=corr.corfunc(npoints, dx);

  for (size_t i=0; i<npoints; ++i)
    res(i) = vec[i];
  return res_;
  }

template<typename To> void fill_zero(
  To *DUCC0_RESTRICT out, const size_t *szo, const ptrdiff_t *stro,
  size_t idim, size_t ndim)
  {
  const size_t lszo=*szo;
  const ptrdiff_t lstro=*stro;
  if (idim+1==ndim)
    {
    if (lstro==1)
      for (size_t i=0; i<lszo; ++i)
        out[i] = To(0);
    else
      for (size_t i=0; i<lszo; ++i)
        out[i*lstro] = To(0);
    }
  else
    for (size_t i=0; i<lszo; ++i)
      fill_zero(out+i*lstro, szo+1, stro+1, idim+1, ndim);
  }
template<typename Ti, typename To> void roll_resize_roll(
  const Ti *DUCC0_RESTRICT inp, const size_t *szi, const ptrdiff_t *stri,
  To *DUCC0_RESTRICT out, const size_t *szo, const ptrdiff_t *stro,
  const size_t *ri, const size_t *ro, size_t idim, size_t ndim)
  {
  const size_t lszi=*szi, lszo=*szo, lri=*ri, lro=*ro;
  const ptrdiff_t lstri=*stri, lstro=*stro;
  const size_t smin=min(lszi,lszo);
  if (idim+1==ndim)
    {
    size_t io=lro, ii=lszi-lri, i=0;
    while(i<smin)
      {
      size_t minstep = smin-i;
      minstep = min(minstep, lszo-io);
      minstep = min(minstep, lszi-ii);
      if ((lstri==1)&&(lstro==1))
        for(size_t ix=0; ix<minstep; ++ix)
          out[io+ix] = To(inp[ii+ix*lstri]);
      else
        for(size_t ix=0; ix<minstep; ++ix)
          out[(io+ix)*lstro] = To(inp[(ii+ix)*lstri]);
      i+=minstep;
      io+=minstep; if (io==lszo) io=0;
      ii+=minstep; if (ii==lszi) ii=0;
      }
    while(i<lszo)
      {
      size_t minstep = lszo-i;
      minstep = min(minstep, lszo-io);
      if (lstro==1)
        for(size_t ix=0; ix<minstep; ++ix)
          out[io+ix] = To(0);
      else
        for(size_t ix=0; ix<minstep; ++ix)
          out[(io+ix)*lstro] = To(0);
      i+=minstep;
      io+=minstep; if (io==lszo) io=0;
      }
    }
  else
    {
    for (size_t i=0; i<smin; ++i)
      {
      size_t io=min(i+lro, i+lro-lszo);
      size_t ii=min(i-lri, i-lri+lszi);
      roll_resize_roll(inp+ii*lstri, szi+1, stri+1, out+io*lstro, szo+1, stro+1, ri+1, ro+1, idim+1, ndim);
      }
    for (size_t i=smin; i<lszo; ++i)
      {
      size_t io=min(i+lro, i+lro-lszo);
      fill_zero(out+ io*lstro, szo+1, stro+1, idim+1, ndim);
      }
    }
  }
template<typename Ti, typename To> void roll_resize_roll_threaded(
  const Ti *DUCC0_RESTRICT inp, const size_t *szi, const ptrdiff_t *stri,
  To *DUCC0_RESTRICT out, const size_t *szo, const ptrdiff_t *stro,
  const size_t *ri, const size_t *ro, size_t ndim, size_t nthreads)
  {
  size_t smin=min(*szi,*szo);
  execParallel(smin, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=lo; i<hi; ++i)
      {
      size_t io=min(i+*ro, i+*ro-*szo);
      size_t ii=min(i-*ri, i-*ri+*szi);
      roll_resize_roll(inp+ ii* *stri, szi+1, stri+1, out+ io* *stro, szo+1, stro+1, ri+1, ro+1, 1, ndim);
      }
    });
  execParallel(*szo-smin, nthreads, [&](size_t lo, size_t hi)
    {
    for (size_t i=smin+lo; i<smin+hi; ++i)
      {
      size_t io=min(i+*ro, i+*ro-*szo);
      fill_zero(out+ io* *stro, szo+1, stro+1, 1, ndim);
      }
    });
  }

template<typename Ti, typename To> py::array roll_resize_roll(const py::array &inp_,
  py::array &out_, const vector<int64_t> &ri_, const vector<int64_t> &ro_, size_t nthreads)
  {
  auto inp(to_cfmav<Ti>(inp_));
  auto out(to_vfmav<To>(out_));
  {
  py::gil_scoped_release release;
  size_t ndim = inp.ndim();
  nthreads = adjust_nthreads(nthreads);
  MR_assert(out.ndim()==ndim, "dimensionality mismatch");
  MR_assert(ri_.size()==ndim, "dimensionality mismatch");
  MR_assert(ro_.size()==ndim, "dimensionality mismatch");
  vector<size_t> ri, ro;
  for (size_t i=0; i<ndim; ++i)
    {
    ptrdiff_t tmp = ri_[i]%ptrdiff_t(inp.shape(i));
    ri.push_back(size_t((tmp<0) ? tmp+inp.shape(i) : tmp));
    tmp = ro_[i]%ptrdiff_t(out.shape(i));
    ro.push_back(size_t((tmp<0) ? tmp+out.shape(i) : tmp));
    }
  if ((ndim>1)&&(nthreads>1))
    roll_resize_roll_threaded(inp.data(), inp.shape().data(), inp.stride().data(),
      out.data(), out.shape().data(), out.stride().data(),
      ri.data(), ro.data(), ndim, nthreads);
  else
    roll_resize_roll(inp.data(), inp.shape().data(), inp.stride().data(),
      out.data(), out.shape().data(), out.stride().data(),
      ri.data(), ro.data(), 0, ndim);
  }
  return out_;
  }

py::array Py_roll_resize_roll(const py::array &inp,
  py::array &out, const vector<int64_t> &ri, const vector<int64_t> &ro,
  size_t nthreads=1)
  {
  if (isPyarr<float>(inp))
    return roll_resize_roll<float,float>(inp, out, ri, ro, nthreads);
  else if (isPyarr<double>(out))
    return roll_resize_roll<double,double>(inp, out, ri, ro, nthreads);
  else if (isPyarr<complex<float>>(inp))
    return roll_resize_roll<complex<float>,complex<float>>(inp, out, ri, ro, nthreads);
  else if (isPyarr<complex<double>>(out))
    return roll_resize_roll<complex<double>,complex<double>>(inp, out, ri, ro, nthreads);
  else
    MR_fail("type matching failed");
  }

constexpr const char *Py_roll_resize_roll_DS = R"""(
Performs operations equivalent to

tmp = np.roll(inp, roll_inp, axis=tuple(range(inp.ndim)))
tmp2 = np.zeros(out.shape, dtype=inp.dtype)
slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(inp.shape, out.shape))
tmp2[slices] = tmp[slices]
out[()] = np.roll(tmp2, roll_out, axis=tuple(range(out.ndim)))
return out

Parameters
----------
inp : numpy.ndarray(any shape, dtype=float or complex)
    input array
out : numpy.ndarray(any shape, same dimensionality and dtype as `in`)
    output array
roll_inp : tuple(int), length=inp.ndim
    amount of rolling for the input array 
roll_out : tuple(int), length=out.ndim
    amount of rolling for the output array 
nthreads : int
    Number of threads to use. If 0, use the system default (typically the number
    of hardware threads on the compute node).

Returns
-------
numpy.ndarray : identical to out

Notes
-----
`inp` and `out` must not overlap in memory.
)""";

constexpr const char *Py_get_deflected_angles_DS = R"""(
Obtains new pointing angles on the sky according to a deflection field on a set of isolatitude rings

Parameters
----------
theta : numpy.ndarray((nrings,), dtype=numpy.float64)
    colatitudes of the rings  (nrings any number)
phi0 : numpy.ndarray((nrings,), dtype=numpy.float64)
    longitude of the first pixel in each ring
ringstart : numpy.ndarray((nrings,), dtype=numpy.uint64)
    index of the first pixel of each ring in output map
deflect : numpy.ndarray((npix, 2), dtype=numpy.float32 or numpy.float64) 
    Spin-1 deflection field, with real and imaginary comp in first and second entry
    (typically, the output of a spin-1 alm2map_spin transform)
    The array layout and npix must be consistent with the given geometry
calc_rotation(optional) : boolean
    If set, also returns the phase correction (gamma in astro-ph/0502469v3)
nthreads(optional): int
    Number of threads to use. Defaults to 1
res(optional): numpy.ndarray((npix, 3 if calc_rotation is set or 2), same dtype as deflect)
    output array, containing new co-latitudes, new longitudes, and rotation gammma if required
dphi(optional): numpy.ndarray((nrings,), dtype=numpy.float64)
    azimuthal distance between pixels in each ring (in radians)

Returns
-------
numpy.ndarray : identical to res

)""";

cmav<double,1> get_dphi_default(const cmav<size_t,1> &nphi)
  {
  vmav<double,1> res(nphi.shape());
  mav_apply([](auto np, auto &dp){dp=2.*pi/np;}, 1, nphi, res);
  return res;
  }

template<typename Tout> py::array Py2_get_deflected_angles(const py::array &theta_,
  const py::array &phi0_, const py::array &nphi_, const py::array &ringstart_,
  const py::array &deflect_, bool calc_rotation, py::object &res__,
  size_t nthreads, const py::object &dphi_)
  {
  auto theta=to_cmav<double,1>(theta_);
  auto phi0=to_cmav<double,1>(phi0_);
  auto nphi=to_cmav<size_t,1>(nphi_);
  auto ringstart=to_cmav<size_t,1>(ringstart_);
  auto deflect=to_cmav<Tout,2>(deflect_);
  auto dphi = dphi_.is(None) ? get_dphi_default(nphi) : to_cmav<double,1>(dphi_);
  size_t nrings = theta.shape(0);
  MR_assert(phi0.shape(0)==nrings, "nrings mismatch");
  MR_assert(nphi.shape(0)==nrings, "nrings mismatch");
  MR_assert(dphi.shape(0)==nrings, "nrings mismatch");
  MR_assert(ringstart.shape(0)==nrings, "nrings mismatch");
  MR_assert(deflect.shape(1)==2, "second dimension of deflect must be 2");
  size_t ncomp = calc_rotation ? 3 : 2;
  auto res_ = get_optional_Pyarr<Tout>(res__, {deflect.shape(0), ncomp});
  auto res = to_vmav<Tout,2>(res_);
  {
  py::gil_scoped_release release;
  execDynamic(nrings, nthreads, 10, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext())
      for (size_t iring=rng.lo; iring<rng.hi; ++iring)
        {
        vec3 e_r(sin(theta(iring)), 0, cos(theta(iring))); 
        for (size_t iphi=0; iphi<nphi(iring); ++iphi)
          {
          double phi = phi0(iring) + iphi*dphi(iring);
          size_t i = ringstart(iring)+iphi;
          double a_theta = deflect(i,0),
                 a_phi = deflect(i,1);
          double d = a_theta*a_theta + a_phi*a_phi;
          double sin_aoa, twohav_aod, cos_a;
          if (d < 0.0025) // largely covers all CMB-lensing relevant cases to double precision
            {
            sin_aoa = 1. - d/6. * (1. - d/20. * (1. - d/42.));         // sin(a) / a
            twohav_aod = -0.5 + d/24. * (1. - d/30. * (1. - d/56.));   // (cos a - 1) / (a* a) (also needed for rotation)      
            cos_a = 1. + d * twohav_aod;                               // cos(a)
            }
          else
            {
            double a = sqrt(d);
            sin_aoa = sin(a)/a;
            cos_a = cos(a);
            twohav_aod = (cos_a -1.) / d;
            }
          vec3 e_a(e_r.z * a_theta, a_phi, -e_r.x * a_theta); 
          pointing n_prime(e_r*cos_a + e_a*sin_aoa);
          double phinew = n_prime.phi+phi;
          phinew = (phinew>=2*pi) ? (phinew-2*pi) : phinew;
          res(i,0) = Tout(n_prime.theta);
          res(i,1) = Tout(phinew);
          if (calc_rotation)
            { 
            if (d > 0.)
              {
              double temp = e_r.x * a_theta * twohav_aod + e_r.z * sin_aoa;
              res(i, 2) = Tout(atan2(a_phi * temp, e_r.x + a_theta * temp));
              } 
            else
              res(i, 2) = Tout(0);
            }
          }
        }
    });
  }
  return res_;
  }
py::array Py_get_deflected_angles(const py::array &theta_,
  const py::array &phi0_, const py::array &nphi_, const py::array &ringstart_,
  const py::array &deflect_, bool calc_rotation, py::object &res__,
  size_t nthreads, const py::object &dphi_)
  {
  if (isPyarr<float>(deflect_))
    return Py2_get_deflected_angles<float>(theta_, phi0_, nphi_, ringstart_,
      deflect_, calc_rotation, res__, nthreads, dphi_);
  else if (isPyarr<double>(deflect_))
    return Py2_get_deflected_angles<double>(theta_, phi0_, nphi_, ringstart_,
      deflect_, calc_rotation, res__, nthreads, dphi_);
  MR_fail("type matching failed: 'deflect' has neither type 'f4' nor 'f8'");
  }

template<typename T> void Py2_lensing_rotate(py::array &values_,
  const py::array &gamma_, int spin, size_t nthreads)
  {
  auto values = to_vfmav<complex<T>>(values_);
  auto gamma = to_cfmav<T>(gamma_);
  {
  py::gil_scoped_release release;
  mav_apply([&](auto &v, const auto &g) { v*=polar(T(1), spin*g); }, nthreads, values, gamma);
  }
  }
void Py_lensing_rotate(py::array &values,
  const py::array &gamma, int spin, size_t nthreads=1)
  {
  if (isPyarr<complex<float>>(values))
    return Py2_lensing_rotate<float>(values, gamma, spin, nthreads);
  else if (isPyarr<complex<double>>(values))
    return Py2_lensing_rotate<double>(values, gamma, spin, nthreads);
  MR_fail("type matching failed: 'values' has neither type 'c8' nor 'c16'");
  }

constexpr const char *Py_lensing_rotate_DS = R"""(
Rotates complex values depending on given angles and spin

Parameters
----------
values : numpy.ndarray(<any shape>, dtype=complex)
    values to rotate; operation is applied in place
gamma : numpy.ndarray(same shape as `values`, dtype=float with same prcision as `values`)
    rotation angles
spin : int
    will be multiplied to `gamma` before rotation
nthreads(optional): int
    Number of threads to use. Defaults to 1
)""";


void coupling_matrix_spin0_nontmpl(const cmav<double,2> &spec,
  size_t lmax, const vmav<double,3> &mat, size_t nthreads)
  {
  size_t nspec=spec.shape(0);
  MR_assert(spec.shape(1)>=1, "spec.shape[1] is too small.");
  auto lmax_spec = spec.shape(1)-1;
  using Tsimd = native_simd<double>;
  constexpr size_t vlen = Tsimd::size();
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,2>::build_noncritical({nspec, lmax_spec_used+1+vlen-1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = spec(i,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(1); ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = 0.;
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    vmav<Tsimd,1> resfullv({lmax+1});
    vmav<Tsimd,1> val_({nspec});
    Tsimd * DUCC0_RESTRICT val = val_.data();
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=k;
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int el2=el1; el2<=int(lmax); el2+=vlen)
        {
        int el3min = el2-el1;
        if (el3min<=int(lmax_spec))
          {
          wigner3j_00_vec_squared_compact(Tsimd(el1), Tsimd(el2)+lofs,
            subarray<1>(resfullv, {{size_t(0), size_t(el1+1)}}));
          const Tsimd * DUCC0_RESTRICT res = resfullv.data();

          for (size_t ispec=0; ispec<nspec; ++ispec)
            val[ispec]=0;
          int max_i = min(el1+el2, int(lmax_spec)) - el3min;
          for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
            {
            int el3 = el3min+i;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
            }
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2+k<=lmax)
                {
                mat(ispec, el1, el2+k) = (2*(el2+k)+1.)*val[ispec][k];
                mat(ispec, el2+k, el1) = (2*el1+1.)*val[ispec][k];
                }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2+k<=lmax)
                mat(ispec, el1, el2+k) = mat(ispec, el2+k, el1) = 0.;
        }
      }
    });
  }
template<size_t nspec> void coupling_matrix_spin0_tmpl(const cmav<double,2> &spec,
  size_t lmax, const vmav<double,3> &mat, size_t nthreads)
  {
  MR_assert(nspec==spec.shape(0), "bad invocation");
  MR_assert(spec.shape(1)>=1, "spec.shape[1] is too small.");
  auto lmax_spec = spec.shape(1)-1;
  using Tsimd = native_simd<double>;
  constexpr size_t vlen = Tsimd::size();
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,2>::build_noncritical({nspec, lmax_spec_used+1+vlen-1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = spec(i,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(1); ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = 0.;
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    vmav<Tsimd,1> resfullv({lmax+1});
    array<Tsimd,nspec> val;
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=k;
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int el2=el1; el2<=int(lmax); el2+=vlen)
        {
        int el3min = el2-el1;
        if (el3min<=int(lmax_spec))
          {
          wigner3j_00_vec_squared_compact(Tsimd(el1), Tsimd(el2)+lofs,
            subarray<1>(resfullv, {{size_t(0), size_t(el1+1)}}));
          const Tsimd * DUCC0_RESTRICT res = resfullv.data();

          for (size_t ispec=0; ispec<nspec; ++ispec)
            val[ispec]=0;
          int max_i = min(el1+el2, int(lmax_spec)) - el3min;
          for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
            {
            int el3 = el3min+i;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
            }
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2+k<=lmax)
                {
                mat(ispec, el1, el2+k) = (2*(el2+k)+1.)*val[ispec][k];
                mat(ispec, el2+k, el1) = (2*el1+1.)*val[ispec][k];
                }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2+k<=lmax)
                mat(ispec, el1, el2+k) = mat(ispec, el2+k, el1) = 0.;
        }
      }
    });
  }

py::array Py_coupling_matrix_spin0(const py::array &spec_, size_t lmax, size_t nthreads, py::object &mat__)
  {
  auto spec = to_cmav<double,2>(spec_);
  auto nspec = spec.shape(0);
  MR_assert(spec.shape(1)>=1, "spec.shape[1] is too small.");
  auto mat_ = get_optional_Pyarr<double>(mat__, {nspec, lmax+1, lmax+1});
  auto mat = to_vmav<double,3>(mat_);
  {
  py::gil_scoped_release release;

  if (nspec==1)
    coupling_matrix_spin0_tmpl<1>(spec, lmax, mat, nthreads);
  else if (nspec==2)
    coupling_matrix_spin0_tmpl<2>(spec, lmax, mat, nthreads);
  else if (nspec==3)
    coupling_matrix_spin0_tmpl<3>(spec, lmax, mat, nthreads);
  else if (nspec==4)
    coupling_matrix_spin0_tmpl<4>(spec, lmax, mat, nthreads);
  else
    coupling_matrix_spin0_nontmpl(spec, lmax, mat, nthreads);
  }
  return mat_;
  }

constexpr const char *Py_coupling_matrix_spin0_DS = R"""(
This is very similar to pspy's calc_coupling_spin0() method, with the following
differences:
- the l values in the output matrix go from 0 to lmax (inclusive) instead of
  2 to lmax (exclusive)
- the input power spectra are multiplied by (2*l+1)
- the output is multiplied by (2*l1+2)/(4*pi) and the full matrix is populated
- the computation can be carried out for more than one power spectrum at the
  same time.

Parameters
----------
spec : numpy.ndarray((nspec, lmax_spec+1), dtype=np.float64)
    the input spectra
lmax : int
    the maximum l moment included in the output matrices
    In principle, this requires the input spectra to be provided with an
    `lmax_spec = 2*lmax`. If `lmax_spec` is smaller, the missing values are
    assumed to be zero.
nthreads : int
    the number of threads to use for the calculations.
res : numpy.ndarray((nspec, lmax+1, lmax+1), dtype=np.float64)
    Optional array to store the output into.

Returns
-------
numpy.ndarray((nspec, lmax+1, lmax+1), dtype=np.float64)
    The coupling matrices. Identical to `res`, if it was provided
)""";

void coupling_matrix_spin0and2_nontmpl(const cmav<double,3> &spec,
  size_t lmax, const vmav<double,4> &mat, size_t nthreads)
  {
  constexpr size_t ncomp_spec=4;
  size_t nspec=spec.shape(0);
  MR_assert(spec.shape(1)==ncomp_spec, "spec.shape[1] must be 4.");
  MR_assert(spec.shape(2)>=1, "lmax_spec is too small.");
  auto lmax_spec = spec.shape(2)-1;
  using Tsimd = native_simd<double>;
  constexpr size_t vlen = Tsimd::size();
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,3>::build_noncritical
    ({nspec, ncomp_spec, lmax_spec_used+1+vlen-1+1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = spec(i,j,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(2); ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = 0.;
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    constexpr size_t ncomp_out=5;
// res arrays are one larger to make loops simpler below
    vmav<Tsimd,2> wig({2, 2*lmax+1+1});
    vmav<array<Tsimd,ncomp_out>,1> val_({nspec});
    array<Tsimd,ncomp_out> * DUCC0_RESTRICT val = val_.data();
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=k;
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int el2=el1; el2<=int(lmax); el2+=vlen)
        {
        int el3min = el2-el1;
        int el3max = el2+el1;
        if (el3min<=int(lmax_spec))
          {
          auto tmp=subarray<2>(wig,{{},{size_t(el3min), size_t(el3max+2)}});
          flexible_wigner3j_vec(Tsimd(el1), Tsimd(el2)+lofs, 0, 0,
            Tsimd(el3min)+lofs, subarray<1>(tmp, {{0}, {}}));
          flexible_wigner3j_vec(Tsimd(el1), Tsimd(el2)+lofs, -2, 2,
            Tsimd(el3min)+lofs, subarray<1>(tmp, {{1}, {}}));

          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<ncomp_out; ++j)
              val[ispec][j]=0;
          int maxidx = min(el3max, int(lmax_spec));
          for (int el3=el3min; el3<=maxidx; el3+=2)
            for (size_t ispec=0; ispec<nspec; ++ispec)
              {
              val[ispec][0] += wig(0,el3)*wig(0,el3)*Tsimd(&spec2(ispec,0,el3), element_aligned_tag());
              val[ispec][1] += wig(0,el3)*wig(1,el3)*Tsimd(&spec2(ispec,1,el3), element_aligned_tag());
              val[ispec][2] += wig(0,el3)*wig(1,el3)*Tsimd(&spec2(ispec,2,el3), element_aligned_tag());
              val[ispec][3] += wig(1,el3)*wig(1,el3)*Tsimd(&spec2(ispec,3,el3), element_aligned_tag());
              val[ispec][4] += wig(1,el3+1)*wig(1,el3+1)*Tsimd(&spec2(ispec,3,el3+1), element_aligned_tag());
              }
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<ncomp_out; ++j)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  {
                  mat(ispec, j, el1, el2+k) = (2*(el2+k)+1.)*val[ispec][j][k];
                  mat(ispec, j, el2+k, el1) = (2*el1+1.)*val[ispec][j][k];
                  }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<ncomp_out; ++j)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, j, el1, el2+k) = mat(ispec, j, el2+k, el1) = 0.;
        }
      }
    });
  }

py::array Py_coupling_matrix_spin0and2(const py::array &spec_, size_t lmax, size_t nthreads, py::object &mat__)
  {
  auto spec = to_cmav<double,3>(spec_);
  auto nspec = spec.shape(0);
  MR_assert(spec.shape(1)==4, "bad ncomp_spec");
  MR_assert(spec.shape(2)>=1, "spec.shape[1] is too small.");
  auto mat_ = get_optional_Pyarr<double>(mat__, {nspec, 5, lmax+1, lmax+1});
  auto mat = to_vmav<double,4>(mat_);
  {
  py::gil_scoped_release release;
  coupling_matrix_spin0and2_nontmpl(spec, lmax, mat, nthreads);
  }
  return mat_;
  }

constexpr const char *Py_coupling_matrix_spin0and2_DS = R"""(
This is very similar to pspy's calc_coupling_spin0and2() method, with the following
differences:
- the l values in the output matrix go from 0 to lmax (inclusive) instead of
  2 to lmax (exclusive)
- the input power spectra are multiplied by (2*l+1)
- the output is multiplied by (2*l1+2)/(4*pi) and the full matrix is populated
- the computation can be carried out for more than one set of power spectra
  at the same time.

Parameters
----------
spec : numpy.ndarray((nspec, 4, lmax_spec+1), dtype=np.float64)
    the input spectra
    the indices of the second dimension correspond to wcl_00, wcl_02, wcl_20,
    and wcl_22 of calc_coupling_spin0and2(), respectively
lmax : int
    the maximum l moment included in the output matrices
    In principle, this requires the input spectra to be provided with an
    `lmax_spec = 2*lmax`. If `lmax_spec` is smaller, the missing values are
    assumed to be zero.
nthreads : int
    the number of threads to use for the calculations.
res : numpy.ndarray((nspec, 5, lmax+1, lmax+1), dtype=np.float64)
    Optional array to store the output into.

Returns
-------
numpy.ndarray((nspec, 5, lmax+1, lmax+1), dtype=np.float64)
    The coupling matrices. Identical to `res`, if it was provided
)""";

void coupling_matrix_spin0and2_pure_nontmpl(const cmav<double,3> &spec,
  size_t lmax, const vmav<double,4> &mat, size_t nthreads)
  {
  using Tsimd = native_simd<double>;
  constexpr size_t vlen=Tsimd::size();
  constexpr size_t ncomp_spec=4;
  constexpr size_t ncomp_out=4;
  size_t nspec=spec.shape(0);
  MR_assert(spec.shape(1)==ncomp_spec, "spec.shape[1] must be 4.");
  MR_assert(spec.shape(2)>=1, "lmax_spec is too small.");
  auto lmax_spec = spec.shape(2)-1;
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,3>::build_noncritical
    ({nspec, ncomp_spec, lmax_spec_used+1+vlen-1+1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = spec(i,j,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(2); ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = 0.;
  vector<double> nom1(2*lmax+1+vlen-1+1), nom2(2*lmax+1+vlen-1+1);
  for (size_t el3=0; el3<nom1.size(); ++el3)
    {
    nom1[el3] = 2.*sqrt((el3+1.)*el3);
    nom2[el3] = sqrt((el3+2.)*(el3+1.)*el3*(el3-1.));
    }
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    // res arrays are one larger to make loops simpler below
    vmav<Tsimd,2> wig({6, 2*lmax+1+1});
    constexpr size_t nvcomp = 7;
    vmav<array<Tsimd,nvcomp>,1> val_({nspec});
    array<Tsimd,nvcomp> * DUCC0_RESTRICT val = val_.data();
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=k;
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int xel2=el1; xel2<=int(lmax); xel2+=vlen)
        {
        Tsimd el2=Tsimd(xel2)+lofs;
        int el3min = abs(xel2-el1);
        int el3max = el1+xel2;
        Tsimd xdenom1 = blend(el2>Tsimd(1.), sqrt(Tsimd(1.) / ((el2-1.)*(el2+2.))), Tsimd(0.)),
              xdenom2 = blend(el2>Tsimd(1.), sqrt(Tsimd(1.) / ((el2+2.)*(el2+1.)*el2*(el2-1.))), Tsimd(0.));
        double xxdenom1 = (el1>1) ? sqrt(1. / ((el1-1.)*(el1+2.))) : 0,
               xxdenom2 = (el1>1) ? sqrt(1. / ((el1+2.)*(el1+1.)*el1*(el1-1.))): 0;
        if (el3min<=int(lmax_spec))
          {
          {
          auto tmp = subarray<2>(wig, {{}, {size_t(el3min), size_t(el3max+2)}});
          constexpr array<int,6> m1 {{0, -2, -2, -2, -2, -2}};
          constexpr array<int,6> m2 {{0,  2,  1,  0,  1,  0}};
          array<Tsimd,6> xl1 {{el1, el1, el1, el1, el2, el2}};
          array<Tsimd,6> xl2 {{el2, el2, el2, el2, el1, el1}};
          for (size_t ii=0; ii<6; ++ii)
            flexible_wigner3j_vec(xl1[ii], xl2[ii], m1[ii], m2[ii],
              Tsimd(el3min)+lofs, subarray<1>(tmp, {{ii}, {}}));
          }

          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<nvcomp; ++j)
              val[ispec][j]=0;
          int maxidx = min(el3max, int(lmax_spec));
          for (int el3=el3min; el3<=maxidx; el3+=2)
            {
            Tsimd fac_b = Tsimd(&nom1[el3],element_aligned_tag())*xdenom1,
                  fac_c = Tsimd(&nom2[el3],element_aligned_tag())*xdenom2,
                  xfac_b = Tsimd(&nom1[el3],element_aligned_tag())*xxdenom1,
                  xfac_c = Tsimd(&nom2[el3],element_aligned_tag())*xxdenom2;
//                  fac_b2 = Tsimd(&nom1[el3+1],element_aligned_tag())*xdenom1,
//                  fac_c2 = Tsimd(&nom2[el3+1],element_aligned_tag())*xdenom2,
//                  xfac_b2 = Tsimd(&nom1[el3+1],element_aligned_tag())*xxdenom1,
//                  xfac_c2 = Tsimd(&nom2[el3+1],element_aligned_tag())*xxdenom2;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              {
              const Tsimd s0(&spec2(ispec,0,el3), element_aligned_tag()),
                          s1(&spec2(ispec,1,el3), element_aligned_tag()),
                          s2(&spec2(ispec,2,el3), element_aligned_tag()),
                          s3(&spec2(ispec,3,el3), element_aligned_tag());
              val[ispec][0] += wig(0,el3)*wig(0,el3)*s0;
              auto combin = wig(1,el3) + fac_b*wig(2,el3) + fac_c*wig(3,el3);
              val[ispec][1] += wig(0,el3)*combin*s1;
              val[ispec][2] += wig(0,el3)*combin*s2;
              val[ispec][3] += combin*combin*Tsimd(&spec2(ispec,3,el3), element_aligned_tag());
              auto xcombin = wig(1,el3) + xfac_b*wig(4,el3) + xfac_c*wig(5,el3);
              val[ispec][4] += wig(0,el3)*xcombin*s1;
              val[ispec][5] += wig(0,el3)*xcombin*s2;
              val[ispec][6] += xcombin*xcombin*s3;
//              auto combin2 = wig(1,el3+1) + fac_b2*wig(2,el3+1) + fac_c2*wig(3,el3+1);
//              val[ispec][7] += combin2*combin2*Tsimd(&spec2(ispec,3,el3+1), element_aligned_tag());
//              auto  xcombin2 = wig(1,el3+1) + xfac_b2*wig(4,el3+1) + xfac_c2*wig(5,el3+1);
//              val[ispec][8] += xcombin2*xcombin2*Tsimd(&spec2(ispec,3,el3+1), element_aligned_tag());
              }
            }
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2[k]<=lmax)
                {
                mat(ispec, 0, xel2+k, el1) = (2*el1+1.)*val[ispec][0][k];
                mat(ispec, 1, xel2+k, el1) = (2*el1+1.)*val[ispec][1][k];
                mat(ispec, 2, xel2+k, el1) = (2*el1+1.)*val[ispec][2][k];
                mat(ispec, 3, xel2+k, el1) = (2*el1+1.)*val[ispec][3][k];
                mat(ispec, 0, el1, xel2+k) = (2*el2[k]+1.)*val[ispec][0][k];
                mat(ispec, 1, el1, xel2+k) = (2*el2[k]+1.)*val[ispec][4][k];
                mat(ispec, 2, el1, xel2+k) = (2*el2[k]+1.)*val[ispec][5][k];
                mat(ispec, 3, el1, xel2+k) = (2*el2[k]+1.)*val[ispec][6][k];
//                mat(ispec, 4, xel2+k, el1) = (2*el1+1.)*val[ispec][4][k];
//                mat(ispec, 4, el1, xel2+k) = (2*el2[k]+1.)*val[ispec][8][k];
                }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<ncomp_out; ++j)
              for (size_t k=0; k<vlen; ++k)
                if (el2[k]<=lmax)
                  mat(ispec, j, xel2+k, el1) = mat(ispec, j, el1, xel2+k) = 0.;
        }
      }
    });
  }

py::array Py_coupling_matrix_spin0and2_pure(const py::array &spec_, size_t lmax, size_t nthreads, py::object &mat__)
  {
  auto spec = to_cmav<double,3>(spec_);
  auto nspec = spec.shape(0);
  MR_assert(spec.shape(1)==4, "bad ncomp_spec");
  MR_assert(spec.shape(2)>=1, "spec.shape[1] is too small.");
  auto mat_ = get_optional_Pyarr<double>(mat__, {nspec, 5, lmax+1, lmax+1});
  auto mat = to_vmav<double,4>(mat_);
  {
  py::gil_scoped_release release;
  coupling_matrix_spin0and2_pure_nontmpl(spec, lmax, mat, nthreads);
  }
  return mat_;
  }

constexpr const char *Py_coupling_matrix_spin0and2_pure_DS = R"""(
This is very similar to pspy's calc_mcm_spin0and2_pure() method, with the following
differences:
- the l values in the output matrix go from 0 to lmax (inclusive) instead of
  2 to lmax (exclusive)
- the input power spectra are multiplied by (2*l+1)
- the output is multiplied by (2*l1+2)/(4*pi) and the full matrix is populated
- the computation can be carried out for more than one set of power spectra
  at the same time.
- the last component is not calculated, since it appears to be zero in all cases

Parameters
----------
spec : numpy.ndarray((nspec, 4, lmax_spec+1), dtype=np.float64)
    the input spectra
    the indices of the second dimension correspond to wcl_00, wcl_02, wcl_20,
    and wcl_22 of calc_mcm_spin0and2_pure(), respectively
lmax : int
    the maximum l moment included in the output matrices
    In principle, this requires the input spectra to be provided with an
    `lmax_spec = 2*lmax`. If `lmax_spec` is smaller, the missing values are
    assumed to be zero.
nthreads : int
    the number of threads to use for the calculations.
res : numpy.ndarray((nspec, 4, lmax+1, lmax+1), dtype=np.float64)
    Optional array to store the output into.

Returns
-------
numpy.ndarray((nspec, 4, lmax+1, lmax+1), dtype=np.float64)
    The coupling matrices. Identical to `res`, if it was provided
)""";

py::object Py_wigner3j_int(int l2, int l3, int m2, int m3)
  {
  size_t ncoef = wigner3j_ncoef_int(l2, l3, m2, m3);
  auto res_ = make_Pyarr<double>({ncoef});
  auto res = to_vmav<double,1>(res_);
  int l1min;
  wigner3j_int (l2, l3, m2, m3, l1min, res);
  return py::make_tuple(py::cast(l1min), res_);
  }

constexpr const char *Py_wigner3j_int_DS = R"""(
Computes Wigner 3j symbols according to the algorithm of
Schulten & Gordon: J. Math. Phys. 16, p. 10 (1975)

This special case only takes integer quantum numbers.

Parameters
----------
l2, l3, m2, m3 : integer
    fixed quantum numbers

Returns
-------
int : the l1 quantum number of the first value in the returned array
numpy.ndarray(dtype=numpy.float64) : 3j symbols in order of increasing l1
)""";

constexpr const char *misc_DS = R"""(
Various unsorted utilities

Notes
-----

The functionality in this module is not considered to have a stable interface
and also may be moved to other modules in the future. If you use it, be prepared
to adjust your code at some point in the future!
)""";

void add_misc(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("misc");
  m.doc() = misc_DS;

  auto m2 = m.def_submodule("experimental");
//  m2.doc() = sht_experimental_DS;

  m.def("vdot", Py_vdot, Py_vdot_DS, "a"_a, "b"_a);
  m.def("l2error",  Py_l2error, Py_l2error_DS, "a"_a, "b"_a);

  m.def("GL_weights", Py_GL_weights, "nlat"_a, "nlon"_a);
  m.def("GL_thetas", Py_GL_thetas, "nlat"_a);

  m.def("transpose", Py_transpose, "in"_a, "out"_a, "nthreads"_a=1);

  m.def("make_noncritical", Py_make_noncritical, Py_make_noncritical_DS,"in"_a);
  m.def("empty_noncritical", Py_empty_noncritical, Py_empty_noncritical_DS, "shape"_a, "dtype"_a);

  py::class_<Py_OofaNoise> (m, "OofaNoise", Py_OofaNoise_DS, py::module_local())
    .def(py::init<double, double, double, double, double>(), Py_OofaNoise_init_DS,
      "sigmawhite"_a, "f_knee"_a, "f_min"_a, "f_samp"_a, "slope"_a)
    .def ("filterGaussian", &Py_OofaNoise::filterGaussian,
      Py_OofaNoise_filterGaussian_DS, "rnd"_a);

  m.def("get_kernel", get_kernel,"beta"_a, "e0"_a, "W"_a, "npoints"_a);
  m.def("get_correction", get_correction,"beta"_a, "e0"_a, "W"_a, "npoints"_a, "dx"_a);

  m.def("roll_resize_roll", Py_roll_resize_roll, Py_roll_resize_roll_DS,
    "inp"_a, "out"_a, "roll_inp"_a, "roll_out"_a, "nthreads"_a=1);

  m.def("get_deflected_angles", Py_get_deflected_angles, Py_get_deflected_angles_DS,
    "theta"_a, "phi0"_a, "nphi"_a, "ringstart"_a, "deflect"_a,
    "calc_rotation"_a=false, "res"_a=py::none(), "nthreads"_a=1, "dphi"_a=None);
  m.def("lensing_rotate", Py_lensing_rotate, Py_lensing_rotate_DS,
    "values"_a, "gamma"_a, "spin"_a, "nthreads"_a=1);

  m.def("wigner3j_int", Py_wigner3j_int, Py_wigner3j_int_DS, "l2"_a, "l3"_a, "m2"_a, "m3"_a);

  m2.def("coupling_matrix_spin0", Py_coupling_matrix_spin0, Py_coupling_matrix_spin0_DS,
    "spec"_a, "lmax"_a, "nthreads"_a=1, "res"_a=None);
  m2.def("coupling_matrix_spin0and2", Py_coupling_matrix_spin0and2, Py_coupling_matrix_spin0and2_DS,
    "spec"_a, "lmax"_a, "nthreads"_a=1, "res"_a=None);
  m2.def("coupling_matrix_spin0and2_pure", Py_coupling_matrix_spin0and2_pure, Py_coupling_matrix_spin0and2_pure_DS,
    "spec"_a, "lmax"_a, "nthreads"_a=1, "res"_a=None);

  m.def("preallocate_memory", preallocate_memory, "gbytes"_a);
  }

}

using detail_pymodule_misc::add_misc;

}

