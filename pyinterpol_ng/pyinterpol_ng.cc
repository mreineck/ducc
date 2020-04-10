/*
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "interpol_ng.h"

using namespace std;
using namespace mr;

namespace py = pybind11;

namespace {

template<typename T> class PyInterpolator: public Interpolator<T>
  {
  protected:
    using Interpolator<T>::lmax;
    using Interpolator<T>::kmax;
    using Interpolator<T>::interpol;
    using Interpolator<T>::deinterpol;
    using Interpolator<T>::getSlm;

vector<Alm<complex<T>>> makevec(const py::array &inp, int64_t lmax, int64_t kmax, bool rw=false)
  {
  auto inp2 = to_mav<complex<T>,2>(inp, rw);
  vector<Alm<complex<T>>> res;
  for (size_t i=0; i<inp2.shape(1); ++i)
    res.push_back(Alm<complex<T>>(inp2.template subarray<1>({0,i},{inp2.shape(0),0}),lmax, kmax));
  return res;
  }
  public:
    PyInterpolator(const py::array &slm, const py::array &blm,
      int64_t lmax, int64_t kmax, double epsilon, int nthreads=0)
      : Interpolator<T>(makevec(slm, lmax, lmax),
                        makevec(blm, lmax, kmax),
                        false, epsilon, nthreads) {}
    PyInterpolator(int64_t lmax, int64_t kmax, double epsilon, int nthreads=0)
      : Interpolator<T>(lmax, kmax, 1, epsilon, nthreads) {}
    py::array pyinterpol(const py::array &ptg) const
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto res = make_Pyarr<double>({ptg2.shape(0),1});
      auto res2 = to_mav<double,2>(res,true);
      interpol(ptg2, res2);
      return res;
      }

    void pydeinterpol(const py::array &ptg, const py::array &data)
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto data2 = to_mav<T,2>(data);
      deinterpol(ptg2, data2);
      }
    py::array pygetSlm(const py::array &blm_)
      {
      auto blm = makevec(blm_, lmax, kmax);
      auto res = make_Pyarr<complex<T>>({Alm_Base::Num_Alms(lmax, lmax),blm.size()});
      auto slm = makevec(res, lmax, lmax, true);
      getSlm(blm, slm);
      return res;
      }
  };

#if 1
template<typename T> py::array pyrotate_alm(const py::array &alm_, int64_t lmax,
  double psi, double theta, double phi)
  {
  auto a1 = to_mav<complex<T>,1>(alm_);
  auto alm = make_Pyarr<complex<T>>({a1.shape(0)});
  auto a2 = to_mav<complex<T>,1>(alm,true);
  for (size_t i=0; i<a1.shape(0); ++i) a2.v(i)=a1(i);
  auto tmp = Alm<complex<T>>(a2,lmax,lmax);
  rotate_alm(tmp, psi, theta, phi);
  return alm;
  }
#endif

constexpr const char *pyinterpol_ng_DS = R"""(
Python interface for total convolution/interpolation library

All arrays containing spherical harmonic coefficients are assumed to have the
following format:
- values for m=0, l going from 0 to lmax
  (these values must have an imaginary part of zero)
- values for m=1, l going from 1 to lmax
  (these values can be fully complex)
- values for m=2, l going from 2 to lmax
- ...
- values for m=mmax, l going from mmax to lmax 

Error conditions are reported by raising exceptions.
)""";

constexpr const char *pyinterpolator_DS = R"""(
Class encapsulating the convolution/interpolation functionality

The class can be configured for interpolation or for adjoint interpolation, by
means of two different constructors.
)""";

constexpr const char *initnormal_DS = R"""(
Constructor for interpolation mode

Parameters
----------
sky : numpy.ndarray(numpy.complex)
    spherical harmonic coefficients of the sky
beam : numpy.ndarray(numpy.complex)
    spherical harmonic coefficients of the sky
lmax : int
    maximum l in the coefficient arays
mmax : int
    maximum m in the sky coefficients
kmax : int
    maximum azimuthal moment in the beam coefficients
epsilon : float
    desired accuracy for the interpolation; a typical value is 1e-5
nthreads : the number of threads to use for computation
)""";

constexpr const char *initadjoint_DS = R"""(
Constructor for adjoint interpolation mode

Parameters
----------
lmax : int
    maximum l in the coefficient arays
mmax : int
    maximum m in the sky coefficients
kmax : int
    maximum azimuthal moment in the beam coefficients
epsilon : float
    desired accuracy for the interpolation; a typical value is 1e-5
nthreads : the number of threads to use for computation
)""";

constexpr const char *interpol_DS = R"""(
Computes the interpolated values for a given set of angle triplets

Parameters
----------
ptg : numpy.ndarray(numpy.float64) of shape(N,3)
    theta, phi and psi angles (in radian) for N pointings
    theta must be in the range [0; pi]
    phi must be in the range [0; 2pi]
    psi should be in the range [-2pi; 2pi]

Returns
-------
numpy.array(numpy.float64) of shape (N,)
    the interpolated values

Notes
-----
    - Can only be called in "normal" (i.e. not adjoint) mode
    - repeated calls to this method are fine, but for good performance the
      number of pointings passed per call should be as large as possible.
)""";

constexpr const char *deinterpol_DS = R"""(
Takes a set of angle triplets and interpolated values and spreads them onto the
data cube.

Parameters
----------
ptg : numpy.ndarray(numpy.float64) of shape(N,3)
    theta, phi and psi angles (in radian) for N pointings
    theta must be in the range [0; pi]
    phi must be in the range [0; 2pi]
    psi should be in the range [-2pi; 2pi]
data : numpy.ndarray(numpy.float64) of shape (N,)
    the interpolated values

Notes
-----
    - Can only be called in adjoint mode
    - repeated calls to this method are fine, but for good performance the
      number of pointings passed per call should be as large as possible.
)""";

constexpr const char *getSlm_DS = R"""(
Returns a set of sky spherical hamonic coefficients resulting from adjoint
interplation

Parameters
----------
blmT : numpy.array(numpy.complex)
    spherical harmonic coefficients of the beam with lmax and kmax defined
    in the constructor call

Returns
-------
numpy.array(numpy.complex):
    spherical harmonic coefficients of the sky with lmax and mmax defined
    in the constructor call

Notes
-----
    - Can only be called in adjoint mode
    - must be the last call to the object
)""";

} // unnamed namespace

PYBIND11_MODULE(pyinterpol_ng, m)
  {
  using namespace pybind11::literals;

  m.doc() = pyinterpol_ng_DS;

  py::class_<PyInterpolator<double>> (m, "PyInterpolator", pyinterpolator_DS)
    .def(py::init<const py::array &, const py::array &, int64_t, int64_t, double, int>(),
      initnormal_DS, "sky"_a, "beam"_a, "lmax"_a, "kmax"_a, "epsilon"_a,
      "nthreads"_a)
    .def(py::init<int64_t, int64_t, double, int>(), initadjoint_DS,
      "lmax"_a, "kmax"_a, "epsilon"_a, "nthreads"_a)
    .def ("interpol", &PyInterpolator<double>::pyinterpol, interpol_DS, "ptg"_a)
    .def ("deinterpol", &PyInterpolator<double>::pydeinterpol, deinterpol_DS, "ptg"_a, "data"_a)
    .def ("getSlm", &PyInterpolator<double>::pygetSlm, getSlm_DS, "blmT"_a);
#if 1
  m.def("rotate_alm", &pyrotate_alm<double>, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a);
#endif
  }
