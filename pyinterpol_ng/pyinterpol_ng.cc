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

using fptype = double;

template<typename T> class PyInterpolator: public Interpolator<T>
  {
  protected:
    using Interpolator<T>::lmax;
    using Interpolator<T>::kmax;
    using Interpolator<T>::ncomp;
    using Interpolator<T>::interpol;
    using Interpolator<T>::deinterpol;
    using Interpolator<T>::getSlm;

vector<Alm<complex<T>>> makevec(const py::array &inp, int64_t lmax, int64_t kmax)
  {
  auto inp2 = to_mav<complex<T>,2>(inp);
  vector<Alm<complex<T>>> res;
  for (size_t i=0; i<inp2.shape(1); ++i)
    res.push_back(Alm<complex<T>>(inp2.template subarray<1>({0,i},{inp2.shape(0),0}),lmax, kmax));
  return res;
  }
void makevec_v(py::array &inp, int64_t lmax, int64_t kmax, vector<Alm<complex<T>>> &res)
  {
  auto inp2 = to_mav<complex<T>,2>(inp, true);
  for (size_t i=0; i<inp2.shape(1); ++i)
    {
    auto xtmp = inp2.template subarray<1>({0,i},{inp2.shape(0),0});
    res.emplace_back(xtmp, lmax, kmax);
    }
  }
  public:
    PyInterpolator(const py::array &slm, const py::array &blm,
      bool separate, int64_t lmax, int64_t kmax, T epsilon, int nthreads=0)
      : Interpolator<T>(makevec(slm, lmax, lmax),
                        makevec(blm, lmax, kmax),
                        separate, epsilon, nthreads) {}
    PyInterpolator(int64_t lmax, int64_t kmax, int64_t ncomp_, T epsilon, int nthreads=0)
      : Interpolator<T>(lmax, kmax, ncomp_, epsilon, nthreads) {}
    py::array pyinterpol(const py::array &ptg) const
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto res = make_Pyarr<T>({ptg2.shape(0),ncomp});
      auto res2 = to_mav<T,2>(res,true);
      interpol(ptg2, res2);
      return move(res);
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
      vector<Alm<complex<T>>> slm;
      makevec_v(res, lmax, lmax, slm);
      getSlm(blm, slm);
      return move(res);
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
  return move(alm);
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
sky : numpy.ndarray((nalm_sky, ncomp), dtype=numpy.complex)
    spherical harmonic coefficients of the sky. ncomp can be 1 or 3.
beam : numpy.ndarray((nalm_beam, ncomp), dtype=numpy.complex)
    spherical harmonic coefficients of the beam. ncomp can be 1 or 3
separate : bool
    whether contributions of individual components should be added together.
lmax : int
    maximum l in the coefficient arays
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
kmax : int
    maximum azimuthal moment in the beam coefficients
ncomp : int
    the number of components which are going to input to `deinterpol`.
    Can be 1 or 3.
epsilon : float
    desired accuracy for the interpolation; a typical value is 1e-5
nthreads : the number of threads to use for computation
)""";

constexpr const char *interpol_DS = R"""(
Computes the interpolated values for a given set of angle triplets

Parameters
----------
ptg : numpy.ndarray((N, 3), dtype=numpy.float64)
    theta, phi and psi angles (in radian) for N pointings
    theta must be in the range [0; pi]
    phi must be in the range [0; 2pi]
    psi should be in the range [-2pi; 2pi]

Returns
-------
numpy.array((N, n2), dtype=numpy.float64)
    the interpolated values
    n2 is either 1 (if separate=True was used in the constructor) or the
    second dimension of the input slm and blm arrays (otherwise)

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
ptg : numpy.ndarray((N,3), dtype=numpy.float64)
    theta, phi and psi angles (in radian) for N pointings
    theta must be in the range [0; pi]
    phi must be in the range [0; 2pi]
    psi should be in the range [-2pi; 2pi]
data : numpy.ndarray((N, n2), dtype=numpy.float64)
    the interpolated values
    n2 must match the `ncomp` value specified in the constructor.

Notes
-----
    - Can only be called in adjoint mode
    - repeated calls to this method are fine, but for good performance the
      number of pointings passed per call should be as large as possible.
)""";

constexpr const char *getSlm_DS = R"""(
Returns a set of sky spherical hamonic coefficients resulting from adjoint
interpolation

Parameters
----------
beam : numpy.array(nalm_beam, nbeam), dtype=numpy.complex)
    spherical harmonic coefficients of the beam with lmax and kmax defined
    in the constructor call
    nbeam must match the ncomp specified in the constructor, unless ncomp was 1.

Returns
-------
numpy.array(nalm_sky, nbeam), dtype=numpy.complex):
    spherical harmonic coefficients of the sky with lmax defined
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

  py::class_<PyInterpolator<fptype>> (m, "PyInterpolator", pyinterpolator_DS)
    .def(py::init<const py::array &, const py::array &, bool, int64_t, int64_t, fptype, int>(),
      initnormal_DS, "sky"_a, "beam"_a, "separate"_a, "lmax"_a, "kmax"_a, "epsilon"_a,
      "nthreads"_a)
    .def(py::init<int64_t, int64_t, int64_t, fptype, int>(), initadjoint_DS,
      "lmax"_a, "kmax"_a, "ncomp"_a, "epsilon"_a, "nthreads"_a)
    .def ("interpol", &PyInterpolator<fptype>::pyinterpol, interpol_DS, "ptg"_a)
    .def ("deinterpol", &PyInterpolator<fptype>::pydeinterpol, deinterpol_DS, "ptg"_a, "data"_a)
    .def ("getSlm", &PyInterpolator<fptype>::pygetSlm, getSlm_DS, "beam"_a);
#if 1
  m.def("rotate_alm", &pyrotate_alm<fptype>, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a);
#endif
  }
