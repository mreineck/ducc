/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "python/totalconvolve.h"

namespace ducc0 {

namespace detail_pymodule_totalconvolve {

using namespace std;

namespace py = pybind11;

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
      bool separate, int64_t lmax, int64_t kmax, T epsilon, T ofactor, int nthreads)
      : Interpolator<T>(makevec(slm, lmax, lmax),
                        makevec(blm, lmax, kmax),
                        separate, epsilon, ofactor, nthreads) {}
    PyInterpolator(int64_t lmax, int64_t kmax, int64_t ncomp_, T epsilon, T ofactor, int nthreads)
      : Interpolator<T>(lmax, kmax, ncomp_, epsilon, ofactor, nthreads) {}

    using Interpolator<T>::support;

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

constexpr const char *totalconvolve_DS = R"""(
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
ofactor : float
    oversampling factor to be used for the interpolation grids.
    Should be in the range [1.2; 2], a typical value is 1.5
    Increasing this factor makes (adjoint) convolution slower and
    increases memory consumption, but speeds up interpolation/deinterpolation.
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
ofactor : float
    oversampling factor to be used for the interpolation grids.
    Should be in the range [1.2; 2], a typical value is 1.5
    Increasing this factor makes (adjoint) convolution slower and
    increases memory consumption, but speeds up interpolation/deinterpolation.
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

void add_totalconvolve(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("totalconvolve");

  m.doc() = totalconvolve_DS;

  using inter_d = PyInterpolator<double>;
  py::class_<inter_d> (m, "Interpolator", py::module_local(), pyinterpolator_DS)
    .def(py::init<const py::array &, const py::array &, bool, int64_t, int64_t, double, double, int>(),
      initnormal_DS, "sky"_a, "beam"_a, "separate"_a, "lmax"_a, "kmax"_a, "epsilon"_a, "ofactor"_a=1.5,
      "nthreads"_a=0)
    .def(py::init<int64_t, int64_t, int64_t, double, double, int>(), initadjoint_DS,
      "lmax"_a, "kmax"_a, "ncomp"_a, "epsilon"_a, "ofactor"_a=1.5, "nthreads"_a=0)
    .def ("interpol", &inter_d::pyinterpol, interpol_DS, "ptg"_a)
    .def ("deinterpol", &inter_d::pydeinterpol, deinterpol_DS, "ptg"_a, "data"_a)
    .def ("getSlm", &inter_d::pygetSlm, getSlm_DS, "beam"_a)
    .def ("support", &inter_d::support);
  using inter_f = PyInterpolator<float>;
  py::class_<inter_f> (m, "Interpolator_f", py::module_local(), pyinterpolator_DS)
    .def(py::init<const py::array &, const py::array &, bool, int64_t, int64_t, float, float, int>(),
      initnormal_DS, "sky"_a, "beam"_a, "separate"_a, "lmax"_a, "kmax"_a, "epsilon"_a, "ofactor"_a=1.5f,
      "nthreads"_a=0)
    .def(py::init<int64_t, int64_t, int64_t, float, float, int>(), initadjoint_DS,
      "lmax"_a, "kmax"_a, "ncomp"_a, "epsilon"_a, "ofactor"_a=1.5f, "nthreads"_a=0)
    .def ("interpol", &inter_f::pyinterpol, interpol_DS, "ptg"_a)
    .def ("deinterpol", &inter_f::pydeinterpol, deinterpol_DS, "ptg"_a, "data"_a)
    .def ("getSlm", &inter_f::pygetSlm, getSlm_DS, "beam"_a)
    .def ("support", &inter_f::support);
  }

}

using detail_pymodule_totalconvolve::add_totalconvolve;

}
