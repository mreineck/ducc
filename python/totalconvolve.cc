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
auto None = py::none();

template<typename T> class PyConvolverPlan: public ConvolverPlan<T>
  {
  private:
    using ConvolverPlan<T>::lmax;
    Alm<complex<T>> getAlm(const py::array &inp, bool write=false) const
      {
      auto inp2 = to_mav<complex<T>,1>(inp, write);
      int mmax = Alm_Base::Get_Mmax(inp2.shape(0), lmax);
      return Alm<complex<T>>(inp2, lmax, mmax);
      }

    using ConvolverPlan<T>::ConvolverPlan;
    using ConvolverPlan<T>::getPlane;
    using ConvolverPlan<T>::interpol;
    using ConvolverPlan<T>::deinterpol;
    using ConvolverPlan<T>::updateSlm;

  public:
    using ConvolverPlan<T>::Ntheta;
    using ConvolverPlan<T>::Nphi;
    void pyGetPlane(const py::array &py_slm, const py::array &py_blm,
      size_t mbeam, py::array &py_re, py::object &py_im) const
      {
      auto slm = getAlm(py_slm);
      auto blm = getAlm(py_blm);
      auto re = to_mav<T,2>(py_re, true);
      auto im = (mbeam==0) ? mav<T,2>::build_empty() : to_mav<T,2>(py_im, true);
      getPlane(slm, blm, mbeam, re, im);
      }
    void pyinterpol(const py::array &pycube, size_t itheta0, size_t iphi0,
      const py::array &pytheta, const py::array &pyphi, const py::array &pypsi,
      py::array &pysignal)
      {
      auto cube = to_mav<T,3>(pycube, false);
      auto theta = to_mav<T,1>(pytheta, false);
      auto phi = to_mav<T,1>(pyphi, false);
      auto psi = to_mav<T,1>(pypsi, false);
      auto signal = to_mav<T,1>(pysignal, true);
      interpol(cube, itheta0, iphi0, theta, phi, psi, signal);
      }
    void pydeinterpol(py::array &pycube, size_t itheta0, size_t iphi0,
      const py::array &pytheta, const py::array &pyphi, const py::array &pypsi,
      const py::array &pysignal)
      {
      auto cube = to_mav<T,3>(pycube, true);
      auto theta = to_mav<T,1>(pytheta, false);
      auto phi = to_mav<T,1>(pyphi, false);
      auto psi = to_mav<T,1>(pypsi, false);
      auto signal = to_mav<T,1>(pysignal, false);
      deinterpol(cube, itheta0, iphi0, theta, phi, psi, signal);
      }
    void pyUpdateSlm(py::array &py_slm, const py::array &py_blm,
      size_t mbeam, py::array &py_re, py::object &py_im) const
      {
      auto slm = getAlm(py_slm, true);
      auto blm = getAlm(py_blm);
      auto re = to_mav<T,2>(py_re, true);
      auto im = (mbeam==0) ? mav<T,2>::build_empty() : to_mav<T,2>(py_im, true);
      updateSlm(slm, blm, mbeam, re, im);
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

void add_totalconvolve(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("totalconvolve");

  m.doc() = totalconvolve_DS;

  using conv_d = PyConvolverPlan<double>;
  py::class_<conv_d> (m, "ConvolverPlan", py::module_local())
    .def(py::init<size_t, double, double, size_t>(),
      "lmax"_a, "sigma"_a, "epsilon"_a, "nthreads"_a=0)
    .def("Ntheta", &conv_d::Ntheta)
    .def("Nphi", &conv_d::Nphi)
    .def("getPlane", &conv_d::pyGetPlane, "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None)
    .def("interpol", &conv_d::pyinterpol, "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("deinterpol", &conv_d::pydeinterpol, "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("updateSlm", &conv_d::pyUpdateSlm, "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None);
  using conv_f = PyConvolverPlan<float>;
  py::class_<conv_f> (m, "ConvolverPlan_f", py::module_local())
    .def(py::init<size_t, double, double, size_t>(),
      "lmax"_a, "sigma"_a, "epsilon"_a, "nthreads"_a=0)
    .def("Ntheta", &conv_f::Ntheta)
    .def("Nphi", &conv_f::Nphi)
    .def("getPlane", &conv_f::pyGetPlane, "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None)
    .def("interpol", &conv_f::pyinterpol, "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("deinterpol", &conv_f::pydeinterpol, "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("updateSlm", &conv_f::pyUpdateSlm, "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None);
  }

}

using detail_pymodule_totalconvolve::add_totalconvolve;

}
