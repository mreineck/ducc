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
    using Interpolator<T>::interpolx;
    using Interpolator<T>::deinterpolx;
    using Interpolator<T>::getSlmx;

  public:
    PyInterpolator(const py::array &slmT, const py::array &blmT,
      int64_t lmax, int64_t kmax, double epsilon, int nthreads=0)
      : Interpolator<T>(Alm<complex<T>>(to_mav<complex<T>,1>(slmT), lmax, lmax),
                        Alm<complex<T>>(to_mav<complex<T>,1>(blmT), lmax, kmax),
                        epsilon, nthreads) {}
    PyInterpolator(int64_t lmax, int64_t kmax, double epsilon, int nthreads=0)
      : Interpolator<T>(lmax, kmax, epsilon, nthreads) {}
    py::array interpol(const py::array &ptg) const
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto res = make_Pyarr<double>({ptg2.shape(0)});
      auto res2 = to_mav<double,1>(res,true);
      interpolx(ptg2, res2);
      return res;
      }

    void deinterpol(const py::array &ptg, const py::array &data)
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto data2 = to_mav<T,1>(data);
      deinterpolx(ptg2, data2);
      }
    py::array getSlm(const py::array &blmT_)
      {
      auto res = make_Pyarr<complex<T>>({Alm_Base::Num_Alms(lmax, lmax)});
      Alm<complex<T>> blmT(to_mav<complex<T>,1>(blmT_, false), lmax, kmax);
      auto slmT_=to_mav<complex<T>,1>(res, true);
      Alm<complex<T>> slmT(slmT_, lmax, lmax);
      getSlmx(blmT, slmT);
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

} // unnamed namespace

PYBIND11_MODULE(pyinterpol_ng, m)
  {
  using namespace pybind11::literals;

  py::class_<PyInterpolator<double>> (m, "PyInterpolator")
    .def(py::init<const py::array &, const py::array &, int64_t, int64_t, double, int>(),
      "sky"_a, "beam"_a, "lmax"_a, "kmax"_a, "epsilon"_a, "nthreads"_a)
    .def(py::init<int64_t, int64_t, double, int>(),
      "lmax"_a, "kmax"_a, "epsilon"_a, "nthreads"_a)
    .def ("interpol", &PyInterpolator<double>::interpol, "ptg"_a)
    .def ("deinterpol", &PyInterpolator<double>::deinterpol, "ptg"_a, "data"_a)
    .def ("getSlm", &PyInterpolator<double>::getSlm, "blmT"_a);
#if 1
  m.def("rotate_alm", &pyrotate_alm<double>, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a);
#endif
  }
