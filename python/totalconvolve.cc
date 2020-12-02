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
#include "ducc0/bindings/pybind_utils.h"
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
    using ConvolverPlan<T>::ConvolverPlan;
    using ConvolverPlan<T>::getPlane;
    using ConvolverPlan<T>::interpol;
    using ConvolverPlan<T>::deinterpol;
    using ConvolverPlan<T>::updateSlm;
    using ConvolverPlan<T>::getPatchInfo;
    using ConvolverPlan<T>::prepPsi;
    using ConvolverPlan<T>::deprepPsi;

  public:
    using ConvolverPlan<T>::Ntheta;
    using ConvolverPlan<T>::Nphi;
    using ConvolverPlan<T>::Npsi;
    vector<size_t> pyGetPatchInfo(T theta_lo, T theta_hi, T phi_lo, T phi_hi)
      { return getPatchInfo(theta_lo, theta_hi, phi_lo, phi_hi); }
    void pyGetPlane(const py::array &py_slm, const py::array &py_blm,
      size_t mbeam, py::array &py_re, py::object &py_im) const
      {
      auto slm = to_mav<complex<T>,1>(py_slm);
      auto blm = to_mav<complex<T>,1>(py_blm);
      auto re = to_mav<T,2>(py_re, true);
      auto im = (mbeam==0) ? mav<T,2>::build_empty() : to_mav<T,2>(py_im, true);
      getPlane(slm, blm, mbeam, re, im);
      }
    void pyPrepPsi(const py::array &py_subcube) const
      {
      auto subcube = to_mav<T,3>(py_subcube, true);
      prepPsi(subcube);
      }
    void pyDeprepPsi(const py::array &py_subcube) const
      {
      auto subcube = to_mav<T,3>(py_subcube, true);
      deprepPsi(subcube);
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
      auto slm = to_mav<complex<T>,1>(py_slm, true);
      auto blm = to_mav<complex<T>,1>(py_blm);
      auto re = to_mav<T,2>(py_re, true);
      auto im = (mbeam==0) ? mav<T,2>::build_empty() : to_mav<T,2>(py_im, true);
      updateSlm(slm, blm, mbeam, re, im);
      }
  };


template<typename T> class PyInterpolator
  {
  private:
    ConvolverPlan<T> conv;
    mav<T,4> cube;

  public:
    PyInterpolator(const py::array &slm, const py::array &blm,
      bool separate, size_t lmax, size_t kmax, T epsilon, T ofactor, int nthreads)
      : conv(lmax, kmax, ofactor, epsilon, nthreads),
        cube({(separate ? size_t(slm.shape(1)) : 1u), conv.Npsi(), conv.Ntheta(), conv.Nphi()})
      {
      auto vslm = to_mav<complex<T>,2>(slm);
      auto vblm = to_mav<complex<T>,2>(blm);
      if (separate)
        for (size_t i=0; i<vslm.shape(1); ++i)
          {
          auto re = subarray<2>(cube, {i,0,0,0},{0, 0, MAXIDX, MAXIDX});
          auto vslmi = subarray<2>(vslm, {0,i},{MAXIDX,1});
          auto vblmi = subarray<2>(vblm, {0,i},{MAXIDX,1});
          conv.getPlane(vslmi, vblmi, 0, re, re);
          for (size_t k=1; k<kmax+1; ++k)
            {
            auto re = subarray<2>(cube, {i,2*k-1,0,0},{0, 0, MAXIDX, MAXIDX});
            auto im = subarray<2>(cube, {i,2*k  ,0,0},{0, 0, MAXIDX, MAXIDX});
            conv.getPlane(vslmi, vblmi, k, re, im);
            }
          }
      else
        {
        auto re = subarray<2>(cube, {0,0,0,0},{0, 0, MAXIDX, MAXIDX});
        conv.getPlane(vslm, vblm, 0, re, re);
        for (size_t k=1; k<kmax+1; ++k)
          {
          auto re = subarray<2>(cube, {0,2*k-1,0,0},{0, 0, MAXIDX, MAXIDX});
          auto im = subarray<2>(cube, {0,2*k  ,0,0},{0, 0, MAXIDX, MAXIDX});
          conv.getPlane(vslm, vblm, k, re, im);
          }
        }
      for (size_t i=0; i<cube.shape(0); ++i)
        {
        auto subcube = subarray<3>(cube, {i,0,0,0},{0, MAXIDX, MAXIDX, MAXIDX});
        conv.prepPsi(subcube);
        }
      }
    PyInterpolator(size_t lmax, size_t kmax, size_t ncomp_, T epsilon, T ofactor, int nthreads)
      : conv(lmax, kmax, ofactor, epsilon, nthreads),
        cube({size_t(ncomp_), conv.Npsi(), conv.Ntheta(), conv.Nphi()})
      {
      }

    py::array pyinterpol(const py::array &ptg) const
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto ptheta = subarray<1>(ptg2, {0,0},{MAXIDX,0});
      auto pphi = subarray<1>(ptg2, {0,1},{MAXIDX,0});
      auto ppsi = subarray<1>(ptg2, {0,2},{MAXIDX,0});
      size_t ncomp = cube.shape(0);
      auto res = make_Pyarr<T>({ptg2.shape(0),ncomp});
      auto res2 = to_mav<T,2>(res,true);
      for (size_t i=0; i<ncomp; ++i)
        {
        auto subcube = subarray<3>(cube, {i,0,0,0},{0, MAXIDX, MAXIDX, MAXIDX});
        auto subres = subarray<1>(res2, {0,i},{MAXIDX,0});
        conv.interpol(subcube, 0, 0, ptheta, pphi, ppsi, subres);
        }
      return move(res);
      }

    void pydeinterpol(const py::array &ptg, const py::array &data)
      {
      auto ptg2 = to_mav<T,2>(ptg);
      auto ptheta = subarray<1>(ptg2, {0,0},{MAXIDX,0});
      auto pphi = subarray<1>(ptg2, {0,1},{MAXIDX,0});
      auto ppsi = subarray<1>(ptg2, {0,2},{MAXIDX,0});
      size_t ncomp = cube.shape(0);
      auto data2 = to_mav<T,2>(data);
      for (size_t i=0; i<ncomp; ++i)
        {
        auto subcube = subarray<3>(cube, {i,0,0,0},{0, MAXIDX, MAXIDX, MAXIDX});
        auto subdata = subarray<1>(data2, {0,i},{MAXIDX,0});
        conv.deinterpol(subcube, 0, 0, ptheta, pphi, ppsi, subdata);
        }
      }
    py::array pygetSlm(const py::array &blm_)
      {
      size_t lmax=conv.Lmax(), kmax=conv.Kmax();
      auto vblm = to_mav<complex<T>,2>(blm_);
      size_t ncomp = vblm.shape(1);
      bool separate = cube.shape(0)>1;
      if (separate) MR_assert(ncomp==cube.shape(0), "dimension mismatch");
      for (size_t i=0; i<cube.shape(0); ++i)
        {
        auto subcube = subarray<3>(cube, {i,0,0,0},{0, MAXIDX, MAXIDX, MAXIDX});
        conv.deprepPsi(subcube);
        }
      auto res = make_Pyarr<complex<T>>({Alm_Base::Num_Alms(lmax, lmax),ncomp});
      auto vslm = to_mav<complex<T>,2>(res, true);
      vslm.fill(T(0));
      if (separate)
        for (size_t i=0; i<ncomp; ++i)
          {
          auto re = subarray<2>(cube, {i,0,0,0},{0, 0, MAXIDX, MAXIDX});
          auto vslmi = subarray<2>(vslm, {0,i},{MAXIDX,1});
          auto vblmi = subarray<2>(vblm, {0,i},{MAXIDX,1});
          conv.updateSlm(vslmi, vblmi, 0, re, re);
          for (size_t k=1; k<kmax+1; ++k)
            {
            auto re = subarray<2>(cube, {i,2*k-1,0,0},{0, 0, MAXIDX, MAXIDX});
            auto im = subarray<2>(cube, {i,2*k  ,0,0},{0, 0, MAXIDX, MAXIDX});
            conv.updateSlm(vslmi, vblmi, k, re, im);
            }
          }
      else
        {
        auto re = subarray<2>(cube, {0,0,0,0},{0, 0, MAXIDX, MAXIDX});
        conv.updateSlm(vslm, vblm, 0, re, re);
        for (size_t k=1; k<kmax+1; ++k)
          {
          auto re = subarray<2>(cube, {0,2*k-1,0,0},{0, 0, MAXIDX, MAXIDX});
          auto im = subarray<2>(cube, {0,2*k  ,0,0},{0, 0, MAXIDX, MAXIDX});
          conv.updateSlm(vslm, vblm, k, re, im);
          }
        }
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

constexpr const char *pyConvolverPlan_DS = R"""(
Class encapsulating the low-level interface for convolution/interpolation.
Computations are performed on double precision data.
)""";

constexpr const char *pyConvolverPlan_f_DS = R"""(
Class encapsulating the low-level interface for convolution/interpolation.
Computations are performed on single precision data.
)""";

constexpr const char *pyConvolverPlan_init_DS = R"""(
ConvolverPlan constructor

Parameters
----------
lmax : int, 0 <= lmax
    maximum l for the sky and beam coefficients; maximum m for sky coefficients
    In other words, the band limit of the involved functions
kmax : int, 0 <= kmax <= lmax
    maximum m (or azimuthal moment) for the beam coefficients
sigma : float, 1.2 <= sigma <= 2.5
    the (approximate) oversampling factor to use for the calculation.
    Lower sigma lead to smaller data cubes, but slower interpolation, and only
    work for relatively low accuracies.
epsilon : float, 1e-12 <= epsilon <= 1e-1
    the desired relative accuracy of the interpolation
    NOTE: epsilons near the accuracy limit can only be reached by choosing
    a sufficiently high value for sigma!
nthreads : int 0 <= nthreads
    the number of threads to use for all computations
    A value of 0 implies that the full number of hardware threads on the system
    will be used.
)""";

constexpr const char *pyConvolverPlan_f_init_DS = R"""(
ConvolverPlan constructor

Parameters
----------
lmax : int, 0 <= lmax
    maximum l for the sky and beam coefficients; maximum m for sky coefficients
    In other words, the band limit of the involved functions
kmax : int, 0 <= kmax <= lmax
    maximum m (or azimuthal moment) for the beam coefficients
sigma : float, 1.2 <= sigma <= 2.5
    the (approximate) oversampling factor to use for the calculation.
    Lower sigma lead to smaller data cubes, but slower interpolation, and only
    work for relatively low accuracies.
epsilon : float, 3e-5 <= epsilon <= 1e-1
    the desired relative accuracy of the interpolation
    NOTE: epsilons near the accuracy limit can only be reached by choosing
    a sufficiently high value for sigma!
nthreads : int 0 <= nthreads
    the number of threads to use for all computations
    A value of 0 implies that the full number of hardware threads on the system
    will be used.
)""";

constexpr const char *pyConvolverPlan_Ntheta_DS = R"""(
Returns
-------
The full data cube dimension in theta direction (second axis)
)""";

constexpr const char *pyConvolverPlan_Nphi_DS = R"""(
Returns
-------
The full data cube dimension in phi direction (third axis)
)""";

constexpr const char *pyConvolverPlan_Npsi_DS = R"""(
Returns
-------
The full data cube dimension in psi direction (first axis)
)""";

constexpr const char *pyConvolverPlan_getPatchInfo_DS = R"""(
Returns information necessary to extract a given sub-area from the data cube.

Parameters
----------
theta_lo, theta_hi : float, 0 <= theta_lo < theta_hi <= pi
    colatitude borders of the requested patch
phi_lo, phi_hi : float, 0 <= phi_lo < phi_hi <= 2*pi
    longitude borders of the requested patch

Returns
-------
tuple(int) with 4 elements itheta_lo, itheta_hi, iphi_lo, iphi_hi
    The sub-array [:, itheta_lo:itheta_hi, iphi_lo:iphi_hi] of a full data cube
    will contain all information necessary to interpolate pointings within the
    specified patch.
)""";

constexpr const char *pyConvolverPlan_getPlane_DS = R"""(
Computes a single (real or complex) sub-plane in (theta, phi) of the data cube

Parameters
----------
slm : numpy.ndarray((nalm_sky), dtype=numpy.complex128), or
      numpy.ndarray((nalm_sky, ncomp), dtype=numpy.complex128)
    spherical harmonic coefficients of the sky.
blm : numpy.ndarray((nalm_beam), dtype=numpy.complex128), or
      numpy.ndarray((nalm_beam, ncomp), dtype=numpy.complex128)
    spherical harmonic coefficients of the beam.
mbeam : int, 0 <= mbeam <= kmax
    requested m moment of the beam
re : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float64)
    will be filled with the real part of the requested sub-plane on exit
im : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float64) or None
    if mbeam > 0,
      will be filled with the imaginary part of the requested sub-plane on exit

Notes
-----
If the `slm` and `blm` arrays have a second dimension, the contributions of all
components will be added together in `re` and `im`.
)""";

constexpr const char *pyConvolverPlan_f_getPlane_DS = R"""(
Computes a single (real or complex) sub-plane in (theta, phi) of the data cube

Parameters
----------
slm : numpy.ndarray((nalm_sky), dtype=numpy.complex64), or
      numpy.ndarray((nalm_sky, ncomp), dtype=numpy.complex)
    spherical harmonic coefficients of the sky.
blm : numpy.ndarray((nalm_beam), dtype=numpy.complex64), or
      numpy.ndarray((nalm_beam, ncomp), dtype=numpy.complex)
    spherical harmonic coefficients of the beam.
mbeam : int, 0 <= mbeam <= kmax
    requested m moment of the beam
re : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float32)
    will be filled with the real part of the requested sub-plane on exit
im : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float32) or None
    if mbeam > 0,
      will be filled with the imaginary part of the requested sub-plane on exit

Notes
-----
If the `slm` and `blm` arrays have a second dimension, the contributions of all
components will be added together in `re` and `im`.
)""";

constexpr const char *pyConvolverPlan_prepPsi_DS = R"""(
Pepares a data cube for for actual interpolation.

Parameters
----------
subcube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float64)
    On entry the part [0:2*kmax+1, :, :] must be filled with results from
    getPlane() calls.
    On exit, the entire array will be filled in a form that can be used for
    subsequent `interpol` calls.
)""";

constexpr const char *pyConvolverPlan_f_prepPsi_DS = R"""(
Pepares a data cube for for actual interpolation.

Parameters
----------
subcube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float32)
    On entry the part [0:2*kmax+1, :, :] must be filled with results from
    getPlane() calls.
    On exit, the entire array will be filled in a form that can be used for
    subsequent `interpol` calls.
)""";

constexpr const char *pyConvolverPlan_deprepPsi_DS = R"""(
Adjoint of `prepPsi`.

Parameters
----------
subcube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float64)
    On entry this must be an array filled by one or more `deinterpol` calls.
    On exit, only the part [0:2*kmax+1, :, :] is relevant and can be used for
    `updateSlm` calls.
)""";

constexpr const char *pyConvolverPlan_f_deprepPsi_DS = R"""(
Adjoint of `prepPsi`.

Parameters
----------
subcube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float32)
    On entry this must be an array filled by one or more `deinterpol` calls.
    On exit, only the part [0:2*kmax+1, :, :] is relevant and can be used for
    `updateSlm` calls.
)""";

constexpr const char *pyConvolverPlan_interpol_DS = R"""(
Computes the interpolated values for a given set of angle triplets

Parameters
----------
cube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float64)
    (Partial) data cube generated with `prepPsi`.
itheta0, iphi0 : int
    starting indices in theta and phi direction of the provided cube relative
    to the full cube.
theta, phi, psi : numpy.ndarray(nptg, dtype=numpy.float64)
    angle triplets at which the interpolated values will be computed
    Theta and phi must lie inside the ranges covered by the supplied cube.
    No constraints on psi.
signal : numpy.ndarray(nptg, dtype=numpy.float64)
    array into which the results will be written

Notes
-----
Repeated calls to this method are fine, but for good performance the
number of pointings passed per call should be as large as possible.
)""";

constexpr const char *pyConvolverPlan_f_interpol_DS = R"""(
Computes the interpolated values for a given set of angle triplets

Parameters
----------
cube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float32)
    (Partial) data cube generated with `prepPsi`.
itheta0, iphi0 : int
    starting indices in theta and phi direction of the provided cube relative
    to the full cube.
theta, phi, psi : numpy.ndarray(nptg, dtype=numpy.float32)
    angle triplets at which the interpolated values will be computed
    Theta and phi must lie inside the ranges covered by the supplied cube.
    No constraints on psi.
signal : numpy.ndarray(nptg, dtype=numpy.float32)
    array into which the results will be written

Notes
-----
Repeated calls to this method are fine, but for good performance the
number of pointings passed per call should be as large as possible.
)""";

constexpr const char *pyConvolverPlan_deinterpol_DS = R"""(
Adjoint of `interpol`.
Spreads the values in `signal` over the appropriate regions of `cube`

Parameters
----------
cube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float64)
    (Partial) data cube to which the deinterpolated values will be added.
    Must be zeroed before the first call to `deinterpol`!
itheta0, iphi0 : int
    starting indices in theta and phi direction of the provided cube relative
    to the full cube.
theta, phi, psi : numpy.ndarray(nptg, dtype=numpy.float64)
    angle triplets at which the interpolated values will be computed
    Theta and phi must lie inside the ranges covered by the supplied cube.
    No constraints on psi.
signal : numpy.ndarray(nptg, dtype=numpy.float64)
    signal values that will be deinterpolated into `cube`.

Notes
-----
Repeated calls to this method are fine, but for good performance the
number of pointings passed per call should be as large as possible.
)""";

constexpr const char *pyConvolverPlan_f_deinterpol_DS = R"""(
Adjoint of `interpol`.
Spreads the values in `signal` over the appropriate regions of `cube`

Parameters
----------
cube : numpy.ndarray((Npsi(), :, :), dtype=numpy.float32)
    (Partial) data cube to which the deinterpolated values will be added.
    Must be zeroed before the first call to `deinterpol`!
itheta0, iphi0 : int
    starting indices in theta and phi direction of the provided cube relative
    to the full cube.
theta, phi, psi : numpy.ndarray(nptg, dtype=numpy.float32)
    angle triplets at which the interpolated values will be computed
    Theta and phi must lie inside the ranges covered by the supplied cube.
    No constraints on psi.
signal : numpy.ndarray(nptg, dtype=numpy.float32)
    signal values that will be deinterpolated into `cube`.

Notes
-----
Repeated calls to this method are fine, but for good performance the
number of pointings passed per call should be as large as possible.
)""";

constexpr const char *pyConvolverPlan_updateSlm_DS = R"""(
Updates a set of sky spherical hamonic coefficients resulting from adjoint
interpolation.

Parameters
----------
slm : numpy.ndarray((nalm_sky), dtype=numpy.complex128), or
      numpy.ndarray((nalm_sky, ncomp), dtype=numpy.complex128)
    The deinterpolated spherical harmonic coefficients will be added to this
    array.
    Must be zeroed before the first call to `updateSlm`!
blm : numpy.ndarray((nalm_beam), dtype=numpy.complex128), or
      numpy.ndarray((nalm_beam, ncomp), dtype=numpy.complex128)
    spherical harmonic coefficients of the beam.
mbeam : int, 0 <= mbeam <= kmax
    requested m moment of the beam
re : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float64)
    real part of the requested plane
im : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float64) or None
    if mbeam > 0,
      imaginary part of the requested plane

Notes
-----
If the `slm` and `blm` arrays have a second dimension, the `slm` will be
computed in a fashion that is adjoint to `getPlane`.
)""";

constexpr const char *pyConvolverPlan_f_updateSlm_DS = R"""(
Updates a set of sky spherical hamonic coefficients resulting from adjoint
interpolation.

Parameters
----------
slm : numpy.ndarray((nalm_sky), dtype=numpy.complex64), or
      numpy.ndarray((nalm_sky, ncomp), dtype=numpy.complex64)
    The deinterpolated spherical harmonic coefficients will be added to this
    array.
    Must be zeroed before the first call to `updateSlm`!
blm : numpy.ndarray((nalm_beam), dtype=numpy.complex64), or
      numpy.ndarray((nalm_beam, ncomp), dtype=numpy.complex64)
    spherical harmonic coefficients of the beam.
mbeam : int, 0 <= mbeam <= kmax
    requested m moment of the beam
re : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float32)
    real part of the requested plane
im : numpy.ndarray((Ntheta(), Nphi()), dtype=numpy.float32) or None
    if mbeam > 0,
      imaginary part of the requested plane

Notes
-----
If the `slm` and `blm` arrays have a second dimension, the `slm` will be
computed in a fashion that is adjoint to `getPlane`.
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

  using conv_d = PyConvolverPlan<double>;
  py::class_<conv_d> (m, "ConvolverPlan", py::module_local(), pyConvolverPlan_DS)
    .def(py::init<size_t, size_t, double, double, size_t>(), pyConvolverPlan_init_DS,
      "lmax"_a, "kmax"_a, "sigma"_a, "epsilon"_a, "nthreads"_a=0)
    .def("Ntheta", &conv_d::Ntheta, pyConvolverPlan_Ntheta_DS)
    .def("Nphi", &conv_d::Nphi, pyConvolverPlan_Nphi_DS)
    .def("Npsi", &conv_d::Npsi, pyConvolverPlan_Npsi_DS)
    .def("getPatchInfo", &conv_d::pyGetPatchInfo, pyConvolverPlan_getPatchInfo_DS,
      "theta_lo"_a, "theta_hi"_a, "phi_lo"_a, "phi_hi"_a)
    .def("getPlane", &conv_d::pyGetPlane, pyConvolverPlan_getPlane_DS,
      "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None)
    .def("prepPsi", &conv_d::pyPrepPsi, pyConvolverPlan_prepPsi_DS, "subcube"_a)
    .def("deprepPsi", &conv_d::pyDeprepPsi, pyConvolverPlan_prepPsi_DS, "subcube"_a)
    .def("interpol", &conv_d::pyinterpol, pyConvolverPlan_interpol_DS,
      "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("deinterpol", &conv_d::pydeinterpol, pyConvolverPlan_deinterpol_DS,
      "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("updateSlm", &conv_d::pyUpdateSlm, pyConvolverPlan_updateSlm_DS,
      "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None);
  using conv_f = PyConvolverPlan<float>;
  py::class_<conv_f> (m, "ConvolverPlan_f", py::module_local(), pyConvolverPlan_f_DS)
    .def(py::init<size_t, size_t, double, double, size_t>(), pyConvolverPlan_f_init_DS,
      "lmax"_a, "kmax"_a, "sigma"_a, "epsilon"_a, "nthreads"_a=0)
    .def("Ntheta", &conv_f::Ntheta, pyConvolverPlan_Ntheta_DS)
    .def("Nphi", &conv_f::Nphi, pyConvolverPlan_Nphi_DS)
    .def("Npsi", &conv_f::Npsi, pyConvolverPlan_Npsi_DS)
    .def("getPatchInfo", &conv_f::pyGetPatchInfo, pyConvolverPlan_getPatchInfo_DS,
      "theta_lo"_a, "theta_hi"_a, "phi_lo"_a, "phi_hi"_a)
    .def("getPlane", &conv_f::pyGetPlane, pyConvolverPlan_f_getPlane_DS,
      "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None)
    .def("prepPsi", &conv_f::pyPrepPsi, pyConvolverPlan_f_prepPsi_DS, "subcube"_a)
    .def("deprepPsi", &conv_f::pyDeprepPsi, pyConvolverPlan_f_deprepPsi_DS, "subcube"_a)
    .def("interpol", &conv_f::pyinterpol, pyConvolverPlan_f_interpol_DS,
      "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("deinterpol", &conv_f::pydeinterpol, pyConvolverPlan_f_deinterpol_DS,
      "cube"_a, "itheta0"_a, "iphi0"_a, "theta"_a, "phi"_a, "psi"_a, "signal"_a)
    .def("updateSlm", &conv_f::pyUpdateSlm, pyConvolverPlan_f_updateSlm_DS,
      "slm"_a, "blm"_a, "mbeam"_a, "re"_a, "im"_a=None);

  using inter_d = PyInterpolator<double>;
  py::class_<inter_d> (m, "Interpolator", py::module_local(), pyinterpolator_DS)
    .def(py::init<const py::array &, const py::array &, bool, size_t, size_t, double, double, int>(),
      initnormal_DS, "sky"_a, "beam"_a, "separate"_a, "lmax"_a, "kmax"_a, "epsilon"_a, "ofactor"_a=1.5,
      "nthreads"_a=0)
    .def(py::init<size_t, size_t, size_t, double, double, int>(), initadjoint_DS,
      "lmax"_a, "kmax"_a, "ncomp"_a, "epsilon"_a, "ofactor"_a=1.5, "nthreads"_a=0)
    .def ("interpol", &inter_d::pyinterpol, interpol_DS, "ptg"_a)
    .def ("deinterpol", &inter_d::pydeinterpol, deinterpol_DS, "ptg"_a, "data"_a)
    .def ("getSlm", &inter_d::pygetSlm, getSlm_DS, "beam"_a);
  using inter_f = PyInterpolator<float>;
  py::class_<inter_f> (m, "Interpolator_f", py::module_local(), pyinterpolator_DS)
    .def(py::init<const py::array &, const py::array &, bool, size_t, size_t, float, float, int>(),
      initnormal_DS, "sky"_a, "beam"_a, "separate"_a, "lmax"_a, "kmax"_a, "epsilon"_a, "ofactor"_a=1.5f,
      "nthreads"_a=0)
    .def(py::init<size_t, size_t, size_t, float, float, int>(), initadjoint_DS,
      "lmax"_a, "kmax"_a, "ncomp"_a, "epsilon"_a, "ofactor"_a=1.5f, "nthreads"_a=0)
    .def ("interpol", &inter_f::pyinterpol, interpol_DS, "ptg"_a)
    .def ("deinterpol", &inter_f::pydeinterpol, deinterpol_DS, "ptg"_a, "data"_a)
    .def ("getSlm", &inter_f::pygetSlm, getSlm_DS, "beam"_a);
  }

}

using detail_pymodule_totalconvolve::add_totalconvolve;

}
