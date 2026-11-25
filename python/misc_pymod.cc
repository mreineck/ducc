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
 *  Copyright (C) 2020-2026 Max-Planck-Society
 *  Author: Martin Reinecke
 */

// IMPORTANT: always include "pybind_utils.h" first, even before standard headers
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/math/constants.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>


namespace DUCC0_NAMESPACE {

namespace detail_pymodule_misc {

using namespace std;

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

using FloatOrComplex = variant<double,complex<double>>;

template<typename T1, typename T2> static FloatOrComplex Py3_vdot(const CNpArr &a_, const CNpArr &b_)
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
  return (acc.imag()==0) ? FloatOrComplex(double(acc.real()))
                         : FloatOrComplex(complex<double>(acc));
  }
template<typename T1> static FloatOrComplex Py2_vdot(const CNpArr &a, const CNpArr &b)
  {
  if (isPyarr<float>(b))
    return Py3_vdot<T1,float>(a,b);
  if (isPyarr<complex<float>>(b))
    return Py3_vdot<T1,complex<float>>(a,b);
  if (isPyarr<double>(b))
    return Py3_vdot<T1,double>(a,b);
  if (isPyarr<complex<double>>(b))
    return Py3_vdot<T1,complex<double>>(a,b);
#ifndef DUCC0_USE_NANOBIND
  if (isPyarr<long double>(b))
    return Py3_vdot<T1,long double>(a,b);
  if (isPyarr<complex<long double>>(b))
    return Py3_vdot<T1,complex<long double>>(a,b);
#endif
  MR_fail("type matching failed");
  }
static FloatOrComplex Py_vdot(const CNpArr &a, const CNpArr &b)
  {
  if (isPyarr<float>(a))
    return Py2_vdot<float>(a,b);
  if (isPyarr<complex<float>>(a))
    return Py2_vdot<complex<float>>(a,b);
  if (isPyarr<double>(a))
    return Py2_vdot<double>(a,b);
  if (isPyarr<complex<double>>(a))
    return Py2_vdot<complex<double>>(a,b);
#ifndef DUCC0_USE_NANOBIND
  if (isPyarr<long double>(a))
    return Py2_vdot<long double>(a,b);
  if (isPyarr<complex<long double>>(a))
    return Py2_vdot<complex<long double>>(a,b);
#endif
  MR_fail("type matching failed");
  }
static FloatOrComplex Py_vdot_scalar(const complex<double> &a,
  const complex<double> &b)
  {
  complex<long double> acc = conj(complex<long double>(a))
                           * complex<long double>(b);
  return (acc.imag()==0) ? FloatOrComplex(double(acc.real()))
                         : FloatOrComplex(complex<double>(acc));
  }

constexpr const char *Py_l2error_DS = R"""(
Compute the L2 error between two arrays or scalars.
More specifically, compute
``sqrt(sum_i(|a_i - b_i|^2) / max(sum_i(|a_i|^2), sum_i(|b_i|^2)))``,
where i goes over all array elements.
In the special case that both a and b are identically zero, the return value is
also zero.

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
template<typename T1, typename T2> static double Py3_l2error(const CNpArr &a_, const CNpArr &b_)
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
  auto maxval = max(acc1,acc2);
  if (maxval==Tacc(0)) return 0.;
  return double(sqrt(acc3/maxval));
  }
template<typename T1> static double Py2_l2error(const CNpArr &a, const CNpArr &b)
  {
  if (isPyarr<float>(b))
    return Py3_l2error<float,T1>(b,a);
  if (isPyarr<complex<float>>(b))
    return Py3_l2error<T1,complex<float>>(a,b);
  if (isPyarr<double>(b))
    return Py3_l2error<double,T1>(b,a);
  if (isPyarr<complex<double>>(b))
    return Py3_l2error<T1,complex<double>>(a,b);
#ifndef DUCC0_USE_NANOBIND
  if (isPyarr<long double>(b))
    return Py3_l2error<long double,T1>(b,a);
  if (isPyarr<complex<long double>>(b))
    return Py3_l2error<T1,complex<long double>>(a,b);
#endif
  MR_fail("type matching failed");
  }
static double Py_l2error(const CNpArr &a, const CNpArr &b)
  {
  if (isPyarr<float>(a))
    return Py2_l2error<float>(a,b);
  if (isPyarr<complex<float>>(a))
    return Py2_l2error<complex<float>>(a,b);
  if (isPyarr<double>(a))
    return Py2_l2error<double>(a,b);
  if (isPyarr<complex<double>>(a))
    return Py2_l2error<complex<double>>(a,b);
#ifndef DUCC0_USE_NANOBIND
  if (isPyarr<long double>(a))
    return Py2_l2error<long double>(a,b);
  if (isPyarr<complex<long double>>(a))
    return Py2_l2error<complex<long double>>(a,b);
#endif
  MR_fail("type matching failed");
  }
double Py_l2error_scalar(const complex<double> &a, const complex<double> &b)
  {
  auto maxval = max(abs(a), abs(b));
  if (maxval==0.) return 0.;
  auto res = abs(a-b)/maxval;
  return double(res);
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
template<typename T> static NpArr Py2_make_noncritical(const CNpArr &in, size_t nthreads)
  {
  auto in2 = to_cfmav<T>(in);
  auto out = make_noncritical_Pyarr<T>(in2.shape(), false, nthreads);
  auto out2 = to_vfmav<T>(out);
  mav_apply([](T &v1, const T &v2) { v1=v2; }, nthreads, out2, in2);
  return out;
  }

static NpArr Py_make_noncritical(const CNpArr &in, size_t nthreads)
  {
  if (isPyarr<float>(in))
    return Py2_make_noncritical<float>(in, nthreads);
  if (isPyarr<complex<float>>(in))
    return Py2_make_noncritical<complex<float>>(in, nthreads);
  if (isPyarr<double>(in))
    return Py2_make_noncritical<double>(in, nthreads);
  if (isPyarr<complex<double>>(in))
    return Py2_make_noncritical<complex<double>>(in, nthreads);
#ifndef DUCC0_USE_NANOBIND
  if (isPyarr<long double>(in))
    return Py2_make_noncritical<long double>(in, nthreads);
  if (isPyarr<complex<long double>>(in))
    return Py2_make_noncritical<complex<long double>>(in, nthreads);
#endif
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
static NpArr Py_empty_noncritical(const vector<size_t> &shape,
  const py::object &dtype_,size_t nthreads)
  {
  auto dtype = normalizeDtype(dtype_);
  if (isDtype<float>(dtype))
    return make_noncritical_Pyarr<float>(shape, false, nthreads);
  if (isDtype<complex<float>>(dtype))
    return make_noncritical_Pyarr<complex<float>>(shape, false, nthreads);
  if (isDtype<double>(dtype))
    return make_noncritical_Pyarr<double>(shape, false, nthreads);
  if (isDtype<complex<double>>(dtype))
    return make_noncritical_Pyarr<complex<double>>(shape, false, nthreads);
#ifndef DUCC0_USE_NANOBIND
  if (isDtype<long double>(dtype))
    return make_noncritical_Pyarr<long double>(shape, false, nthreads);
  if (isDtype<complex<long double>>(dtype))
    return make_noncritical_Pyarr<complex<long double>>(shape, false, nthreads);
#endif
  MR_fail("unsupported datatype");
  }

constexpr const char *available_hardware_threads_DS = R"""(
Returns
-------
int : ducc0's best guess of the number of hardware threads available for the code.


Notes
-----
On Linux, the code checks the thread affinity mask of the current process and
returns the number of (possibly virtual) cores available to the process.
If this cannot be done, std::thread::hardware_concurrency() is returned
)""";

constexpr const char *thread_pool_size_DS = R"""(
Returns
-------
int : the number of threads available for parallel execution

Notes
-----
The default value is determined according to the following pseudo code:

.. code-block:: none

  res = available_hardware_threads();
  if (DUCC0_NUM_THREADS defined)
    res = min(res, $DUCC0_NUM_THREADS)
  else
    if (OMP_NUM_THREADS defined)
      res = min(res, $OMP_NUM_THREADS)
  return max(1, res)
)""";

constexpr const char *resize_thread_pool_DS = R"""(
Parameters
----------
Resizes the thread pool to allow parallel execution with up to `nthreads_new`
threads.

nthreads_new : int >=1
    the desired new number of threads for ducc0 parallel execution

Notes
-----
This function must not be called concurrently;
doing so will result in undefined behaviour.
)""";

using shape_t = fmav_info::shape_t;
template<size_t nd1, size_t nd2> static shape_t repl_dim(const shape_t &s,
  const array<size_t,nd1> &si, const array<size_t,nd2> &so)
  {
  if constexpr (nd1>0)
    {
    MR_assert(s.size()>=nd1,"too few input array dimensions");
    for (size_t i=0; i<nd1; ++i)
      MR_assert(si[i]==s[s.size()-nd1+i], "input dimension mismatch");
    }
  shape_t snew(s.size()-nd1+nd2);
  for (size_t i=0; i<s.size()-nd1; ++i)
    snew[i]=s[i];
  if constexpr (nd2>0)
    for (size_t i=0; i<nd2; ++i)
      snew[i+s.size()-nd1] = so[i];
  return snew;
  }

template<typename T1, typename T2, size_t nd1, size_t nd2>
  static NpArr myprep(const CNpArr &ain, const array<size_t,nd1> &a1,
  const array<size_t,nd2> &a2, const OptNpArr &out)
  {
  auto in = to_cfmav<T1>(ain);
  auto oshp = repl_dim(in.shape(), a1, a2);
  return get_OptNpArr<T2>(out, oshp);
  }

static void print_diagnostics()
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
  const char *version = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY
  cout << "ducc0 v" << version << endl;
  cout << "sizeof(int): " << sizeof(int) << endl;
  cout << "sizeof(size_t): " << sizeof(size_t) << endl;
  cout << "sizeof(float): " << sizeof(float) << endl;
  cout << "alignment of float: " << alignof(float) << endl;
  cout << "sizeof(double): " << sizeof(double) << endl;
  cout << "alignment of double: " << alignof(double) << endl;
  if constexpr (vectorizable<float>)
    {
    cout << "simdlen<float>: " << native_simd<float>::size() << endl;
    cout << "alignment of native_simd<float>: " << alignof(native_simd<float>) << endl;
    }
  else
    cout << "float is not vectorizable" << endl;
  if constexpr (vectorizable<double>)
    {
    cout << "simdlen<double>: " << native_simd<double>::size() << endl;
    cout << "alignment of native_simd<double>: " << alignof(native_simd<double>) << endl;
    }
  else
    cout << "double is not vectorizable" << endl;
  cout << "thread pool size: " << thread_pool_size() << endl;
  }

const char *print_diagnostics_DS = R"""(
Writes miscellaneous diagnostics (type sizes, SIMD lengths, alignments,
thread pool size) to standard output; useful for debugging in CI on exotic
hard- and software.
)""";

static py::tuple native_vector_lengths()
  {
  int vlen_f32=1, vlen_f64=1;
  if constexpr (vectorizable<float>)
    vlen_f32 = int(native_simd<float>::size());
  if constexpr (vectorizable<double>)
    vlen_f64 = int(native_simd<double>::size());
  return py::make_tuple(vlen_f32, vlen_f64);
  }

const char *native_vector_lengths_DS = R"""(
Returns the vector lengths for float32 and float64 supported by this ducc library.

Returns
-------
tuple(int) : supported vector lengths for float32 and float64, respectively
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
  using namespace py::literals;
  auto None = py::none();
  auto m = msup.def_submodule("misc");
  m.doc() = misc_DS;

  auto m2 = m.def_submodule("experimental");

  m.def("vdot", Py_vdot, Py_vdot_DS, "a"_a, "b"_a);
  m.def("vdot", Py_vdot_scalar, Py_vdot_DS, "a"_a, "b"_a);
  m.def("l2error",  Py_l2error, Py_l2error_DS, "a"_a, "b"_a);
  m.def("l2error",  Py_l2error_scalar, Py_l2error_DS, "a"_a, "b"_a);

  m.def("make_noncritical", Py_make_noncritical, Py_make_noncritical_DS,"in"_a, "nthreads"_a=1);
  m.def("empty_noncritical", Py_empty_noncritical, Py_empty_noncritical_DS, "shape"_a, "dtype"_a, "nthreads"_a=1);

  m.def("available_hardware_threads", available_hardware_threads, available_hardware_threads_DS);
  m.def("thread_pool_size", thread_pool_size, thread_pool_size_DS);
  m.def("resize_thread_pool", resize_thread_pool, resize_thread_pool_DS, "nthreads_new"_a);
  m.def("preallocate_memory", preallocate_memory, "gbytes"_a);
  m.def("set_heap_trim_limit", set_heap_trim_limit, "gbytes"_a);

  m.def("print_diagnostics", print_diagnostics, print_diagnostics_DS);
  m.def("native_vector_lengths", native_vector_lengths, native_vector_lengths_DS);
  }

}}
