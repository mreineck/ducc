From mav.h
==========

// idx: 1D mav-like, integer element type, axis length==in.shape(axis), values
//      in [0; out.shape(axis)[
//
// NOTE: "out" is NOT zeroed at the beginning!
template<typename T, typename I> void special_add_at
  (const cfmav<T> &in, size_t axis, const cfmav<I> &idx, vfmav<T> &out)
  {
  MR_assert(in.ndim()==out.ndim(), "dimension mismatch");
  MR_assert(in.ndim()>axis, "input array has too few dimensions");
  MR_assert(idx.size()==in.shape(axis), "idx size mismatch");
  auto idx1 = idx.extend_and_broadcast(in.shape(),axis);
  auto outstr1 = out.stride();
  outstr1[axis] = 0;
  auto axstr = out.stride(axis);
  auto out1 = vfmav<T>(out, in.shape(), outstr1);
  mav_apply([&](T vin, I idx, T &vout)
    {
    *(&vout+idx*axstr) += vin;
    }, 1, in, idx1, out1);
  }


From misc_pymod.cc
==================

template<typename T> py::array Py2_special_add_at(py::array &a_, size_t axis, py::array_t<int64_t> &index_, const py::array &b_)
  {
  auto a = to_vfmav<T>(a_);
  auto b = to_cfmav<T>(b_);
  auto index = to_cfmav<int64_t>(index_);
  special_add_at(b, axis, index, a);
  return a_;
  }
py::array Py_special_add_at(py::array &a, size_t axis, py::array_t<int64_t> &index, const py::array &b)
  {
  if (isPyarr<float>(a))
    return Py2_special_add_at<float>(a, axis, index, b);
  if (isPyarr<double>(a))
    return Py2_special_add_at<double>(a, axis, index, b);
  if (isPyarr<complex<float>>(a))
    return Py2_special_add_at<complex<float>>(a, axis, index, b);
  if (isPyarr<complex<double>>(a))
    return Py2_special_add_at<complex<double>>(a, axis, index, b);
  MR_fail("type matching failed");
  }

constexpr const char *Py_special_add_at_DS = R"""(
Co-add a multidimensional array along a given axis according to a given index.
Iterates over all entries of b and adds them to the specified entries of a,
such that
a[..., index[i], ...] += b[..., i, ...]


Parameters
----------
a : numpy.ndarray(float or complex type)
    the array to which the data is co-added
axis : int
    the number of the axis along which he addition takes place
index : numpy.ndarray(a((b.shape[axis]), dtype=numpy.int64)
    the index array.
    All values must lie in the range [0; a.shape[axis][
b : numpy.ndarray(same ndim and dtype as a, same shape as a except along axis)
    the array over which the addition is performed

Returns
-------
numpy.ndarray(identical to a):
    a plus the co-added values from b
)""";
  m.def("special_add_at", Py_special_add_at, Py_special_add_at_DS,
    "a"_a, "axis"_a, "index"_a, "b"_a);
