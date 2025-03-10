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

/* Copyright (C) 2025 Max-Planck-Society
   Author: Martin Reinecke */

// helper file for ducc0 FFT template instantiations

template void c2c(const cfmav<complex<T>> &in,
  const vfmav<complex<T>> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads);
template void dct(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads);
template void dst(const cfmav<T> &in, const vfmav<T> &out,
  const shape_t &axes, int type, T fct, bool ortho, size_t nthreads);
template void r2c(const cfmav<T> &in,
  const vfmav<complex<T>> &out, size_t axis, bool forward, T fct,
  size_t nthreads);
template void r2c(const cfmav<T> &in,
  const vfmav<complex<T>> &out, const shape_t &axes,
  bool forward, T fct, size_t nthreads);
template void c2r(const cfmav<complex<T>> &in,
  const vfmav<T> &out,  size_t axis, bool forward, T fct, size_t nthreads);
template void c2r(const cfmav<complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads);
template void c2r_mut(const vfmav<complex<T>> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward, T fct,
  size_t nthreads);
template void r2r_fftpack(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool real2hermitian, bool forward,
  T fct, size_t nthreads);
template void r2r_fftw(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, bool forward,
  T fct, size_t nthreads);
template void r2r_separable_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads);
template void r2r_separable_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads);
template void r2r_genuine_hartley(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads);
template void r2r_genuine_fht(const cfmav<T> &in,
  const vfmav<T> &out, const shape_t &axes, T fct, size_t nthreads);
template void convolve_axis(const cfmav<T> &in,
  const vfmav<T> &out, size_t axis, const cmav<T,1> &kernel, size_t nthreads);
template void convolve_axis(const cfmav<complex<T>> &in,
  const vfmav<complex<T>> &out, size_t axis, const cmav<complex<T>,1> &kernel,
  size_t nthreads);
