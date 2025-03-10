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

/* Copyright (C) 2019-2025 Max-Planck-Society
   Author: Martin Reinecke */

#include "ducc0/../../python/module_adders.h"
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/nufft/nufft.h"

namespace ducc0 {

namespace detail_pymodule_nufft {

using namespace std;

using Periodicity = variant<double, vector<double>>;

static vector<double> get_periodicity(const Periodicity &inp, size_t ndim)
  {
  try
    {
    auto val = get<double>(inp);
    vector<double> res;
    for (size_t i=0; i<ndim; ++i) res.push_back(val);
    return res;
    }
  catch(...)
    {}
  auto res = get<vector<double>>(inp);
  MR_assert(res.size()==ndim, "bad size of periodicity argument");
  return res;
  }

template<typename Tgrid, typename Tcoord> static NpArr Py2_u2nu(const CNpArr &grid_,
  const CNpArr &coord_, bool forward, double epsilon, size_t nthreads,
  OptNpArr &out__, size_t verbosity, double sigma_min, double sigma_max,
  const Periodicity &periodicity_, bool fft_order)
  {
  using Tpoints = Tgrid;
  auto coord = to_cmav<Tcoord,2>(coord_, "coord");
  auto periodicity = get_periodicity(periodicity_, coord.shape(1));
  size_t ndim = coord.shape(1);
  auto grid = to_cfmav_with_optional_leading_dimensions<complex<Tgrid>>(grid_,ndim+1);
  MR_assert((grid.ndim()==ndim)||(grid.ndim()==ndim+1), "bad dimensionality of grid");
  auto out_ = (ndim==size_t(grid_.ndim()))
            ? get_optional_Pyarr<complex<Tpoints>>(out__, {coord.shape(0)}, "out")
            : get_optional_Pyarr<complex<Tpoints>>(out__, {grid.shape(0), coord.shape(0)}, "out");
  auto out = to_vmav_with_optional_leading_dimensions<complex<Tpoints>,2>(out_, "out");
  {
  py::gil_scoped_release release;
  vector<slice> slices(grid.ndim(),slice());
  if (grid.shape(0)==1)
    {
    slices[0] = slice(0);
    u2nu<Tgrid,Tgrid>(coord,subarray(grid,slices),forward,epsilon,nthreads,
                      subarray<1>(out,{{0},{}}),verbosity,
                      sigma_min,sigma_max, periodicity, fft_order);
    }
  else
    {
    vector<size_t> shp(grid.shape().begin()+1,grid.shape().end());
    Nufft<Tgrid, Tgrid, Tcoord> nufft (false, coord, shp,
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    for (size_t i=0; i<grid.shape(0); ++i)
      {
      slices[0] = slice(i);
      nufft.u2nu(forward, verbosity, subarray(grid,slices), subarray<1>(out,{{i},{}}));
      }
    }
  }
  return out_;
  }
NpArr Py_u2nu(const CNpArr &grid,
  const CNpArr &coord, bool forward, double epsilon, size_t nthreads,
  OptNpArr &out, size_t verbosity, double sigma_min, double sigma_max,
  const Periodicity &periodicity, bool fft_order)
  {
  if (isPyarr<double>(coord))
    {
    if (isPyarr<complex<double>>(grid))
      return Py2_u2nu<double, double>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(grid))
      return Py2_u2nu<float, double>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (isPyarr<float>(coord))
    {
    if (isPyarr<complex<double>>(grid))
      return Py2_u2nu<double, float>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(grid))
      return Py2_u2nu<float, float>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  MR_fail("not yet supported");
  }

template<typename Tpoints, typename Tcoord> static NpArr Py2_nu2u(const CNpArr &points_,
  const CNpArr &coord_, bool forward, double epsilon, size_t nthreads,
  NpArr &out_, size_t verbosity, double sigma_min, double sigma_max,
  const Periodicity &periodicity_, bool fft_order)
  {
  using Tgrid = Tpoints;
  auto coord = to_cmav<Tcoord,2>(coord_, "coord");
  size_t ndim = coord.shape(1);
  auto points = to_cmav_with_optional_leading_dimensions<complex<Tpoints>,2>(points_, "points");
  auto out = to_vfmav_with_optional_leading_dimensions<complex<Tgrid>>(out_, ndim+1, "out");
  auto periodicity = get_periodicity(periodicity_, coord.shape(1));
  {
  py::gil_scoped_release release;
  vector<slice> slices(out.ndim(),slice());
  if (points.shape(0)==1)
    {
    slices[0] = slice(0);
    nu2u<Tgrid,Tgrid>(coord,subarray<1>(points, {{0},{}}),forward,epsilon,
                      nthreads,subarray(out,slices),verbosity,
                      sigma_min,sigma_max, periodicity, fft_order);
    }
  else
    {
    vector<size_t> shp(out.shape().begin()+1,out.shape().end());
    Nufft<Tgrid, Tgrid, Tcoord> nufft (true, coord, shp,
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    for (size_t i=0; i<points.shape(0); ++i)
      {
      slices[0] = slice(i);
      nufft.nu2u(forward, verbosity, subarray<1>(points,{{i},{}}), subarray(out,slices));
      }
    }
  }
  return out_;
  }
NpArr Py_nu2u(const CNpArr &points,
  const CNpArr &coord, bool forward, double epsilon, size_t nthreads,
  NpArr &out, size_t verbosity, double sigma_min, double sigma_max,
  const Periodicity &periodicity, bool fft_order)
  {
  if (isPyarr<double>(coord))
    {
    if (isPyarr<complex<double>>(points))
      return Py2_nu2u<double, double>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(points))
      return Py2_nu2u<float, double>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (isPyarr<float>(coord))
    {
    if (isPyarr<complex<double>>(points))
      return Py2_nu2u<double, float>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(points))
      return Py2_nu2u<float, float>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  MR_fail("not yet supported");
  }

template<typename Tpoints, typename Tcoord> static NpArr Py2_nu2nu(const CNpArr &points_in_,
  const CNpArr &coord_in_, const CNpArr &coord_out_, bool forward, double epsilon, size_t nthreads,
  OptNpArr &points_out__, size_t verbosity, double sigma_min, double sigma_max)
  {
  using Tgrid = Tpoints;
  auto coord_in = to_cmav<Tcoord,2>(coord_in_, "coord_in");
  auto coord_out = to_cmav<Tcoord,2>(coord_out_, "coord_out");
  auto points_in = to_cmav_with_optional_leading_dimensions<complex<Tpoints>,2>(points_in_, "points_in");
  auto points_out_ = (points_in_.ndim()==1)
    ? get_optional_Pyarr<complex<Tpoints>>(points_out__, {coord_out.shape(0)}, "points_out")
    : get_optional_Pyarr<complex<Tpoints>>(points_out__, {points_in.shape(0),coord_out.shape(0)}, "points_out");
  auto points_out = to_vmav_with_optional_leading_dimensions<complex<Tpoints>,2>(points_out_, "points_out");
  {
  py::gil_scoped_release release;
  if (points_in.shape(0)==1)
    nu2nu<Tgrid, Tgrid>(coord_in,subarray<1>(points_in,{{0},{}}),forward,
      epsilon,nthreads,coord_out,subarray<1>(points_out,{{0},{}}),verbosity,
      sigma_min,sigma_max);
  else
    {
    Nufft3<Tpoints, Tpoints, Tpoints, Tcoord> nufft(coord_in, epsilon, nthreads,
      coord_out, verbosity, sigma_min, sigma_max);
    for (size_t i=0; i<points_in.shape(0); ++i)
      nufft.exec(subarray<1>(points_in,{{i},{}}), subarray<1>(points_out,{{i},{}}), forward);
    }
  return points_out_;
  }
  }
NpArr Py_nu2nu(const CNpArr &points_in,
  const CNpArr &coord_in, const CNpArr &coord_out, bool forward,
  double epsilon, size_t nthreads,
  OptNpArr &points_out, size_t verbosity, double sigma_min, double sigma_max)
  {
  if (isPyarr<double>(coord_in))
    {
    if (isPyarr<complex<double>>(points_in))
      return Py2_nu2nu<double, double>(points_in, coord_in, coord_out, forward, epsilon, nthreads,
        points_out, verbosity, sigma_min, sigma_max);
    else if (isPyarr<complex<float>>(points_in))
      return Py2_nu2nu<float, double>(points_in, coord_in, coord_out, forward, epsilon, nthreads,
        points_out, verbosity, sigma_min, sigma_max);
    }
  else if (isPyarr<float>(coord_in))
    {
    if (isPyarr<complex<double>>(points_in))
      return Py2_nu2nu<double, float>(points_in, coord_in, coord_out, forward, epsilon, nthreads,
        points_out, verbosity, sigma_min, sigma_max);
    else if (isPyarr<complex<float>>(points_in))
      return Py2_nu2nu<float, float>(points_in, coord_in, coord_out, forward, epsilon, nthreads,
        points_out, verbosity, sigma_min, sigma_max);
    }
  MR_fail("not yet supported");
  }

class Py_Nufftplan
  {
  private:
    vector<size_t> uniform_shape;
    size_t npoints;

    unique_ptr<Nufft< float,  float,  float>> pf;
    unique_ptr<Nufft<double, double, double>> pd;

    template<typename T> void construct(
      unique_ptr<Nufft<T,T,T>> &ptr,
      bool gridding, const CNpArr &coord_,
      const vector<size_t> &uniform_shape_,
      double epsilon_,
      size_t nthreads_,
      double sigma_min, double sigma_max,
      const Periodicity &periodicity_, bool fft_order_)
      {
      auto coord = to_cmav<T,2>(coord_, "coord");
      auto shp = uniform_shape_;
      auto periodicity = get_periodicity(periodicity_, coord.shape(1));
      {
      py::gil_scoped_release release;
      ptr = make_unique<Nufft<T,T,T>> (gridding, coord, shp,
        epsilon_, nthreads_, sigma_min, sigma_max, periodicity, fft_order_);
      }
      }
    template<typename T> NpArr do_nu2u(
      const unique_ptr<Nufft<T,T,T>> &ptr,
      bool forward, size_t verbosity, const CNpArr &points_,
      OptNpArr &uniform__) const
      {
      auto points = to_cmav_with_optional_leading_dimensions<complex<T>,2>(points_, "points");
      vector<size_t> uni_shape;
      if (points_.ndim()==2)
        uni_shape.push_back(points.shape(0));
      for(auto v:uniform_shape) uni_shape.push_back(v);
      auto uniform_ = get_optional_Pyarr<complex<T>>(uniform__, uni_shape, "out");
      auto uniform = to_vfmav_with_optional_leading_dimensions<complex<T>>(uniform_,uniform_shape.size()+1, "out");
      {
      py::gil_scoped_release release;
      vector<slice> slices(uniform.ndim(), slice());
      for (size_t i=0; i<points.shape(0); ++i)
        {
        slices[0] = slice(i);
        ptr->nu2u(forward, verbosity, subarray<1>(points,{{i},{}}), subarray(uniform, slices));
        }
      }
      return uniform_;
      }
    template<typename T> NpArr do_u2nu(
      const unique_ptr<Nufft<T,T,T>> &ptr,
      bool forward, size_t verbosity, const CNpArr &uniform_,
      OptNpArr &points__) const
      {
      auto uniform = to_cfmav_with_optional_leading_dimensions<complex<T>>(uniform_, uniform_shape.size()+1, "grid");
      auto points_ = (size_t(uniform_.ndim())==uniform_shape.size())
        ? get_optional_Pyarr<complex<T>>(points__, {npoints}, "out")
        : get_optional_Pyarr<complex<T>>(points__, {uniform.shape(0), npoints}, "out");
      auto points = to_vmav_with_optional_leading_dimensions<complex<T>,2>(points_, "out");
      {
      py::gil_scoped_release release;
      vector<slice> slices(uniform.ndim(), slice());
      for (size_t i=0; i<points.shape(0); ++i)
        {
        slices[0] = slice(i);
        ptr->u2nu(forward, verbosity, subarray(uniform, slices), subarray<1>(points,{{i},{}}));
        }
      }
      return points_;
      }

  public:
    Py_Nufftplan(bool gridding, const CNpArr &coord_,
                 const vector<size_t> &uniform_shape_,
                 double epsilon_,
                 size_t nthreads_,
                 double sigma_min, double sigma_max,
                 const Periodicity &periodicity, bool fft_order_)
      : uniform_shape(uniform_shape_),
        npoints(coord_.shape(0))
      {
      auto ndim = uniform_shape.size();
      MR_assert((ndim>=1)&&(ndim<=3), "unsupported dimensionality");
      if (isPyarr<double>(coord_))
        construct(pd, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
                  sigma_min, sigma_max, periodicity, fft_order_);
      else if (isPyarr<float>(coord_))
        construct(pf, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
            sigma_min, sigma_max, periodicity, fft_order_);
      else
        MR_fail("unsupported");
      }

    NpArr nu2u(bool forward, size_t verbosity,
      const CNpArr &points_, OptNpArr &uniform_)
      {
      if (pd) return do_nu2u(pd, forward, verbosity, points_, uniform_);
      if (pf) return do_nu2u(pf, forward, verbosity, points_, uniform_);
      MR_fail("unsupported");
      }
    NpArr u2nu(bool forward, size_t verbosity,
      const CNpArr &uniform_, OptNpArr &points_)
      {
      if (pd) return do_u2nu(pd, forward, verbosity, uniform_, points_);
      if (pf) return do_u2nu(pf, forward, verbosity, uniform_, points_);
      MR_fail("unsupported");
      }
  };
class Py_incremental_nu2u
  {
  private:
    vector<size_t> uniform_shape;
    vfmav<complex< float>> gridf;
    vfmav<complex<double>> gridd;
    size_t nthreads;
    bool forward;

    unique_ptr<Nufft< float,  float,  float>> pf;
    unique_ptr<Nufft<double, double, double>> pd;

    template<typename T> void construct(
      unique_ptr<Nufft<T,T,T>> &ptr,
      vfmav<complex<T>> &grid,
      size_t npoints_estimate,
      double epsilon,
      double sigma_min, double sigma_max,
      const Periodicity &periodicity_, bool fft_order)
      {
      auto periodicity = get_periodicity(periodicity_, uniform_shape.size());
      {
      py::gil_scoped_release release;
      ptr = make_unique<Nufft<T,T,T>> (true, npoints_estimate, uniform_shape,
        epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      grid.assign(vfmav<complex<T>>(ptr->get_gridsize()));
      }
      }
    template<typename T> void do_add_points(
      const unique_ptr<Nufft<T,T,T>> &ptr,
      const CNpArr &coord_, const CNpArr &values_,
      vfmav<complex<T>> &grid)
      {
      auto coord = to_cmav<T,2>(coord_, "coord");
      auto values = to_cmav<complex<T>,1>(values_, "points");
      {
      py::gil_scoped_release release;
      ptr->spread(coord, values, grid);
      }
      }
    template<typename T> NpArr do_evaluate_and_reset(
      const unique_ptr<Nufft<T,T,T>> &ptr,
      vfmav<complex<T>> &grid,
      OptNpArr &uniform__)
      {
      auto uniform_ = get_optional_Pyarr<complex<T>>(uniform__, uniform_shape, "uniform");
      auto uniform = to_vfmav<complex<T>>(uniform_, "uniform");
      {
      py::gil_scoped_release release;
      ptr->spread_finish(forward, grid, uniform);
      mav_apply([](auto &v){v=0;}, nthreads, grid);
      }
      return uniform_;
      }

  public:
    Py_incremental_nu2u(size_t npoints_estimate,
                 const vector<size_t> &uniform_shape_,
                 bool forward_,
                 double epsilon,
                 size_t nthreads_,
                 double sigma_min, double sigma_max,
                 const Periodicity &periodicity, bool fft_order, bool singleprec)
      : uniform_shape(uniform_shape_),
        nthreads(nthreads_),
        forward(forward_)
      {
      auto ndim = uniform_shape.size();
      MR_assert((ndim>=1)&&(ndim<=3), "unsupported dimensionality");
      if (!singleprec)
        construct(pd, gridd, npoints_estimate, epsilon,
                  sigma_min, sigma_max, periodicity, fft_order);
      else
        construct(pf, gridf, npoints_estimate, epsilon,
                  sigma_min, sigma_max, periodicity, fft_order);
      }

    void add_points(const CNpArr &coord, const CNpArr &values)
      {
      if (pd) return do_add_points(pd, coord, values, gridd);
      if (pf) return do_add_points(pf, coord, values, gridf);
      MR_fail("unsupported");
      }
    NpArr evaluate_and_reset(OptNpArr &uniform)
      {
      if (pd) return do_evaluate_and_reset(pd, gridd, uniform);
      if (pf) return do_evaluate_and_reset(pf, gridf, uniform);
      MR_fail("unsupported");
      }
  };
class Py_incremental_u2nu
  {
  private:
    vector<size_t> uniform_shape;
    vfmav<complex< float>> gridf;
    vfmav<complex<double>> gridd;
    size_t nthreads;

    unique_ptr<Nufft< float,  float,  float>> pf;
    unique_ptr<Nufft<double, double, double>> pd;

    template<typename T> void construct(
      unique_ptr<Nufft<T,T,T>> &ptr,
      vfmav<complex<T>> &grid,
      size_t npoints_estimate,
      const CNpArr &uniform_,
      bool forward,
      double epsilon,
      double sigma_min, double sigma_max,
      const Periodicity &periodicity_, bool fft_order)
      {
      auto uniform = to_cfmav<complex<T>>(uniform_, "grid");
      auto shp = uniform.shape();
      auto periodicity = get_periodicity(periodicity_, shp.size());
      {
      py::gil_scoped_release release;
      ptr = make_unique<Nufft<T,T,T>> (true, npoints_estimate, shp,
        epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      grid.assign(vfmav<complex<T>>(ptr->get_gridsize()));
      ptr->interp_prep(forward, grid, uniform);
      }
      }
    template<typename T> NpArr do_get_points(
      const unique_ptr<Nufft<T,T,T>> &ptr,
      const CNpArr &coord_, OptNpArr &values__,
      const cfmav<complex<T>> &grid) const
      {
      auto coord = to_cmav<T,2>(coord_, "coord");
      auto values_ = get_optional_Pyarr<complex<T>>(values__, {coord.shape(0)}, "points");
      auto values = to_vmav<complex<T>,1>(values_, "points");
      {
      py::gil_scoped_release release;
      ptr->interp(coord, values, grid);
      }
      return values_;
      }

  public:
    Py_incremental_u2nu(size_t npoints_estimate,
                 const CNpArr &uniform,
                 bool forward,
                 double epsilon,
                 size_t nthreads_,
                 double sigma_min, double sigma_max,
                 const Periodicity &periodicity, bool fft_order_)
      : nthreads(nthreads_)
      {
      auto ndim = uniform.ndim();
      MR_assert((ndim>=1)&&(ndim<=3), "unsupported dimensionality");
      if (isPyarr<complex<double>>(uniform))
        construct(pd, gridd, npoints_estimate, uniform, forward, epsilon,
                  sigma_min, sigma_max, periodicity, fft_order_);
      else
        construct(pf, gridf, npoints_estimate, uniform, forward, epsilon,
                  sigma_min, sigma_max, periodicity, fft_order_);
      }

    NpArr get_points(const CNpArr &coord, OptNpArr &values) const
      {
      if (pd) return do_get_points(pd, coord, values, gridd);
      if (pf) return do_get_points(pf, coord, values, gridf);
      MR_fail("unsupported");
      }
  };

class Py_Nufft3plan
  {
  private:
    unique_ptr<Nufft3< float,  float,  float,  float>> pf;
    unique_ptr<Nufft3<double, double, double, double>> pd;
    size_t npoints_in, npoints_out;

    template<typename T> void construct(
      unique_ptr<Nufft3<T,T,T,T>> &ptr,
      const CNpArr &coord_in_,
      const CNpArr &coord_out_,
      double epsilon,
      size_t nthreads,
      double sigma_min, double sigma_max,
      size_t verbosity)
      {
      auto coord_in = to_cmav<T,2>(coord_in_, "coord_in");
      npoints_in = coord_in.shape(0);
      auto coord_out = to_cmav<T,2>(coord_out_, "coord_out");
      npoints_out = coord_out.shape(0);
      {
      py::gil_scoped_release release;
      ptr = make_unique<Nufft3<T,T,T,T>> (coord_in, epsilon, nthreads,
        coord_out, verbosity, sigma_min, sigma_max);
      }
      }
    template<typename T> NpArr do_exec(
      const unique_ptr<Nufft3<T,T,T,T>> &ptr,
      bool forward, const CNpArr &points_in_,
      OptNpArr &points_out__) const
      {
      auto points_in = to_cmav_with_optional_leading_dimensions<complex<T>,2>(points_in_, "points_in");
      auto points_out_ = (points_in_.ndim()==1)
        ? get_optional_Pyarr<complex<T>>(points_out__, {npoints_out}, "points_out")
        : get_optional_Pyarr<complex<T>>(points_out__, {points_in.shape(0),npoints_out}, "points_out");
      auto points_out = to_vmav_with_optional_leading_dimensions<complex<T>,2>(points_out_, "points_out");
      {
      py::gil_scoped_release release;
      for (size_t i=0; i<points_in.shape(0); ++i)
        ptr->exec(subarray<1>(points_in,{{i},{}}), subarray<1>(points_out,{{i},{}}), forward);
      }
      return points_out_;
      }
    template<typename T> NpArr do_exec_adjoint(
      const unique_ptr<Nufft3<T,T,T,T>> &ptr,
      bool forward, const CNpArr &points_in_,
      OptNpArr &points_out__) const
      {
      auto points_in = to_cmav_with_optional_leading_dimensions<complex<T>,2>(points_in_, "points_in");
      auto points_out_ = (points_in_.ndim()==1)
        ? get_optional_Pyarr<complex<T>>(points_out__, {npoints_in}, "points_out")
        : get_optional_Pyarr<complex<T>>(points_out__, {points_in.shape(0),npoints_in}, "points_out");
      auto points_out = to_vmav_with_optional_leading_dimensions<complex<T>,2>(points_out_);
      {
      py::gil_scoped_release release;
      for (size_t i=0; i<points_in.shape(0); ++i)
        ptr->exec_adjoint(subarray<1>(points_in,{{i},{}}), subarray<1>(points_out,{{i},{}}), forward);
      }
      return points_out_;
      }

  public:
    Py_Nufft3plan(const CNpArr &coord_in,
                  const CNpArr &coord_out,
                  double epsilon,
                  size_t nthreads,
                  double sigma_min, double sigma_max,
                  size_t verbosity)
      {
      if (isPyarr<double>(coord_in))
        construct(pd, coord_in, coord_out, epsilon, nthreads,
                  sigma_min, sigma_max, verbosity);
      else if (isPyarr<float>(coord_in))
        construct(pf, coord_in, coord_out, epsilon, nthreads,
                  sigma_min, sigma_max, verbosity);
      else
        MR_fail("unsupported");
      }

    NpArr exec(bool forward,
      const CNpArr &points_in, OptNpArr &points_out)
      {
      if (pd) return do_exec(pd, forward, points_in, points_out);
      if (pf) return do_exec(pf, forward, points_in, points_out);
      MR_fail("unsupported");
      }
    NpArr exec_adjoint(bool forward,
      const CNpArr &points_in, OptNpArr &points_out)
      {
      if (pd) return do_exec_adjoint(pd, forward, points_in, points_out);
      if (pf) return do_exec_adjoint(pf, forward, points_in, points_out);
      MR_fail("unsupported");
      }
  };


constexpr const char *u2nu_DS = R"""(
Type 2 non-uniform FFT (uniform to non-uniform)

Parameters
----------
grid : numpy.ndarray(([ntrans], nx, [ny, [nz]]), dtype=complex)
    the grid(s) of input data
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the npoints non-uniform points.
    ndim must be 1, 2, or 3 and match the shape of `grid`
    Periodicity is assumed; the coordinates don't have to lie inside a
    particular interval, but smaller absolute coordinate values help accuracy
forward : bool
    if True, perform the FFT with exponent -1, else +1.
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
out : numpy.ndarray(([ntrans], npoints,), same data type as grid), optional
    if provided, this will be used to store the result
verbosity: int
    0: no console output
    1: some diagnostic console output
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
periodicity: float or sequence of floats
    periodicity of the coordinates
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode

Returns
-------
numpy.ndarray(([ntrans], npoints,), same data type as grid)
    the computed values at the specified non-uniform grid points.
    Identical to `out` if it was provided
)""";

constexpr const char *nu2u_DS = R"""(
Type 1 non-uniform FFT (non-uniform to uniform)

Parameters
----------
points : numpy.ndarray(([ntrans], npoints,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the npoints non-uniform points.
    ndim must be the same as out.ndim
    Periodicity is assumed; the coordinates don't have to lie inside a
    particular interval, but smaller absolute coordinate values help accuracy
forward : bool
    if True, perform the FFT with exponent -1, else +1.
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
out : numpy.ndarray(([ntrans], nx, [ny, [nz]]), same dtype as points)
    the grid(s) of output data
    Note: this is a mandatory parameter, since its shape defines the grid dimensions!
verbosity: int
    0: no console output
    1: some diagnostic console output
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
periodicity: float or sequence of floats
    periodicity of the coordinates
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode

Returns
-------
numpy.ndarray(([ntrans], nx, [ny, [nz]]), same dtype as points)
    the computed grid values.
    Identical to `out`.
)""";

constexpr const char *plan_init_DS = R"""(
Nufft plan constructor

Parameters
----------
nu2u : bool
    True: plan will be used for nu2u transforms
    False: plan will be used for u2nu transforms
    The resulting plan can actually be used for both transform types, but
    optimization will be better for the requested type.
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the npoints non-uniform points.
    Periodicity is assumed; the coordinates don't have to lie inside a
    particular interval, but smaller absolute coordinate values help accuracy
grid_shape : tuple(int) of length ndim
    the shape of the uniform grid
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
periodicity: float or sequence of floats
    periodicity of the coordinates
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode
)""";

constexpr const char *plan_nu2u_DS = R"""(
Perform a pre-planned nu2u transform.

Parameters
----------
forward : bool
    if True, perform the FFT with exponent -1, else +1.
verbosity: int
    0: no console output
    1: some diagnostic console output
points : numpy.ndarray(([ntrans], npoints,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
out : numpy.ndarray(([ntrans], nx, [ny, [nz]]), same dtype as points)
    if provided, this will be used to store he result.

Returns
-------
numpy.ndarray(([ntrans], nx, [ny, [nz]]), same dtype as points)
    the computed grid values.
    Identical to `out` if it was provided.
)""";

constexpr const char *plan_u2nu_DS = R"""(
Perform a pre-planned u2nu transform.

Parameters
----------
forward : bool
    if True, perform the FFT with exponent -1, else +1.
verbosity: int
    0: no console output
    1: some diagnostic console output
grid : numpy.ndarray(([ntrans], nx, [ny, [nz]]), dtype=complex)
    the grid of input data
out : numpy.ndarray(([ntrans], npoints,), same data type as grid), optional
    if provided, this will be used to store the result

Returns
-------
numpy.ndarray(([ntrans], npoints,), same data type as grid)
    the computed values at the specified non-uniform grid points.
    Identical to `out` if it was provided.
)""";

constexpr const char *bestEpsilon_DS = R"""(
Computes the smallest possible error for the given NUFFT parameters.

Parameters
----------
ndim : int (1-3)
    the dimensionality of the transform
singleprec : bool
    True if np.float32/np.complex64 are used, otherwise False
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5

Returns
-------
float
    the smallest possible error that can be achieved for the given parameters.
)""";


constexpr const char *nu2nu_DS = R"""(
Type 3 non-uniform FFT (non-uniform to non-uniform)

Parameters
----------
points_in : numpy.ndarray(([ntrans], npoints_in,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
coord_in : numpy.ndarray((npoints_in, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the non-uniform input points.
coord_out : numpy.ndarray((npoints_out, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the non-uniform output points.
forward : bool
    if True, perform the FFT with exponent -1, else +1.
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
points_out : numpy.ndarray(([ntrans], npoints_out,), dtype=numpy.complex), optional
    The output values at the specified non-uniform grid points
verbosity: int
    0: no console output
    1: some diagnostic console output
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5

Returns
-------
numpy.ndarray(([ntrans], npoints_out,), same dtype as points_in)
    the computed grid values.
    Identical to `points_out`, if it was provided.
)""";

constexpr const char *plan3_init_DS = R"""(
Nufft3 plan constructor

Parameters
----------
coord_in : numpy.ndarray((npoints_in, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the non-uniform input points.
coord_out : numpy.ndarray((npoints_out, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the non-uniform output points.
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
)""";

constexpr const char *plan3_exec_DS = R"""(
Perform a pre-planned nu2nu transform.

Parameters
----------
forward : bool
    if True, perform the FFT with exponent -1, else +1.
points_in : numpy.ndarray(([ntrans], npoints_in,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
points_out : numpy.ndarray(([ntrans], npoints_out,), dtype=numpy.complex), optional
    The output values at the specified non-uniform grid points.
    if provided, this will be used to store he result.

Returns
-------
numpy.ndarray(([ntrans], npoints_out,), same dtype as points_in)
    the computed nonuniform values.
    Identical to `points_out` if it was provided.
)""";
constexpr const char *plan3_exec_adjoint_DS = R"""(
Perform the adjoint operation of `exec`.

Parameters
----------
forward : bool
    must be the same value as in the corresponding `exec` call to obtain
    the adjoint operation
points_in : numpy.ndarray(([ntrans], npoints_out,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
points_out : numpy.ndarray(([ntrans], npoints_in,), dtype=numpy.complex), optional
    The output values at the specified non-uniform grid points.
    if provided, this will be used to store he result.

Returns
-------
numpy.ndarray(([ntrans], npoints_in,), same dtype as points_in)
    the computed nonuniform values.
    Identical to `points_out` if it was provided.
)""";

constexpr const char *incremental_nu2u_init_DS = R"""(
Incremental nu2u constructor

Parameters
----------
npoints_estimate : int
    estimated total number of nonuniform points
    This is only used for performance optimization; an order-of-magnitude guess
    should be fine, the default should also be OK in most situations
grid_shape : tuple(int) of length ndim
    the shape of the uniform grid
forward : bool
    if True, perform the FFT with exponent -1, else +1.
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
periodicity: float or sequence of floats
    periodicity of the coordinates
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode
singleprec : bool
    True if np.float32/np.complex64 are used, otherwise False
    All variable dtypes in member functions must be consistent with this.
)""";
constexpr const char *incremental_nu2u_add_points_DS = R"""(
Adds nonunifom points to the transform

Parameters
----------
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the added non-uniform points.
points : numpy.ndarray(npoints, dtype=numpy.complex64 or numpy.complex128)
    The input values at the specified non-uniform grid points
)""";
constexpr const char *incremental_nu2u_evaluate_and_reset_DS = R"""(
Finishes the transform and resets it to empty

Parameters
----------
uniform : numpy.ndarray(uniform_shape), dtype=numpy.complex64 or numpy.complex128)
    if provided, this will be used to store he result.

Returns
-------
numpy.ndarray(uniform_shape), dtype=numpy.complex64 or numpy.complex128)
    The result of the transform
)""";

constexpr const char *incremental_u2nu_init_DS = R"""(
Incremental u2nu constructor

Parameters
----------
npoints_estimate : int
    estimated total number of nonuniform points
    This is only used for performance optimization; an order-of-magnitude guess
    should be fine, the default should also be OK in most situations
grid: numpy.ndarray((nx, [ny, [nz]]), dtype=numpy.complex64 or numpy.complex128)
    the grid of input data
    All variable dtypes in member functions must be consistent with the dtype
    of `grid`.
forward : bool
    if True, perform the FFT with exponent -1, else +1.
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
periodicity: float or sequence of floats
    periodicity of the coordinates
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode
)""";
constexpr const char *incremental_u2nu_get_points_DS = R"""(
Returns the result of the transfom at the specified nonunifom points.

Parameters
----------
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the added non-uniform points.
points : numpy.ndarray(npoints, dtype=numpy.complex64 or numpy.complex128)
    if provided, this will be used to store he result.
    The input values at the specified non-uniform grid points

Returns
-------
numpy.ndarray(npoints, dtype=numpy.complex64 or numpy.complex128)
    The result of the transform at the specified non-uniform grid points
)""";

void add_nufft(py::module_ &msup)
  {
  using namespace py::literals;
  auto m = msup.def_submodule("nufft");
  auto m2 = m.def_submodule("experimental");

  m.def("u2nu", &Py_u2nu, u2nu_DS,  py::kw_only(), "grid"_a, "coord"_a,
        "forward"_a, "epsilon"_a, "nthreads"_a=1, "out"_a=None, "verbosity"_a=0,
        "sigma_min"_a=1.2, "sigma_max"_a=2.51, "periodicity"_a=2*pi,
        "fft_order"_a=false);
  m.def("nu2u", &Py_nu2u, nu2u_DS, py::kw_only(), "points"_a, "coord"_a,
        "forward"_a, "epsilon"_a, "nthreads"_a=1, "out"_a=None, "verbosity"_a=0,
        "sigma_min"_a=1.2, "sigma_max"_a=2.51, "periodicity"_a=2*pi,
        "fft_order"_a=false);
  m2.def("nu2nu", &Py_nu2nu, nu2nu_DS, py::kw_only(), "points_in"_a, "coord_in"_a,
        "coord_out"_a, "forward"_a, "epsilon"_a, "nthreads"_a=1,
        "points_out"_a=None, "verbosity"_a=0, "sigma_min"_a=1.2, "sigma_max"_a=2.51);
  m.def("bestEpsilon", &bestEpsilon, bestEpsilon_DS, py::kw_only(),
        "ndim"_a, "singleprec"_a, "sigma_min"_a=1.1, "sigma_max"_a=2.6);

  py::class_<Py_Nufftplan> (m, "plan", /*py::module_local(),*/
                            "Class for repeated execution of type 1/2 NUFFTs")
    .def(py::init<bool, const CNpArr &, const vector<size_t> &,
                  double, size_t, double, double, const Periodicity &, bool>(),
      plan_init_DS, py::kw_only(), "nu2u"_a, "coord"_a, "grid_shape"_a,
        "epsilon"_a, "nthreads"_a=0, "sigma_min"_a=1.1, "sigma_max"_a=2.6,
        "periodicity"_a=2*pi, "fft_order"_a=false)
    .def("nu2u", &Py_Nufftplan::nu2u, plan_nu2u_DS, py::kw_only(), "forward"_a,
      "verbosity"_a=0, "points"_a, "out"_a=None)
    .def("u2nu", &Py_Nufftplan::u2nu, plan_u2nu_DS, py::kw_only(), "forward"_a,
      "verbosity"_a=0, "grid"_a, "out"_a=None);

  py::class_<Py_incremental_nu2u> (m2, "incremental_nu2u", /*py::module_local(),*/
                                   "Class for incremental execution of a type 1 NUFFT")
    .def(py::init<size_t, const vector<size_t> &, bool,
                  double, size_t, double, double, const Periodicity &, bool, bool>(),
      incremental_nu2u_init_DS,
      py::kw_only(), "npoints_estimate"_a=1000000000, "grid_shape"_a, "forward"_a,
        "epsilon"_a, "nthreads"_a=0, "sigma_min"_a=1.1, "sigma_max"_a=2.6,
        "periodicity"_a=2*pi, "fft_order"_a=false, "singleprec"_a=false)
    .def("add_points", &Py_incremental_nu2u::add_points,
      incremental_nu2u_add_points_DS, py::kw_only(), "coord"_a, "points"_a)
    .def("evaluate_and_reset", &Py_incremental_nu2u::evaluate_and_reset,
      incremental_nu2u_evaluate_and_reset_DS, py::kw_only(), "uniform"_a=None);

  py::class_<Py_incremental_u2nu> (m2, "incremental_u2nu", /*py::module_local(),*/
                                   "Class for incremental execution of a type 2 NUFFT")
    .def(py::init<size_t, const CNpArr &, bool,
                  double, size_t, double, double, const Periodicity &, bool>(),
      incremental_u2nu_init_DS,
      py::kw_only(), "npoints_estimate"_a=1000000000, "grid"_a, "forward"_a,
        "epsilon"_a, "nthreads"_a=0, "sigma_min"_a=1.1, "sigma_max"_a=2.6,
        "periodicity"_a=2*pi, "fft_order"_a=false)
    .def("get_points", &Py_incremental_u2nu::get_points,
      incremental_u2nu_get_points_DS, py::kw_only(),
      "coord"_a, "points"_a=None);

  py::class_<Py_Nufft3plan> (m2, "plan3", /*py::module_local(),*/
                             "Class for repeated execution of type 3 NUFFTs")
    .def(py::init<const CNpArr &, const CNpArr &,
                  double, size_t, double, double, size_t>(),
      plan3_init_DS, py::kw_only(), "coord_in"_a, "coord_out"_a,
        "epsilon"_a, "nthreads"_a=0, "sigma_min"_a=1.1, "sigma_max"_a=2.6,
        "verbosity"_a=0)
    .def("exec", &Py_Nufft3plan::exec, plan3_exec_DS, py::kw_only(), "forward"_a,
      "points_in"_a, "points_out"_a=None)
    .def("exec_adjoint", &Py_Nufft3plan::exec_adjoint, plan3_exec_adjoint_DS, py::kw_only(), "forward"_a,
      "points_in"_a, "points_out"_a=None);
  }

}

using detail_pymodule_nufft::add_nufft;

}
