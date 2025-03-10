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

#ifndef DUCC0_NUFFT_SPREADINTERP_IMPL_H
#define DUCC0_NUFFT_SPREADINTERP_IMPL_H

#include <algorithm>
#include "ducc0/infra/simd.h"
#include "ducc0/infra/bucket_sort.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/nufft/spreadinterp.h"
#include "ducc0/nufft/nufft_common.h"

namespace ducc0 {

namespace detail_nufft {

using namespace std;

template<typename T> complex<T> hsum_cmplx(mysimd<T> vr, mysimd<T> vi)
  { return complex<T>(reduce(vr, plus<>()), reduce(vi, plus<>())); }

#if (!defined(DUCC0_NO_SIMD))
#if (!defined(__AVX512F__))
#if (defined(__AVX__))
static_assert(mysimd<float>::size()==8, "must not happen");
#if 1
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm256_hadd_ps(__m256(vr), __m256(vi));
  auto t2 = _mm_hadd_ps(_mm256_extractf128_ps(t1, 0), _mm256_extractf128_ps(t1, 1));
  t2 += _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,0,3,2));
  return complex<float>(t2[0], t2[1]);
  }
#else
// this version may be slightly faster, but this needs more benchmarking
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm256_shuffle_ps(vr, vi, _MM_SHUFFLE(0,2,0,2));
  auto t2 = _mm256_shuffle_ps(vr, vi, _MM_SHUFFLE(1,3,1,3));
  auto t3 = _mm256_add_ps(t1,t2);
  t3 = _mm256_shuffle_ps(t3, t3, _MM_SHUFFLE(3,0,2,1));
  auto t4 = _mm_add_ps(_mm256_extractf128_ps(t3, 1), _mm256_castps256_ps128(t3));
  auto t5 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
  return complex<float>(t5[0], t5[1]);
  }
#endif
#elif defined(__SSE3__)
static_assert(mysimd<float>::size()==4, "must not happen");
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm_hadd_ps(__m128(vr), __m128(vi));
  t1 += _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2,3,0,1));
  return complex<float>(t1[0], t1[2]);
  }
#endif
#endif
#endif


template<typename Tacc, size_t ndim> constexpr inline int log2tile_=-1;
template<> constexpr inline int log2tile_<long double, 1> = 9;
template<> constexpr inline int log2tile_<double, 1> = 9;
template<> constexpr inline int log2tile_<float , 1> = 9;
template<> constexpr inline int log2tile_<long double, 2> = 4;
template<> constexpr inline int log2tile_<double, 2> = 4;
template<> constexpr inline int log2tile_<float , 2> = 5;
#ifdef NEW_DUMP
template<> constexpr inline int log2tile_<double, 3> = 5;
template<> constexpr inline int log2tile_<float , 3> = 5;
#else
template<> constexpr inline int log2tile_<double, 3> = 4;
template<> constexpr inline int log2tile_<float , 3> = 4;
#endif
template<> constexpr inline int log2tile_<long double, 3> = 4;

template<size_t ndim> constexpr inline size_t max_ntile=-1;
template<> constexpr inline size_t max_ntile<1> = (~uint32_t(0))-10;
template<> constexpr inline size_t max_ntile<2> = (uint32_t(1<<16))-10;
template<> constexpr inline size_t max_ntile<3> = (uint32_t(1<<10))-10;

template<typename Tcalc, typename Tacc, typename Tidx, size_t ndim> class Spreadinterp_ancestor
  {
  protected:
    // number of threads to use for this transform.
    size_t nthreads;

    // 1./<periodicity of coordinates>
    array<double, ndim> coordfct;

    // oversampled grid dimensions
    array<size_t, ndim> nover;

    // holds the indices of the nonuniform points in the order in which they
    // should be processed
    quick_array<Tidx> coord_idx;

    shared_ptr<PolynomialKernel> krn;

    size_t supp, nsafe;
    array<double, ndim> shift;
    // Origin of the irregular coords. This is only different from 0 when used
    // inside a type 3 transform.
    array<double, ndim> corigin;

    array<int64_t, ndim> maxi0;

    // the base-2 logarithm of the linear dimension of a computational tile.
    constexpr static int log2tile = log2tile_<Tacc,ndim>;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc),
      "Tacc must be at least as accurate as Tcalc");

    /*! Compute minimum index in the oversampled grid touched by the kernel
        around coordinate \a in. */
    template<typename Tcoord> [[gnu::always_inline]] void getpix(array<double,ndim> in,
      array<double,ndim> &out, array<int64_t,ndim> &out0) const
      {
      // do range reduction in long double when Tcoord is double,
      // to avoid inaccuracies with very large grids
      using Tbig = typename conditional<is_same<Tcoord,double>::value, long double, double>::type;
      for (size_t i=0; i<ndim; ++i)
        {
        auto tmp = (in[i]-corigin[i])*coordfct[i];
        auto tmp2 = Tbig(tmp-floor(tmp))*nover[i];
        out0[i] = min(int64_t(tmp2+shift[i])-int64_t(nover[i]), maxi0[i]);
        out[i] = double(tmp2-out0[i]);
        }
      }

    /*! Compute index of the tile into which \a in falls. */
    template<typename Tcoord> [[gnu::always_inline]] array<Tidx,ndim> get_tile(const array<double,ndim> &in) const
      {
      array<double,ndim> dum;
      array<int64_t,ndim> i0;
      getpix<Tcoord>(in, dum, i0);
      array<Tidx,ndim> res;
      for (size_t i=0; i<ndim; ++i)
        res[i] = Tidx((i0[i]+nsafe)>>log2tile);
      return res;
      }
    template<typename Tcoord> [[gnu::always_inline]] array<Tidx,ndim> get_tile(const array<double,ndim> &in, size_t lsq2) const
      {
      array<double,ndim> dum;
      array<int64_t,ndim> i0;
      getpix<Tcoord>(in, dum, i0);
      array<Tidx,ndim> res;
      for (size_t i=0; i<ndim; ++i)
        res[i] = Tidx((i0[i]+nsafe)>>lsq2);
      return res;
      }

    template<typename Tcoord> void sort_coords(const cmav<Tcoord,2> &coords,
      const vmav<Tcoord,2> &coords_sorted)
      {
      execParallel(coords.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          for (size_t d=0; d<ndim; ++d)
            coords_sorted(i,d) = coords(coord_idx[i],d);
        });
      }

    static array<double, ndim> get_coordfct(const vector<double> &periodicity)
      {
      MR_assert(periodicity.size()==ndim, "periodicity size mismatch");
      array<double, ndim> res;
      for (size_t i=0; i<ndim; ++i)
        res[i] = 1./periodicity[i];
      return res;
      }

  public:
    Spreadinterp_ancestor(size_t npoints,
      const array<size_t,ndim> &over_shape, size_t kidx,
      size_t nthreads_,
      const vector<double> &periodicity,
      const vector<double> &corigin_)
      : nthreads(adjust_nthreads(nthreads_)), coordfct(get_coordfct(periodicity)),
        nover(over_shape)
      {
//FIXME
      MR_assert(npoints<=(~Tidx(0)), "too many nonuniform points");

      for (size_t i=0; i<ndim; ++i)
        MR_assert((nover[i]>>log2tile)<=max_ntile<ndim>, "oversampled grid too large");

      krn = selectKernel(kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;

      if (corigin_.empty())
        fill(corigin.begin(), corigin.end(), 0.);
      else
        {
        MR_assert(corigin_.size()==ndim, "bad corigin size");
        for (size_t i=0; i<ndim; ++i)
          corigin[i] = corigin_[i];
        }
      for (size_t i=0; i<ndim; ++i)
        {
        shift[i] = supp*(-0.5)+1+nover[i];
        maxi0[i] = (nover[i]+nsafe)-supp;
        MR_assert(nover[i]>=2*nsafe, "oversampled length too small");
        MR_assert((nover[i]&1)==0, "oversampled dimensions must be even");
        }
      }
  };

#define DUCC0_SPREADINTERP_BOILERPLATE \
  private: \
    using parent=Spreadinterp_ancestor<Tcalc, Tacc, Tidx, ndim>; \
    using parent::coord_idx, parent::nthreads, parent::supp, \
          parent::krn, \
          parent::nover, parent::shift, parent::corigin, parent::maxi0, \
          parent::log2tile, parent::sort_coords; \
 \
    vmav<Tcoord,2> coords_sorted; \
 \
  public: \
    using parent::parent; /* inherit constructor */ \
    Spreadinterp(const cmav<Tcoord,2> &coords, \
          const array<size_t, ndim> &over_shape_, size_t kidx,  \
          size_t nthreads_, const vector<double> &periodicity, \
          const vector<double> &corigin_=vector<double>()) \
      : parent(coords.shape(0), over_shape_, kidx, nthreads_, \
               periodicity, corigin_), \
        coords_sorted({coords.shape(0),ndim},UNINITIALIZED) \
      { \
      build_index(coords); \
      sort_coords(coords, coords_sorted); \
      } \
 \
    template<typename Tpoints, typename Tgrid> void spread( \
      const cmav<complex<Tpoints>,1> &points, const vmav<complex<Tgrid>,ndim> &grid) \
      { \
      MR_assert(coords_sorted.shape(0)==points.shape(0), "npoints mismatch"); \
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc), \
        "Tcalc must be at least as accurate as Tpoints"); \
      MR_assert(grid.shape()==nover, "oversampled grid dimensions mismatch"); \
      if (coords_sorted.size()==0) return; \
      spreading_helper<16>(supp, coords_sorted, points, grid); \
      } \
    template<typename Tpoints, typename Tgrid> void interp( \
      const cmav<complex<Tgrid>,ndim> &grid, const vmav<complex<Tpoints>,1> &points) \
      { \
      MR_assert(coords_sorted.shape(0)==points.shape(0), "npoints mismatch"); \
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc), \
        "Tcalc must be at least as accurate as Tpoints"); \
      MR_assert(grid.shape()==nover, "oversampled grid dimensions mismatch"); \
      if (coords_sorted.size()==0) return; \
      interpolation_helper<16>(supp, grid, coords_sorted, points); \
      } \
    template<typename Tpoints, typename Tgrid> void spread( \
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points, \
      const vmav<complex<Tgrid>,ndim> &grid) \
      { \
      MR_assert(coords.shape(0)==points.shape(0), "npoints mismatch"); \
      MR_assert(coords_sorted.size()==0, "bad usage"); \
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc), \
        "Tcalc must be at least as accurate as Tpoints"); \
      MR_assert(grid.shape()==nover, "oversampled grid dimensions mismatch"); \
      if (coords.size()==0) return; \
      build_index(coords); \
      spreading_helper<16>(supp, coords, points, grid); \
      } \
    template<typename Tpoints, typename Tgrid> void interp( \
      const cmav<complex<Tgrid>,ndim> &grid, const cmav<Tcoord,2> &coords, \
      const vmav<complex<Tpoints>,1> &points) \
      { \
      MR_assert(coords.shape(0)==points.shape(0), "npoints mismatch"); \
      MR_assert(coords_sorted.size()==0, "bad usage"); \
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc), \
        "Tcalc must be at least as accurate as Tpoints"); \
      MR_assert(grid.shape()==nover, "oversampled grid dimensions mismatch"); \
      if (coords.size()==0) return; \
      build_index(coords); \
      interpolation_helper<16>(supp, grid, coords, points); \
      }

/*! Helper class for carrying out 1D nonuniform FFTs of types 1 and 2.
    Tcalc: the floating-point type in which all kernel-related calculations
           are performed
    Tacc:  the floating-point type used for the grid on which data is
           accumulated in nu2u transforms. Can usually be the same as Tcalc,
           but may be chosen to be more accurate in specific situations.
    Tpoints: the floating-point type used for storing the values at the
           non-uniform points
    Tgrid: the floating-point type used for storing the values on the uniform
           grid.
    Tcoord: the floating-point type used for storing the coordinates of the
           non-uniform points.
 */
template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> class Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 1>: public Spreadinterp_ancestor<Tcalc, Tacc, Tidx, 1>
  {
  private:
    static constexpr size_t ndim=1;

  DUCC0_SPREADINTERP_BOILERPLATE

  private:
    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int suvec = su+vlen-1;
        static constexpr double xsupp=2./supp;
        const Spreadinterp *parent;
        const vmav<complex<Tcalc>,ndim> &grid;
        array<int64_t,ndim> i0; // start index of the current nonuniform point
        array<int64_t,ndim> b0; // start index of the current buffer

        vmav<Tacc,ndim> bufr, bufi;
        Tacc *px0r, *px0i;
        Mutex &mylock;

        // add the acumulated local tile to the global oversampled grid
        DUCC0_NOINLINE void dump()
          {
          if (b0[0]<-nsafe) return; // nothing written into buffer yet
          int64_t inu = int(parent->nover[0]);
          {
          LockGuard lock(mylock);
          for (int64_t iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            {
            grid(idxu) += complex<Tcalc>(Tcalc(bufr(iu)), Tcalc(bufi(iu)));
            bufr(iu) = bufi(iu) = 0;
            }
          }
          }

      public:
        Tacc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;

        HelperNu2u(const Spreadinterp *parent_, const vmav<complex<Tcalc>,ndim> &grid_,
          Mutex &mylock_)
          : parent(parent_), grid(grid_),
            i0{-1000000}, b0{-1000000},
            bufr({size_t(suvec)}), bufi({size_t(suvec)}),
            px0r(bufr.data()), px0i(bufi.data()), mylock(mylock_) {}
        ~HelperNu2u() { dump(); }

        [[gnu::always_inline]] [[gnu::hot]] void prep_for_index(array<int64_t,ndim> ind)
          {
          if (ind==i0) return;
          i0 = ind;
          if ((i0[0]<b0[0]) || (i0[0]+int(supp)>b0[0]+su))
            {
            dump();
            b0[0]=((((i0[0]+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
          auto ofs = i0[0]-b0[0];
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t supp> class HelperU2nu
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int suvec = su+vlen-1;
        static constexpr double xsupp=2./supp;
        const Spreadinterp *parent;

        const cmav<complex<Tcalc>,ndim> &grid;
        array<int64_t,ndim> i0; // start index of the current nonuniform point
        array<int64_t,ndim> b0; // start index of the current buffer

        vmav<Tcalc,ndim> bufr, bufi;
        const Tcalc *px0r, *px0i;

        // load a tile from the global oversampled grid into local buffer
        DUCC0_NOINLINE void load()
          {
          int64_t inu = int(parent->nover[0]);
          for (int64_t iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            { bufr(iu) = grid(idxu).real(); bufi(iu) = grid(idxu).imag(); }
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;

        HelperU2nu(const Spreadinterp *parent_, const cmav<complex<Tcalc>,ndim> &grid_)
          : parent(parent_), grid(grid_),
            i0{-1000000}, b0{-1000000},
            bufr({size_t(suvec)}), bufi({size_t(suvec)}),
            px0r(bufr.data()), px0i(bufi.data()) {}

        [[gnu::always_inline]] [[gnu::hot]] void prep_for_index(array<int64_t,ndim> ind)
          {
          if (ind==i0) return;
          i0 = ind;
          if ((i0[0]<b0[0]) || (i0[0]+int(supp)>b0[0]+su))
            {
            b0[0]=((((i0[0]+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = i0[0]-b0[0];
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void spreading_helper
      (size_t supp, const cmav<Tcoord,2> &coords,
      const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tcalc>,ndim> &grid) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, coords, points, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, coords, points, grid);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;

      Mutex mylock;
      size_t npoints = points.shape(0);

      TemplateKernel<SUPP, mysimd<Tacc>> tkrn(*parent::krn);
      size_t chunksz = max<size_t>(1000, npoints/(10*nthreads));
      execDynamic(npoints, nthreads, chunksz, [&](Scheduler &sched)
        {
        HelperNu2u<SUPP> hlp(this, grid, mylock);

        constexpr size_t batchsize=3;
        array<array<int64_t,1>,batchsize> index;
        array<array<double,1>,batchsize> frac;
        mysimd<Tacc> kubuf[batchsize*hlp.nvec];

        constexpr size_t lookahead=10;

        while (auto rng=sched.getNext())
          {
          auto ix = rng.lo;
          for(; ix+batchsize<=rng.hi; ix+=batchsize)
            {
            for(size_t k=0; k<batchsize; ++k)
              {
              if (ix+k+lookahead<npoints)
                {
                auto nextidx = coord_idx[ix+k+lookahead];
                points.prefetch_r(nextidx);
                if (!sorted) coords.prefetch_r(nextidx,0);
                }
              parent::template getpix<Tcoord>(
                {coords(sorted ? ix+k : coord_idx[ix+k],0)}, frac[k], index[k]);
              }

            if constexpr(batchsize==3)
              tkrn.eval3(Tacc(supp-1-2*frac[0][0]),
                         Tacc(supp-1-2*frac[1][0]),
                         Tacc(supp-1-2*frac[2][0]), &kubuf[0]);
            else
              MR_fail("bad batchsize");

            for (size_t k=0; k<batchsize; ++k)
              {
              auto * DUCC0_RESTRICT ku = &kubuf[k*hlp.nvec];
              hlp.prep_for_index(index[k]);
              auto v(points(coord_idx[ix+k]));
  
              Tacc vr(v.real()), vi(v.imag());
              for (size_t cu=0; cu<hlp.nvec; ++cu)
                {
                auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*hlp.vlen;
                auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*hlp.vlen;
                auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
                tr += vr*ku[cu];
                tr.copy_to(pxr,element_aligned_tag());
                auto ti = mysimd<Tacc>(pxi, element_aligned_tag());
                ti += vi*ku[cu];
                ti.copy_to(pxi,element_aligned_tag());
                }
              }
            }
          for(; ix<rng.hi; ++ix)
            {
            if (ix+lookahead<npoints)
              {
              auto nextidx = coord_idx[ix+lookahead];
              points.prefetch_r(nextidx);
              if (!sorted) coords.prefetch_r(nextidx,0);
              }
            auto v(points(coord_idx[ix]));
            parent::template getpix<Tcoord>(
              {coords(sorted ? ix : coord_idx[ix],0)}, frac[0], index[0]);
            auto * DUCC0_RESTRICT ku = &kubuf[0];
            tkrn.eval1(Tacc(supp-1-2*frac[0][0]), &ku[0]);
            hlp.prep_for_index(index[0]);
  
            Tacc vr(v.real()), vi(v.imag());
            for (size_t cu=0; cu<hlp.nvec; ++cu)
              {
              auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*hlp.vlen;
              auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*hlp.vlen;
              auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
              tr += vr*ku[cu];
              tr.copy_to(pxr,element_aligned_tag());
              auto ti = mysimd<Tacc>(pxi, element_aligned_tag());
              ti += vi*ku[cu];
              ti.copy_to(pxi,element_aligned_tag());
              }
            }
          }
        });
      }

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,ndim> &grid,
      const cmav<Tcoord,2> &coords, const vmav<complex<Tpoints>,1> &points) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid, coords, points);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid, coords, points);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;
      size_t npoints = points.shape(0);

      TemplateKernel<SUPP, mysimd<Tcalc>> tkrn(*parent::krn);
      size_t chunksz = max<size_t>(1000, npoints/(10*nthreads));
      execDynamic(npoints, nthreads, chunksz, [&](Scheduler &sched)
        {
        HelperU2nu<SUPP> hlp(this, grid);

        constexpr size_t batchsize=3;
        array<array<int64_t,1>,batchsize> index;
        array<array<double,1>,batchsize> frac;
        mysimd<Tcalc> kubuf[batchsize*hlp.nvec];

        constexpr size_t lookahead=10;
        while (auto rng=sched.getNext())
          {
          auto ix = rng.lo;

          for(; ix+batchsize<=rng.hi; ix+=batchsize)
            {
            for(size_t k=0; k<batchsize; ++k)
              {
              if (ix+k+lookahead<npoints)
                {
                auto nextidx = coord_idx[ix+k+lookahead];
                points.prefetch_w(nextidx);
                if (!sorted)
                  coords.prefetch_r(nextidx,0);
                }
              parent::template getpix<Tcoord>(
                {coords(sorted ? ix+k : coord_idx[ix+k],0)}, frac[k], index[k]);
              }

            if constexpr(batchsize==3)
              tkrn.eval3(Tcalc(supp-1-2*frac[0][0]),
                         Tcalc(supp-1-2*frac[1][0]),
                         Tcalc(supp-1-2*frac[2][0]), &kubuf[0]);
            else
              MR_fail("bad batchsize");

            for (size_t k=0; k<batchsize; ++k)
              {
              auto * DUCC0_RESTRICT ku = &kubuf[k*hlp.nvec];
              hlp.prep_for_index(index[k]);
              size_t row = coord_idx[ix+k];

              mysimd<Tcalc> rr=0, ri=0;
              for (size_t cu=0; cu<hlp.nvec; ++cu)
                {
                const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*hlp.vlen;
                const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*hlp.vlen;
                rr += ku[cu]*mysimd<Tcalc>(pxr,element_aligned_tag());
                ri += ku[cu]*mysimd<Tcalc>(pxi,element_aligned_tag());
                }
              points(row) = hsum_cmplx<Tcalc>(rr,ri);
              }
            }
          for(; ix<rng.hi; ++ix)
            {
            if (ix+lookahead<npoints)
              {
              auto nextidx = coord_idx[ix+lookahead];
              points.prefetch_w(nextidx);
              if (!sorted) coords.prefetch_r(nextidx,0);
              }
            size_t row = coord_idx[ix];
            parent::template getpix<Tcoord>({coords(sorted ? ix : coord_idx[ix],0)}, frac[0], index[0]);
            auto * DUCC0_RESTRICT ku = &kubuf[0];
            tkrn.eval1(Tcalc(supp-1-2*frac[0][0]), &ku[0]);
            hlp.prep_for_index(index[0]);

            mysimd<Tcalc> rr=0, ri=0;
            for (size_t cu=0; cu<hlp.nvec; ++cu)
              {
              const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*hlp.vlen;
              const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*hlp.vlen;
              rr += ku[cu]*mysimd<Tcalc>(pxr,element_aligned_tag());
              ri += ku[cu]*mysimd<Tcalc>(pxi,element_aligned_tag());
              }
            points(row) = hsum_cmplx<Tcalc>(rr,ri);
            }
          }
        });
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      MR_assert(coords.shape(1)==ndim, "ndim mismatch");
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      coord_idx.resize(coords.shape(0));
      quick_array<Tidx> key(coords.shape(0));
      execParallel(coords.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          key[i] = parent::template get_tile<Tcoord>({coords(i,0)})[0];
        });
      bucket_sort2(key, coord_idx, ntiles_u, nthreads);
      }
  };

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> class Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 2>: public Spreadinterp_ancestor<Tcalc, Tacc, Tidx, 2>
  {
  private:
    static constexpr size_t ndim=2;

  DUCC0_SPREADINTERP_BOILERPLATE

    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile), sv = su;
        static constexpr double xsupp=2./supp;
        const Spreadinterp *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        const vmav<complex<Tcalc>,ndim> &grid;
        array<int64_t,ndim> i0; // start index of the current nonuniform point
        array<int64_t,ndim> b0; // start index of the current buffer

        vmav<complex<Tacc>,ndim> gbuf;
        complex<Tacc> *px0;
        vector<Mutex> &locks;

        DUCC0_NOINLINE void dump()
          {
          if (b0[0]<-nsafe) return; // nothing written into buffer yet
          int64_t inu = int(parent->nover[0]);
          int64_t inv = int(parent->nover[1]);

          int64_t idxv0 = (b0[1]+inv)%inv;
          for (int64_t iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            {
            LockGuard lock(locks[idxu]);
            for (int64_t iv=0, idxv=idxv0; iv<sv; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              {
              grid(idxu,idxv) += complex<Tcalc>(gbuf(iu,iv));
              gbuf(iu,iv) = 0;
              }
            }
          }

      public:
        complex<Tacc> * DUCC0_RESTRICT p0;
        union kbuf {
          Tacc scalar[2*nvec*vlen];
          mysimd<Tacc> simd[2*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperNu2u(const Spreadinterp *parent_, const vmav<complex<Tcalc>,ndim> &grid_,
          vector<Mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            i0{-1000000, -1000000}, b0{-1000000, -1000000},
            gbuf({size_t(su+1),size_t(sv)}),
            px0(gbuf.data()), locks(locks_) {}
        ~HelperNu2u() { dump(); }

        constexpr int lineJump() const { return sv; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(array<double,ndim> in)
          {
          array<double,ndim> frac;
          auto i0old = i0;
          parent->template getpix<Tcoord>(in, frac, i0);
          auto x0 = -frac[0]*2+(supp-1);
          auto y0 = -frac[1]*2+(supp-1);
          tkrn.eval2(Tacc(x0), Tacc(y0), &buf.simd[0]);
          if (i0==i0old) return;
          if ((i0[0]<b0[0]) || (i0[1]<b0[1]) || (i0[0]+int(supp)>b0[0]+su) || (i0[1]+int(supp)>b0[1]+sv))
            {
            dump();
            b0[0]=((((i0[0]+nsafe)>>log2tile)<<log2tile))-nsafe;
            b0[1]=((((i0[1]+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
          p0 = px0 + (i0[0]-b0[0])*sv + i0[1]-b0[1];
          }
      };

    template<size_t supp> class HelperU2nu
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile), sv = su;
        static constexpr int svvec = max<size_t>(sv, ((supp+2*vlen-2)/vlen)*vlen);
        static constexpr double xsupp=2./supp;
        const Spreadinterp *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,ndim> &grid;
        array<int64_t,ndim> i0; // start index of the current nonuniform point
        array<int64_t,ndim> b0; // start index of the current buffer

        vmav<Tcalc,ndim> bufri;
        const Tcalc *px0r, *px0i;

        DUCC0_NOINLINE void load()
          {
          int64_t inu = int(parent->nover[0]);
          int64_t inv = int(parent->nover[1]);
          int64_t idxv0 = (b0[1]+inv)%inv;
          for (int64_t iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            for (int64_t iv=0, idxv=idxv0; iv<sv; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              {
              bufri(2*iu  ,iv) = grid(idxu, idxv).real();
              bufri(2*iu+1,iv) = grid(idxu, idxv).imag();
              }
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tcalc scalar[2*nvec*vlen];
          mysimd<Tcalc> simd[2*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperU2nu(const Spreadinterp *parent_, const cmav<complex<Tcalc>,ndim> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            i0{-1000000, -1000000}, b0{-1000000, -1000000},
            bufri({size_t(2*su+1),size_t(svvec)}),
            px0r(bufri.data()), px0i(bufri.data()+svvec) {}

        constexpr int lineJump() const { return 2*svvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(array<double,ndim> in)
          {
          array<double,ndim> frac;
          auto i0old = i0;
          parent->template getpix<Tcoord>(in, frac, i0);
          auto x0 = -frac[0]*2+(supp-1);
          auto y0 = -frac[1]*2+(supp-1);
          tkrn.eval2(Tcalc(x0), Tcalc(y0), &buf.simd[0]);
          if (i0==i0old) return;
          if ((i0[0]<b0[0]) || (i0[1]<b0[1]) || (i0[0]+int(supp)>b0[0]+su) || (i0[1]+int(supp)>b0[1]+sv))
            {
            b0[0]=((((i0[0]+nsafe)>>log2tile)<<log2tile))-nsafe;
            b0[1]=((((i0[1]+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = (i0[0]-b0[0])*2*svvec + i0[1]-b0[1];
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void spreading_helper
      (size_t supp, const cmav<Tcoord,2> &coords,
      const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tcalc>,ndim> &grid) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, coords, points, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, coords, points, grid);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;

      vector<Mutex> locks(nover[0]);

      size_t chunksz = max<size_t>(1000, coord_idx.size()/(10*nthreads));
      execDynamic(coord_idx.size(), nthreads, chunksz, [&](Scheduler &sched)
        {
        HelperNu2u<SUPP> hlp(this, grid, locks);
        constexpr auto jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+hlp.nvec*hlp.vlen;
        constexpr size_t NVEC2 = (2*SUPP+hlp.vlen-1)/hlp.vlen;
        array<complex<Tacc>,SUPP> cdata;
        array<mysimd<Tacc>,NVEC2> vdata;
        for (size_t i=0; i<vdata.size(); ++i) vdata[i]=0;

        constexpr size_t lookahead=3;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            points.prefetch_r(nextidx);
            if (!sorted)
              for (size_t d=0; d<ndim; ++d) coords.prefetch_r(nextidx,d);
            }
          size_t row = coord_idx[ix];
          sorted ? hlp.prep({coords(ix,0), coords(ix,1)})
                 : hlp.prep({coords(row,0), coords(row,1)});
          complex<Tacc> v(points(row));

          for (size_t cv=0; cv<SUPP; ++cv)
            cdata[cv] = kv[cv]*v;

          // really ugly, but attemps with type-punning via union fail on some platforms
          memcpy(reinterpret_cast<void *>(vdata.data()),
                 reinterpret_cast<const void *>(cdata.data()),
                 SUPP*sizeof(complex<Tacc>));

          Tacc * DUCC0_RESTRICT xpx = reinterpret_cast<Tacc *>(hlp.p0);
// It seems that performance is slightly better if we don't work in
// memory-contiguous fashion, probably due to the unaligned accesses.
#if 0  // old version, leaving it in for now
          for (size_t cu=0; cu<SUPP; ++cu)
            {
            Tacc tmpx=ku[cu];
            for (size_t cv=0; cv<NVEC2; ++cv)
              {
              auto * DUCC0_RESTRICT px = xpx+cu*2*jump+cv*hlp.vlen;
              auto tval = mysimd<Tacc>(px,element_aligned_tag());
              tval += tmpx*vdata[cv];
              tval.copy_to(px,element_aligned_tag());
              }
            }
#else
          for (size_t cv=0; cv<NVEC2; ++cv)
            {
            auto tmpx=vdata[cv];
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              auto * DUCC0_RESTRICT px = xpx+cu*2*jump+cv*hlp.vlen;
              auto tval = mysimd<Tacc>(px,element_aligned_tag());
              tval += tmpx*ku[cu];
              tval.copy_to(px,element_aligned_tag());
              }
            }
#endif
          }
        });
      }

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,ndim> &grid,
      const cmav<Tcoord,2> &coords, const vmav<complex<Tpoints>,1> &points) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid, coords, points);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid, coords, points);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;
      size_t npoints = points.shape(0);

      size_t chunksz = max<size_t>(1000, coord_idx.size()/(10*nthreads));
      execDynamic(npoints, nthreads, chunksz, [&](Scheduler &sched)
        {
        HelperU2nu<SUPP> hlp(this, grid);
        constexpr int jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+hlp.nvec;

        constexpr size_t lookahead=3;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<npoints)
            {
            auto nextidx = coord_idx[ix+lookahead];
            points.prefetch_w(nextidx);
            if (!sorted)
              for (size_t d=0; d<ndim; ++d) coords.prefetch_r(nextidx,d);
            }
          size_t row = coord_idx[ix];
          sorted ? hlp.prep({coords(ix,0), coords(ix,1)})
                 : hlp.prep({coords(row,0), coords(row,1)});
          mysimd<Tcalc> rr=0, ri=0;
          if constexpr (hlp.nvec==1)
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*jump;
              const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*jump;
              rr += mysimd<Tcalc>(pxr,element_aligned_tag())*ku[cu];
              ri += mysimd<Tcalc>(pxi,element_aligned_tag())*ku[cu];
              }
            rr *= kv[0];
            ri *= kv[0];
            }
          else
            {
// The repeated addition to tmpr and tmpi may be a bottleneck ...
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> tmpr(0), tmpi(0);
              for (size_t cv=0; cv<hlp.nvec; ++cv)
                {
                const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*jump + hlp.vlen*cv;
                const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*jump + hlp.vlen*cv;
                tmpr += kv[cv]*mysimd<Tcalc>(pxr,element_aligned_tag());
                tmpi += kv[cv]*mysimd<Tcalc>(pxi,element_aligned_tag());
                }
              rr += ku[cu]*tmpr;
              ri += ku[cu]*tmpi;
              }
            }
          points(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      size_t ntiles_v = (nover[1]>>log2tile) + 3;
      coord_idx.resize(coords.shape(0));
      quick_array<Tidx> key(coords.shape(0));
      execParallel(coords.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tile = parent::template get_tile<Tcoord>({coords(i,0), coords(i,1)});
          key[i] = tile[0]*ntiles_v + tile[1];
          }
        });
      bucket_sort2(key, coord_idx, ntiles_u*ntiles_v, nthreads);
      }
  };

template<typename Tcalc, typename Tacc, typename Tcoord,typename Tidx> class Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 3>: public Spreadinterp_ancestor<Tcalc, Tacc, Tidx, 3>
  {
  private:
    static constexpr size_t ndim=3;

  DUCC0_SPREADINTERP_BOILERPLATE

    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile), sv = su, sw = su;
        static constexpr double xsupp=2./supp;
        const Spreadinterp *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        const vmav<complex<Tcalc>,ndim> &grid;
        array<int64_t,ndim> i0; // start index of the current nonuniform point
        array<int64_t,ndim> b0; // start index of the current buffer
#ifdef NEW_DUMP
        array<int64_t,ndim> imin,imax;
#endif

        vmav<complex<Tacc>,ndim> gbuf;
        complex<Tacc> *px0;
        vector<Mutex> &locks;

        DUCC0_NOINLINE void dump()
          {
          if (b0[0]<-nsafe) return; // nothing written into buffer yet
          int64_t inu = int(parent->nover[0]);
          int64_t inv = int(parent->nover[1]);
          int64_t inw = int(parent->nover[2]);

#ifdef NEW_DUMP
          int64_t idxv0 = (imin[1]+b0[1]+inv)%inv;
          int64_t idxw0 = (imin[2]+b0[2]+inw)%inw;
          for (int64_t iu=imin[0], idxu=(imin[0]+b0[0]+inu)%inu; iu<imax[0]; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            {
            LockGuard lock(locks[idxu]);
            for (int64_t iv=imin[1], idxv=idxv0; iv<imax[1]; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              for (int64_t iw=imin[2], idxw=idxw0; iw<imax[2]; ++iw, idxw=(idxw+1<inw)?(idxw+1):0)
                {
                auto t=gbuf(iu,iv,iw);
                grid(idxu,idxv,idxw) += complex<Tcalc>(t);
                gbuf(iu,iv,iw) = 0;
                }
            }
          imin={1000,1000,1000}; imax={-1000,-1000,-1000};
#else
          int64_t idxv0 = (b0[1]+inv)%inv;
          int64_t idxw0 = (b0[2]+inw)%inw;
          for (int64_t iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            {
            LockGuard lock(locks[idxu]);
            for (int64_t iv=0, idxv=idxv0; iv<sv; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              for (int64_t iw=0, idxw=idxw0; iw<sw; ++iw, idxw=(idxw+1<inw)?(idxw+1):0)
                {
                auto t=gbuf(iu,iv,iw);
                grid(idxu,idxv,idxw) += complex<Tcalc>(t);
                gbuf(iu,iv,iw) = 0;
                }
            }
#endif
          }

      public:
        complex<Tacc> * DUCC0_RESTRICT p0;
        union kbuf {
          Tacc scalar[3*nvec*vlen];
          mysimd<Tacc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperNu2u(const Spreadinterp *parent_, const vmav<complex<Tcalc>,ndim> &grid_,
          vector<Mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            i0{-1000000, -1000000, -1000000}, b0{-1000000, -1000000, -1000000},
#ifdef NEW_DUMP
            imin{1000,1000,1000},imax{-1000,-1000,-1000},
#endif
            gbuf({size_t(su),size_t(sv),size_t(sw)}),
            px0(gbuf.data()), locks(locks_) {}
        ~HelperNu2u() { dump(); }

        constexpr int lineJump() const { return sw; }
        constexpr int planeJump() const { return sv*sw; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(array<double,ndim> in)
          {
          array<double,ndim> frac;
          
          auto i0old = i0;
          parent->template getpix<Tcoord>(in, frac, i0);
          auto x0 = -frac[0]*2+(supp-1);
          auto y0 = -frac[1]*2+(supp-1);
          auto z0 = -frac[2]*2+(supp-1);
          tkrn.eval3(Tacc(x0), Tacc(y0), Tacc(z0), &buf.simd[0]);
          if (i0==i0old) return;
          if ((i0[0]<b0[0]) || (i0[1]<b0[1]) || (i0[2]<b0[2])
           || (i0[0]+int(supp)>b0[0]+su) || (i0[1]+int(supp)>b0[1]+sv) || (i0[2]+int(supp)>b0[2]+sw))
            {
            dump();
            b0[0]=((((i0[0]+nsafe)>>log2tile)<<log2tile))-nsafe;
            b0[1]=((((i0[1]+nsafe)>>log2tile)<<log2tile))-nsafe;
            b0[2]=((((i0[2]+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
#ifdef NEW_DUMP
          for (size_t i=0; i<ndim; ++i)
            {
            imin[i]=min<int>(imin[i],i0[i]-b0[i]);
            imax[i]=max<int>(imax[i],i0[i]-b0[i]+supp);
            }
#endif
          p0 = px0 + (i0[0]-b0[0])*sv*sw + (i0[1]-b0[1])*sw + (i0[2]-b0[2]);
          }
      };

    template<size_t supp> class HelperU2nu
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile), sv = su, sw = su;
        static constexpr int swvec = max<size_t>(sw, ((supp+2*nvec-2)/nvec)*nvec);
        static constexpr double xsupp=2./supp;
        const Spreadinterp *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,ndim> &grid;
        array<int64_t,ndim> i0; // start index of the nonuniform point
        array<int64_t,ndim> b0; // start index of the current buffer

        vmav<Tcalc,ndim> bufri;
        const Tcalc *px0r, *px0i;

        DUCC0_NOINLINE void load()
          {
          int64_t inu = int(parent->nover[0]);
          int64_t inv = int(parent->nover[1]);
          int64_t inw = int(parent->nover[2]);
          int64_t idxv0 = (b0[1]+inv)%inv;
          int64_t idxw0 = (b0[2]+inw)%inw;
          for (int64_t iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            for (int64_t iv=0, idxv=idxv0; iv<sv; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              for (int64_t iw=0, idxw=idxw0; iw<sw; ++iw, idxw=(idxw+1<inw)?(idxw+1):0)
                {
                bufri(iu,2*iv,iw) = grid(idxu, idxv, idxw).real();
                bufri(iu,2*iv+1,iw) = grid(idxu, idxv, idxw).imag();
                }
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tcalc scalar[3*nvec*vlen];
          mysimd<Tcalc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperU2nu(const Spreadinterp *parent_, const cmav<complex<Tcalc>,ndim> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            i0{-1000000, -1000000, -1000000}, b0{-1000000, -1000000, -1000000},
            bufri({size_t(su+1),size_t(2*sv),size_t(swvec)}),
            px0r(bufri.data()), px0i(bufri.data()+swvec) {}

        constexpr int lineJump() const { return 2*swvec; }
        constexpr int planeJump() const { return 2*sv*swvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(array<double,ndim> in)
          {
          array<double,ndim> frac;
          auto i0old = i0;
          parent->template getpix<Tcoord>(in, frac, i0);
          auto x0 = -frac[0]*2+(supp-1);
          auto y0 = -frac[1]*2+(supp-1);
          auto z0 = -frac[2]*2+(supp-1);
          tkrn.eval3(Tcalc(x0), Tcalc(y0), Tcalc(z0), &buf.simd[0]);
          if (i0==i0old) return;
          if ((i0[0]<b0[0]) || (i0[1]<b0[1]) || (i0[2]<b0[2])
           || (i0[0]+int(supp)>b0[0]+su) || (i0[1]+int(supp)>b0[1]+sv) || (i0[2]+int(supp)>b0[2]+sw))
            {
            b0[0]=((((i0[0]+nsafe)>>log2tile)<<log2tile))-nsafe;
            b0[1]=((((i0[1]+nsafe)>>log2tile)<<log2tile))-nsafe;
            b0[2]=((((i0[2]+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = (i0[0]-b0[0])*2*sv*swvec + (i0[1]-b0[1])*2*swvec + (i0[2]-b0[2]);
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void spreading_helper
      (size_t supp, const cmav<Tcoord,2> &coords,
      const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tcalc>,ndim> &grid) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, coords, points, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, coords, points, grid);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;
      size_t npoints = points.shape(0);

      vector<Mutex> locks(nover[0]);

      size_t chunksz = max<size_t>(1000, npoints/(10*nthreads));
      execDynamic(npoints, nthreads, chunksz, [&](Scheduler &sched)
        {
        HelperNu2u<SUPP> hlp(this, grid, locks);
        constexpr auto ljump = hlp.lineJump();
        constexpr auto pjump = hlp.planeJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+hlp.vlen*hlp.nvec;
        const auto * DUCC0_RESTRICT kw = hlp.buf.scalar+2*hlp.vlen*hlp.nvec;
        using Tsimd = mysimd<Tacc>;
        constexpr size_t vlen = Tsimd::size();
        constexpr size_t nvec2 = (2*SUPP+vlen-1)/vlen;
        union Txdata{
          array<complex<Tacc>,SUPP> c;
          array<Tacc,2*SUPP> f;
          array<Tsimd,nvec2> v;
          Txdata(){for (size_t i=0; i<v.size(); ++i) v[i]=0;}
          };
        Txdata xdata;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          constexpr size_t lookahead=3;
          if (ix+lookahead<npoints)
            {
            auto nextidx = coord_idx[ix+lookahead];
            points.prefetch_r(nextidx);
            if (!sorted)
              for (size_t d=0; d<ndim; ++d) coords.prefetch_r(nextidx,d);
            }
          size_t row = coord_idx[ix];
          sorted ? hlp.prep({coords(ix,0), coords(ix,1), coords(ix,2)})
                 : hlp.prep({coords(row,0), coords(row,1), coords(row,2)});
          complex<Tacc> v(points(row));

          for (size_t cw=0; cw<SUPP; ++cw)
            xdata.c[cw]=kw[cw]*v;
          Tacc * DUCC0_RESTRICT fptr2=reinterpret_cast<Tacc *>(hlp.p0);
// this is quite voodoo, but helps a lot, at least on my machine
if constexpr(SUPP<=8)
  {
          const Tsimd * DUCC0_RESTRICT fptr1=xdata.v.data();
          for (size_t cu=0; cu<SUPP; ++cu)
            for (size_t cw=0; cw<nvec2; ++cw)
              {
              auto tmp2x=ku[cu]*fptr1[cw];
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                Tsimd tmp(fptr2+cw*vlen+cv*2*ljump + cu*2*pjump, element_aligned_tag());
                tmp += tmp2x*kv[cv];
                tmp.copy_to(fptr2+cw*vlen+cv*2*ljump + cu*2*pjump, element_aligned_tag());
                }
              }
  }
else
  {
          const Tacc * DUCC0_RESTRICT fptr1=xdata.f.data();
          const auto j1 = 2*ljump;
          const auto j2 = 2*(pjump-SUPP*ljump);
// We might want to try the 2D non-contiguous approach here at some point,
// but it doesn't work if we use the current unvectorized loops.
          for (size_t cu=0; cu<SUPP; ++cu, fptr2+=j2)
            for (size_t cv=0; cv<SUPP; ++cv, fptr2+=j1)
              {
              Tacc tmp2x=ku[cu]*kv[cv];
              for (size_t cw=0; cw<2*SUPP; ++cw)
                fptr2[cw] += tmp2x*fptr1[cw];
              }
  }
          }
        });
      }

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,ndim> &grid,
      const cmav<Tcoord,2> &coords, const vmav<complex<Tpoints>,1> &points) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid, coords, points);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid, coords, points);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;
      size_t npoints = points.shape(0);

      size_t chunksz = max<size_t>(1000, npoints/(10*nthreads));
      execDynamic(npoints, nthreads, chunksz, [&](Scheduler &sched)
        {
        HelperU2nu<SUPP> hlp(this, grid);
        constexpr auto ljump = hlp.lineJump();
        constexpr auto pjump = hlp.planeJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+hlp.vlen*hlp.nvec;
        const auto * DUCC0_RESTRICT kw = hlp.buf.simd+2*hlp.nvec;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          constexpr size_t lookahead=3;
          if (ix+lookahead<npoints)
            {
            auto nextidx = coord_idx[ix+lookahead];
            points.prefetch_w(nextidx);
            if (!sorted)
              for (size_t d=0; d<ndim; ++d) coords.prefetch_r(nextidx,d);
            }
          size_t row = coord_idx[ix];
          sorted ? hlp.prep({coords(ix,0), coords(ix,1), coords(ix,2)})
                 : hlp.prep({coords(row,0), coords(row,1), coords(row,2)});
          mysimd<Tcalc> rr=0, ri=0;
          if constexpr (hlp.nvec==1)
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> r2r=0, r2i=0;
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*pjump + cv*ljump;
                const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*pjump + cv*ljump;
                r2r += mysimd<Tcalc>(pxr,element_aligned_tag())*kv[cv];
                r2i += mysimd<Tcalc>(pxi,element_aligned_tag())*kv[cv];
                }
              rr += r2r*ku[cu];
              ri += r2i*ku[cu];
              }
            rr *= kw[0];
            ri *= kw[0];
            }
          else
            {
// The repeated addition to tmp2r and tmp2i may be a bottleneck ...
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> tmpr(0), tmpi(0);
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                mysimd<Tcalc> tmp2r(0), tmp2i(0);
                for (size_t cw=0; cw<hlp.nvec; ++cw)
                  {
                  const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*pjump + cv*ljump + hlp.vlen*cw;
                  const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*pjump + cv*ljump + hlp.vlen*cw;
                  tmp2r += kw[cw]*mysimd<Tcalc>(pxr,element_aligned_tag());
                  tmp2i += kw[cw]*mysimd<Tcalc>(pxi,element_aligned_tag());
                  }
                tmpr += kv[cv]*tmp2r;
                tmpi += kv[cv]*tmp2i;
                }
              rr += ku[cu]*tmpr;
              ri += ku[cu]*tmpi;
              }
            }
          points(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      size_t ntiles_v = (nover[1]>>log2tile) + 3;
      size_t ntiles_w = (nover[2]>>log2tile) + 3;
      size_t lsq2 = log2tile;
      while ((lsq2>=1) && (((ntiles_u*ntiles_v*ntiles_w)<<(3*(log2tile-lsq2)))<(size_t(1)<<28)))
        --lsq2;
      auto ssmall = log2tile-lsq2;
      auto msmall = (size_t(1)<<ssmall) - 1;

      coord_idx.resize(coords.shape(0));
      quick_array<Tidx> key(coords.shape(0));
      execParallel(coords.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tile = parent::template get_tile<Tcoord>({coords(i,0),coords(i,1),coords(i,2)},lsq2);
          auto lowkey = ((tile[0]&msmall)<<(2*ssmall))
                      | ((tile[1]&msmall)<<   ssmall)
                      |  (tile[2]&msmall);
          auto hikey = ((tile[0]>>ssmall)*ntiles_v*ntiles_w)
                     + ((tile[1]>>ssmall)*ntiles_w)
                     +  (tile[2]>>ssmall);
          key[i] = (hikey<<(3*ssmall)) | lowkey;
          }
        });
      bucket_sort2(key, coord_idx, (ntiles_u*ntiles_v*ntiles_w)<<(3*ssmall), nthreads);
      }
  };

#undef DUCC0_SPREADINTERP_BOILERPLATE

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx>
Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::Spreadinterp2(size_t npoints,
  const vector<size_t> &over_shape, size_t kidx,
  size_t nthreads,
  const vector<double> &periodicity,
  const vector<double> &corigin)
  {
  size_t ndim = over_shape.size();
  if (ndim==1)
    si1 = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 1>>
      (npoints, array<size_t,1>{over_shape[0]}, kidx, nthreads, periodicity, corigin);
  else if (ndim==2)
    si2 = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 2>>
      (npoints, array<size_t,2>{over_shape[0],over_shape[1]}, kidx, nthreads, periodicity, corigin);
  else if (ndim==3)
    si3 = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 3>>
      (npoints, array<size_t,3>{over_shape[0],over_shape[1],over_shape[2]}, kidx, nthreads, periodicity, corigin);
  }

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx>
Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::Spreadinterp2(const cmav<Tcoord,2> &coords,
  const vector<size_t> &over_shape, size_t kidx,
  size_t nthreads, const vector<double> &periodicity,
  const vector<double> &corigin)
  {
  size_t ndim = over_shape.size();
  if (ndim==1)
    si1 = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 1>>
      (coords, array<size_t,1>{over_shape[0]}, kidx, nthreads, periodicity, corigin);
  else if (ndim==2)
    si2 = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 2>>
      (coords, array<size_t,2>{over_shape[0],over_shape[1]}, kidx, nthreads, periodicity, corigin);
  else if (ndim==3)
    si3 = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, Tidx, 3>>
      (coords, array<size_t,3>{over_shape[0],over_shape[1],over_shape[2]}, kidx, nthreads, periodicity, corigin);
  }

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx>
Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::~Spreadinterp2(){}

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> template<typename Tpoints, typename Tgrid>
void Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::spread(
  const cmav<complex<Tpoints>,1> &points, const vfmav<complex<Tgrid>> &grid)
  {
  if (si1) si1->spread(points, vmav<complex<Tgrid>,1>(grid));
  if (si2) si2->spread(points, vmav<complex<Tgrid>,2>(grid));
  if (si3) si3->spread(points, vmav<complex<Tgrid>,3>(grid));
  }

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> template<typename Tpoints, typename Tgrid>
void Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::interp(
  const cfmav<complex<Tgrid>> &grid, const vmav<complex<Tpoints>,1> &points)
  {
  if (si1) si1->interp(cmav<complex<Tgrid>,1>(grid), points);
  if (si2) si2->interp(cmav<complex<Tgrid>,2>(grid), points);
  if (si3) si3->interp(cmav<complex<Tgrid>,3>(grid), points);
  }

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> template<typename Tpoints, typename Tgrid>
void Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::spread(
  const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
  const vfmav<complex<Tgrid>> &grid)
  {
  if (si1) si1->spread(coords, points, vmav<complex<Tgrid>,1>(grid));
  if (si2) si2->spread(coords, points, vmav<complex<Tgrid>,2>(grid));
  if (si3) si3->spread(coords, points, vmav<complex<Tgrid>,3>(grid));
  }

template<typename Tcalc, typename Tacc, typename Tcoord, typename Tidx> template<typename Tpoints, typename Tgrid>
void Spreadinterp2<Tcalc, Tacc, Tcoord, Tidx>::interp(
  const cfmav<complex<Tgrid>> &grid, const cmav<Tcoord,2> &coords,
  const vmav<complex<Tpoints>,1> &points)
  {
  if (si1) si1->interp(cmav<complex<Tgrid>,1>(grid), coords, points);
  if (si2) si2->interp(cmav<complex<Tgrid>,2>(grid), coords, points);
  if (si3) si3->interp(cmav<complex<Tgrid>,3>(grid), coords, points);
  }
 
}}

#endif
