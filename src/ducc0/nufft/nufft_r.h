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

/* Copyright (C) 2019-2024 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_NUFFT_R_H
#define DUCC0_NUFFT_R_H

#include "ducc0/nufft/nufft.h"

namespace ducc0 {

namespace detail_nufft {

using namespace std;
// the next line is necessary to address some sloppy name choices in AdaptiveCpp
using std::min, std::max;

template<typename Tcalc, typename Tacc, typename Tcoord, size_t ndim> class RNufft;

template<typename Tcalc, typename Tacc, typename Tcoord> class RNufft<Tcalc, Tacc, Tcoord, 3>: public Nufft_ancestor<Tcalc, Tacc, 3>
  {
  private:
    static constexpr size_t ndim=3;

    template<typename Tpoints, typename Tgrid> bool prep_nu2u
      (const cmav<Tpoints,1> &points, const vmav<complex<Tgrid>,ndim> &uniform)
      {
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tpoints");
      static_assert(sizeof(Tgrid)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tgrid");
      MR_assert(points.shape(0)==npoints, "number of points mismatch");
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      if (npoints==0)
        {
        mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform);
        return true;
        }
      return false;
      }
    template<typename Tpoints, typename Tgrid> bool prep_u2nu
      (const cmav<Tpoints,1> &points, const cmav<complex<Tgrid>,ndim> &uniform)
      {
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tpoints");
      static_assert(sizeof(Tgrid)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tgrid");
      MR_assert(points.shape(0)==npoints, "number of points mismatch");
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      return npoints==0;
      }

  private: \
    using parent=Nufft_ancestor<Tcalc, Tacc, ndim>; \
    using parent::coord_idx, parent::nthreads, parent::npoints, parent::supp, \
          parent::timers, parent::krn, parent::fft_order, parent::nuni, \
          parent::nover, parent::shift, parent::maxi0, parent::report, \
          parent::log2tile, parent::corfac, parent::sort_coords; \
 \
    vmav<Tcoord,2> coords_sorted; \
 \
  public: \
    using parent::parent; /* inherit constructor */ \
    RNufft(bool gridding, const cmav<Tcoord,2> &coords, \
          const array<size_t, ndim> &uniform_shape_, double epsilon_,  \
          size_t nthreads_, double sigma_min, double sigma_max, \
          const vector<double> &periodicity, bool fft_order_) \
      : parent(gridding, coords.shape(0), uniform_shape_, epsilon_, nthreads_, \
               sigma_min, sigma_max, periodicity, fft_order_), \
        coords_sorted({npoints,ndim},UNINITIALIZED) \
      { \
      build_index(coords); \
      sort_coords(coords, coords_sorted); \
      } \
 \
    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity, \
      const cmav<Tpoints,1> &points, const vmav<complex<Tgrid>,ndim> &uniform) \
      { \
      if (prep_nu2u(points, uniform)) return; \
      MR_assert(coords_sorted.size()!=0, "bad call"); \
      if (verbosity>0) report(true); \
      nonuni2uni(forward, coords_sorted, points, uniform); \
      if (verbosity>0) timers.report(cout); \
      } \
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity, \
      const cmav<complex<Tgrid>,ndim> &uniform, const vmav<Tpoints,1> &points) \
      { \
      if (prep_u2nu(points, uniform)) return; \
      MR_assert(coords_sorted.size()!=0, "bad call"); \
      if (verbosity>0) report(false); \
      uni2nonuni(forward, uniform, coords_sorted, points); \
      if (verbosity>0) timers.report(cout); \
      } \
    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity, \
      const cmav<Tcoord,2> &coords, const cmav<Tpoints,1> &points, \
      const vmav<complex<Tgrid>,ndim> &uniform) \
      { \
      if (prep_nu2u(points, uniform)) return; \
      MR_assert(coords_sorted.size()==0, "bad call"); \
      if (verbosity>0) report(true); \
      build_index(coords); \
      nonuni2uni(forward, coords, points, uniform); \
      if (verbosity>0) timers.report(cout); \
      } \
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity, \
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords, \
      const vmav<Tpoints,1> &points) \
      { \
      if (prep_u2nu(points, uniform)) return; \
      MR_assert(coords_sorted.size()==0, "bad call"); \
      if (verbosity>0) report(false); \
      build_index(coords); \
      uni2nonuni(forward, uniform, coords, points); \
      if (verbosity>0) timers.report(cout); \
      }

    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;
int ru0, ru1, rv0, rv1, rw0, rw1;
double c0, c1;
      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile), sv = su, sw = su;
        static constexpr double xsupp=2./supp;
        const RNufft *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        const vmav<complex<Tcalc>,ndim> &grid;
        array<int,ndim> i0; // start index of the current nonuniform point
        array<int,ndim> b0; // start index of the current buffer

        vmav<Tacc,ndim> bufr;
        Tacc *px0;
        vector<Mutex> &locks;

        DUCC0_NOINLINE void dump()
          {
          if (b0[0]<-nsafe) return; // nothing written into buffer yet
          int inu = int(parent->nover[0]);
          int inv = int(parent->nover[1]);
          int inw = int(parent->nover[2]);
//cout << "dump" << endl;
//cout << ru0 << " " << ru1 << " " << rv0 << " " << rv1 << " " << rw0 << " " << rw1 << endl;
          int idxv0 = (b0[1]+rv0+inv)%inv;
          int idxw0 = (b0[2]+rw0+inw)%inw;
          for (int iu=ru0, idxu=(b0[0]+ru0+inu)%inu; iu<ru1; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            {
            LockGuard lock(locks[idxu]);
            for (int iv=rv0, idxv=idxv0; iv<rv1; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              for (int iw=rw0, idxw=idxw0; iw<rw1; ++iw, idxw=(idxw+1<inw)?(idxw+1):0)
                {
                grid(idxu,idxv,idxw) += Tcalc(bufr(iu,iv,iw));
                bufr(iu,iv,iw) = 0;
                }
            }
//c0+=su*sv*sw; c1+=(ru1-ru0)*(rv1-rv0)*(rw1-rw0);
//cout << c1/c0 << endl;
ru0=rv0=rw0=100000; ru1=rv1=rw1=-10;
          }

      public:
        Tacc * DUCC0_RESTRICT p0;
        union kbuf {
          Tacc scalar[3*nvec*vlen];
          mysimd<Tacc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperNu2u(const RNufft *parent_, const vmav<complex<Tcalc>,ndim> &grid_,
          vector<Mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            i0{-1000000, -1000000, -1000000}, b0{-1000000, -1000000, -1000000},
            bufr({size_t(su),size_t(sv),size_t(sw)}),
            px0(bufr.data()), locks(locks_) {ru0=rv0=rw0=100000; ru1=rv1=rw1=-10; c0=c1=0;}
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
ru0=min<int>(ru0,i0[0]-b0[0]);
ru1=max<int>(ru1,i0[0]-b0[0]+supp);
rv0=min<int>(rv0,i0[1]-b0[1]);
rv1=max<int>(rv1,i0[1]-b0[1]+supp);
rw0=min<int>(rw0,i0[2]-b0[2]);
rw1=max<int>(rw1,i0[2]-b0[2]+supp);
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
        const RNufft *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,ndim> &grid;
        array<int,ndim> i0; // start index of the nonuniform point
        array<int,ndim> b0; // start index of the current buffer

        vmav<Tcalc,ndim> bufr;
        const Tcalc *px0;

        DUCC0_NOINLINE void load()
          {
          int inu = int(parent->nover[0]);
          int inv = int(parent->nover[1]);
          int inw = int(parent->nover[2]);
          int idxv0 = (b0[1]+inv)%inv;
          int idxw0 = (b0[2]+inw)%inw;
          for (int iu=0, idxu=(b0[0]+inu)%inu; iu<su; ++iu, idxu=(idxu+1<inu)?(idxu+1):0)
            for (int iv=0, idxv=idxv0; iv<sv; ++iv, idxv=(idxv+1<inv)?(idxv+1):0)
              for (int iw=0, idxw=idxw0; iw<sw; ++iw, idxw=(idxw+1<inw)?(idxw+1):0)
                bufr(iu,iv,iw) = grid(idxu, idxv, idxw).real();
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0;
        union kbuf {
          Tcalc scalar[3*nvec*vlen];
          mysimd<Tcalc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperU2nu(const RNufft *parent_, const cmav<complex<Tcalc>,ndim> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            i0{-1000000, -1000000, -1000000}, b0{-1000000, -1000000, -1000000},
            bufr({size_t(su+1),size_t(sv),size_t(swvec)}),
            px0(bufr.data()) {}

        constexpr int lineJump() const { return swvec; }
        constexpr int planeJump() const { return sv*swvec; }

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
          auto ofs = (i0[0]-b0[0])*sv*swvec + (i0[1]-b0[1])*swvec + (i0[2]-b0[2]);
          p0 = px0+ofs;
          }
      };

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void spreading_helper
      (size_t supp, const cmav<Tcoord,2> &coords,
      const cmav<Tpoints,1> &points,
      const vmav<complex<Tcalc>,ndim> &grid) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, coords, points, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, coords, points, grid);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;

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
        array<Tacc,SUPP> xdata;

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
          auto v(points(row));

          for (size_t cw=0; cw<SUPP; ++cw)
            xdata[cw]=kw[cw]*v;
          const Tacc * DUCC0_RESTRICT fptr1=xdata.data();
          Tacc * DUCC0_RESTRICT fptr2=hlp.p0;
          const auto j1 = ljump;
          const auto j2 = pjump-SUPP*ljump;
          for (size_t cu=0; cu<SUPP; ++cu, fptr2+=j2)
            for (size_t cv=0; cv<SUPP; ++cv, fptr2+=j1)
              {
              Tacc tmp2x=ku[cu]*kv[cv];
              for (size_t cw=0; cw<SUPP; ++cw)
                fptr2[cw] += tmp2x*fptr1[cw];
              }
          }
        });
      }

    template<size_t SUPP, typename Tpoints> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,ndim> &grid,
      const cmav<Tcoord,2> &coords, const vmav<Tpoints,1> &points) const
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid, coords, points);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid, coords, points);
      MR_assert(supp==SUPP, "requested support out of range");
      bool sorted = coords_sorted.size()!=0;

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
          mysimd<Tcalc> r=0;
          if constexpr (hlp.nvec==1)
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> r2=0;
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                const auto * DUCC0_RESTRICT px = hlp.p0 + cu*pjump + cv*ljump;
                r2 += mysimd<Tcalc>(px,element_aligned_tag())*kv[cv];
                }
              r += r2*ku[cu];
              }
            r *= kw[0];
            }
          else
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> tmp(0);
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                mysimd<Tcalc> tmp2(0);
                for (size_t cw=0; cw<hlp.nvec; ++cw)
                  {
                  const auto * DUCC0_RESTRICT px = hlp.p0 + cu*pjump + cv*ljump + hlp.vlen*cw;
                  tmp2 += kw[cw]*mysimd<Tcalc>(px,element_aligned_tag());
                  }
                tmp += kv[cv]*tmp2;
                }
              r += ku[cu]*tmp;
              }
            }
          points(row) = reduce(r, plus<>());
          }
        });
      }

    template<typename Tpoints, typename Tgrid> void nonuni2uni(bool forward,
      const cmav<Tcoord,2> &coords, const cmav<Tpoints,1> &points,
      const vmav<complex<Tgrid>,ndim> &uniform)
      {
      timers.push("nu2u proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("spreading");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      spreading_helper<maxsupp>(supp, coords, points, grid);
      timers.poppush("FFT");
      {
      vfmav<complex<Tcalc>> fgrid(grid);
      slice slz{0,(nuni[2]+2)/2}, shz{fgrid.shape(2)-nuni[2]/2,MAXIDX};
      slice sly{0,(nuni[1]+2)/2}, shy{fgrid.shape(1)-nuni[1]/2,MAXIDX};
      c2c(fgrid, fgrid, {2}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{},slz});
      c2c(fgridl, fgridl, {1}, forward, Tcalc(1), nthreads);
      auto fgridh=fgrid.subarray({{},{},shz});
      c2c(fgridh, fgridh, {1}, forward, Tcalc(1), nthreads);
      auto fgridll=fgrid.subarray({{},sly,slz});
      c2c(fgridll, fgridll, {0}, forward, Tcalc(1), nthreads);
      auto fgridlh=fgrid.subarray({{},sly,shz});
      c2c(fgridlh, fgridlh, {0}, forward, Tcalc(1), nthreads);
      auto fgridhl=fgrid.subarray({{},shy,slz});
      c2c(fgridhl, fgridhl, {0}, forward, Tcalc(1), nthreads);
      auto fgridhh=fgrid.subarray({{},shy,shz});
      c2c(fgridhh, fgridhh, {0}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("symmetrizing");
      if ((nuni[0]&1)==0)  // mirror yz Nyquist plane
        {
        size_t isrc = fft_order ? nover[0]-nuni[0]/2 : 0;
        size_t idst = fft_order ? nuni[0]/2 : nuni[0];
        for (size_t j=0; j<nover[1]; ++j)
          for (size_t k=0; k<nover[2]; ++k)
            grid(isrc,j,k)+=grid(idst,j,k);
        }
      if ((nuni[1]&1)==0)  // mirror xz Nyquist plane
        {
        size_t jsrc = fft_order ? nover[1]-nuni[1]/2 : 0;
        size_t jdst = fft_order ? nuni[1]/2 : nuni[1];
        for (size_t i=0; i<nover[0]; ++i)
          for (size_t k=0; k<nover[2]; ++k)
            grid(i,jsrc,k)+=grid(i,jdst,k);
        }
      if ((nuni[2]&1)==0)  // mirror xy Nyquist plane
        {
        size_t ksrc = fft_order ? nover[2]-nuni[2]/2 : 0;
        size_t kdst = fft_order ? nuni[2]/2 : nuni[2];
        for (size_t i=0; i<nover[0]; ++i)
          for (size_t j=0; j<nover[1]; ++j)
            grid(i,j,ksrc)+=grid(i,j,kdst);
        }
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iout, iin] = comp_indices(i, nuni[0], nover[0], fft_order);
          for (size_t j=0; j<nuni[1]; ++j)
            {
            auto [icfv, jout, jin] = comp_indices(j, nuni[1], nover[1], fft_order);
            for (size_t k=0; k<nuni[2]; ++k)
              {
              auto [icfw, kout, kin] = comp_indices(k, nuni[2], nover[2], fft_order);
              uniform(iout,jout,kout) = complex<Tgrid>(grid(iin,jin,kin)
                *Tgrid(corfac[0][icfu]*corfac[1][icfv]*corfac[2][icfw]));
              }
            }
          }
        });
      timers.pop();
      timers.pop();
      }

    template<typename Tpoints, typename Tgrid> void uni2nonuni(bool forward,
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<Tpoints,1> &points)
      {
      timers.push("u2nu proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      // TODO: not all entries need to be zeroed, perhaps some time can be saved here
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iin, iout] = comp_indices(i, nuni[0], nover[0], fft_order);
          for (size_t j=0; j<nuni[1]; ++j)
            {
            auto [icfv, jin, jout] = comp_indices(j, nuni[1], nover[1], fft_order);
            for (size_t k=0; k<nuni[2]; ++k)
              {
              auto [icfw, kin, kout] = comp_indices(k, nuni[2], nover[2], fft_order);
              grid(iout,jout,kout) = uniform(iin,jin,kin)
                *Tcalc(corfac[0][icfu]*corfac[1][icfv]*corfac[2][icfw]);
              }
            }
          }
        });
      timers.poppush("symmetrizing");
      if ((nuni[0]&1)==0)  // mirror yz Nyquist plane
        {
        size_t isrc = fft_order ? nover[0]-nuni[0]/2 : 0;
        size_t idst = fft_order ? nuni[0]/2 : nuni[0];
        for (size_t j=0; j<nover[1]; ++j)
          for (size_t k=0; k<nover[2]; ++k)
            {
//            grid(isrc,j,k)*=0.5;
            grid(idst,j,k)=grid(isrc,j,k);
            }
        }
      if ((nuni[1]&1)==0)  // mirror xz Nyquist plane
        {
        size_t jsrc = fft_order ? nover[1]-nuni[1]/2 : 0;
        size_t jdst = fft_order ? nuni[1]/2 : nuni[1];
        for (size_t i=0; i<nover[0]; ++i)
          for (size_t k=0; k<nover[2]; ++k)
            {
//            grid(i,jsrc,k)*=0.5;
            grid(i,jdst,k)=grid(i,jsrc,k);
            }
        }
      if ((nuni[2]&1)==0)  // mirror xy Nyquist plane
        {
        size_t ksrc = fft_order ? nover[2]-nuni[2]/2 : 0;
        size_t kdst = fft_order ? nuni[2]/2 : nuni[2];
        for (size_t i=0; i<nover[0]; ++i)
          for (size_t j=0; j<nover[1]; ++j)
            {
//            grid(i,j,ksrc)*=0.5;
            grid(i,j,kdst)=grid(i,j,ksrc);
            }
        }
      timers.poppush("FFT");
      {
      vfmav<complex<Tcalc>> fgrid(grid);
      slice slz{0,(nuni[2]+2)/2}, shz{fgrid.shape(2)-nuni[2]/2,MAXIDX};
      slice sly{0,(nuni[1]+2)/2}, shy{fgrid.shape(1)-nuni[1]/2,MAXIDX};
      auto fgridll=fgrid.subarray({{},sly,slz});
      c2c(fgridll, fgridll, {0}, forward, Tcalc(1), nthreads);
      auto fgridlh=fgrid.subarray({{},sly,shz});
      c2c(fgridlh, fgridlh, {0}, forward, Tcalc(1), nthreads);
      auto fgridhl=fgrid.subarray({{},shy,slz});
      c2c(fgridhl, fgridhl, {0}, forward, Tcalc(1), nthreads);
      auto fgridhh=fgrid.subarray({{},shy,shz});
      c2c(fgridhh, fgridhh, {0}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{},slz});
      c2c(fgridl, fgridl, {1}, forward, Tcalc(1), nthreads);
      auto fgridh=fgrid.subarray({{},{},shz});
      c2c(fgridh, fgridh, {1}, forward, Tcalc(1), nthreads);
      c2c(fgrid, fgrid, {2}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("interpolation");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      interpolation_helper<maxsupp>(supp, grid, coords, points);
      timers.pop();
      timers.pop();
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      timers.push("building index");
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      size_t ntiles_v = (nover[1]>>log2tile) + 3;
      size_t ntiles_w = (nover[2]>>log2tile) + 3;
      size_t lsq2 = log2tile;
      while ((lsq2>=1) && (((ntiles_u*ntiles_v*ntiles_w)<<(3*(log2tile-lsq2)))<(size_t(1)<<28)))
        --lsq2;
      auto ssmall = log2tile-lsq2;
      auto msmall = (size_t(1)<<ssmall) - 1;

      coord_idx.resize(npoints);
      quick_array<uint32_t> key(npoints);
      execParallel(npoints, nthreads, [&](size_t lo, size_t hi)
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
      timers.pop();
      }
  };

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord>
  void nu2u_r(const cmav<Tcoord,2> &coord, const cmav<Tpoints,1> &points,
    bool forward, double epsilon, size_t nthreads,
    const vfmav<complex<Tgrid>> &uniform, size_t verbosity,
    double sigma_min, double sigma_max, const vector<double> &periodicity, bool fft_order)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=3) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  if (ndim==3)
    {
    vmav<complex<Tgrid>,3> uniform2(uniform);
    RNufft<Tcalc, Tacc, Tcoord, 3> nufft(true, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.nu2u(forward, verbosity, coord, points, uniform2); 
    }
  }
template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord>
  void u2nu_r(const cmav<Tcoord,2> &coord, const cfmav<complex<Tgrid>> &uniform,
    bool forward, double epsilon, size_t nthreads,
    const vmav<Tpoints,1> &points, size_t verbosity,
    double sigma_min, double sigma_max, const vector<double> &periodicity, bool fft_order)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=3) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  if (ndim==3)
    {
    cmav<complex<Tgrid>,3> uniform2(uniform);
    RNufft<Tcalc, Tacc, Tcoord, 3> nufft(false, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.u2nu(forward, verbosity, uniform2, coord, points); 
    }
  }
} // namespace detail_nufft

// public names
using detail_nufft::u2nu_r;
using detail_nufft::nu2u_r;

}

#endif
