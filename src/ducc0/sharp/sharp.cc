/*
 *  This file is part of libsharp2.
 *
 *  libsharp2 is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp2 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp2; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* libsharp2 is being developed at the Max-Planck-Institut fuer Astrophysik */

/*! \file sharp.cc
 *  Spherical transform library
 *
 *  Copyright (C) 2006-2021 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#include <cmath>
#include <algorithm>
#include <memory>
#include "ducc0/math/math_utils.h"
#include "ducc0/math/fft1d.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/sharp/sht.h"

namespace ducc0 {

namespace detail_sharp {

using namespace std;

using dcmplx = complex<double>;
using fcmplx = complex<float>;

static size_t chunksize_min=500, nchunks_max=10;

static void get_chunk_info (size_t ndata, size_t nmult, size_t &nchunks, size_t &chunksize)
  {
  chunksize = (ndata+nchunks_max-1)/nchunks_max;
  if (chunksize>=chunksize_min) // use max number of chunks
    chunksize = ((chunksize+nmult-1)/nmult)*nmult;
  else // need to adjust chunksize and nchunks
    {
    nchunks = (ndata+chunksize_min-1)/chunksize_min;
    chunksize = (ndata+nchunks-1)/nchunks;
    if (nchunks>1)
      chunksize = ((chunksize+nmult-1)/nmult)*nmult;
    }
  nchunks = (ndata+chunksize-1)/chunksize;
  }

class sharp_job
  {
  private:
    std::vector<std::any> alm;
    std::vector<std::any> map;
    sharp_jobtype type;
    size_t spin;
    size_t flags;
    const sharp_geom_info &ginfo;
    const sharp_alm_info &ainfo;
    int nthreads;

    size_t nmaps() const { return 1+(spin>0); }
    size_t nalm() const { return (type==SHARP_ALM2MAP_DERIV1) ? 1 : (1+(spin>0)); }

    void init_output()
      {
      if (flags&SHARP_ADD) return;
      if (type == SHARP_MAP2ALM)
        for (size_t i=0; i<alm.size(); ++i)
          ainfo.clear_alm (alm[i]);
      else
        for (size_t i=0; i<map.size(); ++i)
          ginfo.clear_map(map[i]);
      }

    DUCC0_NOINLINE void alm2almtmp (size_t mi, mav<dcmplx,2> &almtmp,
      const vector<double> norm_l)
      {
      size_t nalm_ = nalm();
      size_t lmax = ainfo.lmax();
      if (type!=SHARP_MAP2ALM)
        {
        auto m=ainfo.mval(mi);
        auto lmin=(m<spin) ? spin : m;
        for (size_t i=0; i<nalm_; ++i)
          {
          auto sub = subarray<1>(almtmp, {0,i},{MAXIDX,0});
          ainfo.get_alm(mi, alm[i], sub);
          }
        for (auto l=m; l<lmin; ++l)
          for (size_t i=0; i<nalm_; ++i)
            almtmp.v(l,i) = 0;
        for (size_t i=0; i<nalm_; ++i)
          almtmp.v(lmax+1,i) = 0;
        if (spin>0)
          for (auto l=lmin; l<=lmax; ++l)
            for (size_t i=0; i<nalm_; ++i)
              almtmp.v(l,i) *= norm_l[l];
        }
      else
        for (size_t l=ainfo.mval(mi); l<(lmax+2); ++l)
          for (size_t i=0; i<nalm(); ++i)
            almtmp.v(l,i)=0;
      }

    DUCC0_NOINLINE void almtmp2alm (size_t mi,
      mav<dcmplx,2> &almtmp, const vector<double> norm_l)
      {
      if (type != SHARP_MAP2ALM) return;
      size_t lmax = ainfo.lmax();
      auto m=ainfo.mval(mi);
      auto lmin=(m<spin) ? spin : m;
      size_t nalm_ = nalm();
      if (spin>0)
        for (auto l=lmin; l<=lmax; ++l)
          for (size_t i=0; i<nalm_; ++i)
            almtmp.v(l,i) *= norm_l[l];
      for (size_t i=0; i<nalm_; ++i)
        ainfo.add_alm(mi, subarray<1>(almtmp, {0,i},{MAXIDX,0}), alm[i]);
      }

    DUCC0_NOINLINE void ringtmp2ring (size_t iring,
      const mav<double,2> &ringtmp)
      {
      for (size_t i=0; i<nmaps(); ++i)
        ginfo.add_ring(flags&SHARP_USE_WEIGHTS, iring, subarray<1>(ringtmp, {i,1},{0,MAXIDX}), map[i]);
      }

    DUCC0_NOINLINE void ring2ringtmp (size_t iring,
      mav<double,2> &ringtmp)
      {
      for (size_t i=0; i<nmaps(); ++i)
        {
        auto rtmp = subarray<1>(ringtmp, {i,1},{0,MAXIDX});
        ginfo.get_ring(flags&SHARP_USE_WEIGHTS, iring, map[i], rtmp);
        }
      }

    //FIXME: set phase to zero if not SHARP_MAP2ALM?
    DUCC0_NOINLINE void map2phase (size_t mmax, size_t llim, size_t ulim, mav<dcmplx,3> &phase)
      {
      if (type != SHARP_MAP2ALM) return;
      execDynamic(ulim-llim, nthreads, 1, [&](Scheduler &sched)
        {
        detail_sht::ringhelper helper;
        size_t rstride=ginfo.nphmax()+2;
        mav<double,2> ringtmp({nmaps(), rstride});

        while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
          {
          size_t iring = ginfo.pair(ith).r1;
          ring2ringtmp(iring,ringtmp);
          for (size_t i=0; i<nmaps(); ++i)
            {
            auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
            auto ptmp = subarray<1>(phase, {i, 2*(ith-llim), 0}, {0, 0, MAXIDX});
            helper.ring2phase (ginfo.nph(iring),ginfo.phi0(iring),rtmp,mmax,ptmp);
            }
          if (ginfo.pair(ith).r2!=~size_t(0))
            {
            size_t iring = ginfo.pair(ith).r2;
            ring2ringtmp(iring,ringtmp);
            for (size_t i=0; i<nmaps(); ++i)
              {
              auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
              auto ptmp = subarray<1>(phase, {i, 2*(ith-llim)+1, 0}, {0, 0, MAXIDX});
              helper.ring2phase (ginfo.nph(iring),ginfo.phi0(iring),rtmp,mmax,ptmp);
              }
            }
          }
        }); /* end of parallel region */
      }

    DUCC0_NOINLINE void phase2map (size_t mmax, size_t llim, size_t ulim, const mav<dcmplx,3> &phase)
      {
      if (type == SHARP_MAP2ALM) return;
      execDynamic(ulim-llim, nthreads, 1, [&](Scheduler &sched)
        {
        detail_sht::ringhelper helper;
        size_t rstride=ginfo.nphmax()+2;
        mav<double,2> ringtmp({nmaps(), rstride});

        while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
          {
          size_t iring = ginfo.pair(ith).r1;
          for (size_t i=0; i<nmaps(); ++i)
            {
            auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
            auto ptmp = subarray<1>(phase, {i, 2*(ith-llim), 0}, {0, 0, MAXIDX});
            helper.phase2ring (ginfo.nph(iring),ginfo.phi0(iring),rtmp,mmax,ptmp);
            }
          ringtmp2ring(iring,ringtmp);
          if (ginfo.pair(ith).r2!=~size_t(0))
            {
            size_t iring = ginfo.pair(ith).r2;
            for (size_t i=0; i<nmaps(); ++i)
              {
              auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
              auto ptmp = subarray<1>(phase, {i, 2*(ith-llim)+1, 0}, {0, 0, MAXIDX});
              helper.phase2ring (ginfo.nph(iring),ginfo.phi0(iring),rtmp,mmax,ptmp);
              }
            ringtmp2ring(iring,ringtmp);
            }
          }
        }); /* end of parallel region */
      }

  public:
    DUCC0_NOINLINE void execute()
      {
      size_t lmax = ainfo.lmax(),
             mmax = ainfo.mmax();

      auto norm_l = (type==SHARP_ALM2MAP_DERIV1) ?
        detail_sht::YlmBase::get_d1norm (lmax) :
        detail_sht::YlmBase::get_norm (lmax, spin);

    /* clear output arrays if requested */
      init_output();

      size_t nchunks, chunksize;
      get_chunk_info(ginfo.npairs(), (spin==0) ? 128 : 64,
                     nchunks,chunksize);
    //FIXME: needs to be changed to "nm"
      auto phase = mav<dcmplx,3>::build_noncritical({nmaps(),2*chunksize,mmax+1});
      detail_sht::YlmBase ylmbase(lmax,mmax,spin);
      detail_sht::SHT_mode mode = (type==SHARP_MAP2ALM) ? detail_sht::MAP2ALM : 
                                 ((type==SHARP_ALM2MAP) ? detail_sht::ALM2MAP : detail_sht::ALM2MAP_DERIV1);
    /* chunk loop */
      for (size_t chunk=0; chunk<nchunks; ++chunk)
        {
        size_t llim=chunk*chunksize, ulim=min(llim+chunksize,ginfo.npairs());
        vector<detail_sht::ringdata> rdata(ulim-llim);
        for (size_t i=0; i<ulim-llim; ++i)
          {
          double cth = ginfo.cth(ginfo.pair(i+llim).r1);
          double sth = ginfo.sth(ginfo.pair(i+llim).r1);
          auto mlim = detail_sht::get_mlim(lmax, spin, sth, cth);
          size_t idx = 2*i;
          size_t midx = 2*i+1;
          if (ginfo.pair(i+llim).r2==~size_t(0)) midx=idx;
          rdata[i] = { mlim, idx, midx, cth, sth };
          }

    /* map->phase where necessary */
        map2phase(mmax, llim, ulim, phase);

        execDynamic(ainfo.nm(), nthreads, 1, [&](Scheduler &sched)
          {
          detail_sht::Ylmgen ylmgen(ylmbase);
          auto almtmp = mav<dcmplx,2>({lmax+2, nalm()});

          while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
            {
    /* alm->alm_tmp where necessary */
            alm2almtmp(mi, almtmp, norm_l);
            ylmgen.prepare(ainfo.mval(mi));
            detail_sht::inner_loop(mode, almtmp, phase, rdata, ylmgen, mi);

    /* alm_tmp->alm where necessary */
            almtmp2alm(mi, almtmp, norm_l);
            }
          }); /* end of parallel region */

    /* phase->map where necessary */
        phase2map (mmax, llim, ulim, phase);
        } /* end of chunk loop */
      }

    sharp_job (sharp_jobtype type_,
      size_t spin_, const vector<any> &alm_, const vector<any> &map_,
      const sharp_geom_info &geom_info, const sharp_alm_info &alm_info, size_t flags_, int nthreads_)
      : alm(alm_), map(map_), type(type_), spin(spin_), flags(flags_), ginfo(geom_info), ainfo(alm_info),
        nthreads(nthreads_)
      {
      if (type==SHARP_ALM2MAP_DERIV1) spin_=1;
      if (type==SHARP_MAP2ALM) flags|=SHARP_USE_WEIGHTS;
      if (type==SHARP_Yt) type=SHARP_MAP2ALM;
      if (type==SHARP_WY) { type=SHARP_ALM2MAP; flags|=SHARP_USE_WEIGHTS; }

      MR_assert(spin<=ainfo.lmax(), "bad spin");
      MR_assert(alm.size()==nalm(), "incorrect # of a_lm components");
      MR_assert(map.size()==nmaps(), "incorrect # of a_lm components");
      }
  };

void sharp_execute (sharp_jobtype type, size_t spin, const vector<any> &alm,
  const vector<any> &map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads)
  {
  sharp_job job(type, spin, alm, map, geom_info, alm_info, flags, nthreads);
  job.execute();
  }

void sharp_set_chunksize_min(size_t new_chunksize_min)
  { chunksize_min=new_chunksize_min; }
void sharp_set_nchunks_max(size_t new_nchunks_max)
  { nchunks_max=new_nchunks_max; }

sharp_standard_alm_info::sharp_standard_alm_info (size_t lmax__, size_t nm_, ptrdiff_t stride_,
  const size_t *mval__, const ptrdiff_t *mstart)
  : lmax_(lmax__), mval_(nm_), mvstart(nm_), stride(stride_)
  {
  for (size_t mi=0; mi<nm_; ++mi)
    {
    mval_[mi] = mval__[mi];
    mvstart[mi] = mstart[mi];
    }
  }

sharp_standard_alm_info::sharp_standard_alm_info (size_t lmax__, size_t mmax_, ptrdiff_t stride_,
  const ptrdiff_t *mstart)
  : lmax_(lmax__), mval_(mmax_+1), mvstart(mmax_+1), stride(stride_)
  {
  for (size_t i=0; i<=mmax_; ++i)
    {
    mval_[i]=i;
    mvstart[i] = mstart[i];
    }
  }

template<typename T> void sharp_standard_alm_info::tclear (T *alm) const
  {
  for (size_t mi=0;mi<mval_.size();++mi)
    for (size_t l=mval_[mi];l<=lmax_;++l)
      reinterpret_cast<T *>(alm)[mvstart[mi]+ptrdiff_t(l)*stride]=0.;
  }
void sharp_standard_alm_info::clear_alm(const any &alm) const
  {
  auto hc = alm.type().hash_code();
  if (hc==typeid(dcmplx *).hash_code())
    tclear(any_cast<dcmplx *>(alm));
  else if (hc==typeid(fcmplx *).hash_code())
    tclear(any_cast<fcmplx *>(alm));
  else MR_fail("bad a_lm data type");
  }
template<typename T> void sharp_standard_alm_info::tget(size_t mi, const T *alm, mav<dcmplx,1> &almtmp) const
  {
  for (auto l=mval_[mi]; l<=lmax_; ++l)
    almtmp.v(l) = alm[mvstart[mi]+ptrdiff_t(l)*stride];
  }
void sharp_standard_alm_info::get_alm(size_t mi, const any &alm, mav<dcmplx,1> &almtmp) const
  {
  auto hc = alm.type().hash_code();
  if (hc==typeid(dcmplx *).hash_code())
    tget(mi, any_cast<dcmplx *>(alm), almtmp);
  else if (hc==typeid(const dcmplx *).hash_code())
    tget(mi, any_cast<const dcmplx *>(alm), almtmp);
  else if (hc==typeid(fcmplx *).hash_code())
    tget(mi, any_cast<fcmplx *>(alm), almtmp);
  else if (hc==typeid(const fcmplx *).hash_code())
    tget(mi, any_cast<const fcmplx *>(alm), almtmp);
  else MR_fail("bad a_lm data type");
  }
template<typename T> void sharp_standard_alm_info::tadd(size_t mi, const mav<dcmplx,1> &almtmp, T *alm) const
  {
  for (auto l=mval_[mi]; l<=lmax_; ++l)
    alm[mvstart[mi]+ptrdiff_t(l)*stride] += T(almtmp(l));
  }
void sharp_standard_alm_info::add_alm(size_t mi, const mav<dcmplx,1> &almtmp, const any &alm) const
  {
  auto hc = alm.type().hash_code();
  if (hc==typeid(dcmplx *).hash_code())
    tadd(mi, almtmp, any_cast<dcmplx *>(alm));
  else if (hc==typeid(fcmplx *).hash_code())
    tadd(mi, almtmp, any_cast<fcmplx *>(alm));
  else MR_fail("bad a_lm data type");
  }

ptrdiff_t sharp_standard_alm_info::index (size_t l, size_t mi)
  { return mvstart[mi]+stride*ptrdiff_t(l); }
/* This currently requires all m values from 0 to nm-1 to be present.
   It might be worthwhile to relax this criterion such that holes in the m
   distribution are permissible. */
size_t sharp_standard_alm_info::mmax() const
  {
  //FIXME: if gaps are allowed, we have to search the maximum m in the array
  auto nm_=mval_.size();
  vector<bool> mcheck(nm_,false);
  for (auto m_cur : mval_)
    {
    MR_assert(m_cur<nm_, "not all m values are present");
    MR_assert(mcheck[m_cur]==false, "duplicate m value");
    mcheck[m_cur]=true;
    }
  return nm_-1;
  }

unique_ptr<sharp_standard_alm_info> sharp_make_triangular_alm_info (size_t lmax, size_t mmax, ptrdiff_t stride)
  {
  vector<ptrdiff_t> mvstart(mmax+1);
  size_t tval = 2*lmax+1;
  for (size_t m=0; m<=mmax; ++m)
    mvstart[m] = stride*ptrdiff_t((m*(tval-m))>>1);
  return make_unique<sharp_standard_alm_info>(lmax, mmax, stride, mvstart.data());
  }

sharp_standard_geom_info::sharp_standard_geom_info(size_t nrings, const size_t *nph, const ptrdiff_t *ofs,
  ptrdiff_t stride_, const double *phi0, const double *theta, const double *wgt)
  : ring(nrings), stride(stride_)
  {
  size_t pos=0;

  nphmax_=0;

  for (size_t m=0; m<nrings; ++m)
    {
    ring[m].theta = theta[m];
    ring[m].cth = cos(theta[m]);
    ring[m].sth = sin(theta[m]);
    ring[m].weight = (wgt != nullptr) ? wgt[m] : 1.;
    ring[m].phi0 = phi0[m];
    ring[m].ofs = ofs[m];
    ring[m].nph = nph[m];
    if (nphmax_<nph[m]) nphmax_=nph[m];
    }
  sort(ring.begin(), ring.end(),[](const Tring &a, const Tring &b)
    { return (a.sth<b.sth); });
  while (pos<nrings)
    {
    pair_.push_back(Tpair());
    pair_.back().r1=pos;
    if ((pos<nrings-1) && approx(ring[pos].cth,-ring[pos+1].cth,1e-12))
      {
      if (ring[pos].cth>0)  // make sure northern ring is in r1
        pair_.back().r2=pos+1;
      else
        {
        pair_.back().r1=pos+1;
        pair_.back().r2=pos;
        }
      ++pos;
      }
    else
      pair_.back().r2=size_t(~0);
    ++pos;
    }

  sort(pair_.begin(), pair_.end(), [this] (const Tpair &a, const Tpair &b)
    {
    if (ring[a.r1].nph==ring[b.r1].nph)
    return (ring[a.r1].phi0 < ring[b.r1].phi0) ? true :
      ((ring[a.r1].phi0 > ring[b.r1].phi0) ? false :
        (ring[a.r1].cth>ring[b.r1].cth));
    return ring[a.r1].nph<ring[b.r1].nph;
    });
  }

template<typename T> void sharp_standard_geom_info::tclear(T *map) const
  {
  for (const auto &r: ring)
    {
    if (stride==1)
      memset(&map[r.ofs],0,r.nph*sizeof(T));
    else
      for (size_t i=0;i<r.nph;++i)
        map[r.ofs+ptrdiff_t(i)*stride]=T(0);
    }
  }

void sharp_standard_geom_info::clear_map (const any &map) const
  {
  auto hc = map.type().hash_code();
  if (hc==typeid(double *).hash_code())
    tclear(any_cast<double *>(map));
  else if (hc==typeid(float *).hash_code())
    tclear(any_cast<float *>(map));
  else MR_fail("bad map data type");
  }

template<typename T> void sharp_standard_geom_info::tadd(bool weighted, size_t iring, const mav<double,1> &ringtmp, T *map) const
  {
  T *DUCC0_RESTRICT p1=&map[ring[iring].ofs];
  double wgt = weighted ? ring[iring].weight : 1.;
  for (size_t m=0; m<ring[iring].nph; ++m)
    p1[ptrdiff_t(m)*stride] += T(ringtmp(m)*wgt);
  }
//virtual
void sharp_standard_geom_info::add_ring(bool weighted, size_t iring, const mav<double,1> &ringtmp, const any &map) const
  {
  auto hc = map.type().hash_code();
  if (hc==typeid(double *).hash_code())
    tadd(weighted, iring, ringtmp, any_cast<double *>(map));
  else if (hc==typeid(float *).hash_code())
    tadd(weighted, iring, ringtmp, any_cast<float *>(map));
  else MR_fail("bad map data type");
  }
template<typename T> void sharp_standard_geom_info::tget(bool weighted, size_t iring, const T *map, mav<double,1> &ringtmp) const
  {
  const T *DUCC0_RESTRICT p1=&map[ring[iring].ofs];
  double wgt = weighted ? ring[iring].weight : 1.;
  for (size_t m=0; m<ring[iring].nph; ++m)
    ringtmp.v(m) = p1[ptrdiff_t(m)*stride]*wgt;
  }
//virtual
void sharp_standard_geom_info::get_ring(bool weighted, size_t iring, const any &map, mav<double,1> &ringtmp) const
  {
  auto hc = map.type().hash_code();
  if (hc==typeid(const double *).hash_code())
    tget(weighted, iring, any_cast<const double *>(map), ringtmp);
  else if (hc==typeid(double *).hash_code())
    tget(weighted, iring, any_cast<double *>(map), ringtmp);
  else if (hc==typeid(const float *).hash_code())
    tget(weighted, iring, any_cast<const float *>(map), ringtmp);
  else if (hc==typeid(float *).hash_code())
    tget(weighted, iring, any_cast<float *>(map), ringtmp);
  else MR_fail("bad map data type",map.type().name());
  }

unique_ptr<sharp_geom_info> sharp_make_subset_healpix_geom_info (size_t nside, ptrdiff_t stride, size_t nrings,
  const size_t *rings, const double *weight)
  {
  size_t npix=nside*nside*12;
  size_t ncap=2*nside*(nside-1);

  vector<double> theta(nrings), weight_(nrings), phi0(nrings);
  vector<size_t> nph(nrings);
  vector<ptrdiff_t> ofs(nrings);
  ptrdiff_t curofs=0, checkofs; /* checkofs used for assertion introduced when adding rings arg */
  for (size_t m=0; m<nrings; ++m)
    {
    auto ring = (rings==nullptr)? (m+1) : rings[m];
    size_t northring = (ring>2*nside) ? 4*nside-ring : ring;
    if (northring < nside)
      {
      theta[m] = 2*asin(northring/(sqrt(6.)*nside));
      nph[m] = 4*northring;
      phi0[m] = pi/nph[m];
      checkofs = ptrdiff_t(2*northring*(northring-1))*stride;
      }
    else
      {
      double fact1 = (8.*nside)/npix;
      double costheta = (2*nside-northring)*fact1;
      theta[m] = acos(costheta);
      nph[m] = 4*nside;
      if ((northring-nside) & 1)
        phi0[m] = 0;
      else
        phi0[m] = pi/nph[m];
      checkofs = ptrdiff_t(ncap + (northring-nside)*nph[m])*stride;
      ofs[m] = curofs;
      }
    if (northring != ring) /* southern hemisphere */
      {
      theta[m] = pi-theta[m];
      checkofs = ptrdiff_t(npix - nph[m])*stride - checkofs;
      ofs[m] = curofs;
      }
    weight_[m]=4.*pi/npix*((weight==nullptr) ? 1. : weight[northring-1]);
    if (rings==nullptr)
      MR_assert(curofs==checkofs, "Bug in computing ofs[m]");
    ofs[m] = curofs;
    curofs+=ptrdiff_t(nph[m]);
    }

  return make_unique<sharp_standard_geom_info>(nrings, nph.data(), ofs.data(), stride, phi0.data(), theta.data(), weight_.data());
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
static vector<double> get_dh_weights(size_t nrings)
  {
  vector<double> weight(nrings);

  weight[0]=2.;
  for (size_t k=1; k<=(nrings/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k);
  weight[2*(nrings/2)-1]=(nrings-3.)/(2*(nrings/2)-1) -1.;
  pocketfft_r<double> plan(nrings);
  plan.exec(weight.data(), 1., false);
  return weight;
  }

void get_gridinfo(const string &type,
  mav<double, 1> &theta, mav<double, 1> &wgt)
  {
  auto nrings = theta.shape(0);
  bool do_wgt = (wgt.shape(0)!=0);
  if (do_wgt)
    MR_assert(wgt.shape(0)==nrings, "array size mismatch");

  if (type=="GL") // Gauss-Legendre
    {
    ducc0::GL_Integrator integ(nrings);
    auto cth = integ.coords();
    for (size_t m=0; m<nrings; ++m)
      theta.v(m) = acos(-cth[m]);
    if (do_wgt)
      {
      auto xwgt = integ.weights();
      for (size_t m=0; m<nrings; ++m)
        wgt.v(m) = 2*pi*xwgt[m];
      }
    }
  else if (type=="F1") // Fejer 1
    {
    for (size_t m=0; m<(nrings+1)/2; ++m)
      {
      theta.v(m)=pi*(m+0.5)/nrings;
      theta.v(nrings-1-m)=pi-theta(m);
      }
    if (do_wgt)
      {
      /* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
      vector<double> xwgt(nrings);
      xwgt[0]=2.;
      for (size_t k=1; k<=(nrings-1)/2; ++k)
        {
        xwgt[2*k-1]=2./(1.-4.*k*k)*cos((k*pi)/nrings);
        xwgt[2*k  ]=2./(1.-4.*k*k)*sin((k*pi)/nrings);
        }
      if ((nrings&1)==0) xwgt[nrings-1]=0.;
      pocketfft_r<double> plan(nrings);
      plan.exec(xwgt.data(), 1., false);
      for (size_t m=0; m<(nrings+1)/2; ++m)
        wgt.v(m)=wgt.v(nrings-1-m)=xwgt[m]*2*pi/nrings;
      }
    }
  else if (type=="CC") // Clenshaw-Curtis
    {
    for (size_t m=0; m<(nrings+1)/2; ++m)
      {
      theta.v(m)=max(1e-15,pi*m/(nrings-1.));
      theta.v(nrings-1-m)=pi-theta(m);
      }
    if (do_wgt)
      {
      /* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
      size_t n=nrings-1;
      double dw=-1./(n*n-1.+(n&1));
      vector<double> xwgt(nrings);
      xwgt[0]=2.+dw;
      for (size_t k=1; k<=(n/2-1); ++k)
        xwgt[2*k-1]=2./(1.-4.*k*k) + dw;
      //FIXME if (n>1) ???
      xwgt[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
      pocketfft_r<double> plan(n);
      plan.exec(xwgt.data(), 1., false);
      for (size_t m=0; m<(nrings+1)/2; ++m)
        wgt.v(m)=wgt.v(nrings-1-m)=xwgt[m]*2*pi/n;
      }
    }
  else if (type=="F2") // Fejer 2
    {
    for (size_t m=0; m<nrings; ++m)
      theta.v(m)=pi*(m+1)/(nrings+1.);
    if (do_wgt)
      {
      auto xwgt = get_dh_weights(nrings+1);
      for (size_t m=0; m<nrings; ++m)
        wgt.v(m) = xwgt[m+1]*2*pi/(nrings+1);
      }
    }
  else if (type=="DH") // Driscoll-Healy
    {
    for (size_t m=0; m<nrings; ++m)
      theta.v(m) = m*pi/nrings;
    if (do_wgt)
      {
      auto xwgt = get_dh_weights(nrings);
      for (size_t m=0; m<nrings; ++m)
        wgt.v(m) = xwgt[m]*2*pi/nrings;
      }
    }
  else if (type=="MW") // McEwen-Wiaux
    {
    for (size_t m=0; m<nrings; ++m)
      theta.v(m)=pi*(2.*m+1.)/(2.*nrings-1.);
    MR_assert(!do_wgt, "no quadrature weights exist for the MW grid");
    }
  else
    MR_fail("unsupported grid type");
  }

unique_ptr<sharp_geom_info> sharp_make_2d_geom_info
  (size_t nrings, size_t ppring, double phi0, ptrdiff_t stride_lon,
  ptrdiff_t stride_lat, const string &type, bool with_weight)
  {
  vector<size_t> nph(nrings, ppring);
  vector<double> phi0_(nrings, phi0);
  vector<ptrdiff_t> ofs(nrings);
  mav<double,1> theta({nrings}), weight({with_weight ? nrings : 0});
  get_gridinfo(type, theta, weight);
  for (size_t m=0; m<nrings; ++m)
    {
    ofs[m]=ptrdiff_t(m)*stride_lat;
    if (with_weight) weight.v(m) /= ppring;
    }
  return make_unique<sharp_standard_geom_info>(nrings, nph.data(), ofs.data(),
    stride_lon, phi0_.data(), theta.cdata(), with_weight ? weight.cdata() : nullptr);
  }

}}
