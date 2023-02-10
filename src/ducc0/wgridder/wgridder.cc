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

/* Copyright (C) 2019-2023 Max-Planck-Society
   Author: Martin Reinecke */

#include "ducc0/wgridder/wgridder.h"

namespace ducc0 {

namespace detail_gridder {

using namespace std;

auto get_winfo(const cmav<double,2> &uvw, const cmav<double,1> &freq,
               const cmav<uint8_t,2> &mask_, size_t nbin, size_t nthreads)
  {
  MR_assert(nbin<255, "too many bins requested");
  Baselines bl(uvw, freq, false);

  size_t nrow=bl.Nrows(),
         nchan=bl.Nchannels();
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform({nrow,nchan}, 1));
  checkShape(mask.shape(), {nrow,nchan});

  vmav<uint8_t,2> bin({nrow,nchan}, UNINITIALIZED);
  vmav<size_t,1> hist({nbin}, UNINITIALIZED);

  double wmin=1e300;
  double wmax=-1e300;
  Mutex mut;

  // determine wmin, wmax
  execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
    {
    double lwmin=1e300, lwmax=-1e300;
    for (auto irow=lo; irow<hi; ++irow)
      for (size_t ichan=0; ichan<nchan; ++ichan)
        if (mask(irow,ichan))
          {
          double w = bl.absEffectiveW(irow, ichan);
          lwmin = min(lwmin, w);
          lwmax = max(lwmax, w);
          }
    {
    LockGuard lock(mut);
    wmin = min(wmin, lwmin);
    wmax = max(wmax, lwmax);
    }
    });

  // do binning and histogram
  for (size_t i=0; i<nbin; ++i) hist(i) = 0;
  double xdw = nbin/(wmax-wmin);
  execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
    {
    vector<size_t> lhist(hist.shape(0),0);
    for (auto irow=lo; irow<hi; ++irow)
      for (size_t ichan=0; ichan<nchan; ++ichan)
        if (mask(irow,ichan))
          {
          double w = bl.absEffectiveW(irow, ichan);
          uint8_t ibin = min(size_t((w-wmin)*xdw), nbin-1);
          bin(irow,ichan) = ibin;
          ++lhist[ibin];
          }
    {
    LockGuard lock(mut);
    for (size_t i=0; i<nbin; ++i) hist(i) += lhist[i];
    }
    });

  return make_tuple(wmin,wmax,hist,bin);
  }

tuple<size_t, size_t, size_t, size_t, double, double>
  get_facet_data(size_t npix_x, size_t npix_y, size_t nfx, size_t nfy, size_t ifx, size_t ify,
  double pixsize_x, double pixsize_y, double center_x, double center_y)
  {
  size_t istep = (npix_x+nfx-1) / nfx;
  istep += istep%2;  // make even
  size_t jstep = (npix_y+nfy-1) / nfy;
  jstep += jstep%2;  // make even

  MR_assert((istep<=npix_x) && (jstep<=npix_y), "bad istep, jstep");

  size_t startx=ifx*istep, stopx=min((ifx+1)*istep, npix_x);
  MR_assert((startx+32<=stopx) && ((stopx&1)==0), "bad facet x length");
  size_t starty=ify*jstep, stopy=min((ify+1)*jstep, npix_y);
  MR_assert((starty+32<=stopy) && ((stopy&1)==0), "bad facet y length");

  double cx = center_x + pixsize_x*0.5*(startx+stopx-double(npix_x));
  double cy = center_y + pixsize_y*0.5*(starty+stopy-double(npix_y));
  return make_tuple(startx, starty, stopx, stopy, cx, cy);
  }

auto get_nminmax_rectangle(double xmin, double xmax, double ymin, double ymax)
  {
  vector<double> xext{xmin, xmax},
                 yext{ymin, ymax};
  if (xmin*xmax<0) xext.push_back(0);
  if (ymin*ymax<0) yext.push_back(0);
  double nm1min = 1e300, nm1max = -1e300;
  for (auto xc: xext)
    for (auto yc: yext)
      {
      double tmp = xc*xc+yc*yc;
      double nval = (tmp<=1.) ?  (sqrt(1.-tmp)-1.) : (-sqrt(tmp-1.)-1.);
      nm1min = min(nm1min, nval);
      nm1max = max(nm1max, nval);
      }
  return make_pair(nm1min, nm1max);
  }

double get_sum_nminmax(const vector<double> &xlist, const vector<double> &ylist)
  {
  auto nx=xlist.size();
  auto ny=ylist.size();
  if ((nx<2) || (ny<2)) return 0.;

  double res=0.;
  for (size_t i=0; i+1<nx; ++i)
    for (size_t j=0; j+1<ny; ++j)
      {
      auto [nmin,nmax] = get_nminmax_rectangle(xlist[i], xlist[i+1], ylist[j], ylist[j+1]);
      res += (nmax-nmin) * (xlist[i+1]-xlist[i]) * (ylist[j+1]-ylist[j]);
      }
  return res;
  }

tuple<vmav<uint8_t,2>,size_t,size_t,size_t> get_tuning_parameters(
  const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<uint8_t,2> &mask_,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads,
  size_t verbosity, double center_x, double center_y)
  {
  if ((!do_wgridding) || (npix_x<500) || (npix_y<500))
    {
    if (verbosity>0)
      cout << "Tuning decision:\n  no subdivision" << endl;
    return make_tuple(vmav<uint8_t,2>::build_empty(), size_t(0), size_t(0), size_t(0));
    }
  auto W = size_t(max(3, min(16, int(2-log10(epsilon)))));

  constexpr double sigma = 1.5;

  constexpr size_t nbin=50;
  auto [wmin, wmax, whist, wbin] = get_winfo(uvw, freq, mask_, nbin, nthreads);
  vmav<size_t,1> whist_acc({nbin}, UNINITIALIZED);
  for (size_t i=0; i<whist.shape(0); ++i)
    whist_acc(i) = whist(i) + ((i==0) ? 0 : whist_acc(i-1));
  vector<double> wborders(nbin+1);
  for (size_t i=0; i<wborders.size(); ++i)
    wborders[i] = wmin + i*(wmax-wmin)/nbin;

  double xmin = center_x - 0.5*npix_x*pixsize_x,
         ymin = center_y - 0.5*npix_y*pixsize_y,
         xmax = xmin + (npix_x-1)*pixsize_x,
         ymax = ymin + (npix_y-1)*pixsize_y;

  auto [nm1min, nm1max] = get_nminmax_rectangle(xmin, xmax, ymin, ymax);
  double dw = 1./sigma/abs(nm1min-nm1max);

  constexpr size_t vlen=4;
  size_t nvec = (W+vlen-1)/vlen;
  double gridcost0 = 2.2e-10*W*(W*nvec*vlen + ((2*nvec+1)*(W+3)*vlen));
  constexpr double nref_fft = 2048;
  constexpr double costref_fft = 0.0693;
  double nu=sigma*npix_x, nv=sigma*npix_y;
  double logterm = log(nu*nv)/log(nref_fft*nref_fft);
  double fftcost0 = npix_x/nref_fft*nv/nref_fft*logterm*costref_fft * 1.3;
  double overhead = 4e-9*uvw.shape(0)*freq.shape(0);

  {
  // check for early exit
  auto nplanes_naive = (wmax-wmin)/dw+W;
  auto gridcost_naive = gridcost0*whist_acc(nbin-1);
  auto fftcost_naive = fftcost0*nplanes_naive;
  if ((nplanes_naive<=2*W) || (gridcost_naive>2*fftcost_naive))
    {
    if (verbosity>0)
      cout << "Tuning decision:\n  no subdivision" << endl;
    return make_tuple(vmav<uint8_t,2>::build_empty(), size_t(0), size_t(0), size_t(0));
    }
  }

  double minmincost=1e300;
  size_t minminnfx=0, minminnfy=0, miniwcut=0;
  for (size_t iwcut=0; iwcut<wborders.size(); ++iwcut)
    {
    double wcut = wborders[iwcut];
    auto nvis1 = (iwcut==0) ? 0 : whist_acc(iwcut-1);
    auto nvis2 = whist_acc(nbin-1)-nvis1;
//cout << "nvis: " << nvis1 << " " << nvis2 << " " << nvis1+nvis2 << endl;
    double mincost=1e300;
    size_t minnfx=0, minnfy=0;
    if (nvis2>0)
      {
      for (size_t nfx=1; nfx<8; ++nfx)
        for (size_t nfy=(nfx==1)?2:1; nfy<8; ++nfy)
          {
          double gridcost = nfx*nfy*nvis2*gridcost0;
          vector<double> xlist(nfx+1), ylist(nfy+1);
          for (size_t i=0; i<=nfx; ++i)
            xlist[i] = xmin + i*(xmax-xmin)/nfx;
          for (size_t i=0; i<=nfy; ++i)
            ylist[i] = ymin + i*(ymax-ymin)/nfy;
          double totplanes=0;
          for (size_t i=0; i<nfx; ++i)
            for (size_t j=0; j<nfy; ++j)
              {
              auto [lnm1min, lnm1max] = get_nminmax_rectangle
                (xlist[i], xlist[i+1], ylist[j], ylist[j+1]);
              double ldw = 1./sigma/abs(lnm1min-lnm1max);
              totplanes += (wmax-wcut)/ldw+W;
              }
          double fftcost=(fftcost0*totplanes)/(nfx*nfy)*log(nu*nv*1./(nfx*nfy))/log(nu*nv);
          double cost = gridcost+fftcost+nfx*nfy*overhead;
          if (cost < mincost)
            {
            mincost = cost;
            minnfx = nfx;
            minnfy = nfy;
            }
          }
      }
    else
      mincost=minnfx=minnfy = 0;

    if (iwcut != 0)
      {
      // FIXME: fudge factor to account for the fact that FFT is typically sparse in this situation
      double fudge = (nvis2==0) ? 0.75 : 1.;
      mincost += fudge*fftcost0*((wcut-wmin)/dw+W) + gridcost0*nvis1 + overhead;
      }
//cout << "iwcut: " << iwcut << " " << mincost << " " << minnfx << " " << minnfy << endl;
    if (mincost<minmincost)
      {
      minmincost = mincost;
      minminnfx = minnfx;
      minminnfy = minnfy;
      miniwcut = iwcut;
      }
    }

  if (verbosity>0)
    {
    cout << "Tuning decision:" << endl;
    if (miniwcut==0)  // only facets
      cout << "  subdividing into " << minminnfx << "x" << minminnfy << " facets." << endl;
    else if (miniwcut+1==wborders.size())  // no facets
      cout << "  no subdivision" << endl;
    else
      cout << "  subdividing w range at w=" << wborders[miniwcut]
           << ", subdividing high w part into " << minminnfx << "x" << minminnfy << " facets." << endl;
    }

  if (miniwcut==0)  // only facets
    return make_tuple(vmav<uint8_t,2>::build_empty(), size_t(0), minminnfx, minminnfy);
  else if (miniwcut+1==wborders.size())  // no facets
    return make_tuple(vmav<uint8_t,2>::build_empty(), size_t(0), size_t(0), size_t(0));
  else
    return make_tuple(wbin, miniwcut, minminnfx, minminnfy);
  }

}}
