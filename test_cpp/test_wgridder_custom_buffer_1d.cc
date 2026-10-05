/* Unit test: Wgridder must be instantiable with a wsclean-style callback
   visibility buffer (deriving from ducc0::mav_info instead of vmav) for the
   1-D (BDA) path. This is the path fixed by commit b60cbec9; the test guards
   against reintroducing the `(gridding ? ms_in : ms_out)->shape()` pattern. */
#include "ducc0/wgridder/wgridder_impl.h"
#include <complex>
#include <vector>

using namespace ducc0;
using std::vector;

struct VisBuf1D : public mav_info<1>
  {
  VisBuf1D(size_t n) : mav_info<1>({n}) {}
  std::complex<float> operator()(size_t) const { return {}; }
  void prefetch_r(size_t) const {}
  };

int main()
  {
  vector<double> uvwd(6);
  vector<uint64_t> idd(2), nfd(1);
  vector<double> fqd(1);
  vector<float> wd(2);
  vector<uint8_t> md(2);
  cmav<double, 2> uvw(uvwd.data(), {2, 3});
  cmav<uint64_t, 1> freqlist_id(idd.data(), {2});
  cmav<uint64_t, 1> freqlist_nfreqs(nfd.data(), {1});
  cmav<double, 1> freqlist_freqs(fqd.data(), {1});
  cmav<float, 1> wgt1d(wd.data(), {2});
  cmav<uint8_t, 1> mask1d(md.data(), {2});
  VisBuf1D ms1d(2);
  auto dirty_in(vmav<float, 2>::build_empty());
  auto dirty_out(vmav<float, 2>::build_empty());
  detail_gridder::Wgridder<float, float, float, float,
                           VisBuf1D, cmav<std::complex<float>, 2>>
    par(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, &ms1d, nullptr,
        nullptr, nullptr, dirty_in, dirty_out, &wgt1d, nullptr, &mask1d,
        nullptr, 1., 1., 1e-6, false, 1, 0, false, false, false, true,
        1.1, 2.6, 0., 0., true);
  return 0;
  }
