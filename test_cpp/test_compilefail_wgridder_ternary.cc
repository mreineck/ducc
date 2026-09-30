/* Compile-fail proof: wgridder's 2D visibility check still uses the
   `(gridding ? ms2d_in : ms2d_out)->shape()` pattern that commit b60cbec9
   fixed for the BDA (1-D) path (src/ducc0/wgridder/wgridder_impl.h:719 and
   :1631).

   The conditional operator needs a composite pointer type even though only
   one branch is ever taken at runtime. For wsclean-style callback buffers
   (deriving from ducc0::mav_info, not from vmav) there is no composite type,
   so Wgridder cannot be instantiated with such a buffer for the 2-D path,
   while the 1-D path works after b60cbec9.

   Compile without -DBUGGY_2D: the 1-D (BDA) control instantiation must build.
   Compile with -DBUGGY_2D:    the 2-D instantiation must fail with
   "conditional expression between distinct pointer types".
*/
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

struct VisBuf2D : public mav_info<2>
  {
  VisBuf2D(size_t r, size_t c) : mav_info<2>({r, c}) {}
  std::complex<float> operator()(size_t, size_t) const { return {}; }
  void prefetch_r(size_t, size_t) const {}
  };

int main()
  {
  vector<double> uvwd(6);
  vector<uint64_t> idd(2), nfd(1);
  vector<double> fqd(1);
  cmav<double, 2> uvw(uvwd.data(), {2, 3});
  cmav<uint64_t, 1> freqlist_id(idd.data(), {2});
  cmav<uint64_t, 1> freqlist_nfreqs(nfd.data(), {1});
  cmav<double, 1> freqlist_freqs(fqd.data(), {1});
  auto dirty_in(vmav<float, 2>::build_empty());
  auto dirty_out(vmav<float, 2>::build_empty());

#ifdef BUGGY_2D
  VisBuf2D ms2d(2, 1);
  vector<float> wd(2);
  vector<uint8_t> md(2);
  cmav<float, 2> wgt2d(wd.data(), {2, 1});
  cmav<uint8_t, 2> mask2d(md.data(), {2, 1});
  detail_gridder::Wgridder<float, float, float, float, cmav<std::complex<float>, 1>, VisBuf2D>
    par(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, nullptr, &ms2d,
        nullptr, nullptr, dirty_in, dirty_out, nullptr, &wgt2d, nullptr,
        &mask2d, 1., 1., 1e-6, false, 1, 0, false, false, false, true,
        1.1, 2.6, 0., 0., true);
#else
  VisBuf1D ms1d(2);
  vector<float> wd(2);
  vector<uint8_t> md(2);
  cmav<float, 1> wgt1d(wd.data(), {2});
  cmav<uint8_t, 1> mask1d(md.data(), {2});
  detail_gridder::Wgridder<float, float, float, float, VisBuf1D, cmav<std::complex<float>, 2>>
    par(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, &ms1d, nullptr,
        nullptr, nullptr, dirty_in, dirty_out, &wgt1d, nullptr, &mask1d,
        nullptr, 1., 1., 1e-6, false, 1, 0, false, false, false, true,
        1.1, 2.6, 0., 0., true);
#endif
  return 0;
  }
