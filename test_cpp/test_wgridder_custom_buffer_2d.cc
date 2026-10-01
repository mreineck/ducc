/* Unit test: Wgridder must be instantiable with a wsclean-style callback
   visibility buffer (deriving from ducc0::mav_info instead of vmav) for the
   2-D path as well.

   Currently fails to compile: wgridder_impl.h:719 and :1631 still use the
   `(gridding ? ms2d_in : ms2d_out)->shape()` pattern, which has no
   composite pointer type for such buffers - the same pattern that commit
   b60cbec9 removed from the 1-D path. */
#include "ducc0/wgridder/wgridder_impl.h"
#include <complex>
#include <vector>

using namespace ducc0;
using std::vector;

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
  vector<float> wd(2);
  vector<uint8_t> md(2);
  cmav<double, 2> uvw(uvwd.data(), {2, 3});
  cmav<uint64_t, 1> freqlist_id(idd.data(), {2});
  cmav<uint64_t, 1> freqlist_nfreqs(nfd.data(), {1});
  cmav<double, 1> freqlist_freqs(fqd.data(), {1});
  cmav<float, 2> wgt2d(wd.data(), {2, 1});
  cmav<uint8_t, 2> mask2d(md.data(), {2, 1});
  VisBuf2D ms2d(2, 1);
  auto dirty_in(vmav<float, 2>::build_empty());
  auto dirty_out(vmav<float, 2>::build_empty());
  detail_gridder::Wgridder<float, float, float, float,
                           cmav<std::complex<float>, 1>, VisBuf2D>
    par(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, nullptr, &ms2d,
        nullptr, nullptr, dirty_in, dirty_out, nullptr, &wgt2d, nullptr,
        &mask2d, 1., 1., 1e-6, false, 1, 0, false, false, false, true,
        1.1, 2.6, 0., 0., true);
  return 0;
  }
