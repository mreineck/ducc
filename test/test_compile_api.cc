/* Compile-only checks for public and template APIs that need to coexist in
   one translation unit but do not require a linked test executable. */
#include "ducc0/infra/timers.h"
#include "ducc0/sht/sphere_interpol.h"
#include "ducc0/wgridder/wgridder_impl.h"

#include <complex>
#include <cstddef>
#include <vector>

using namespace ducc0;
using std::vector;

void compile_sphere_interpol_api()
  {
  detail_sphereinterpol::SphereInterpol<double> inter(
    8, 8, 0, 0, 1.1, 2.6, 1e-8, 1);
  std::vector<std::complex<double>> almd(8);
  std::vector<double> planed(64);
  std::vector<size_t> msd(1);
  cmav<std::complex<double>, 2> valm(almd.data(), {1, 8});
  vmav<std::complex<double>, 2> valmw(almd.data(), {1, 8});
  vmav<double, 3> planes(planed.data(), {1, 8, 8});
  cmav<size_t, 1> mstart(msd.data(), {1});
  TimerHierarchy timers("test");
  inter.getPlane(valm, mstart, 1, planes, SHT_mode::STANDARD, timers);
  inter.updateAlm(valmw, mstart, 1, planes, SHT_mode::STANDARD, timers);
  }

struct VisBuf1D : public mav_info<1>
  {
  VisBuf1D(size_t n) : mav_info<1>({n}) {}
  std::complex<float> operator()(size_t) const { return {}; }
  void prefetch_r(size_t) const {}
  };

void compile_wgridder_custom_buffer_1d()
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
  }

struct VisBuf2D : public mav_info<2>
  {
  VisBuf2D(size_t r, size_t c) : mav_info<2>({r, c}) {}
  std::complex<float> operator()(size_t, size_t) const { return {}; }
  void prefetch_r(size_t, size_t) const {}
  };

void compile_wgridder_custom_buffer_2d()
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
  }
