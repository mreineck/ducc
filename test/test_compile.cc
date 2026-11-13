/* Compile-only checks that keep selected C++ interfaces and templates
   instantiable with the current DUCC interfaces. */
#include "ducc0/infra/timers.h"
#include "ducc0/sht/sphere_interpol.h"
#include "ducc0/wgridder/wgridder_impl.h"

#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

using namespace ducc0;
using std::vector;

/* SphereInterpol's current getPlane/updateAlm signatures must stay usable;
   the obsolete overloads removed for PR #79 forwarded to nonexistent APIs. */
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

/* The callback visibility-buffer path must remain instantiable for 1-D BDA
   buffers deriving from mav_info rather than vmav; this guards against
   relying on `(gridding ? ms_in : ms_out)->shape()` for those buffer types. */
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

/* The custom callback buffer must remain instantiable on the separate 2-D
   path too; the input/output buffer pointers have no common pointer type for
   a `(gridding ? ms2d_in : ms2d_out)->shape()` expression. */
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

/* A writable vfmav must not be constructible from the buffer of a read-only
   array; sharing the data of another writable array must keep working. */
static_assert(!std::is_constructible_v<vfmav<double>, cfmav<double> &,
  fmav_info::shape_t, fmav_info::stride_t>);
static_assert(!std::is_constructible_v<vfmav<double>, cmav<double, 2> &,
  fmav_info::shape_t, fmav_info::stride_t>);
static_assert(std::is_constructible_v<vfmav<double>, const vfmav<double> &,
  fmav_info::shape_t, fmav_info::stride_t>);
static_assert(std::is_convertible_v<vmav<double, 2>, vfmav<double>>);

static_assert(std::is_same_v<decltype(std::declval<vfmav<double>>()
  .extend_and_broadcast(fmav_info::shape_t{}, 0)), cfmav<double>>);
static_assert(std::is_same_v<decltype(std::declval<vmav<double, 1>>()
  .extend_and_broadcast(std::array<size_t, 2>{}, 0)), cmav<double, 2>>);

/* mav_info_proto handles fixed and dynamic rank in one class template and
   must keep working with shape/stride containers other than std::array
   (e.g. cuda::std::array in GPU code). */
template<typename T, size_t N> struct dev_array
  {
  T v[N];
  constexpr size_t size() const { return N; }
  T *begin() { return v; }
  T *end() { return v+N; }
  const T *begin() const { return v; }
  const T *end() const { return v+N; }
  T *data() { return v; }
  const T *data() const { return v; }
  T &operator[](size_t i) { return v[i]; }
  const T &operator[](size_t i) const { return v[i]; }
  const T &back() const { return v[N-1]; }
  bool operator==(const dev_array &) const = default;
  };
template<typename C> concept has_rank
  = requires { detail_mav::rank_of_impl<C>::value; };
static_assert(detail_mav::rank_of<std::vector<size_t>> == dynamic_rank);
static_assert(detail_mav::rank_of<std::array<size_t, 3>> == 3);
static_assert(detail_mav::rank_of<dev_array<size_t, 2>> == 2);
static_assert(!has_rank<const std::array<size_t, 3>>);
static_assert(!has_rank<std::array<size_t, 3> &>);
static_assert(!has_rank<const std::vector<size_t> &>);
static_assert(!has_rank<int>);
static_assert(std::is_same_v<fmav_info, mav_info<dynamic_rank>>);
static_assert(std::is_convertible_v<mav_info<2>, fmav_info>);
static_assert(!std::is_convertible_v<fmav_info, mav_info<2>>);
static_assert(std::is_constructible_v<mav_info<2>, fmav_info>);
static_assert(!std::is_constructible_v<mav_info<2>, mav_info<3>>);
void compile_mav_info_ranks()
  {
  using dev_info = detail_mav::mav_info_proto<dev_array, 2>;
  dev_info d({2, 3});
  fmav_info f = d;
  dev_info d2(f);
  (void)d2.idx(1, 2);
  (void)d.contiguous();
  (void)d.transpose({1, 0});
  (void)d.prepend_1();
  (void)d.extend_and_broadcast(dev_array<size_t, 3>{4, 2, 3}, 1);
  (void)f.idx(1, 2);
  (void)f.contiguous();
  (void)f.extend_and_broadcast(fmav_info::shape_t{4, 2, 3}, 1);
  }

/* cfmav is the dynamic-rank cmav: conversions between ranks are implicit
   towards dynamic rank and explicit towards fixed rank, deduction through
   the alias keeps working, and rank-specific helpers exist for both. */
static_assert(std::is_same_v<cfmav<double>, cmav<double, dynamic_rank>>);
static_assert(std::is_convertible_v<cmav<double, 2>, cfmav<double>>);
static_assert(!std::is_convertible_v<cfmav<double>, cmav<double, 2>>);
static_assert(std::is_constructible_v<cmav<double, 2>, cfmav<double>>);
static_assert(!std::is_constructible_v<cmav<double, 2>, cmav<double, 3>>);
template<typename T> size_t takes_cfmav(const cfmav<T> &a) { return a.ndim(); }
void compile_cmav_ranks()
  {
  std::vector<double> d(6);
  cmav<double, 2> a(d.data(), {2, 3});
  cfmav<double> f(d.data(), {2, 3});
  (void)takes_cfmav(f);
  (void)takes_cfmav(a.to_fmav());
  (void)a.subarray(vector<slice>{{0, 1}, {}});
  (void)f.sub(1, slice());
  (void)a.extend_and_broadcast(std::array<size_t, 3>{4, 2, 3}, 1);
  (void)f.extend_and_broadcast(fmav_info::shape_t{4, 2, 3}, 1);
  (void)f.prepend_1();
  (void)cfmav<double>::build_uniform({2, 3}, 1.);
  }

/* vfmav is the dynamic-rank vmav, with the same conversion rules as cfmav. */
static_assert(std::is_same_v<vfmav<double>, vmav<double, dynamic_rank>>);
static_assert(std::is_convertible_v<vmav<double, 2>, vfmav<double>>);
static_assert(std::is_convertible_v<vmav<double, 2>, cfmav<double>>);
static_assert(!std::is_convertible_v<vfmav<double>, vmav<double, 2>>);
static_assert(std::is_constructible_v<vmav<double, 2>, vfmav<double>>);
static_assert(!std::is_constructible_v<vmav<double, 2>, cmav<double, 2>>);
template<typename T> size_t takes_vfmav(const vfmav<T> &a) { return a.ndim(); }
void compile_vmav_ranks()
  {
  std::vector<double> d(6);
  vmav<double, 2> a(d.data(), {2, 3});
  vfmav<double> f(d.data(), {2, 3});
  (void)takes_vfmav(f);
  (void)takes_vfmav(a.to_fmav());
  vmav<double, 1> s(f.sub(1, slice()));
  s(0) = 1.;
  f(fmav_info::shape_t{1, 2}) = 2.;
  (void)f.prepend_1();
  (void)vfmav<double>::build_noncritical({2, 3}, UNINITIALIZED);
  }

/* sub(): integers drop an axis, slices keep it; the result rank is known at
   compile time for fixed rank and stays dynamic for dynamic rank. */
template<typename A, typename... S> concept has_sub
  = requires(const A &a, S... s) { a.sub(s...); };
static_assert(std::is_same_v<decltype(std::declval<cmav<double, 3>>()
  .sub(1, slice(), slice(0, 2))), cmav<double, 2>>);
static_assert(std::is_same_v<decltype(std::declval<vmav<double, 3>>()
  .sub(1, size_t(2), slice())), vmav<double, 1>>);
static_assert(std::is_same_v<decltype(std::declval<vfmav<double>>()
  .sub(1, slice())), vfmav<double>>);
static_assert(!has_sub<cmav<double, 3>, int, slice>);      // wrong count
static_assert(!has_sub<cmav<double, 2>, double, slice>);   // not an index
