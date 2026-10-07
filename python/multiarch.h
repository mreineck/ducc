#ifndef DUCC0_MULTIARCH_H
#define DUCC0_MULTIARCH_H

#include <cstdint>
#include <string>
#include <vector>

namespace ducc0_multiarch {

using profile_mask = std::uint32_t;

constexpr profile_mask profile_bit(int psabi_level)
  { return (psabi_level >= 1 && psabi_level <= 4) ? (1u << psabi_level) : 0; }

// No v2 build is compiled; x86-64-v2 hosts use the v1 build.
inline constexpr profile_mask ducc_compiled_profiles_mask =
    profile_bit(1) | profile_bit(3) | profile_bit(4);

struct profile_state
  {
  int host_psabi_level;
  int configured_limit;
  int active_profile;
  };

namespace detail {

struct x86_features
  {
  bool sse3 = false;
  bool ssse3 = false;
  bool sse41 = false;
  bool sse42 = false;
  bool cx16 = false;
  bool lahf_sahf = false;
  bool popcnt = false;
  bool avx = false;
  bool xsave = false;
  bool osxsave = false;
  bool ymm_state = false;
  bool avx2 = false;
  bool bmi1 = false;
  bool bmi2 = false;
  bool f16c = false;
  bool fma = false;
  bool lzcnt = false;
  bool movbe = false;
  bool avx512f = false;
  bool avx512dq = false;
  bool avx512cd = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool zmm_state = false;
  };

constexpr bool usable_avx(const x86_features &features)
  {
  return features.avx && features.xsave && features.osxsave
      && features.ymm_state;
  }

constexpr bool usable_avx2(const x86_features &features)
  { return usable_avx(features) && features.avx2; }

constexpr bool usable_avx512(const x86_features &features)
  {
  return usable_avx(features) && features.avx512f && features.avx512dq
      && features.avx512cd && features.avx512bw && features.avx512vl
      && features.zmm_state;
  }

constexpr bool usable_psabi_v2(const x86_features &features)
  {
  return features.cx16 && features.lahf_sahf && features.popcnt
      && features.sse3 && features.ssse3 && features.sse41 && features.sse42;
  }

constexpr bool usable_psabi_v3(const x86_features &features)
  {
  return usable_psabi_v2(features) && usable_avx2(features)
      && features.bmi1 && features.bmi2 && features.f16c && features.fma
      && features.lzcnt && features.movbe;
  }

constexpr bool usable_psabi_v4(const x86_features &features)
  { return usable_psabi_v3(features) && usable_avx512(features); }

constexpr int psabi_level(const x86_features &features)
  {
  if (usable_psabi_v4(features)) return 4;
  if (usable_psabi_v3(features)) return 3;
  if (usable_psabi_v2(features)) return 2;
  return 1; // Linux x86-64 guarantees the x86-64-v1 ABI baseline.
  }

} // namespace detail

struct cpu_capabilities
  {
  int x86_psabi_level = 0;
  std::vector<std::string> features;
  };

const char *architecture_name();
cpu_capabilities detect_cpu_capabilities();
int configured_psabi_limit();
std::vector<int> compiled_profiles(
  profile_mask profiles = ducc_compiled_profiles_mask);
std::vector<int> available_profiles(profile_mask profiles,
                                    int host_psabi_level);
int select_profile(int host_psabi_level, int configured_limit,
                   profile_mask profiles);
const char *profile_name(int psabi_level);
profile_state current_profile_state(int host_psabi_level,
  profile_mask profiles = ducc_compiled_profiles_mask);

} // namespace ducc0_multiarch

#endif
