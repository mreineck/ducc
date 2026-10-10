#include "multiarch.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

#if defined(__linux__) && defined(__x86_64__)
#include <cpuid.h>
#elif defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace ducc0_multiarch {

const char *architecture_name()
  {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__i386__) || defined(_M_IX86)
  return "i686";
#elif defined(__powerpc64__) || defined(__ppc64__)
  return "ppc64";
#elif defined(__powerpc__) || defined(__ppc__)
  return "ppc";
#elif defined(__wasm64__)
  return "wasm64";
#elif defined(__wasm32__)
  return "wasm32";
#elif defined(__riscv)
  return "riscv";
#else
  return "unknown";
#endif
  }

const char *profile_name(int psabi_level)
  {
  switch (psabi_level)
    {
    case 1: return "x86-64";
    case 2: return "x86-64-v2";
    case 3: return "x86-64-v3";
    case 4: return "x86-64-v4";
    default: return "unknown";
    }
  }

std::vector<int> compiled_profiles(profile_mask profiles)
  {
  std::vector<int> result;
  for (int psabi_level=1; psabi_level<=4; ++psabi_level)
    if (profiles & profile_bit(psabi_level)) result.push_back(psabi_level);
  return result;
  }

std::vector<int> available_profiles(profile_mask profiles,
                                    int host_psabi_level)
  {
  std::vector<int> result;
  const int limit = std::min(host_psabi_level, 4);
  for (int psabi_level=1; psabi_level<=limit; ++psabi_level)
    if (profiles & profile_bit(psabi_level)) result.push_back(psabi_level);
  return result;
  }

int select_profile(int host_psabi_level, int configured_limit,
                   profile_mask profiles)
  {
  const int limit = std::min(std::min(host_psabi_level, configured_limit), 4);
  for (int psabi_level=limit; psabi_level>=1; --psabi_level)
    if (profiles & profile_bit(psabi_level)) return psabi_level;
  return 0;
  }

int configured_psabi_limit()
  {
  const char *value = std::getenv("DUCC0_MAX_PSABI_LEVEL");
  if (value == nullptr) return 4;

  char *end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  while (end != value && *end != '\0'
      && std::isspace(static_cast<unsigned char>(*end)))
    ++end;

  if (end == value || *end != '\0' || errno == ERANGE)
    throw std::runtime_error(
      "invalid DUCC0_MAX_PSABI_LEVEL: expected an integer");
  if (parsed < 1) return 1;
  if (parsed > 4) return 4;
  return static_cast<int>(parsed);
  }

#if defined(__linux__) && defined(__x86_64__)

namespace {

struct cpuid_result
  { std::uint32_t eax, ebx, ecx, edx; };

cpuid_result cpuid(std::uint32_t leaf, std::uint32_t subleaf)
  {
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return {eax, ebx, ecx, edx};
  }

bool has_bit(std::uint32_t value, unsigned bit)
  { return (value & (std::uint32_t(1) << bit)) != 0; }

std::uint64_t read_xcr(std::uint32_t index)
  {
  std::uint32_t eax, edx;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
    : "=a"(eax), "=d"(edx) : "c"(index));
  return (std::uint64_t(edx)<<32) | eax;
  }

detail::x86_features detect_x86_features()
  {
  detail::x86_features features;
  const auto leaf0 = cpuid(0, 0);
  const auto leaf1 = (leaf0.eax>=1) ? cpuid(1, 0) : cpuid_result{0,0,0,0};

  features.sse3 = has_bit(leaf1.ecx, 0);
  features.ssse3 = has_bit(leaf1.ecx, 9);
  features.fma = has_bit(leaf1.ecx, 12);
  features.cx16 = has_bit(leaf1.ecx, 13);
  features.sse41 = has_bit(leaf1.ecx, 19);
  features.sse42 = has_bit(leaf1.ecx, 20);
  features.movbe = has_bit(leaf1.ecx, 22);
  features.popcnt = has_bit(leaf1.ecx, 23);
  features.xsave = has_bit(leaf1.ecx, 26);
  features.osxsave = has_bit(leaf1.ecx, 27);
  features.avx = has_bit(leaf1.ecx, 28);
  features.f16c = has_bit(leaf1.ecx, 29);

  const auto leaf7 = (leaf0.eax>=7) ? cpuid(7, 0) : cpuid_result{0,0,0,0};
  features.bmi1 = has_bit(leaf7.ebx, 3);
  features.avx2 = has_bit(leaf7.ebx, 5);
  features.bmi2 = has_bit(leaf7.ebx, 8);
  features.avx512f = has_bit(leaf7.ebx, 16);
  features.avx512dq = has_bit(leaf7.ebx, 17);
  features.avx512cd = has_bit(leaf7.ebx, 28);
  features.avx512bw = has_bit(leaf7.ebx, 30);
  features.avx512vl = has_bit(leaf7.ebx, 31);

  constexpr std::uint32_t high_leaf = 0x80000000;
  const auto leaf_high0 = cpuid(high_leaf, 0);
  const auto leaf_high1 = (leaf_high0.eax>=high_leaf+1)
    ? cpuid(high_leaf+1, 0) : cpuid_result{0,0,0,0};
  features.lahf_sahf = has_bit(leaf_high1.ecx, 0);
  features.lzcnt = has_bit(leaf_high1.ecx, 5);

  std::uint64_t xcr0 = 0;
  if (features.xsave && features.osxsave) xcr0 = read_xcr(0);
  features.ymm_state = (xcr0 & 0x6)==0x6;
  features.zmm_state = (xcr0 & 0xe6)==0xe6;
  return features;
  }

} // namespace

cpu_capabilities detect_cpu_capabilities()
  {
  const auto features = detect_x86_features();
  std::vector<std::string> names{"sse2"};
  if (features.sse3) names.emplace_back("sse3");
  if (features.ssse3) names.emplace_back("ssse3");
  if (features.sse41) names.emplace_back("sse4.1");
  if (features.sse42) names.emplace_back("sse4.2");
  if (detail::usable_avx(features)) names.emplace_back("avx");
  if (detail::usable_avx2(features)) names.emplace_back("avx2");
  if (detail::usable_avx512(features)) names.emplace_back("avx512");
  return {detail::psabi_level(features), names};
  }

#else

cpu_capabilities detect_cpu_capabilities()
  {
  std::vector<std::string> names;
#if defined(__linux__) && defined(__aarch64__)
  const auto hwcap = getauxval(AT_HWCAP);
#ifdef HWCAP_ASIMD
  if (hwcap & HWCAP_ASIMD) names.emplace_back("neon");
#endif
#ifdef HWCAP_SVE
  if (hwcap & HWCAP_SVE) names.emplace_back("sve");
#endif
#ifdef HWCAP2_SVE2
  unsigned long hwcap2 = 0;
#ifdef AT_HWCAP2
  hwcap2 = getauxval(AT_HWCAP2);
#endif
  if (hwcap2 & HWCAP2_SVE2) names.emplace_back("sve2");
#endif
#endif
  return {0, names};
  }

#endif

profile_state current_profile_state(int host_psabi_level,
                                    profile_mask profiles)
  {
  const int configured_limit = configured_psabi_limit();
  return {host_psabi_level, configured_limit,
          select_profile(host_psabi_level, configured_limit, profiles)};
  }

} // namespace ducc0_multiarch
