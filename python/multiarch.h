#ifndef DUCC0_MULTIARCH_H
#define DUCC0_MULTIARCH_H

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__linux__) && defined(__x86_64__)
#include <cpuid.h>
#endif

namespace ducc0_multiarch {

inline const char *architecture_name()
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

constexpr unsigned psabi_level_bit(int level)
  { return 1u << level; }

inline constexpr unsigned ducc_compiled_psabi_mask =
    psabi_level_bit(1) | psabi_level_bit(3) | psabi_level_bit(4);

inline std::vector<int> compiled_levels(unsigned profile_mask =
                                       ducc_compiled_psabi_mask)
  {
  std::vector<int> result;
  for (int level=1; level<=4; ++level)
    if (profile_mask & psabi_level_bit(level)) result.push_back(level);
  return result;
  }

inline std::vector<int> available_levels(unsigned profile_mask,
                                         int usable_level)
  {
  std::vector<int> result;
  const int limit = std::min(usable_level, 4);
  for (int level=1; level<=limit; ++level)
    if (profile_mask & psabi_level_bit(level)) result.push_back(level);
  return result;
  }

inline int select_psabi_level(int usable_level, int max_level,
                              unsigned profile_mask)
  {
  const int limit = std::min(std::min(usable_level, max_level), 4);
  for (int level=limit; level>=1; --level)
    if (profile_mask & psabi_level_bit(level)) return level;
  return 0;
  }

inline const char *level_name(int level)
  {
  switch (level)
    {
    case 1: return "x86-64";
    case 2: return "x86-64-v2";
    case 3: return "x86-64-v3";
    case 4: return "x86-64-v4";
    default: return "unknown";
    }
  }

inline int max_psabi_level()
  {
  const char *value = std::getenv("DUCC0_MAX_PSABI_LEVEL");
  if (value == nullptr) return 4;

  const char *first = value;
  while (*first != '\0' && std::isspace(static_cast<unsigned char>(*first)))
    ++first;

  char *end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  while (end != value && *end != '\0'
      && std::isspace(static_cast<unsigned char>(*end)))
    ++end;

  if (end == value || *end != '\0')
    {
    std::fprintf(stderr,
      "warning: invalid DUCC0_MAX_PSABI_LEVEL; using x86-64-v4\n");
    return 4;
    }

  if (errno == ERANGE)
    return (*first == '-') ? 1 : 4;
  if (parsed < 1) return 1;
  if (parsed > 4) return 4;
  return static_cast<int>(parsed);
  }

#if defined(__linux__) && defined(__x86_64__)

namespace detail {

struct cpuid_result
  { std::uint32_t eax, ebx, ecx, edx; };

inline cpuid_result cpuid(std::uint32_t leaf, std::uint32_t subleaf)
  {
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return {eax, ebx, ecx, edx};
  }

inline bool has_bit(std::uint32_t value, unsigned bit)
  { return (value & (std::uint32_t(1) << bit)) != 0; }

inline std::uint64_t read_xcr(std::uint32_t index)
  {
  std::uint32_t eax, edx;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
    : "=a"(eax), "=d"(edx) : "c"(index));
  return (std::uint64_t(edx)<<32) | eax;
  }

} // namespace detail

inline int usable_psabi_level()
  {
  using namespace detail;

  const auto leaf0 = cpuid(0, 0);
  const auto leaf1 = (leaf0.eax>=1) ? cpuid(1, 0) : cpuid_result{0,0,0,0};

  const bool sse3 = has_bit(leaf1.ecx, 0);
  const bool ssse3 = has_bit(leaf1.ecx, 9);
  const bool fma = has_bit(leaf1.ecx, 12);
  const bool cx16 = has_bit(leaf1.ecx, 13);
  const bool sse41 = has_bit(leaf1.ecx, 19);
  const bool sse42 = has_bit(leaf1.ecx, 20);
  const bool movbe = has_bit(leaf1.ecx, 22);
  const bool popcnt = has_bit(leaf1.ecx, 23);
  const bool xsave = has_bit(leaf1.ecx, 26);
  const bool osxsave = has_bit(leaf1.ecx, 27);
  const bool avx = has_bit(leaf1.ecx, 28);
  const bool f16c = has_bit(leaf1.ecx, 29);

  const auto leaf7 = (leaf0.eax>=7) ? cpuid(7, 0) : cpuid_result{0,0,0,0};
  const bool bmi1 = has_bit(leaf7.ebx, 3);
  const bool avx2 = has_bit(leaf7.ebx, 5);
  const bool bmi2 = has_bit(leaf7.ebx, 8);
  const bool avx512f = has_bit(leaf7.ebx, 16);
  const bool avx512dq = has_bit(leaf7.ebx, 17);
  const bool avx512cd = has_bit(leaf7.ebx, 28);
  const bool avx512bw = has_bit(leaf7.ebx, 30);
  const bool avx512vl = has_bit(leaf7.ebx, 31);

  constexpr std::uint32_t high_leaf = 0x80000000;
  const auto leaf_high0 = cpuid(high_leaf, 0);
  const auto leaf_high1 = (leaf_high0.eax>=high_leaf+1)
    ? cpuid(high_leaf+1, 0) : cpuid_result{0,0,0,0};
  const bool lahf_sahf = has_bit(leaf_high1.ecx, 0);
  const bool lzcnt = has_bit(leaf_high1.ecx, 5);

  std::uint64_t xcr0 = 0;
  if (xsave && osxsave) xcr0 = read_xcr(0);
  const bool ymm_state = (xcr0 & 0x6)==0x6;
  const bool zmm_state = (xcr0 & 0xe6)==0xe6;
  const bool avx_usable = avx && xsave && osxsave && ymm_state;

  int level = 1; // The dispatcher itself is compiled for x86-64-v1.
  if (cx16 && lahf_sahf && popcnt && sse3 && ssse3 && sse41 && sse42)
    level = 2;
  if ((level==2) && avx_usable && avx2 && bmi1 && bmi2 && f16c && fma
      && lzcnt && movbe)
    level = 3;
  if ((level==3) && avx512f && avx512dq && avx512cd && avx512bw && avx512vl
      && zmm_state)
    level = 4;

  return level;
  }

#else

inline int usable_psabi_level() { return 0; }

#endif

} // namespace ducc0_multiarch

#endif
