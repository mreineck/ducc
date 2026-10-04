#ifndef DUCC0_MULTIARCH_H
#define DUCC0_MULTIARCH_H

#include <algorithm>

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

inline int select_psabi_level(int usable_level, int selection_cap,
                              bool has_v3, bool has_v4)
  {
  const int limit = std::min(usable_level, selection_cap);
  if ((limit >= 4) && has_v4) return 4;
  if ((limit >= 3) && has_v3) return 3;
  return 1;
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

} // namespace ducc0_multiarch

#endif
