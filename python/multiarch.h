#ifndef DUCC0_MULTIARCH_H
#define DUCC0_MULTIARCH_H

#include <algorithm>

namespace ducc0_multiarch {

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
