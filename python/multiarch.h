#ifndef DUCC0_MULTIARCH_H
#define DUCC0_MULTIARCH_H

#include <cstdint>
#include <vector>

namespace ducc0_multiarch {

using profile_mask = std::uint32_t;

constexpr profile_mask profile_bit(int psabi_level)
  { return (psabi_level >= 1 && psabi_level <= 4) ? (1u << psabi_level) : 0; }

inline constexpr profile_mask ducc_compiled_profiles_mask =
    profile_bit(1) | profile_bit(3) | profile_bit(4);

struct profile_state
  {
  int host_psabi_level;
  int configured_limit;
  int active_profile;
  };

const char *architecture_name();
int usable_psabi_level();
int configured_psabi_limit();
std::vector<int> compiled_profiles(
  profile_mask profiles = ducc_compiled_profiles_mask);
std::vector<int> available_profiles(profile_mask profiles,
                                    int host_psabi_level);
int select_profile(int host_psabi_level, int configured_limit,
                   profile_mask profiles);
const char *profile_name(int psabi_level);
profile_state current_profile_state(
  profile_mask profiles = ducc_compiled_profiles_mask);

} // namespace ducc0_multiarch

#endif
