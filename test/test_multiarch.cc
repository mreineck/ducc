#include "multiarch.h"

#include <iostream>
#include <vector>

int main()
  {
  constexpr auto all_profiles = ducc0_multiarch::profile_bit(1)
    | ducc0_multiarch::profile_bit(2)
    | ducc0_multiarch::profile_bit(3)
    | ducc0_multiarch::profile_bit(4);
  constexpr auto v2_baseline_profiles = ducc0_multiarch::profile_bit(2)
    | ducc0_multiarch::profile_bit(3)
    | ducc0_multiarch::profile_bit(4);
  constexpr auto v1_v3_profiles = ducc0_multiarch::profile_bit(1)
    | ducc0_multiarch::profile_bit(3);
  constexpr auto v1_only_profiles = ducc0_multiarch::profile_bit(1);

  struct TestCase
    {
    int host_psabi_level;
    int configured_limit;
    ducc0_multiarch::profile_mask profiles;
    int expected;
    const char *name;
    };

  const TestCase cases[] = {
    {1, 4, ducc0_multiarch::ducc_compiled_profiles_mask, 1,
      "host v1, max v4, compiled {1,3,4}"},
    {2, 4, ducc0_multiarch::ducc_compiled_profiles_mask, 1,
      "host v2, max v4, compiled {1,3,4}"},
    {3, 4, ducc0_multiarch::ducc_compiled_profiles_mask, 3,
      "host v3, max v4, compiled {1,3,4}"},
    {4, 4, ducc0_multiarch::ducc_compiled_profiles_mask, 4,
      "host v4, max v4, compiled {1,3,4}"},
    {4, 3, ducc0_multiarch::ducc_compiled_profiles_mask, 3,
      "host v4, max v3, compiled {1,3,4}"},
    {4, 2, ducc0_multiarch::ducc_compiled_profiles_mask, 1,
      "host v4, max v2, compiled {1,3,4}"},
    {3, 2, ducc0_multiarch::ducc_compiled_profiles_mask, 1,
      "host v3, max v2, compiled {1,3,4}"},
    {4, 4, v1_v3_profiles, 3,
      "v4 implementation absent, host v4, max v4"},
    {4, 4, v1_only_profiles, 1,
      "v3 and v4 implementations absent"},
    {2, 4, all_profiles, 2, "host v2, max v4, compiled {1,2,3,4}"},
    {4, 2, all_profiles, 2, "host v4, max v2, compiled {1,2,3,4}"},
    {2, 4, v2_baseline_profiles, 2,
      "host v2, max v4, compiled {2,3,4}"},
    {2, 4, v1_v3_profiles, 1,
      "host v2, max v4, compiled {1,3}"},
    };

  bool ok = true;
  for (const auto &test : cases)
    {
    const auto result = ducc0_multiarch::select_profile(
      test.host_psabi_level, test.configured_limit, test.profiles);
    if (result != test.expected)
      {
      std::cerr << "FAIL " << test.name << ": got " << result
                << ", expected " << test.expected << "\n";
      ok = false;
      }
    }

  const auto check_profiles = [&ok](const char *name,
                                    const std::vector<int> &actual,
                                    const std::vector<int> &expected)
    {
    if (actual != expected)
      {
      std::cerr << "FAIL " << name << "\n";
      ok = false;
      }
    };
  check_profiles("compiled DUCC profiles",
    ducc0_multiarch::compiled_profiles(), {1, 3, 4});
  check_profiles("available DUCC profiles on v2 host",
    ducc0_multiarch::available_profiles(
      ducc0_multiarch::ducc_compiled_profiles_mask, 2), {1});
  check_profiles("available DUCC profiles on v3 host",
    ducc0_multiarch::available_profiles(
      ducc0_multiarch::ducc_compiled_profiles_mask, 3), {1, 3});
  check_profiles("available DUCC profiles on v4 host",
    ducc0_multiarch::available_profiles(
      ducc0_multiarch::ducc_compiled_profiles_mask, 4), {1, 3, 4});
  check_profiles("available all profiles on v2 host",
    ducc0_multiarch::available_profiles(all_profiles, 2), {1, 2});

  if (ok) std::cout << "PASS multiarch selection policy\n";
  return ok ? 0 : 1;
  }
