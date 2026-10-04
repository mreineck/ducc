#include "multiarch.h"

#include <iostream>

int main()
  {
  struct TestCase
    {
    int usable;
    int cap;
    bool has_v3;
    bool has_v4;
    int expected;
    const char *name;
    };

  const TestCase cases[] = {
    {1, 4, true, true, 1, "host v1, cap v4"},
    {3, 4, true, true, 3, "host v3, cap v4"},
    {4, 4, true, true, 4, "host v4, cap v4"},
    {4, 3, true, true, 3, "host v4, cap v3"},
    {4, 2, true, true, 1, "host v4, cap v2"},
    {4, 1, true, true, 1, "host v4, cap v1"},
    {3, 2, true, true, 1, "host v3, cap v2"},
    {4, 4, true, false, 3, "v4 implementation absent"},
    {4, 4, false, false, 1, "v3 and v4 implementations absent"},
    };

  bool ok = true;
  for (const auto &test : cases)
    {
    const auto result = ducc0_multiarch::select_psabi_level(
      test.usable, test.cap, test.has_v3, test.has_v4);
    if (result != test.expected)
      {
      std::cerr << "FAIL " << test.name << ": got " << result
                << ", expected " << test.expected << "\n";
      ok = false;
      }
    }
  if (ok) std::cout << "PASS multiarch selection policy\n";
  return ok ? 0 : 1;
  }
