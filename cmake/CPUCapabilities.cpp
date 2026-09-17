/*
 * Small, standalone CPU capability diagnostic for CI and developer use.
 *
 * This file is intentionally not part of the DUCC0 library or its build.
 */

#include <cstdint>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
#define DUCC0_CPU_X86
#define DUCC0_CPU_X86_64
#elif defined(__i386__) || defined(_M_IX86)
#define DUCC0_CPU_X86
#endif

#if defined(DUCC0_CPU_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace {

#if defined(DUCC0_CPU_X86)

struct cpuid_result
  {
  std::uint32_t eax, ebx, ecx, edx;
  };

cpuid_result cpuid(std::uint32_t leaf, std::uint32_t subleaf)
  {
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  return {static_cast<std::uint32_t>(regs[0]),
          static_cast<std::uint32_t>(regs[1]),
          static_cast<std::uint32_t>(regs[2]),
          static_cast<std::uint32_t>(regs[3])};
#else
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return {eax, ebx, ecx, edx};
#endif
  }

bool has_bit(std::uint32_t value, unsigned bit)
  { return (value & (std::uint32_t(1) << bit)) != 0; }

std::uint64_t read_xcr(std::uint32_t index)
  {
#if defined(_MSC_VER)
  return static_cast<std::uint64_t>(_xgetbv(index));
#else
  std::uint32_t eax, edx;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
    : "=a"(eax), "=d"(edx) : "c"(index));
  return (std::uint64_t(edx)<<32) | eax;
#endif
  }

void print_flag(const char *name, bool value)
  { std::cout << ' ' << name << '=' << (value ? 1 : 0); }

#endif

}

int main()
  {
#if defined(DUCC0_CPU_X86)
  const auto leaf0 = cpuid(0, 0);
  const auto leaf1 = (leaf0.eax>=1) ? cpuid(1, 0) : cpuid_result{0,0,0,0};
  const auto leaf7 = (leaf0.eax>=7) ? cpuid(7, 0) : cpuid_result{0,0,0,0};

  const bool sse = has_bit(leaf1.edx, 25);
  const bool sse2 = has_bit(leaf1.edx, 26);
  const bool sse3 = has_bit(leaf1.ecx, 0);
  const bool ssse3 = has_bit(leaf1.ecx, 9);
  const bool fma = has_bit(leaf1.ecx, 12);
  const bool sse41 = has_bit(leaf1.ecx, 19);
  const bool sse42 = has_bit(leaf1.ecx, 20);
  const bool xsave = has_bit(leaf1.ecx, 26);
  const bool osxsave = has_bit(leaf1.ecx, 27);
  const bool avx = has_bit(leaf1.ecx, 28);
  const bool f16c = has_bit(leaf1.ecx, 29);

  const bool bmi1 = has_bit(leaf7.ebx, 3);
  const bool avx2 = has_bit(leaf7.ebx, 5);
  const bool bmi2 = has_bit(leaf7.ebx, 8);
  const bool avx512f = has_bit(leaf7.ebx, 16);
  const bool avx512dq = has_bit(leaf7.ebx, 17);
  const bool avx512cd = has_bit(leaf7.ebx, 28);
  const bool avx512bw = has_bit(leaf7.ebx, 30);
  const bool avx512vl = has_bit(leaf7.ebx, 31);

  std::uint64_t xcr0 = 0;
  if (xsave && osxsave)
    xcr0 = read_xcr(0);

  const bool ymm_state = (xcr0 & 0x6)==0x6;
  const bool zmm_state = (xcr0 & 0xe6)==0xe6;
  const bool avx_usable = avx && xsave && osxsave && ymm_state;
  const bool avx2_usable = avx_usable && avx2;
  const bool fma_usable = avx_usable && fma;
  const bool f16c_usable = avx_usable && f16c;
  const bool avx512f_usable = avx_usable && avx512f && zmm_state;
  const bool avx512_full_usable = avx512f_usable && avx512dq
    && avx512cd && avx512bw && avx512vl;

  std::cout << "arch="
#if defined(DUCC0_CPU_X86_64)
    << "x86_64";
#else
    << "x86";
#endif
  print_flag("sse", sse);
  print_flag("sse2", sse2);
  print_flag("sse3", sse3);
  print_flag("ssse3", ssse3);
  print_flag("sse4_1", sse41);
  print_flag("sse4_2", sse42);
  print_flag("avx_cpu", avx);
  print_flag("avx2_cpu", avx2);
  print_flag("fma_cpu", fma);
  print_flag("f16c_cpu", f16c);
  print_flag("bmi1", bmi1);
  print_flag("bmi2", bmi2);
  print_flag("avx512f_cpu", avx512f);
  print_flag("avx512dq_cpu", avx512dq);
  print_flag("avx512cd_cpu", avx512cd);
  print_flag("avx512bw_cpu", avx512bw);
  print_flag("avx512vl_cpu", avx512vl);
  print_flag("xsave", xsave);
  print_flag("osxsave", osxsave);
  std::cout << " xcr0=0x" << std::hex << xcr0 << std::dec;
  print_flag("avx_usable", avx_usable);
  print_flag("avx2_usable", avx2_usable);
  print_flag("fma_usable", fma_usable);
  print_flag("f16c_usable", f16c_usable);
  print_flag("avx512f_usable", avx512f_usable);
  print_flag("avx512_full_usable", avx512_full_usable);
  std::cout << '\n';
#elif defined(__aarch64__) || defined(_M_ARM64)
  std::cout << "arch=arm64 neon=1";
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE
  std::cout << " sve_compile_target=1";
#else
  std::cout << " sve_compile_target=0";
#endif
  std::cout << '\n';
#elif defined(__arm__) || defined(_M_ARM)
  std::cout << "arch=arm";
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  std::cout << " neon=1";
#else
  std::cout << " neon=0";
#endif
  std::cout << '\n';
#elif defined(__powerpc64__)
  std::cout << "arch=ppc64 detailed_simd_detection=not_implemented\n";
#elif defined(__powerpc__)
  std::cout << "arch=ppc detailed_simd_detection=not_implemented\n";
#elif defined(__riscv)
  std::cout << "arch=riscv detailed_simd_detection=not_implemented\n";
#elif defined(__s390x__)
  std::cout << "arch=s390x detailed_simd_detection=not_implemented\n";
#elif defined(__wasm64__)
  std::cout << "arch=wasm64 native_cpu_simd=not_applicable\n";
#elif defined(__wasm32__)
  std::cout << "arch=wasm32 native_cpu_simd=not_applicable\n";
#elif defined(__loongarch64)
  std::cout << "arch=loongarch64 detailed_simd_detection=not_implemented\n";
#else
  std::cout << "arch=unknown detailed_simd_detection=not_implemented\n";
#endif
  return 0;
  }
