#ifdef DUCC0_USE_NANOBIND
#include <nanobind/nanobind.h>
#else
#include <pybind11/pybind11.h>
#endif
#include <string>
#include <cstdlib>
#include <cstdint>
#include <iostream>

using namespace std;

#ifdef DUCC0_USE_NANOBIND
namespace py = nanobind;
#else
namespace py = pybind11;
#endif

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

#endif

}

#if defined(DUCC0_CPU_X86_64)

int psabi_level()
  {
  const auto leaf0 = cpuid(0, 0);
  const auto leaf1 = (leaf0.eax>=1) ? cpuid(1, 0) : cpuid_result{0,0,0,0};

  const bool fpu = has_bit(leaf1.edx, 0);
  const bool cx8 = has_bit(leaf1.edx, 8);
  const bool cmov = has_bit(leaf1.edx, 15);
  const bool mmx = has_bit(leaf1.edx, 23);
  const bool fxsr = has_bit(leaf1.edx, 24);
  const bool sse = has_bit(leaf1.edx, 25);
  const bool sse2 = has_bit(leaf1.edx, 26);

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

  const uint32_t high = 0x80000000;
  const auto leaf_high0 = cpuid(high, 0);
  const auto leaf_high1 = (leaf_high0.eax>=high+1) ? cpuid(high+1, 0) : cpuid_result{0,0,0,0};

  const bool lzcnt = has_bit(leaf_high1.ecx, 5);

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

  int lvl = 0;
  // Level 1: always present on x86_64
  // includes CMOV, CX8, FPU, FXSR, MMX, OSFXSR, SCE, SSE, SSE2
  // MR: no idea how to check for OSFXSR and SCE ...
  if (cmov && cx8 && fpu && fxsr && mmx && sse && sse2)
    lvl=1;
  // Level 2:
  // includes CMPXCHG16B, LAHF-SAHF, POPCNT, SSE3, SSE4_1, SSE4_2, SSSE3
  // MR: no idea how to check for LAHF-SAHF ...
  if ((lvl==1) && cx16 && popcnt && sse3 && sse41 && sse42 && ssse3)
    lvl=2;
  // Level 3:
  // includes AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE
  if (avx_usable && avx2_usable && bmi1 && bmi2 && f16c && fma_usable && lzcnt && movbe && osxsave)
    lvl = 3;
  if ((lvl>=3) && avx512_full_usable)
    lvl = 4;

  // check if maximum level is limted by environment variable (typically for testing)
  const auto *evar=getenv("DUCC0_MAX_PSABI_LEVEL");
  if (evar!=nullptr)
    {
    int maxlvl = atoi(evar);
    if (maxlvl<1) maxlvl=1;
    if (lvl>maxlvl) lvl = maxlvl;
    }
  return lvl;
  }

#else

int psabi_level() { return 0; }

#endif

namespace ducc0_v1 { void add_ducc0(py::module_ &m); }
namespace ducc0_v3 { void add_ducc0(py::module_ &m); }
namespace ducc0_v4 { void add_ducc0(py::module_ &m); }

#ifdef DUCC0_USE_NANOBIND
NB_MODULE(PKGNAME, m)
#else
PYBIND11_MODULE(PKGNAME, m, py::mod_gil_not_used())
#endif
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
  m.attr("__version__") = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY
#ifdef DUCC0_USE_NANOBIND
  m.attr("__wrapper__") = "nanobind";
#else
  m.attr("__wrapper__") = "pybind11";
#endif

  auto lvl = psabi_level();
  if (lvl>=4) return ducc0_v4::add_ducc0(m);
  if (lvl>=3) return ducc0_v3::add_ducc0(m);
  ducc0_v1::add_ducc0(m);
  }
