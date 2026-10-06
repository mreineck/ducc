/* Unit test: SphereInterpol's getPlane/updateAlm must be usable via their
   current signatures. (The broken 1-D/3-argument legacy overloads that
   forwarded to nonexistent functions were removed in response to
   mreineck/ducc#79.) */
#include "ducc0/sht/sphere_interpol.h"
#include "ducc0/infra/timers.h"
#include <complex>
#include <cstddef>
#include <vector>

using namespace ducc0;

int main()
  {
  detail_sphereinterpol::SphereInterpol<double> inter(8, 8, 0, 0, 1.1, 2.6, 1e-8, 1);
  std::vector<std::complex<double>> almd(8);
  std::vector<double> planed(64);
  std::vector<size_t> msd(1);
  cmav<std::complex<double>, 2> valm(almd.data(), {1, 8});
  vmav<std::complex<double>, 2> valmw(almd.data(), {1, 8});
  vmav<double, 3> planes(planed.data(), {1, 8, 8});
  cmav<size_t, 1> mstart(msd.data(), {1});
  TimerHierarchy timers("test");
  inter.getPlane(valm, mstart, 1, planes, SHT_mode::STANDARD, timers);
  inter.updateAlm(valmw, mstart, 1, planes, SHT_mode::STANDARD, timers);
  return 0;
  }
