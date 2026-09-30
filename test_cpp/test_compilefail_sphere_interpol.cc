/* Compile-fail proof: the legacy convenience overloads of
   ducc0::detail_sphereinterpol::SphereInterpol<T> are broken.

   - getPlane(const cmav<complex<T>,1> &, const vmav<T,3> &)
     (src/ducc0/sht/sphere_interpol.h:539) forwards to a 2-argument
     getPlane(valm, planes) that does not exist; the real overload takes six
     arguments (valm, mstart, lstride, planes, mode, timers).
   - updateAlm(const vmav<complex<T>,1> &, const vmav<T,3> &, SHT_mode)
     (sphere_interpol.h:640) forwards to a 3-argument updateAlm that does not
     exist; the real overload takes seven arguments.

   These are public member functions that fail to compile whenever they are
   used, i.e. they are unusable API. This translation unit must fail to
   compile with "no matching function for call to getPlane/updateAlm".
*/
#include "ducc0/sht/sphere_interpol.h"
#include <complex>

using namespace ducc0;

int main()
  {
  detail_sphereinterpol::SphereInterpol<double> inter(8, 8, 0, 0, 1.1, 2.6, 1e-8, 1);
  std::vector<std::complex<double>> almd(8);
  std::vector<double> planed(64);
  cmav<std::complex<double>, 1> almc(almd.data(), {8});
  vmav<std::complex<double>, 1> almv(almd.data(), {8});
  vmav<double, 3> planes(planed.data(), {1, 8, 8});
  inter.getPlane(almc, planes);
  inter.updateAlm(almv, planes, SHT_mode::STANDARD);
  return 0;
  }
