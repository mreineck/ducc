#include "ducc0/wgridder/wgridder_impl.h"

namespace ducc0 {
namespace detail_gridder {

using namespace std;

#define Tcalc float
#define Tacc double
#define Tms float
#define Timg float
#define Tms_in cmav<complex<Tms>,2>
#include "ducc0/wgridder/wgridder_inst_inc.h"
#undef Tms_in
#undef Timg
#undef Tms
#undef Tacc
#undef Tcalc

}}
