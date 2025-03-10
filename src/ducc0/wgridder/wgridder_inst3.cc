#include "ducc0/wgridder/wgridder_impl.h"

namespace ducc0 {
namespace detail_gridder {

using namespace std;

#define Tcalc double
#define Tacc double
#define Tms double
#define Timg double
#define Tms_in cmav<complex<Tms>,2>
#include "ducc0/wgridder/wgridder_inst_inc.h"
#undef Tms_in
#undef Timg
#undef Tms
#undef Tacc
#undef Tcalc

}}
