#include "mr_util/error_handling.h"

using namespace std;

namespace mr {

namespace error_handling {

namespace detail {

bool abort_in_progress__ = false;

ostream &CodeLocation::print(ostream &os) const
  {
  os << "file: " << file <<  ", line: " <<  line;
  if (func) os << ", function: " << func;
  return os;
  }

void killjob__()
  {
  // perhaps print stack trace?
  exit(1);
  }

}}}
