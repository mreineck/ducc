/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  Copyright (C) 2023 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#define _GNU_SOURCE
#include <cfenv>
#include "ducc0/infra/error_handling.h"

#include <iostream>

namespace ducc0 {
namespace detail_startup_checks {

using namespace std;

class Startup
  {
  public:
    Startup()
      {
      //fexcept_t fpe_flags;
      //MR_assert(fegetexceptflag(&fpe_flags, FE_ALL_EXCEPT)==0,
        //"error while getting FPE flags");
      //cout << "FE_DIVBYZERO: " << bool(fpe_flags&FE_DIVBYZERO)<<endl;
      //cout << "FE_OVERFLOW: " << bool(fpe_flags&FE_OVERFLOW)<<endl;
      //cout << "FE_UNDERFLOW: " << bool(fpe_flags&FE_UNDERFLOW)<<endl;
      //cout << "FE_INEXACT: " << bool(fpe_flags&FE_INEXACT)<<endl;
      //fpe_flags |= FE_DIVBYZERO|FE_UNDERFLOW;
      //MR_assert(fesetexceptflag(&fpe_flags, FE_ALL_EXCEPT)==0,
        //"error while setting FPE flags");
      //MR_assert(fegetexceptflag(&fpe_flags, FE_ALL_EXCEPT)==0,
        //"error while getting FPE flags");
      //cout << "FE_DIVBYZERO: " << bool(fpe_flags&FE_DIVBYZERO)<<endl;
      //cout << "FE_OVERFLOW: " << bool(fpe_flags&FE_OVERFLOW)<<endl;
      //cout << "FE_UNDERFLOW: " << bool(fpe_flags&FE_UNDERFLOW)<<endl;
      //cout << "FE_INEXACT: " << bool(fpe_flags&FE_INEXACT)<<endl;
//feenableexcept(FE_DIVBYZERO|FE_UNDERFLOW);
      }
  };

static Startup startup;

}}
