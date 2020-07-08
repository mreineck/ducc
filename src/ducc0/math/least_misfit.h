#ifndef DUCC0_LEAST_MISFIT_H
#define DUCC0_LEAST_MISFIT_H

/*
 *  This file is part of nifty_gridder.
 *
 *  nifty_gridder is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  nifty_gridder is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nifty_gridder; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2020 Max-Planck-Society
   Author: Martin Reinecke */

#include <vector>

//#include "least_misfit.h"

namespace ducc0 {

namespace detail_least_misfit {

using namespace std;

struct PolyKernel
  {
  size_t W;
  double ofactor;
  size_t D;
  double epsilon;
  vector<double> coeff;
  };

const PolyKernel &selectLeastMisfitKernel(double ofactor, double epsilon);

} // namespace detail_least_misfit

using detail_least_misfit::PolyKernel;
using detail_least_misfit::selectLeastMisfitKernel;

} // namespace ducc0

#endif
