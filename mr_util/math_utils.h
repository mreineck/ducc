/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file math_utils.h
 *  Various convenience mathematical functions.
 *
 *  Copyright (C) 2002-2015 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef MRUTIL_MATH_UTILS_H
#define MRUTIL_MATH_UTILS_H

#include <cmath>

namespace mr {

/*! Returns \e true if | \a a-b | <= \a epsilon * | \a b |, else \e false. */
template<typename F> inline bool approx (F a, F b, F epsilon=1e-5)
  {
  return std::abs(a-b) <= (epsilon*std::abs(b));
  }

}

#endif
