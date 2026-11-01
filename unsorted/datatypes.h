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

/*! \file datatypes.h
 *  This file defines various platform-independent data types.
 *  If any of the requested types is not available, compilation aborts
 *  with an error (unfortunately a rather obscure one).
 *
 *  Copyright (C) 2004-2015 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_DATATYPES_H
#define DUCC0_DATATYPES_H

#include <string>
#include <cstddef>
#include <cstdint>
#include "ducc0/error_handling.h"

namespace ducc0 {

namespace datatypes {

/*! Returns a C string describing the data type \a T. */
template<typename T> inline const char *type2typename ()
  { MR_fail(T::UNSUPPORTED_DATA_TYPE); }
template<> inline const char *type2typename<signed char> ()
  { return "signed char"; }
template<> inline const char *type2typename<unsigned char> ()
  { return "unsigned char"; }
template<> inline const char *type2typename<short> ()
  { return "short"; }
template<> inline const char *type2typename<unsigned short> ()
  { return "unsigned short"; }
template<> inline const char *type2typename<int> ()
  { return "int"; }
template<> inline const char *type2typename<unsigned int> ()
  { return "unsigned int"; }
template<> inline const char *type2typename<long> ()
  { return "long"; }
template<> inline const char *type2typename<unsigned long> ()
  { return "unsigned long"; }
template<> inline const char *type2typename<long long> ()
  { return "long long"; }
template<> inline const char *type2typename<unsigned long long> ()
  { return "unsigned long long"; }
template<> inline const char *type2typename<float> ()
  { return "float"; }
template<> inline const char *type2typename<double> ()
  { return "double"; }
template<> inline const char *type2typename<long double> ()
  { return "long double"; }
template<> inline const char *type2typename<bool> ()
  { return "bool"; }
template<> inline const char *type2typename<std::string> ()
  { return "std::string"; }

/*! mapping of "native" data types to integer constants */
enum NDT {
       NAT_CHAR,
       NAT_SCHAR,
       NAT_UCHAR,
       NAT_SHORT,
       NAT_USHORT,
       NAT_INT,
       NAT_UINT,
       NAT_LONG,
       NAT_ULONG,
       NAT_LONGLONG,
       NAT_ULONGLONG,
       NAT_FLOAT,
       NAT_DOUBLE,
       NAT_LONGDOUBLE,
       NAT_BOOL,
       NAT_STRING };

/*! Returns the \a NDT constant associated with \a T. */
template<typename T> inline NDT nativeType()
  { MR_fail(T::UNSUPPORTED_DATA_TYPE); }
template<> inline NDT nativeType<char>              () { return NAT_CHAR;      }
template<> inline NDT nativeType<signed char>       () { return NAT_SCHAR;     }
template<> inline NDT nativeType<unsigned char>     () { return NAT_UCHAR;     }
template<> inline NDT nativeType<short>             () { return NAT_SHORT;     }
template<> inline NDT nativeType<unsigned short>    () { return NAT_USHORT;    }
template<> inline NDT nativeType<int>               () { return NAT_INT;       }
template<> inline NDT nativeType<unsigned int>      () { return NAT_UINT;      }
template<> inline NDT nativeType<long>              () { return NAT_LONG;      }
template<> inline NDT nativeType<unsigned long>     () { return NAT_ULONG;     }
template<> inline NDT nativeType<long long>         () { return NAT_LONGLONG;  }
template<> inline NDT nativeType<unsigned long long>() { return NAT_ULONGLONG; }
template<> inline NDT nativeType<float>             () { return NAT_FLOAT;     }
template<> inline NDT nativeType<double>            () { return NAT_DOUBLE;    }
template<> inline NDT nativeType<long double>       () { return NAT_LONGDOUBLE;}
template<> inline NDT nativeType<bool>              () { return NAT_BOOL;      }
template<> inline NDT nativeType<std::string>       () { return NAT_STRING;    }

/*! Returns the size (in bytes) of the native data type \a type. */
inline int ndt2size (NDT type)
  {
  switch (type)
    {
    case NAT_CHAR      :
    case NAT_SCHAR     :
    case NAT_UCHAR     : return sizeof(char);
    case NAT_SHORT     :
    case NAT_USHORT    : return sizeof(short);
    case NAT_INT       :
    case NAT_UINT      : return sizeof(int);
    case NAT_LONG      :
    case NAT_ULONG     : return sizeof(long);
    case NAT_LONGLONG  :
    case NAT_ULONGLONG : return sizeof(long long);
    case NAT_FLOAT     : return sizeof(float);
    case NAT_DOUBLE    : return sizeof(double);
    case NAT_LONGDOUBLE: return sizeof(long double);
    case NAT_BOOL      : return sizeof(bool);
    default:
      MR_fail ("ndt2size: unsupported data type");
    }
  }

}}

#endif
