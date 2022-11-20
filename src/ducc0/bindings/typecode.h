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

/* Copyright (C) 2022 Max-Planck-Society
   Author: Martin Reinecke */


#ifndef DUCC0_TYPECODE_H
#define DUCC0_TYPECODE_H

#include <cstddef>
#include <type_traits>
#include <complex>

namespace ducc0 {

namespace detail_typecode {

using namespace std;

// bits [0-4] contain the type size in bytes
// bit 5 is 1 iff the type is unsigned
// bit 6 is 1 iff the type is floating point
// bit 7 is 1 iff the type is complex-valued
constexpr size_t sizemask = 0x1f;
constexpr size_t unsigned_bit = 0x20;
constexpr size_t floating_point_bit = 0x40;
constexpr size_t complex_bit = 0x80;

template<typename T> class Typecode
  {
  private:
    static constexpr size_t compute()
      {
      static_assert(!is_same<T,bool>::value, "no bools allowed");
      static_assert(is_integral<T>::value||is_floating_point<T>::value,
        "need integral or floating point type");
      static_assert(sizeof(T)<=8, "type size must be at most 8 bytes");
      if constexpr(is_integral<T>::value)
        return sizeof(T) + unsigned_bit*(!is_signed<T>::value);
      if constexpr(is_floating_point<T>::value)
        {
        static_assert(is_signed<T>::value,
          "no support for unsigned floating point types");
        return sizeof(T)+floating_point_bit;
        }
      }

  public:
    static constexpr size_t value = compute();
  };

template<typename T> class Typecode<complex<T>>
  {
  private:
    static constexpr size_t compute()
      {
      static_assert(is_floating_point<T>::value, "need a floating point type");
      return Typecode<T>::value + sizeof(T) + complex_bit;
      }

  public:
    static constexpr size_t value = compute();
  };

//constexpr size_t type_size(size_t tc) { return tc&sizemask; }
//constexpr bool type_is_unsigned(size_t tc) { return tc&unsigned_bit; }
//constexpr bool type_is_floating_point(size_t tc) { return tc&floating_point_bit; }
//constexpr bool type_is_complex(size_t tc) { return tc&complex_bit; }

}

using detail_typecode::Typecode;

}

#endif
