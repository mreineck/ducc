/*
 *  This file is part of the MR utility library.
 *
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

/* Copyright (C) 2019 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef MRUTIL_ERROR_HANDLING_H
#define MRUTIL_ERROR_HANDLING_H

#include <iostream>
#include <cstdlib>

namespace mr {

namespace error_handling {

namespace detail {

#if defined (__GNUC__)
#define MR_ERROR_HANDLING_LOC_ ::mr::error_handling::detail::CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__)
#else
#define MRERROR_HANDLING_LOC_ ::mr::error_handling::detail::CodeLocation(__FILE__, __LINE__)
#endif

#define MR_fail(...) \
  do { \
    if (!::mr::error_handling::detail::abort_in_progress__) \
      { \
      ::mr::error_handling::detail::abort_in_progress__ = true; \
      ::mr::error_handling::detail::streamDump__(::std::cerr, MR_ERROR_HANDLING_LOC_, "\n", ##__VA_ARGS__, "\n"); \
      ::mr::error_handling::detail::killjob__(); \
      } \
    ::std::exit(1); \
    } while(0)

#define MR_assert(cond,...) \
  do { \
    if (cond); \
    else { MR_fail("Assertion failure\n", ##__VA_ARGS__); } \
    } while(0)

// to be replaced with std::source_location once generally available
class CodeLocation
  {
  private:
    const char *file, *func;
    int line;

  public:
    CodeLocation(const char *file_, int line_, const char *func_=nullptr)
      : file(file_), func(func_), line(line_) {}

    ::std::ostream &print(::std::ostream &os) const;
  };

inline ::std::ostream &operator<<(::std::ostream &os, const CodeLocation &loc)
  { return loc.print(os); }

extern bool abort_in_progress__;
void killjob__();

#if (__cplusplus>=201703L) // hyper-elegant C++2017 version
template<typename ...Args>
inline void streamDump__(::std::ostream &os, Args&&... args)
  { (os << ... << args); }
#else
template<typename T>
inline void streamDump__(::std::ostream &os, const T& value)
  { os << value; }

template<typename T, typename ... Args>
inline void streamDump__(::std::ostream &os, const T& value,
  const Args& ... args)
  {
  os << value;
  streamDump__(os, args...);
  }
#endif

}}}

#endif
