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

/* Copyright (C) 2019-2020 Peter Bell, Max-Planck-Society
   Authors: Peter Bell, Martin Reinecke */

#ifndef DUCC0_THREADING_H
#define DUCC0_THREADING_H

#include <functional>

namespace ducc0 {

namespace detail_threading {

using std::size_t;

struct Range
  {
  size_t lo, hi;
  Range() : lo(0), hi(0) {}
  Range(size_t lo_, size_t hi_) : lo(lo_), hi(hi_) {}
  operator bool() const { return hi>lo; }
  };

class Scheduler
  {
  public:
    virtual ~Scheduler() {}
    virtual size_t num_threads() const = 0;
    virtual size_t thread_num() const = 0;
    virtual Range getNext() = 0;
  };

size_t max_threads();
void set_default_nthreads(size_t new_default_nthreads);
size_t get_default_nthreads();

void execSingle(size_t nwork,
  std::function<void(Scheduler &)> func);
void execStatic(size_t nwork, size_t nthreads, size_t chunksize,
  std::function<void(Scheduler &)> func);
void execDynamic(size_t nwork, size_t nthreads, size_t chunksize,
  std::function<void(Scheduler &)> func);
void execGuided(size_t nwork, size_t nthreads, size_t chunksize_min,
  double fact_max, std::function<void(Scheduler &)> func);
void execParallel(size_t nthreads, std::function<void(Scheduler &)> func);
void execParallel(size_t work_lo, size_t work_hi, size_t nthreads,
  std::function<void(size_t, size_t)> func);
inline void execParallel(size_t nwork, size_t nthreads,
  std::function<void(size_t, size_t)> func)
  { execParallel(0, nwork, nthreads, func); }
void execParallel(size_t work_lo, size_t work_hi, size_t nthreads,
  std::function<void(size_t, size_t, size_t)> func);
inline void execParallel(size_t nwork, size_t nthreads,
  std::function<void(size_t, size_t, size_t)> func)
  { execParallel(0, nwork, nthreads, func); }

} // end of namespace detail_threading

using detail_threading::max_threads;
using detail_threading::get_default_nthreads;
using detail_threading::set_default_nthreads;
using detail_threading::Scheduler;
using detail_threading::execSingle;
using detail_threading::execStatic;
using detail_threading::execDynamic;
using detail_threading::execGuided;
using detail_threading::execParallel;

} // end of namespace ducc0

#endif
