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

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_ALIGNED_ARRAY_H
#define DUCC0_ALIGNED_ARRAY_H

#include <cstdlib>
#include <memory>

namespace ducc0 {

namespace detail_aligned_array {

using namespace std;

/*! Simple array class guaranteeing 64-byte alignment of the data pointer.
    Mostly useful for storing data accessed by SIMD instructions. */
template<typename T> class aligned_array
  {
  private:
    T *p;
    size_t sz;

// FIXME: let's not use aligned_alloc on Apple for the moment,
// it's only supported from 10.15 on...
#if ((__cplusplus >= 201703L) && (!defined(__APPLE__)))
    static T *ralloc(size_t num)
      {
      if (num==0) return nullptr;
      // aligned_alloc requires the allocated size to be a multiple of the
      // requested alignment, so increase size if necessary
      void *res = aligned_alloc(64,((num*sizeof(T)+63)/64)*64);
      if (!res) throw bad_alloc();
      return reinterpret_cast<T *>(res);
      }
    static void dealloc(T *ptr)
      { free(ptr); }
#else // portable emulation
    static T *ralloc(size_t num)
      {
      if (num==0) return nullptr;
      void *ptr = malloc(num*sizeof(T)+64);
      if (!ptr) throw bad_alloc();
      T *res = reinterpret_cast<T *>
        ((reinterpret_cast<size_t>(ptr) & ~(size_t(63))) + 64);
      (reinterpret_cast<void**>(res))[-1] = ptr;
      return res;
      }
    static void dealloc(T *ptr)
      { if (ptr) free((reinterpret_cast<void**>(ptr))[-1]); }
#endif

  public:
    aligned_array() : p(nullptr), sz(0) {}
    aligned_array(size_t n) : p(ralloc(n)), sz(n) {}
    aligned_array(aligned_array &&other)
      : p(other.p), sz(other.sz)
      { other.p=nullptr; other.sz=0; }
    ~aligned_array() { dealloc(p); }

    void resize(size_t n)
      {
      if (n==sz) return;
      dealloc(p);
      p = ralloc(n);
      sz = n;
      }

    T &operator[](size_t idx) { return p[idx]; }
    const T &operator[](size_t idx) const { return p[idx]; }

    T *data() { return p; }
    const T *data() const { return p; }

    size_t size() const { return sz; }
  };

}

using detail_aligned_array::aligned_array;

}

#endif

