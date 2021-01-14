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
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC0_BUCKET_SORT_H
#define DUCC0_BUCKET_SORT_H

#include <vector>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/aligned_array.h"
#include "ducc0/math/math_utils.h"

namespace ducc0 {

namespace detail_bucket_sort {

using namespace std;

template<typename RAidx, typename Tkey, typename Tidx> void subsort
  (RAidx idx, aligned_array<Tkey> &keys, size_t keybits, size_t lo,
   size_t hi, vector<Tidx> &numbers, aligned_array<Tidx> &idxbak,
   aligned_array<Tkey> &keybak)
  {
  auto nval = hi-lo;
  if (nval<=1) return;
  size_t keyshift = (keybits<=8) ? 0 : keybits-8;
  size_t nkeys = min<size_t>(size_t(1)<<keybits, 256);
  size_t keymask = nkeys-1;
  if (keybak.size()<nval) keybak.resize(nval);
  if (idxbak.size()<nval) idxbak.resize(nval);
  if (numbers.size()<nkeys) numbers.resize(nkeys);
  for (size_t i=0; i<nkeys; ++i) numbers[i]=0;
  for (size_t i=0; i<nval; ++i)
    {
    keybak[i] = keys[i+lo];
    idxbak[i] = idx[i+lo];
    ++numbers[(keys[i+lo]>>keyshift)&keymask];
    }
  Tidx ofs=0;
  for (auto &x: numbers)
    {
    auto tmp = x;
    x = ofs;
    ofs += tmp;
    }
  for (size_t i=0; i<nval; ++i)
    {
    auto loc = (keybak[i]>>keyshift)&keymask;
    keys[lo+numbers[loc]] = keybak[i];
    idx[lo+numbers[loc]] = idxbak[i];
    ++numbers[loc];
    }
  if (keyshift==0) return;
  keybits -= 8;
  vector<Tidx> newnumbers;
  for (size_t i=0; i<nkeys; ++i)
    subsort(idx, keys, keybits, lo + ((i==0) ? 0 : numbers[i-1]),
      lo+numbers[i], newnumbers, idxbak, keybak);
  }

template<typename RAidx, typename RAkey> void bucket_sort
  (RAkey keys, RAidx res, size_t nval, size_t max_key, size_t nthreads)
  {
  using Tidx = typename remove_reference<decltype(*res)>::type;
  using Tkey = typename remove_reference<decltype(*keys)>::type;
  struct vbuf
    {
    vector<Tidx> v;
    array<uint64_t,8> dummy;
    };
  vector<vbuf> numbers(nthreads);
  auto keybits = ilog2(max_key)+1;
  size_t keyshift = (keybits<=8) ? 0 : keybits-8;
  size_t nkeys = min<size_t>(size_t(1)<<keybits, 256);
  execParallel(nval, nthreads, [&](size_t tid, size_t lo, size_t hi)
    {
    auto &mybuf(numbers[tid].v);
    mybuf.resize(nkeys,0);
    for (size_t i=lo; i<hi; ++i)
      {
      MR_assert(keys[i]<=max_key, "key too large");
      ++mybuf[(keys[i]>>keyshift)];
      }
    });
  size_t ofs=0;
  for (size_t i=0; i<numbers[0].v.size(); ++i)
    for (size_t t=0; t<nthreads; ++t)
      {
      auto tmp=numbers[t].v[i];
      numbers[t].v[i]=ofs;
      ofs+=tmp;
      }
  aligned_array<Tkey> keys2(nval);
  execParallel(nval, nthreads, [&](size_t tid, size_t lo, size_t hi)
    {
    auto &mybuf(numbers[tid].v);
    for (size_t i=lo; i<hi; ++i)
      {
      auto loc = (keys[i]>>keyshift);
      res[mybuf[loc]] = i;
      keys2[mybuf[loc]] = keys[i];
      ++mybuf[loc];
      }
    });
  if (keyshift==0) return;
  keybits -= 8;
  execDynamic(nkeys, nthreads, 1, [&](Scheduler &sched)
    {
    vector<Tidx> newnumbers;
    aligned_array<Tkey> keybak;
    aligned_array<Tidx> idxbak;
    while (auto rng=sched.getNext())
      for(auto i=rng.lo; i<rng.hi; ++i)
        subsort(res, keys2, keybits, (i==0) ? 0 : numbers[nthreads-1].v[i-1],
          numbers[nthreads-1].v[i], newnumbers, idxbak, keybak);
    });
  }

}

using detail_bucket_sort::bucket_sort;

}

#endif
