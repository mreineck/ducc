/** \file ducc0/infra/threading.cc
 *
 *  \copyright Copyright (C) 2019-2024 Peter Bell, Max-Planck-Society
 *  \authors Peter Bell, Martin Reinecke
 */

/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

#include "ducc0/infra/threading.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/string_utils.h"
#include <atomic>
#include <exception>
#include <utility>

#ifdef DUCC0_STDCXX_LOWLEVEL_THREADING
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <thread>
#include <queue>
#include <vector>
#include <errno.h>
#include <string>
#include <string.h>
#if __has_include(<pthread.h>)
#include <pthread.h>
#if __has_include(<pthread.h>) && defined(__linux__) && defined(_GNU_SOURCE)
#include <unistd.h>
#endif
#endif
#include <latch>
#endif

namespace ducc0 {

namespace detail_threading {

#ifdef DUCC0_STDCXX_LOWLEVEL_THREADING

using std::latch;

size_t ducc0_max_threads()
  {
  static const size_t max_threads_ = []()
    {
#if __has_include(<pthread.h>) && defined(__linux__) && defined(_GNU_SOURCE)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    size_t res=0;
    for (size_t i=0; i<CPU_SETSIZE; ++i)
      if (CPU_ISSET(i, &cpuset)) ++res;
#else
    size_t res = std::max<size_t>(1, std::thread::hardware_concurrency());
#endif
    auto evar=getenv("DUCC0_NUM_THREADS");
    // fallback
    if (!evar)
      evar=getenv("OMP_NUM_THREADS");
    if (!evar)
      return res;
    auto res2 = stringToData<long>(trim(std::string(evar)));
    MR_assert(res2>=0, "invalid value in DUCC0_NUM_THREADS/OMP_NUM_THREADS");
    if (res2==0)
      return res;
    return std::min<size_t>(res, res2);
    }();
  return max_threads_;
  }
 
static thread_local bool in_parallel_region = false;
int pin_info()
  {
  static const int pin_info_ = []()
    {
    auto evar=getenv("DUCC0_PIN_DISTANCE");
    if (!evar)
      return -1; // do nothing at all
    auto res = stringToData<long>(trim(std::string(evar)));
    return int(res);
    }();
  return pin_info_;
  }
int pin_offset()
  {
  static const int pin_offset_ = []()
    {
    auto evar=getenv("DUCC0_PIN_OFFSET");
    if (!evar)
      return 0;
    auto res = stringToData<long>(trim(std::string(evar)));
    return int(res);
    }();
  return pin_offset_;
  }

#if __has_include(<pthread.h>) && defined(__linux__) && defined(_GNU_SOURCE)
static void do_pinning(int ithread)
  {
  if (pin_info()==-1) return;
  int num_proc = sysconf(_SC_NPROCESSORS_ONLN);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int cpu_wanted = pin_offset() + ithread*pin_info();
  MR_assert((cpu_wanted>=0)&&(cpu_wanted<num_proc), "bad CPU number requested");
  CPU_SET(cpu_wanted, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  }
#else
static void do_pinning(int /*ithread*/)
  { return; }
#endif

class ducc_thread_pool: public thread_pool
  {
  private:
    // A reasonable guess, probably close enough for most hardware
    static constexpr size_t cache_line_size = 64;
    // align members with cache lines
    struct alignas(cache_line_size) worker
      {
      std::thread thread;

      void worker_main(const std::atomic<bool> &news,
        std::function<void(size_t)> &work_, size_t ithread, const size_t &nthreads)
        {
        bool nextnews = true;
        in_parallel_region = true;
        do_pinning(ithread);
        while (true)
          {
          // Wait until something happens
          news.wait(!nextnews);
          nextnews=!nextnews;
          // shutting down?
          if (!work_) return;
          // if not, do the work
          work_(ithread);
          }
        }
      };

    Mutex mut_;
    std::vector<worker> workers_;
    std::atomic<bool> news_=false;
    size_t nthreads_ = 0;
    using lock_t = LockGuard;
    std::function<void(size_t)> work_;

    void create_threads()
      {
      lock_t lock(mut_);
      news_=false;
      size_t nthreads=workers_.size();
      for (size_t i=0; i<nthreads; ++i)
        {
        try
          {
          auto *worker = &workers_[i];
          worker->thread = std::thread(
            [worker, this, i]{ worker->worker_main(news_, work_, i+1, nthreads_); });
          }
        catch (...)
          {
          shutdown_locked();
          throw;
          }
        }
      }

    void shutdown_locked()
      {
      // we send a nullptr work to the workers, signalling shutdown
      work_ = nullptr;
      news_ = !news_;
      news_.notify_all();

      for (auto &worker : workers_)
        if (worker.thread.joinable())
          worker.thread.join();
      }

  public:
    explicit ducc_thread_pool(size_t nthreads):
      workers_(nthreads)
      { do_pinning(0); create_threads(); }

    //virtual
    ~ducc_thread_pool() { shutdown(); }

    //virtual
    size_t nthreads() const { return workers_.size(); }

    //virtual
    size_t adjust_nthreads(size_t nthreads_in) const
      {
      if (in_parallel_region)
        return 1;
      if (nthreads_in==0)
        return ducc0_max_threads();
      return std::min(ducc0_max_threads(), nthreads_in);
      }

    //virtual
    void submit(std::function<void(Scheduler &)> work, Distribution &dist,
      size_t nthreads);

    void shutdown()
      {
      lock_t lock(mut_);
      shutdown_locked();
      }

    void restart()
      {
      create_threads();
      }
  };

// return a pointer to a singleton thread_pool, which is always available
inline ducc_thread_pool *get_master_pool()
  {
  static auto master_pool = new ducc_thread_pool(ducc0_max_threads()-1);
#if __has_include(<pthread.h>)
  static std::once_flag f;
  call_once(f,
    []{
    pthread_atfork(
      +[]{ get_master_pool()->shutdown(); },  // prepare
      +[]{ get_master_pool()->restart(); },   // parent
      +[]{ get_master_pool()->restart(); }    // child
      );
    });
#endif
  return master_pool;
  }

thread_local thread_pool *active_pool = get_master_pool();

thread_pool *set_active_pool(thread_pool *new_pool)
  { return std::exchange(active_pool, new_pool); }
thread_pool *get_active_pool()
  {
  if (!active_pool) active_pool = get_master_pool();
  MR_assert(active_pool, "no thread pool active");
  return active_pool;
  }

#endif

#ifdef DUCC0_NO_LOWLEVEL_THREADING

class ducc_pseudo_thread_pool: public thread_pool
  {
  public:
    ducc_pseudo_thread_pool() {}

    //virtual
    size_t nthreads() const { return 1; }

    //virtual
    size_t adjust_nthreads(size_t /*nthreads_in*/) const
      { return 1; }
    //virtual
    void submit(std::function<void()> work)
      { work(); }
    //virtual
    void submit(std::function<void()> work, size_t)
      { work(); }
  };

// return a pointer to a singleton thread_pool, which is always available
inline ducc_pseudo_thread_pool *get_master_pool()
  {
  static auto master_pool = new ducc_pseudo_thread_pool();
  return master_pool;
  }

thread_local thread_pool *active_pool = get_master_pool();

thread_pool *set_active_pool(thread_pool *new_pool)
  { return std::exchange(active_pool, new_pool); }
thread_pool *get_active_pool()
  {
  MR_assert(active_pool!=nullptr, "no thread pool active");
  return active_pool;
  }

static bool in_parallel_region=false;

#endif

size_t max_threads()
  { return get_active_pool()->nthreads()+1; }
size_t adjust_nthreads(size_t nthreads_in)
  { return get_active_pool()->adjust_nthreads(nthreads_in); }

class Distribution
  {
  private:
    size_t nthreads_;
    Mutex mut_;
    size_t nwork_;
    size_t cur_;
    std::atomic<size_t> cur_dynamic_;
    size_t chunksize_;
    double fact_max_;
    struct alignas(64) spaced_size_t { size_t v; }; 
    std::vector<spaced_size_t> nextstart;
    enum SchedMode { SINGLE, STATIC, DYNAMIC, GUIDED };
    SchedMode mode;
    bool single_done;

    void thread_map(std::function<void(Scheduler &)> f);

  public:
    size_t nthreads() const { return nthreads_; }

    void execSingle(size_t nwork, std::function<void(Scheduler &)> f)
      {
      mode = SINGLE;
      single_done = false;
      nwork_ = nwork;
      nthreads_ = 1;
      thread_map(std::move(f));
      }
    void execStatic(size_t nwork, size_t nthreads, size_t chunksize,
      std::function<void(Scheduler &)> f)
      {
      mode = STATIC;
      nthreads_ = adjust_nthreads(nthreads);
      nwork_ = nwork;
      chunksize_ = (chunksize<1) ? (nwork_+nthreads_-1)/nthreads_
                                 : chunksize;
      if (chunksize_>=nwork_)
        return execSingle(nwork_, std::move(f));
// if there are fewer chunks than threads, reduce nthreads
      nthreads_ = std::min(nthreads_, (nwork_+chunksize_-1)/chunksize_);
      nextstart.resize(nthreads_);
      for (size_t i=0; i<nextstart.size(); ++i)
        nextstart[i].v = i*chunksize_;
      thread_map(std::move(f));
      }
    void execDynamic(size_t nwork, size_t nthreads, size_t chunksize,
      std::function<void(Scheduler &)> f)
      {
      mode = DYNAMIC;
      nthreads_ = adjust_nthreads(nthreads);
      nwork_ = nwork;
      chunksize_ = (chunksize<1) ? 1 : chunksize;
      if (chunksize_ >= nwork)
        return execSingle(nwork, std::move(f));
      if (chunksize_*nthreads_>=nwork_)
        return execStatic(nwork, nthreads, chunksize_, std::move(f));
      cur_dynamic_ = 0;
      thread_map(std::move(f));
      }
    void execGuided(size_t nwork, size_t nthreads, size_t chunksize_min,
      double fact_max, std::function<void(Scheduler &)> f)
      {
      mode = GUIDED;
      nthreads_ = adjust_nthreads(nthreads);
      nwork_ = nwork;
      chunksize_ = (chunksize_min<1) ? 1 : chunksize_min;
      if (chunksize_*nthreads_>=nwork_)
        return execStatic(nwork, nthreads, chunksize_, std::move(f));
      fact_max_ = fact_max;
      cur_ = 0;
      thread_map(std::move(f));
      }
    void execParallel(size_t nthreads, std::function<void(Scheduler &)> f)
      {
      mode = STATIC;
      nthreads_ = adjust_nthreads(nthreads);
      nwork_ = nthreads_;
      chunksize_ = 1;
      thread_map(std::move(f));
      }
    Range getNext(size_t thread_id)
      {
      switch (mode)
        {
        case SINGLE:
          {
          if (single_done) return Range();
          single_done=true;
          return Range(0, nwork_);
          }
        case STATIC:
          {
          if (nextstart[thread_id].v>=nwork_) return Range();
          size_t lo=nextstart[thread_id].v;
          size_t hi=std::min(lo+chunksize_,nwork_);
          nextstart[thread_id].v += nthreads_*chunksize_;
          return Range(lo, hi);
          }
        case DYNAMIC:
          {
          auto curval = cur_dynamic_.fetch_add(chunksize_);
          return Range(std::min(curval, nwork_),
                       std::min(curval+chunksize_, nwork_));
          }
        case GUIDED:
          {
          LockGuard lck(mut_);
          if (cur_>=nwork_) return Range();
          auto rem = nwork_-cur_;
          size_t tmp = size_t((fact_max_*double(rem))/double(nthreads_));
          auto sz = std::min(rem, std::max(chunksize_, tmp));
          size_t lo=cur_;
          cur_+=sz;
          size_t hi=cur_;
          return Range(lo, hi);
          }
        }
      return Range();
      }
  };

class MyScheduler: public Scheduler
  {
  private:
    Distribution &dist_;
    size_t ithread_;

  public:
    MyScheduler(Distribution &dist, size_t ithread)
      : dist_(dist), ithread_(ithread) {}
    virtual size_t num_threads() const { return dist_.nthreads(); }
    virtual size_t thread_num() const { return ithread_; }
    virtual Range getNext() { return dist_.getNext(ithread_); }
  };

template<typename T> class ScopedValueChanger
  {
  private:
    T &object;
    T original_value;

  public:
    ScopedValueChanger(T &object_, T new_value)
      : object(object_), original_value(object_) { object=new_value; }
    ~ScopedValueChanger()
      { object=original_value; }
  };

void ducc_thread_pool::submit(std::function<void(Scheduler &)> work, Distribution &dist,
  size_t nthreads)
  {
  lock_t lock(mut_);
//     if (shutdown_)
//       throw std::runtime_error("Work items submitted after shutdown");

  MR_assert(nthreads<=workers_.size()+1, "too many threads requested");

  std::exception_ptr ex;
  Mutex ex_mut;
  nthreads_ = nthreads;
  latch counter(workers_.size());
  work_ = [this, &work, nthreads, &counter, &dist, &ex, &ex_mut](size_t i)
    {
    if (i<nthreads)
      {
      try
        {
        MyScheduler sched(dist, i);
        work(sched);
        }
      catch (...)
        {
        LockGuard lock(ex_mut);
        ex = std::current_exception();
        }
      }
    counter.count_down();
    };
  news_ = !news_;
  news_.notify_all();
  {
  ScopedValueChanger<bool> changer(in_parallel_region, true);
  MyScheduler sched(dist, 0);
  work(sched);
  }
  counter.wait();
  if (ex)
    std::rethrow_exception(ex);
  }

void Distribution::thread_map(std::function<void(Scheduler &)> f)
  {
  if (nthreads_ == 1)
    {
    MyScheduler sched(*this, 0);
    f(sched);
    }
#ifndef DUCC0_NO_LOWLEVEL_THREADING
  else
    get_active_pool()->submit(f, *this, nthreads_);
#endif
  }

void execSingle(size_t nwork, std::function<void(Scheduler &)> func)
  {
  Distribution dist;
  dist.execSingle(nwork, std::move(func));
  }
void execStatic(size_t nwork, size_t nthreads, size_t chunksize,
  std::function<void(Scheduler &)> func)
  {
  Distribution dist;
  dist.execStatic(nwork, nthreads, chunksize, std::move(func));
  }
void execDynamic(size_t nwork, size_t nthreads, size_t chunksize,
  std::function<void(Scheduler &)> func)
  {
  Distribution dist;
  dist.execDynamic(nwork, nthreads, chunksize, std::move(func));
  }
void execGuided(size_t nwork, size_t nthreads, size_t chunksize_min,
  double fact_max, std::function<void(Scheduler &)> func)
  {
  Distribution dist;
  dist.execGuided(nwork, nthreads, chunksize_min, fact_max, std::move(func));
  }
void execParallel(size_t nthreads, std::function<void(Scheduler &)> func)
  {
  nthreads = adjust_nthreads(nthreads);
  Distribution dist;
  dist.execParallel(nthreads, std::move(func));
  }
void execParallel(size_t nthreads, std::function<void(size_t)> func)
  {
  Distribution dist;
  MR_assert(nthreads==adjust_nthreads(nthreads), "bad nthreads value");
  dist.execParallel(nthreads, [&](Scheduler &sched)
    { func(sched.thread_num()); });
  }
void execParallel(size_t work_lo, size_t work_hi, size_t nthreads,
  std::function<void(size_t, size_t)> func)
  {
  nthreads = adjust_nthreads(nthreads);
  execParallel(nthreads, [&](Scheduler &sched)
    {
    auto tid = sched.thread_num();
    auto [lo, hi] = calcShare(nthreads, tid, work_lo, work_hi);
    func(lo, hi);
    });
  }
void execParallel(size_t work_lo, size_t work_hi, size_t nthreads,
  std::function<void(size_t, size_t, size_t)> func)
  {
  MR_assert(nthreads==adjust_nthreads(nthreads), "bad nthreads value");
  execParallel(nthreads, [&](Scheduler &sched)
    {
    auto tid = sched.thread_num();
    auto [lo, hi] = calcShare(nthreads, tid, work_lo, work_hi);
    func(tid, lo, hi);
    });
  }

}}
