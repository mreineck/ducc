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

#ifndef MRUTIL_THREADING_H
#define MRUTIL_THREADING_H

#ifndef MRUTIL_NO_THREADING
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <functional>
#include <vector>
#if __has_include(<pthread.h>)
#include <pthread.h>
#endif
#endif

namespace mr {

namespace detail_threading {

using namespace std;

#ifndef MRUTIL_NO_THREADING
static const size_t max_threads_ = max(1u, thread::hardware_concurrency());

inline atomic<size_t> &default_nthreads()
  {
  static atomic<size_t> default_nthreads_=max_threads_;
  return default_nthreads_;
  }

inline size_t get_default_nthreads()
  { return default_nthreads(); }

inline void set_default_nthreads(size_t new_default_nthreads)
  { default_nthreads() = max<size_t>(1, new_default_nthreads); }

class latch
  {
    atomic<size_t> num_left_;
    mutex mut_;
    condition_variable completed_;
    using lock_t = unique_lock<mutex>;

  public:
    latch(size_t n): num_left_(n) {}

    void count_down()
      {
      lock_t lock(mut_);
      if (--num_left_)
        return;
      completed_.notify_all();
      }

    void wait()
      {
      lock_t lock(mut_);
      completed_.wait(lock, [this]{ return is_ready(); });
      }
    bool is_ready() { return num_left_ == 0; }
  };

template <typename T> class concurrent_queue
  {
    queue<T> q_;
    mutex mut_;
    condition_variable item_added_;
    bool shutdown_;
    using lock_t = unique_lock<mutex>;

  public:
    concurrent_queue(): shutdown_(false) {}

    void push(T val)
      {
      {
      lock_t lock(mut_);
      if (shutdown_)
        throw runtime_error("Item added to queue after shutdown");
      q_.push(move(val));
      }
      item_added_.notify_one();
      }

    bool pop(T & val)
      {
      lock_t lock(mut_);
      item_added_.wait(lock, [this] { return (!q_.empty() || shutdown_); });
      if (q_.empty())
        return false;  // We are shutting down

      val = std::move(q_.front());
      q_.pop();
      return true;
      }

    void shutdown()
      {
      {
      lock_t lock(mut_);
      shutdown_ = true;
      }
      item_added_.notify_all();
      }

    void restart() { shutdown_ = false; }
  };

class thread_pool
  {
    concurrent_queue<function<void()>> work_queue_;
    vector<thread> threads_;

    void worker_main()
      {
      function<void()> work;
      while (work_queue_.pop(work))
        work();
      }

    void create_threads()
      {
      size_t nthreads = threads_.size();
      for (size_t i=0; i<nthreads; ++i)
        {
        try { threads_[i] = thread([this]{ worker_main(); }); }
        catch (...)
          {
          shutdown();
          throw;
          }
        }
      }

  public:
    explicit thread_pool(size_t nthreads):
      threads_(nthreads)
      { create_threads(); }

    thread_pool(): thread_pool(max_threads_) {}

    ~thread_pool() { shutdown(); }

    void submit(function<void()> work)
      {
      work_queue_.push(move(work));
      }

    void shutdown()
      {
      work_queue_.shutdown();
      for (auto &thread : threads_)
        if (thread.joinable())
          thread.join();
      }

    void restart()
      {
      work_queue_.restart();
      create_threads();
      }
  };

inline thread_pool &get_pool()
  {
  static thread_pool pool;
#if __has_include(<pthread.h>)
  static once_flag f;
  call_once(f,
    []{
    pthread_atfork(
      +[]{ get_pool().shutdown(); },  // prepare
      +[]{ get_pool().restart(); },   // parent
      +[]{ get_pool().restart(); }    // child
      );
    });
#endif

  return pool;
  }

struct Range
  {
  size_t lo, hi;
  Range() : lo(0), hi(0) {}
  Range(size_t lo_, size_t hi_) : lo(lo_), hi(hi_) {}
  operator bool() const { return hi>lo; }
  };

class Distribution
  {
  private:
    size_t nthreads_;
    mutex mut_;
    size_t nwork_;
    size_t cur_;
    size_t chunksize_;
    double fact_max_;
    vector<size_t> nextstart;
    enum SchedMode { SINGLE, STATIC, DYNAMIC };
    SchedMode mode;
    bool single_done;

    template <typename Func> void thread_map(Func f);

  public:
    size_t nthreads() const { return nthreads_; }

    template<typename Func> void execSingle(size_t nwork, Func f)
      {
      mode = SINGLE;
      single_done = false;
      nwork_ = nwork;
      nthreads_ = 1;
      thread_map(move(f));
      }
    template<typename Func> void execStatic(size_t nwork,
      size_t nthreads, size_t chunksize, Func f)
      {
      mode = STATIC;
      nthreads_ = (nthreads==0) ? get_default_nthreads() : nthreads;
      nwork_ = nwork;
      chunksize_ = (chunksize<1) ? (nwork_+nthreads_-1)/nthreads_
                                 : chunksize;
      if (chunksize_>=nwork_) return execSingle(nwork_, move(f));
      nextstart.resize(nthreads_);
      for (size_t i=0; i<nextstart.size(); ++i)
        nextstart[i] = i*chunksize_;
      thread_map(move(f));
      }
    template<typename Func> void execDynamic(size_t nwork,
      size_t nthreads, size_t chunksize_min, double fact_max, Func f)
      {
      mode = DYNAMIC;
      nthreads_ = (nthreads==0) ? get_default_nthreads() : nthreads;
      nwork_ = nwork;
      chunksize_ = (chunksize_min<1) ? 1 : chunksize_min;
      if (chunksize_*nthreads_>=nwork_)
        return execStatic(nwork, nthreads, 0, move(f));
      fact_max_ = fact_max;
      cur_ = 0;
      thread_map(move(f));
      }
    template<typename Func> void execParallel(size_t nthreads, Func f)
      {
      mode = STATIC;
      nthreads_ = (nthreads==0) ? get_default_nthreads() : nthreads;
      nwork_ = nthreads_;
      chunksize_ = 1;
      thread_map(move(f));
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
          if (nextstart[thread_id]>=nwork_) return Range();
          size_t lo=nextstart[thread_id];
          size_t hi=min(lo+chunksize_,nwork_);
          nextstart[thread_id] += nthreads_*chunksize_;
          return Range(lo, hi);
          }
        case DYNAMIC:
          {
          unique_lock<mutex> lck(mut_);
          if (cur_>=nwork_) return Range();
          auto rem = nwork_-cur_;
          size_t tmp = size_t((fact_max_*rem)/nthreads_);
          auto sz = min(rem, max(chunksize_, tmp));
          size_t lo=cur_;
          cur_+=sz;
          size_t hi=cur_;
          return Range(lo, hi);
          }
        }
      return Range();
      }
  };

class Scheduler
  {
  private:
    Distribution &dist_;
    size_t ithread_;

  public:
    Scheduler(Distribution &dist, size_t ithread)
      : dist_(dist), ithread_(ithread) {}
    size_t num_threads() const { return dist_.nthreads(); }
    size_t thread_num() const { return ithread_; }
    Range getNext() { return dist_.getNext(ithread_); }
  };

template <typename Func> void Distribution::thread_map(Func f)
  {
  auto & pool = get_pool();
  latch counter(nthreads_);
  exception_ptr ex;
  mutex ex_mut;
  for (size_t i=0; i<nthreads_; ++i)
    {
    pool.submit(
      [this, &f, i, &counter, &ex, &ex_mut] {
      try
        {
        Scheduler sched(*this, i);
        f(sched);
        }
      catch (...)
        {
        lock_guard<mutex> lock(ex_mut);
        ex = current_exception();
        }
      counter.count_down();
      });
    }
  counter.wait();
  if (ex)
    rethrow_exception(ex);
  }

#else

constexpr size_t max_threads_ = 1;

class Sched0
  {
  private:
    size_t nwork_;

  public:
    size_t nthreads() const { return 1; }

    template<typename Func> void execSingle(size_t nwork, Func f)
      {
      nwork_ = nwork;
      f(Scheduler(*this,0));
      }
    template<typename Func> void execStatic(size_t nwork,
      size_t /*nthreads*/, size_t /*chunksize*/, Func f)
      {  execSingle(nwork, move(f)); }
    template<typename Func> void execDynamic(size_t nwork,
      size_t /*nthreads*/, size_t /*chunksize_min*/, double /*fact_max*/,
      Func f)
      {  execSingle(nwork, move(f)); }
    Range getNext()
      {
      Range res(0, nwork_);
      nwork_=0;
      return res;
      }
  };

template<typename Func> void execParallel(size_t /*nthreads*/, Func f)
  { f(); }

inline size_t get_default_nthreads()
  { return 1; }

inline void set_default_nthreads(size_t /*new_default_nthreads*/)
  {}

#endif

namespace {

template<typename Func> void execSingle(size_t nwork, Func f)
  {
  Distribution dist;
  dist.execSingle(nwork, move(f));
  }
template<typename Func> void execStatic(size_t nwork,
  size_t nthreads, size_t chunksize, Func f)
  {
  Distribution dist;
  dist.execStatic(nwork, nthreads, chunksize, move(f));
  }
template<typename Func> void execDynamic(size_t nwork,
  size_t nthreads, size_t chunksize_min, Func f)
  {
  Distribution dist;
  dist.execDynamic(nwork, nthreads, chunksize_min, 0., move(f));
  }
template<typename Func> void execGuided(size_t nwork,
  size_t nthreads, size_t chunksize_min, double fact_max, Func f)
  {
  Distribution dist;
  dist.execDynamic(nwork, nthreads, chunksize_min, fact_max, move(f));
  }

template<typename Func> static void execParallel(size_t nthreads, Func f)
  {
  Distribution dist;
  dist.execParallel(nthreads, move(f));
  }

inline size_t max_threads()
  { return max_threads_; }

}

} // end of namespace detail

using detail_threading::get_default_nthreads;
using detail_threading::set_default_nthreads;
using detail_threading::Scheduler;
using detail_threading::execSingle;
using detail_threading::execStatic;
using detail_threading::execDynamic;
using detail_threading::execGuided;
using detail_threading::max_threads;
using detail_threading::execParallel;

} // end of namespace mr

#endif
