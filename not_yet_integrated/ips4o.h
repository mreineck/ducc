/******************************************************************************
 * ips4o.h
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 *
 ******************************************************************************
 * BSD 2-Clause License
 *
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#ifndef DUCC0_IPS4O_H
#define DUCC0_IPS4O_H

#include <random>
#include <functional>
#include <iterator>
#include <type_traits>
#include <cstddef>
#include <limits>
#include <cassert>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>

#define IPS4OML_ASSUME_NOT(c) if (c) __builtin_unreachable()
#define IPS4OML_IS_NOT(c) assert(!(c))
namespace ducc0 {
namespace detail_sort {
namespace ips4o {
namespace detail {

template<typename T> class concurrent_queue
  {
  private:
    std::queue<T> q;
    mutable std::mutex mut;

  public:
    bool empty() const
      {
      std::lock_guard<std::mutex> lock(mut);
      return q.empty();
      }
    bool try_pop(T &res)
      {
      std::lock_guard<std::mutex> lock(mut);
      if (q.empty()) return false;
      res = q.front();
      q.pop();
      return true;
      }
    void push(const T &val)
      {
      std::lock_guard<std::mutex> lock(mut);
      q.push(val);
      }
  };

template <typename T> class PrivateQueue
  {
  public:
    PrivateQueue(size_t init_size = ((1ul<<12)+sizeof(T)-1)/sizeof(T))
      : m_v(), m_off(0)
      {
      // Preallocate memory. By default, the vector covers at least one page.
      m_v.reserve(init_size);
      }

    template <typename Iterator> void push(Iterator begin, Iterator end)
      { m_v.insert(m_v.end(), begin, end); }

    template <typename T1> void push(const T1&& e)
      { m_v.emplace_back(std::forward(e)); }

    template <typename... Args> void emplace(Args... args)
      { m_v.emplace_back(args...); }

    size_t size() const { return m_v.size() - m_off; }

    bool empty() const { return size() == 0; }

    T popBack()
      {
      assert(m_v.size() > m_off);
      const T e = m_v.back();
      m_v.pop_back();
      if (m_v.size() == m_off)
        clear();
      return e;
      }

    T popFront()
      {
      assert(m_v.size() > m_off);
      const T e = m_v[m_off];
      ++m_off;
      if (m_v.size() == m_off)
        clear();
      return e;
      }

    void clear()
      {
      m_off = 0;
      m_v.clear();
      }

  protected:
    std::vector<T> m_v;
    size_t m_off;
  };

template <typename Job> class Scheduler {
  public:
    Scheduler(size_t num_threads) : m_num_idle_threads(0), m_num_threads(num_threads) {}

    bool getJob(PrivateQueue<Job>& my_queue, Job& j)
      {
      // Try to get local job.
      if (!my_queue.empty())
        {
        j = my_queue.popBack();
        return true;
        }

      // Try to get global job.
      const bool succ = m_glob_queue.try_pop(j);
      if (succ) return succ;

      // Signal idle.
      m_num_idle_threads.fetch_add(1, std::memory_order_relaxed);

      while (m_num_idle_threads.load(std::memory_order_relaxed) != m_num_threads)
        if (!m_glob_queue.empty())
          {
          m_num_idle_threads.fetch_sub(1, std::memory_order_relaxed);

          const bool succ = m_glob_queue.try_pop(j);
          if (succ) return succ;

          m_num_idle_threads.fetch_add(1, std::memory_order_relaxed);
          }

      return false;
      }

    void offerJob(PrivateQueue<Job>& my_queue)
      {
      if (my_queue.size() > 1 && m_num_idle_threads.load(std::memory_order_relaxed) > 0
        && m_glob_queue.empty())
        addJob(my_queue.popFront());
      }

    void addJob(const Job& j) { m_glob_queue.push(j); }

    void addJob(const Job&& j) { m_glob_queue.push(j); }

    void reset() { m_num_idle_threads.store(0, std::memory_order_relaxed); }

  protected:
    concurrent_queue<Job> m_glob_queue;
    std::atomic_uint64_t m_num_idle_threads;
    const size_t m_num_threads;
  };

inline constexpr unsigned long log2(unsigned long n)
  {
  return (std::numeric_limits<unsigned long>::digits - 1 - __builtin_clzl(n));
  }
/** A subtask in the parallel algorithm.
    Uses indices instead of iterators to avoid unnecessary template instantiations. */
struct Task
  {
  Task() {}
  Task(std::ptrdiff_t begin, std::ptrdiff_t end) : begin(begin), end(end) {}

  std::ptrdiff_t begin;
  std::ptrdiff_t end;
  };

/** Thread barrier, also supports single() execution. */
class Barrier {
  public:
    explicit Barrier(int num_threads)
      : init_count_(num_threads), hit_count_(num_threads), flag_(false) {}

    void barrier()
      {
      std::unique_lock<std::mutex> lk(mutex_);
      if (--hit_count_ == 0)
        notify_all(lk);
      else
        cv_.wait(lk, [this, f = flag_] { return f != flag_; });
      }

    template <typename F> void single(F&& func)
      {
      std::unique_lock<std::mutex> lk(mutex_);
      if (hit_count_-- == init_count_)
        {
        lk.unlock();
        func();
        lk.lock();
        --hit_count_;
        }
      if (hit_count_ < 0)
        notify_all(lk);
      else
        cv_.wait(lk, [this, f = flag_] { return f != flag_; });
      }

    /** Reset the number of threads.
        No thread must currently be waiting at this barrier. */
    void setNumThreads(int num_threads)
      { hit_count_ = init_count_ = num_threads; }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int init_count_;
    int hit_count_;
    bool flag_;

    void notify_all(std::unique_lock<std::mutex>& lk)
      {
      hit_count_ = init_count_;
      flag_ = !flag_;
      lk.unlock();
      cv_.notify_all();
      }
  };

/** General synchronization support. */
class Sync: public Barrier
  {
  public:
    explicit Sync(int num_threads) : Barrier(num_threads) {}
  };

//base case

/** Insertion sort. */
template <typename It, typename Comp>
void insertionSort(const It begin, const It end, Comp comp) {
  IPS4OML_ASSUME_NOT(begin >= end);

  for (It it=begin+1; it<end; ++it) {
    typename std::iterator_traits<It>::value_type val = std::move(*it);
    if (comp(val, *begin)) {
      std::move_backward(begin, it, it+1);
      *begin = std::move(val);
    } else {
      auto cur = it;
      for (auto next=it-1; comp(val, *next); --next) {
        *cur = std::move(*next);
        cur = next;
        }
      *cur = std::move(val);
    }
  }
}

/** Wrapper for base case sorter, for easier swapping. */
template <typename It, typename Comp>
inline void baseCaseSort(It begin, It end, Comp&& comp) {
  if (begin == end) return;
  detail::insertionSort(std::move(begin), std::move(end), std::forward<Comp>(comp));
}

template <typename It, typename Comp, typename ThreadPool>
inline bool isSorted(It begin, It end, Comp&& comp, ThreadPool& thread_pool) {
    // Do nothing if input is already sorted.
    std::vector<bool> is_sorted(thread_pool.numThreads());
    thread_pool(
            [begin, end, &is_sorted, &comp](int my_id, int num_threads) {
                const auto size = end - begin;
                const auto stripe = (size + num_threads - 1) / num_threads;
                const auto my_begin = begin + std::min(stripe * my_id, size);
                const auto my_end = begin + std::min(stripe * (my_id + 1) + 1, size);
                is_sorted[my_id] = std::is_sorted(my_begin, my_end, comp);
            },
            thread_pool.numThreads());

    return std::all_of(is_sorted.begin(), is_sorted.end(), [](bool res) { return res; });
}

template <typename It, typename Comp>
inline bool sortSimpleCases(It begin, It end, Comp&& comp) {
  if (begin == end) return true;

  // If last element is not smaller than first element,
  // test if input is sorted (input is not reverse sorted).
  if (!comp(*(end-1), *begin)) {
    if (std::is_sorted(begin, end, comp)) return true;
  } else {
    // Check whether the input is reverse sorted.
    for (It it = begin; (it + 1) != end; ++it)
      if (comp(*it, *(it + 1))) return false;
    std::reverse(begin, end);
    return true;
  }

  return false;
}

/** Selects a random sample in-place. */
template <typename It, typename RandomGen>
void selectSample(It begin, const It end,
                  typename std::iterator_traits<It>::difference_type num_samples,
                  RandomGen&& gen)
  {
    using std::swap;

  auto n = end - begin;
  while (num_samples--)
    {
    const auto i = std::uniform_int_distribution<
                typename std::iterator_traits<It>::difference_type>(0, --n)(gen);
      swap(*begin, begin[i]);
      ++begin;
    }
  }

}

#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))

/** A thread pool using std::thread. */
class StdThreadPool
  {
  public:
    using Sync = detail::Sync;

    explicit StdThreadPool(int num_threads = StdThreadPool::maxNumThreads())
        : impl_(new Impl(num_threads)) {}

    template <typename F>
    void operator()(F&& func, int num_threads = std::numeric_limits<int>::max())
      {
      num_threads = std::min(num_threads, numThreads());
      (num_threads > 1) ? impl_.get()->run(std::forward<F>(func), num_threads)
                        : func(0, 1);
      }

    Sync& sync() { return impl_.get()->sync_; }

    int numThreads() const { return impl_.get()->threads_.size() + 1; }

    static int maxNumThreads() { return std::thread::hardware_concurrency(); }

 private:
    struct Impl
      {
      Sync sync_;
      detail::Barrier pool_barrier_;
      std::vector<std::thread> threads_;
      std::function<void(int, int)> func_;
      int num_threads_;
      bool done_ = false;

      Impl(int num_threads)
        : sync_(std::max(1, num_threads))
        , pool_barrier_(std::max(1, num_threads))
        , num_threads_(num_threads)
        {
        num_threads = std::max(1, num_threads);
        threads_.reserve(num_threads - 1);
        for (int i = 1; i < num_threads; ++i)
            threads_.emplace_back(&Impl::main, this, i);
        }
      ~Impl()
        {
        done_ = true;
        pool_barrier_.barrier();
        for (auto& t : threads_)
          t.join();
        }

      template <typename F>
      void run(F&& func, const int num_threads)
        {
        func_ = func;
        num_threads_ = num_threads;
        sync_.setNumThreads(num_threads);
    
        pool_barrier_.barrier();
        func_(0, num_threads);
        pool_barrier_.barrier();
        }

      void main(const int my_id)
        {
        for (;;)
          {
          pool_barrier_.barrier();
          if (done_) break;
          if (my_id < num_threads_)
            func_(my_id, num_threads_);
          pool_barrier_.barrier();
          }
        }
    };

    std::unique_ptr<Impl> impl_;
  };

/** A thread pool to which external threads can join. */
class ThreadJoiningThreadPool
  {
  public:
    using Sync = detail::Sync;

    explicit ThreadJoiningThreadPool(int num_threads) : impl_(new Impl(num_threads))
      { assert(num_threads >= 2); }

    void join(int my_id) { impl_->join(my_id); }

    void release_threads() { impl_->release_threads(); }

    template <typename F>
    void operator()(F&& func, int num_threads = std::numeric_limits<int>::max())
      {
      num_threads = std::min(num_threads, numThreads());
      (num_threads > 1) ? impl_->run(std::forward<F>(func), num_threads)
                        : func(0, 1);
      }

    Sync& sync() { return impl_.get()->sync_; }

    int numThreads() const { return impl_.get()->num_threads_; }

  private:
    struct Impl
      {
      Sync sync_;
      detail::Barrier pool_barrier_;
      std::function<void(int, int)> func_;
      int num_threads_;
      bool done_ = false;

      Impl(int num_threads)
        : sync_(num_threads), pool_barrier_(num_threads), num_threads_(num_threads) {}
      ~Impl()
        { assert(done_ == true); }

      template <typename F> void run(F&& func, const int num_threads)
        {
        func_ = func;
        num_threads_ = num_threads;
        sync_.setNumThreads(num_threads);
    
        pool_barrier_.barrier();
        func_(0, num_threads);
        pool_barrier_.barrier();
        }

      void join(int my_id)
        { main(my_id); }
      void release_threads()
        {
        done_ = true;
        pool_barrier_.barrier();
        }

      void main(const int my_id)
        {
        for (;;)
          {
          pool_barrier_.barrier();
          if (done_) break;
          if (my_id < num_threads_)
            func_(my_id, num_threads_);
          pool_barrier_.barrier();
          }
        }
      };

    std::unique_ptr<Impl> impl_;
  };

using DefaultThreadPool = StdThreadPool;

#endif  // threading

#ifndef IPS4OML_ALLOW_EQUAL_BUCKETS
#define IPS4OML_ALLOW_EQUAL_BUCKETS true
#endif

#ifndef IPS4OML_BASE_CASE_SIZE
#define IPS4OML_BASE_CASE_SIZE 16
#endif

#ifndef IPS4OML_BASE_CASE_MULTIPLIER
#define IPS4OML_BASE_CASE_MULTIPLIER 16
#endif

#ifndef IPS4OML_BLOCK_SIZE
#define IPS4OML_BLOCK_SIZE (2 << 10)
#endif

#ifndef IPS4OML_BUCKET_TYPE
#define IPS4OML_BUCKET_TYPE std::ptrdiff_t
#endif

#ifndef IPS4OML_DATA_ALIGNMENT
#define IPS4OML_DATA_ALIGNMENT (4 << 10)
#endif

#ifndef IPS4OML_EQUAL_BUCKETS_THRESHOLD
#define IPS4OML_EQUAL_BUCKETS_THRESHOLD 5
#endif

#ifndef IPS4OML_LOG_BUCKETS
#define IPS4OML_LOG_BUCKETS 8
#endif

#ifndef IPS4OML_MIN_PARALLEL_BLOCKS_PER_THREAD
#define IPS4OML_MIN_PARALLEL_BLOCKS_PER_THREAD 4
#endif

#ifndef IPS4OML_OVERSAMPLING_FACTOR_PERCENT
#define IPS4OML_OVERSAMPLING_FACTOR_PERCENT 20
#endif

#ifndef IPS4OML_UNROLL_CLASSIFIER
#define IPS4OML_UNROLL_CLASSIFIER 7
#endif

template <bool AllowEqualBuckets_     = IPS4OML_ALLOW_EQUAL_BUCKETS
        , std::ptrdiff_t BaseCase_    = IPS4OML_BASE_CASE_SIZE
        , std::ptrdiff_t BaseCaseM_   = IPS4OML_BASE_CASE_MULTIPLIER
        , std::ptrdiff_t BlockSize_   = IPS4OML_BLOCK_SIZE
        , typename BucketT_           = IPS4OML_BUCKET_TYPE
        , std::size_t DataAlign_      = IPS4OML_DATA_ALIGNMENT
        , std::ptrdiff_t EqualBuckTh_ = IPS4OML_EQUAL_BUCKETS_THRESHOLD
        , int LogBuckets_             = IPS4OML_LOG_BUCKETS
        , std::ptrdiff_t MinParBlks_  = IPS4OML_MIN_PARALLEL_BLOCKS_PER_THREAD
        , int OversampleF_            = IPS4OML_OVERSAMPLING_FACTOR_PERCENT
        , int UnrollClass_            = IPS4OML_UNROLL_CLASSIFIER
        >
struct Config {
    /** The type used for bucket indices in the classifier. */
    using bucket_type = BucketT_;

    /** Whether we are on 64 bit or 32 bit. */
    static constexpr const bool kIs64Bit = sizeof(std::uintptr_t) == 8;
    static_assert(kIs64Bit || sizeof(std::uintptr_t) == 4,
                  "Architecture must be 32 or 64 bit");

    /** Whether equal buckets can be used. */
    static constexpr const bool kAllowEqualBuckets = AllowEqualBuckets_;
    /** Desired base case size. */
    static constexpr const std::ptrdiff_t kBaseCaseSize = BaseCase_;
    /** Multiplier for base case threshold. */
    static constexpr const int kBaseCaseMultiplier = BaseCaseM_;
    /** Number of bytes in one block. */
    static constexpr const std::ptrdiff_t kBlockSizeInBytes = BlockSize_;
    /** Alignment for shared and thread-local data. */
    static constexpr const std::size_t kDataAlignment = DataAlign_;
    /** Number of splitters that must be equal before equality buckets are enabled. */
    static constexpr const std::ptrdiff_t kEqualBucketsThreshold = EqualBuckTh_;
    /** Logarithm of the maximum number of buckets (excluding equality buckets). */
    static constexpr const int kLogBuckets = LogBuckets_;
    /** Minimum number of blocks per thread for which parallelism is used. */
    static constexpr const std::ptrdiff_t kMinParallelBlocksPerThread = MinParBlks_;
    static_assert(kMinParallelBlocksPerThread > 0,
                  "Min. blocks per thread must be at least 1.");
    /** How many times the classification loop is unrolled. */
    static constexpr const int kUnrollClassifier = UnrollClass_;

    static constexpr const std::ptrdiff_t kSingleLevelThreshold =
            kBaseCaseSize * (1ul << kLogBuckets);
    static constexpr const std::ptrdiff_t kTwoLevelThreshold =
            kSingleLevelThreshold * (1ul << kLogBuckets);

    /** The oversampling factor to be used for input of size n. */
    static constexpr double oversamplingFactor(std::ptrdiff_t n) {
        const double f = OversampleF_ / 100.0 * detail::log2(n);
        return f < 1.0 ? 1.0 : f;
    }

    /** Computes the logarithm of the number of buckets to use for input size n. */
    static int logBuckets(const std::ptrdiff_t n) {
        if (n <= kSingleLevelThreshold) {
            // Only one more level until the base case, reduce the number of buckets
            return std::max(1ul, detail::log2(n / kBaseCaseSize));
        } else if (n <= kTwoLevelThreshold) {
            // Only two more levels until we reach the base case, split the buckets evenly
            return std::max(1ul, (detail::log2(n / kBaseCaseSize) + 1) / 2);
        } else {
            // Use the maximum number of buckets
            return kLogBuckets;
        }
    }

    /**
     * Returns the number of threads that should be used for the given input range.
     */
    template <typename It>
#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))
    static constexpr int numThreadsFor(const It& begin, const It& end, int max_threads) {
        const std::ptrdiff_t blocks =
                (end - begin) * sizeof(decltype(*begin)) / kBlockSizeInBytes;
        return (blocks < (kMinParallelBlocksPerThread * max_threads)) ? 1 : max_threads;
#else
    static constexpr int numThreadsFor(const It&, const It&, int) {
        return 1;
#endif
    }
};

template <typename It_, typename Comp_, typename Cfg = Config<>
#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))
          , typename ThreadPool_ = DefaultThreadPool
#endif
        >
struct ExtendedConfig : public Cfg {
    /** Base config containing user-specified parameters. */
    using BaseConfig = Cfg;
    /** The iterator type for the input data. */
    using iterator = It_;
    /** The difference type for the iterator. */
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    /** The value type of the input data. */
    using value_type = typename std::iterator_traits<iterator>::value_type;
    /** The comparison operator. */
    using less = Comp_;

#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))

    /** Thread pool for parallel algorithm. */
    using ThreadPool = ThreadPool_;

    using SubThreadPool = ThreadJoiningThreadPool;

    /** Synchronization support for parallel algorithm. */
    using Sync = decltype(std::declval<ThreadPool&>().sync());

#else

    struct Sync {
        constexpr void barrier() const {}
        template <typename F>
        constexpr void single(F&&) const {}
    };

    /** Dummy thread pool. */
    class SubThreadPool {
      public:
        explicit SubThreadPool(int) {}

        void join(int) {}

        void release_threads() {}

        template <typename F>
        void operator()(F&&, int) {}

        Sync& sync() { return sync_; }

        int numThreads() const { return 1; }

      private:
        Sync sync_;
    };

#endif

    /** Maximum number of buckets (including equality buckets). */
    static constexpr const int kMaxBuckets =
            1ul << (Cfg::kLogBuckets + Cfg::kAllowEqualBuckets);

    /** Number of elements in one block. */
    static constexpr const difference_type kBlockSize =
            1ul << (detail::log2(
                    Cfg::kBlockSizeInBytes < sizeof(value_type)
                            ? 1
                            : (Cfg::kBlockSizeInBytes / sizeof(value_type))));

    // Redefine applicable constants as difference_type.
    static constexpr const difference_type kBaseCaseSize = Cfg::kBaseCaseSize;
    static constexpr const difference_type kEqualBucketsThreshold =
            Cfg::kEqualBucketsThreshold;

    // Cannot sort without random access.
    static_assert(std::is_same<typename std::iterator_traits<iterator>::iterator_category,
                               std::random_access_iterator_tag>::value,
                  "Iterator must be a random access iterator.");
    // Number of buckets is limited by switch in classifier
    static_assert(Cfg::kLogBuckets >=1, "Max. bucket count must be <= 512.");
    // The implementation of the block alignment limits the possible block sizes.
    static_assert((kBlockSize & (kBlockSize - 1)) == 0,
                  "Block size must be a power of two.");
    // The main classifier function assumes that the loop can be unrolled at least once.
    static_assert(Cfg::kUnrollClassifier <= kBaseCaseSize,
                  "Base case size must be larger than unroll factor.");

    /** Aligns an offset to the next block boundary, upwards. */
    static constexpr difference_type alignToNextBlock(difference_type p) {
        return (p + kBlockSize - 1) & ~(kBlockSize - 1);
    }
};

#undef IPS4OML_ALLOW_EQUAL_BUCKETS
#undef IPS4OML_BASE_CASE_SIZE
#undef IPS4OML_BASE_CASE_MULTIPLIER
#undef IPS4OML_BLOCK_SIZE
#undef IPS4OML_BUCKET_TYPE
#undef IPS4OML_DATA_ALIGNMENT
#undef IPS4OML_EQUAL_BUCKETS_THRESHOLD
#undef IPS4OML_LOG_BUCKETS
#undef IPS4OML_MIN_PARALLEL_BLOCKS_PER_THREAD
#undef IPS4OML_OVERSAMPLING_FACTOR_PERCENT
#undef IPS4OML_UNROLL_CLASSIFIER

namespace detail {

/** Data describing a parallel task and the corresponding threads. */
struct BigTask {
    BigTask() : has_task{false} {}
    // TODO or Cfg::iterator???
    std::ptrdiff_t begin;
    std::ptrdiff_t end;
    // My thread id of this task.
    int task_thread_id;
    // Index of the thread owning the thread pool used by this task.
    int root_thread;
    // Indicates whether this is a task or not
    bool has_task;
};
/** Aligns a pointer. */
template <typename T> static T* alignPointer(T* ptr, std::size_t alignment) {
  uintptr_t v = reinterpret_cast<std::uintptr_t>(ptr);
  v = (v - 1 + alignment) & ~(alignment - 1);
  return reinterpret_cast<T*>(v);
  }

/** Constructs an object at the specified alignment. */
template <typename T> class AlignedPtr
  {
  public:
    AlignedPtr() {}

    template <typename... Args>
    explicit AlignedPtr(std::size_t alignment, Args&&... args)
        : alloc_(new char[sizeof(T) + alignment])
        , value_(new (alignPointer(alloc_, alignment)) T(std::forward<Args>(args)...)) {}

    AlignedPtr(const AlignedPtr&) = delete;
    AlignedPtr& operator=(const AlignedPtr&) = delete;

    AlignedPtr(AlignedPtr&& rhs) : alloc_(rhs.alloc_), value_(rhs.value_)
      { rhs.alloc_ = nullptr; }
    AlignedPtr& operator=(AlignedPtr&& rhs) {
        std::swap(alloc_, rhs.alloc_);
        std::swap(value_, rhs.value_);
        return *this;
    }

    ~AlignedPtr() {
        if (alloc_) {
            value_->~T();
            delete[] alloc_;
        }
    }

    T& get() { return *value_; }

  private:
    char* alloc_ = nullptr;
    T* value_;
  };

/** Provides aligned storage without constructing an object. */
template <> class AlignedPtr<void>
  {
  public:
    AlignedPtr() {}

    template <typename... Args>
    explicit AlignedPtr(std::size_t alignment, std::size_t size)
      : alloc_(new char[size + alignment]), value_(alignPointer(alloc_, alignment)) {}

    AlignedPtr(const AlignedPtr&) = delete;
    AlignedPtr& operator=(const AlignedPtr&) = delete;

    AlignedPtr(AlignedPtr&& rhs) : alloc_(rhs.alloc_), value_(rhs.value_)
      { rhs.alloc_ = nullptr; }
    AlignedPtr& operator=(AlignedPtr&& rhs)
      {
      std::swap(alloc_, rhs.alloc_);
      std::swap(value_, rhs.value_);
      return *this;
      }

    ~AlignedPtr()
      { if (alloc_) delete[] alloc_; }

    char* get() { return value_; }

  private:
    char* alloc_ = nullptr;
    char* value_;
  };

template <typename Cfg> class Sorter {
  public:
    using iterator = typename Cfg::iterator;
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;
    using SubThreadPool = typename Cfg::SubThreadPool;

    class BufferStorage;
    class Block;
    class Buffers;
    class BucketPointers;
    class Classifier;
    struct LocalData;
    struct SharedData;
    explicit Sorter(LocalData& local) : local_(local) {}

    void sequential(iterator begin, iterator end)
      {
      // Check for base case
      const auto n = end - begin;
      if (n <= 2 * Cfg::kBaseCaseSize)
        {
        detail::baseCaseSort(begin, end, local_.classifier.getComparator());
        return;
        }
      sequential_rec(begin, end);
      }

    void sequential(const iterator begin, const Task& task, PrivateQueue<Task>& queue)
      {
      // Check for base case
      const auto n = task.end - task.begin;
      IPS4OML_IS_NOT(n <= 2 * Cfg::kBaseCaseSize);

      diff_t bucket_start[Cfg::kMaxBuckets + 1];

      // Do the partitioning
      const auto res =
            partition<false>(begin + task.begin, begin + task.end, bucket_start, 0, 1);
      const int num_buckets = std::get<0>(res);
      const bool equal_buckets = std::get<1>(res);

      // Final base case is executed in cleanup step, so we're done here
      if (n <= Cfg::kSingleLevelThreshold)
        return;

      // Recurse
      if (equal_buckets)
        {
        const auto start = bucket_start[num_buckets - 1];
        const auto stop = bucket_start[num_buckets];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
          queue.emplace(task.begin + start, task.begin + stop);
        }
      for (int i = num_buckets - 1 - equal_buckets; i >= 0; i -= 1 + equal_buckets)
        {
        const auto start = bucket_start[i];
        const auto stop = bucket_start[i + 1];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
          queue.emplace(task.begin + start, task.begin + stop);
        }
      }

    void sequential_rec(iterator begin, iterator end)
      {
      // Check for base case
      const auto n = end - begin;
      IPS4OML_IS_NOT(n <= 2 * Cfg::kBaseCaseSize);

      diff_t bucket_start[Cfg::kMaxBuckets + 1];

      // Do the partitioning
      const auto res = partition<false>(begin, end, bucket_start, 0, 1);
      const int num_buckets = std::get<0>(res);
      const bool equal_buckets = std::get<1>(res);

      // Final base case is executed in cleanup step, so we're done here
      if (n <= Cfg::kSingleLevelThreshold)
        return;

      // Recurse
      for (int i = 0; i < num_buckets; i += 1 + equal_buckets)
        {
        const auto start = bucket_start[i];
        const auto stop = bucket_start[i + 1];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
            sequential(begin + start, begin + stop);
        }
      if (equal_buckets)
        {
        const auto start = bucket_start[num_buckets - 1];
        const auto stop = bucket_start[num_buckets];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
          sequential(begin + start, begin + stop);
        }
      }

#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))
    void parallelSortPrimary(iterator begin, iterator end, int num_threads,
                             BufferStorage& buffer_storage,
                             std::vector<std::shared_ptr<SubThreadPool>>& tp_trash)
      {
      const auto res = partition<true>(begin, end, shared_->bucket_start, 0, num_threads);
      const bool is_last_level = end - begin <= Cfg::kSingleLevelThreshold;
      const auto stripe = ((end - begin) + num_threads - 1) / num_threads;

      if (!is_last_level)
        {
        const int num_buckets = std::get<0>(res);
        const bool equal_buckets = std::get<1>(res);

        queueTasks(stripe, 0, num_threads, end - begin, begin - begin,
                   shared_->bucket_start, num_buckets, equal_buckets);
        }

      shared_->reset();
      shared_->sync.barrier();

      processBigTasks(begin, stripe, 0, buffer_storage, tp_trash);
      processSmallTasks(begin);
      }

    void parallelSortSecondary(iterator begin, iterator end, int id, int num_threads,
                               BufferStorage& buffer_storage,
                               std::vector<std::shared_ptr<SubThreadPool>>& tp_trash)
      {
      shared_->local[id] = &local_;

      partition<true>(begin, end, shared_->bucket_start, id, num_threads);
      shared_->sync.barrier();

      const auto stripe = ((end - begin) + num_threads - 1) / num_threads;
      processBigTasks(begin, stripe, id, buffer_storage, tp_trash);
      processSmallTasks(begin);
      }

    std::pair<std::vector<diff_t>, bool> parallelPartitionPrimary(iterator begin,
                                                                  iterator end,
                                                                  int num_threads)
      {
      const auto res = partition<true>(begin, end, shared_->bucket_start, 0, num_threads);
      const int num_buckets = std::get<0>(res);
      const bool equal_buckets = std::get<1>(res);

      std::vector<diff_t> bucket_start(shared_->bucket_start,
                                       shared_->bucket_start + num_buckets + 1);

      shared_->reset();
      shared_->sync.barrier();
      return {bucket_start, equal_buckets};
      }

    void parallelPartitionSecondary(iterator begin, iterator end, int id,
                                    int num_threads)
      {
      shared_->local[id] = &local_;
      partition<true>(begin, end, shared_->bucket_start, id, num_threads);
      shared_->sync.barrier();
      }

    void setShared(SharedData* shared)
      { shared_ = shared; }
#endif

 private:
    LocalData& local_;
    SharedData* shared_;
    Classifier* classifier_;

    diff_t* bucket_start_;
    BucketPointers* bucket_pointers_;
    Block* overflow_;

    iterator begin_;
    iterator end_;
    int num_buckets_;
    int my_id_;
    int num_threads_;

    std::pair<int, bool> buildClassifier(iterator begin, iterator end,
                                         Classifier& classifier)
      {
      const auto n = end - begin;
      int log_buckets = Cfg::logBuckets(n);
      int num_buckets = 1 << log_buckets;
      const auto step = std::max<diff_t>(1, diff_t(Cfg::oversamplingFactor(n)));
      const auto num_samples = std::min(step * num_buckets - 1, n / 2);
  
      // Select the sample
      detail::selectSample(begin, end, num_samples, local_.random_generator);
  
      // Sort the sample
      sequential(begin, begin + num_samples);
      auto splitter = begin + step - 1;
      auto sorted_splitters = classifier.getSortedSplitters();
      const auto comp = classifier.getComparator();
  
      // Choose the splitters
      IPS4OML_ASSUME_NOT(sorted_splitters == nullptr);
      new (sorted_splitters) typename Cfg::value_type(*splitter);
      for (int i = 2; i < num_buckets; ++i)
        {
        splitter += step;
        // Skip duplicates
        if (comp(*sorted_splitters, *splitter))
          {
          IPS4OML_ASSUME_NOT(sorted_splitters + 1 == nullptr);
          new (++sorted_splitters) typename Cfg::value_type(*splitter);
          }
        }
  
      // Check for duplicate splitters
      const auto diff_splitters = sorted_splitters - classifier.getSortedSplitters() + 1;
      const bool use_equal_buckets = Cfg::kAllowEqualBuckets
              && num_buckets - 1 - diff_splitters >= Cfg::kEqualBucketsThreshold;
  
      // Fill the array to the next power of two
      log_buckets = log2(diff_splitters) + 1;
      num_buckets = 1 << log_buckets;
      for (int i = diff_splitters + 1; i < num_buckets; ++i)
        {
        IPS4OML_ASSUME_NOT(sorted_splitters + 1 == nullptr);
        new (++sorted_splitters) typename Cfg::value_type(*splitter);
        }
  
      // Build the tree
      classifier.build(log_buckets);
      this->classifier_ = &classifier;
  
      const int used_buckets = num_buckets * (1 + use_equal_buckets);
      return {used_buckets, use_equal_buckets};
      }

    template <bool kEqualBuckets>
    __attribute__((flatten)) diff_t classifyLocally(iterator my_begin, iterator my_end)
      {
      auto write = my_begin;
      auto& buffers = local_.buffers;

      // Do the classification
      classifier_->template classify<kEqualBuckets>(
            my_begin, my_end, [&](typename Cfg::bucket_type bucket, iterator it)
        {
        // Only flush buffers on overflow
        if (buffers.isFull(bucket))
          {
          buffers.writeTo(bucket, write);
          write += Cfg::kBlockSize;
          local_.bucket_size[bucket] += Cfg::kBlockSize;
          }
        buffers.push(bucket, std::move(*it));
        });

      // Update bucket sizes to account for partially filled buckets
      for (int i = 0, end = num_buckets_; i < end; ++i)
        local_.bucket_size[i] += local_.buffers.size(i);

      return write - begin_;
      }

    void parallelClassification(bool use_equal_buckets)
      {
      // Compute stripe for each thread
      const auto elements_per_thread = static_cast<double>(end_ - begin_) / num_threads_;
      const auto my_begin =
              begin_ + Cfg::alignToNextBlock(typename Cfg::difference_type(my_id_ * elements_per_thread + 0.5));
      const auto my_end = [&] {
          auto e = begin_ + Cfg::alignToNextBlock(typename Cfg::difference_type((my_id_ + 1) * elements_per_thread + 0.5));
          e = end_ < e ? end_ : e;
          return e;
      }();
  
      local_.first_block = my_begin - begin_;
  
      // Do classification
      if (my_begin >= my_end) {
        // Small input (less than two blocks per thread), wait for other threads to finish
        local_.first_empty_block = my_begin - begin_;
        shared_->sync.barrier();
        shared_->sync.barrier();
      } else {
        const auto my_first_empty_block =
                  use_equal_buckets ? classifyLocally<true>(my_begin, my_end)
                                    : classifyLocally<false>(my_begin, my_end);
  
        // Find bucket boundaries
        diff_t sum = 0;
        for (int i = 0, end = num_buckets_; i < end; ++i)
          {
          sum += local_.bucket_size[i];
          __atomic_fetch_add(&bucket_start_[i + 1], sum, __ATOMIC_RELAXED);
          }
  
        local_.first_empty_block = my_first_empty_block;
  
        shared_->sync.barrier();
  
        // Move empty blocks and set bucket write/read pointers
        moveEmptyBlocks(my_begin - begin_, my_end - begin_, my_first_empty_block);
  
        shared_->sync.barrier();
        }
      }

    void sequentialClassification(bool use_equal_buckets)
      {
      const auto my_first_empty_block = use_equal_buckets
                                              ? classifyLocally<true>(begin_, end_)
                                              : classifyLocally<false>(begin_, end_);

      // Find bucket boundaries
      diff_t sum = 0;
      bucket_start_[0] = 0;
      for (int i = 0, end = num_buckets_; i < end; ++i)
        {
        sum += local_.bucket_size[i];
        bucket_start_[i + 1] = sum;
        }
      IPS4OML_ASSUME_NOT(bucket_start_[num_buckets_] != end_ - begin_);

      // Set write/read pointers for all buckets
      for (int bucket = 0, end = num_buckets_; bucket < end; ++bucket)
        {
        const auto start = Cfg::alignToNextBlock(bucket_start_[bucket]);
        const auto stop = Cfg::alignToNextBlock(bucket_start_[bucket + 1]);
        bucket_pointers_[bucket].set(
                start,
                (start >= my_first_empty_block
                         ? start
                         : (stop <= my_first_empty_block ? stop : my_first_empty_block))
                        - Cfg::kBlockSize);
        }
      }

    void moveEmptyBlocks(diff_t my_begin, diff_t my_end, diff_t my_first_empty_block)
{
    // Find range of buckets that start in this stripe
    const int bucket_range_start = [&](int i) {
        while (Cfg::alignToNextBlock(bucket_start_[i]) < my_begin) ++i;
        return i;
    }(0);
    const int bucket_range_end = [&](int i) {
        const auto num_buckets = num_buckets_;
        if (my_id_ == num_threads_ - 1) return num_buckets;
        while (i < num_buckets && Cfg::alignToNextBlock(bucket_start_[i]) < my_end) ++i;
        return i;
    }(bucket_range_start);

    /*
     * After classification, a stripe consists of full blocks followed by empty blocks.
     * This means that the invariant above already holds for all buckets except those that
     * cross stripe boundaries.
     *
     * The following cases exist:
     * 1)  The bucket is fully contained within one stripe.
     *     In this case, nothing needs to be done, just set the bucket pointers.
     *
     * 2)  The bucket starts in stripe i, and ends in stripe i+1.
     *     In this case, thread i moves full blocks from the end of the bucket (from the
     *     stripe of thread i+1) to fill the holes at the end of its stripe.
     *
     * 3)  The bucket starts in stripe i, crosses more than one stripe boundary, and ends
     *     in stripe i+k. This is an extension of case 2. In this case, multiple threads
     *     work on the same bucket. Each thread is responsible for filling the empty
     *     blocks in its stripe. The left-most thread will take the right-most blocks.
     *     Therefore, we count how many blocks are fetched by threads to our left before
     *     moving our own blocks.
     */

    // Check if last bucket overlaps the end of the stripe
    const auto bucket_end = Cfg::alignToNextBlock(bucket_start_[bucket_range_end]);
    const bool last_bucket_is_overlapping = bucket_end > my_end;

    // Case 1)
    for (int b = bucket_range_start; b < bucket_range_end - last_bucket_is_overlapping;
         ++b) {
        const auto start = Cfg::alignToNextBlock(bucket_start_[b]);
        const auto stop = Cfg::alignToNextBlock(bucket_start_[b + 1]);
        auto read = stop;
        if (my_first_empty_block <= start) {
            // Bucket is completely empty
            read = start;
        } else if (my_first_empty_block < stop) {
            // Bucket is partially empty
            read = my_first_empty_block;
        }
        bucket_pointers_[b].set(start, read - Cfg::kBlockSize);
    }

    // Cases 2) and 3)
    if (last_bucket_is_overlapping) {
        const int overlapping_bucket = bucket_range_end - 1;
        const auto bucket_start =
                Cfg::alignToNextBlock(bucket_start_[overlapping_bucket]);

        // If it is a very large bucket, other threads will also move blocks around in it
        // (case 3) Count how many filled blocks are in this bucket
        diff_t flushed_elements_in_bucket = 0;
        if (bucket_start < my_begin) {
            int prev_id = my_id_ - 1;
            // Iterate over stripes which are completely contained in this bucket
            while (bucket_start < shared_->local[prev_id]->first_block) {
                const auto eb = shared_->local[prev_id]->first_empty_block;
                flushed_elements_in_bucket += eb - shared_->local[prev_id]->first_block;
                --prev_id;
            }
            // Count blocks in stripe where bucket starts
            const auto eb = shared_->local[prev_id]->first_empty_block;
            // Check if there are any filled blocks in this bucket
            if (eb > bucket_start) flushed_elements_in_bucket += eb - bucket_start;
        }

        // Threads to our left will move this many blocks (0 if we are the left-most)
        diff_t elements_reserved = 0;
        if (my_begin > bucket_start) {
            // Threads to the left of us get priority
            elements_reserved = my_begin - bucket_start - flushed_elements_in_bucket;

            // Count how many elements we flushed into this bucket
            flushed_elements_in_bucket += my_first_empty_block - my_begin;
        } else if (my_first_empty_block > bucket_start) {
            // We are the left-most thread
            // Count how many elements we flushed into this bucket
            flushed_elements_in_bucket += my_first_empty_block - bucket_start;
        }

        // Find stripe which contains last block of this bucket (off by one)
        // Also continue counting how many filled blocks are in this bucket
        int read_from_thread = my_id_ + 1;
        while (read_from_thread < num_threads_
               && bucket_end > shared_->local[read_from_thread]->first_block) {
            const auto eb = std::min<diff_t>(
                    shared_->local[read_from_thread]->first_empty_block, bucket_end);
            flushed_elements_in_bucket +=
                    eb - shared_->local[read_from_thread]->first_block;
            ++read_from_thread;
        }

        // After moving blocks, this will be the first empty block in this bucket
        const auto first_empty_block_in_bucket =
                bucket_start + flushed_elements_in_bucket;

        // This is the range of blocks we want to fill
        auto write_ptr = begin_ + std::max(my_first_empty_block, bucket_start);
        const auto write_ptr_end = begin_ + std::min(first_empty_block_in_bucket, my_end);

        // Read from other stripes until we filled our blocks
        while (write_ptr < write_ptr_end) {
            --read_from_thread;
            // This is the range of blocks we can read from stripe 'read_from_thread'
            auto read_ptr = std::min(shared_->local[read_from_thread]->first_empty_block,
                                     bucket_end);
            auto read_range_size =
                    read_ptr - shared_->local[read_from_thread]->first_block;

            // Skip reserved blocks
            if (elements_reserved >= read_range_size) {
                elements_reserved -= read_range_size;
                continue;
            }
            read_ptr -= elements_reserved;
            read_range_size -= elements_reserved;
            elements_reserved = 0;

            // Move blocks
            const auto size = std::min(read_range_size, write_ptr_end - write_ptr);
            write_ptr = std::move(begin_ + read_ptr - size, begin_ + read_ptr, write_ptr);
        }

        // Set bucket pointers if the bucket starts in this stripe
        if (my_begin <= bucket_start) {
            bucket_pointers_[overlapping_bucket].set(
                    bucket_start, first_empty_block_in_bucket - Cfg::kBlockSize);
        }
    }
}

    int computeOverflowBucket()
      {
      int bucket = num_buckets_ - 1;
      while (bucket >= 0
             && (bucket_start_[bucket + 1] - bucket_start_[bucket]) <= Cfg::kBlockSize)
          --bucket;
      return bucket;
      }

    template <bool kEqualBuckets, bool kIsParallel> int classifyAndReadBlock(int read_bucket)
      {
      auto& bp = bucket_pointers_[read_bucket];

      diff_t write, read;
      std::tie(write, read) = bp.template decRead<kIsParallel>();

      if (read < write)
        {
        // No more blocks in this bucket
        if (kIsParallel) bp.stopRead();
        return -1;
        }

      // Read block
      local_.swap[0].readFrom(begin_ + read);
      if (kIsParallel) bp.stopRead();

      return classifier_->template classify<kEqualBuckets>(local_.swap[0].head());
      }

    template <bool kEqualBuckets, bool kIsParallel>
    int swapBlock(diff_t max_off, int dest_bucket, bool current_swap)
      {
      diff_t write, read;
      int new_dest_bucket;
      auto& bp = bucket_pointers_[dest_bucket];
      do
        {
        std::tie(write, read) = bp.template incWrite<kIsParallel>();
        if (write > read)
          {
          // Destination block is empty
          if (write >= max_off)
            {
            // Out-of-bounds; write to overflow buffer instead
            local_.swap[current_swap].writeTo(local_.overflow);
            overflow_ = &local_.overflow;
            return -1;
            }
          // Make sure no one is currently reading this block
          while (kIsParallel && bp.isReading()) {}
          // Write block
          local_.swap[current_swap].writeTo(begin_ + write);
          return -1;
          }
        // Check if block needs to be moved
        new_dest_bucket = classifier_->template classify<kEqualBuckets>(begin_[write]);
        } while (new_dest_bucket == dest_bucket);

      // Swap blocks
      local_.swap[!current_swap].readFrom(begin_ + write);
      local_.swap[current_swap].writeTo(begin_ + write);

      return new_dest_bucket;
      }

    template <bool kEqualBuckets, bool kIsParallel> void permuteBlocks()
      {
      const auto num_buckets = num_buckets_;
      // Distribute starting points of threads
      int read_bucket = (my_id_ * num_buckets / num_threads_) % num_buckets;
      // Not allowed to write to this offset, to avoid overflow
      const diff_t max_off = Cfg::alignToNextBlock(end_ - begin_ + 1) - Cfg::kBlockSize;
  
      // Go through all buckets
      for (int count = num_buckets; count; --count)
        {
        int dest_bucket;
        // Try to read a block ...
        while ((dest_bucket =
                          classifyAndReadBlock<kEqualBuckets, kIsParallel>(read_bucket))
                 != -1)
          {
          bool current_swap = false;
          // ... then write it to the correct bucket
          while ((dest_bucket = swapBlock<kEqualBuckets, kIsParallel>(
                  max_off, dest_bucket, current_swap))
                     != -1)
            current_swap = !current_swap; // Read another block, keep going
          }
        read_bucket = (read_bucket + 1) % num_buckets;
        }
      }

    std::pair<int, diff_t> saveMargins(int last_bucket)
      {
      // Find last bucket boundary in this thread's area
      diff_t tail = bucket_start_[last_bucket];
      const diff_t end = Cfg::alignToNextBlock(tail);

      // Don't need to do anything if there is no overlap, or we are in the overflow case
      if (tail == end || end > (end_ - begin_))
        return {-1, 0};

      // Find bucket this last block belongs to
      {
      const auto start_of_last_block = end - Cfg::kBlockSize;
      diff_t last_start;
      do {
        --last_bucket;
        last_start = bucket_start_[last_bucket];
        } while (last_start > start_of_last_block);
      }

      // Check if the last block has been written
      const auto write = shared_->bucket_pointers[last_bucket].getWrite();
      if (write < end)
        return {-1, 0};

      // Read excess elements, if necessary
      tail = bucket_start_[last_bucket + 1];
      local_.swap[0].readFrom(begin_ + tail, end - tail);

      return {last_bucket, end - tail};
      }

    template <bool kIsParallel>
    void writeMargins(int first_bucket, int last_bucket, int overflow_bucket,
                      int swap_bucket, diff_t in_swap_buffer)
      {
      const bool is_last_level = end_ - begin_ <= Cfg::kSingleLevelThreshold;
      const auto comp = classifier_->getComparator();

      for (int i = first_bucket; i < last_bucket; ++i)
        {
        // Get bucket information
        const auto bstart = bucket_start_[i];
        const auto bend = bucket_start_[i + 1];
        const auto bwrite = bucket_pointers_[i].getWrite();
        // Destination where elements can be written
        auto dst = begin_ + bstart;
        auto remaining = Cfg::alignToNextBlock(bstart) - bstart;

        if (i == overflow_bucket && overflow_)
          {
          // Is there overflow?

          // Overflow buffer has been written => write pointer must be at end of bucket
          IPS4OML_ASSUME_NOT(Cfg::alignToNextBlock(bend) != bwrite);

          auto src = overflow_->data();
          // There must be space for at least BlockSize elements
          IPS4OML_ASSUME_NOT((bend - (bwrite - Cfg::kBlockSize)) + remaining
                             < Cfg::kBlockSize);
          auto tail_size = Cfg::kBlockSize - remaining;

          // Fill head
          std::move(src, src + remaining, dst);
          src += remaining;
          remaining = std::numeric_limits<diff_t>::max();

          // Write remaining elements into tail
          dst = begin_ + (bwrite - Cfg::kBlockSize);
          dst = std::move(src, src + tail_size, dst);

          overflow_->reset(Cfg::kBlockSize);
          }
        else if (i == swap_bucket && in_swap_buffer)
          {
          // Did we save this in saveMargins?

          // Bucket of last block in this thread's area => write swap buffer
          auto src = local_.swap[0].data();
          // All elements from the buffer must fit
          IPS4OML_ASSUME_NOT(in_swap_buffer > remaining);

          // Write to head
          dst = std::move(src, src + in_swap_buffer, dst);
          remaining -= in_swap_buffer;

          local_.swap[0].reset(in_swap_buffer);
          }
        else if (bwrite > bend && bend - bstart > Cfg::kBlockSize)
          {
          // Final block has been written => move excess elements to head
          IPS4OML_ASSUME_NOT(Cfg::alignToNextBlock(bend) != bwrite);

          auto src = begin_ + bend;
          auto head_size = bwrite - bend;
          // Must fit, no other empty space left
          IPS4OML_ASSUME_NOT(head_size > remaining);

          // Write to head
          dst = std::move(src, src + head_size, dst);
          remaining -= head_size;
          }

        // Write elements from buffers
        for (int t = 0; t < num_threads_; ++t)
          {
          auto& buffers = kIsParallel ? shared_->local[t]->buffers : local_.buffers;
          auto src = buffers.data(i);
          auto count = buffers.size(i);

          if (count <= remaining)
            {
            dst = std::move(src, src + count, dst);
            remaining -= count;
            }
          else
            {
            std::move(src, src + remaining, dst);
            src += remaining;
            count -= remaining;
            remaining = std::numeric_limits<diff_t>::max();

            dst = begin_ + bwrite;
            dst = std::move(src, src + count, dst);
            }

          buffers.reset(i);
          }

        // Perform final base case sort here, while the data is still cached
        if (is_last_level
            || ((bend - bstart <= 2 * Cfg::kBaseCaseSize) && !kIsParallel))
            detail::baseCaseSort(begin_ + bstart, begin_ + bend, comp);
        }
      }


    template <bool kIsParallel>
    std::pair<int, bool> partition(iterator begin, iterator end, diff_t* bucket_start,
                                   int my_id, int num_threads)
      {
      // Sampling
      bool use_equal_buckets = false;
        {
        if (!kIsParallel)
          std::tie(this->num_buckets_, use_equal_buckets) =
                  buildClassifier(begin, end, local_.classifier);
        else
          {
          shared_->sync.single([&]
            {
            std::tie(this->num_buckets_, use_equal_buckets) =
                    buildClassifier(begin, end, shared_->classifier);
            shared_->num_buckets = this->num_buckets_;
            shared_->use_equal_buckets = use_equal_buckets;
            });
          this->num_buckets_ = shared_->num_buckets;
          use_equal_buckets = shared_->use_equal_buckets;
          }
        }

      // Set parameters for this partitioning step
      // Must do this AFTER sampling, because sampling will recurse to sort splitters.
      this->classifier_ = kIsParallel ? &shared_->classifier : &local_.classifier;
      this->bucket_start_ = bucket_start;
      this->bucket_pointers_ =
            kIsParallel ? shared_->bucket_pointers : local_.bucket_pointers;
      this->overflow_ = nullptr;
      this->begin_ = begin;
      this->end_ = end;
      this->my_id_ = my_id;
      this->num_threads_ = num_threads;

      // Local Classification
      kIsParallel ? parallelClassification(use_equal_buckets)
                  : sequentialClassification(use_equal_buckets);

      // Compute which bucket can cause overflow
      const int overflow_bucket = computeOverflowBucket();

      // Block Permutation
      use_equal_buckets ? permuteBlocks<true, kIsParallel>()
                        : permuteBlocks<false, kIsParallel>();

      if (kIsParallel && overflow_)
        shared_->overflow = &local_.overflow;

      if (kIsParallel) shared_->sync.barrier();

      // Cleanup
      {
      if (kIsParallel) overflow_ = shared_->overflow;

      // Distribute buckets among threads
      const int num_buckets = num_buckets_;
      const int buckets_per_thread = (num_buckets + num_threads_ - 1) / num_threads_;
      int my_first_bucket = my_id_ * buckets_per_thread;
      int my_last_bucket = (my_id_ + 1) * buckets_per_thread;
      my_first_bucket = num_buckets < my_first_bucket ? num_buckets : my_first_bucket;
      my_last_bucket = num_buckets < my_last_bucket ? num_buckets : my_last_bucket;

      // Save excess elements at right end of stripe
      const auto in_swap_buffer = !kIsParallel
                                          ? std::pair<int, diff_t>(-1, 0)
                                          : saveMargins(my_last_bucket);
      if (kIsParallel) shared_->sync.barrier();

      // Write remaining elements
      writeMargins<kIsParallel>(my_first_bucket, my_last_bucket, overflow_bucket,
                                in_swap_buffer.first, in_swap_buffer.second);
      }

      if (kIsParallel) shared_->sync.barrier();
      local_.reset();

      return {this->num_buckets_, use_equal_buckets};
      }

    void processSmallTasks(iterator begin)
      {
      auto& scheduler = shared_->scheduler;
      auto& my_queue = local_.seq_task_queue;
      Task task;
      auto comp = local_.classifier.getComparator();

      while (scheduler.getJob(my_queue, task))
        {
        scheduler.offerJob(my_queue);
        if (task.end - task.begin <= 2 * Cfg::kBaseCaseSize)
          detail::baseCaseSort(begin + task.begin, begin + task.end, comp);
        else
          sequential(begin, task, my_queue);
        }
      }

    void processBigTasks(const iterator begin, const diff_t stripe, const int id,
                         BufferStorage& buffer_storage,
                         std::vector<std::shared_ptr<SubThreadPool>>& tp_trash)
      {
      BigTask& task = shared_->big_tasks[id];

      while (task.has_task)
        if (task.root_thread == id)
          // Only thread 0 passes a task sorter (the one stored in this
          // object). The other threads have to create a task sorter if
          // required.
          processBigTaskPrimary(begin, stripe, id, buffer_storage, tp_trash);
        else
          processBigTasksSecondary(id);
      }

    void processBigTaskPrimary(const iterator begin, const diff_t stripe, const int id,
                               BufferStorage& buffer_storage,
                               std::vector<std::shared_ptr<SubThreadPool>>& tp_trash)
      {
      BigTask& task = shared_->big_tasks[id];

      // Thread pool of this task.
      auto partial_thread_pool = shared_->thread_pools[id];

      using Sorter =
              Sorter<ExtendedConfig<iterator, decltype(shared_->classifier.getComparator()),
                                    Config<>
#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))
                                  , SubThreadPool
#endif
                                   >>;

      // Create shared data.
      detail::AlignedPtr<typename Sorter::SharedData> partial_shared_ptr(
              Cfg::kDataAlignment, shared_->classifier.getComparator(),
              partial_thread_pool->sync(), partial_thread_pool->numThreads());
      auto& partial_shared = partial_shared_ptr.get();

      // Create local data.
      typename Sorter::BufferStorage partial_buffer_storage(
              partial_thread_pool->numThreads());
      std::unique_ptr<detail::AlignedPtr<typename Sorter::LocalData>[]> partial_local_ptrs(
              new detail::AlignedPtr<
                      typename Sorter::LocalData>[partial_thread_pool->numThreads()]);

      for (int i = 0; i != partial_thread_pool->numThreads(); ++i)
        {
        partial_local_ptrs[i] = detail::AlignedPtr<typename Sorter::LocalData>(
                Cfg::kDataAlignment, shared_->classifier.getComparator(),
                buffer_storage.forThread(task.root_thread + i));
        partial_shared.local[i] = &partial_local_ptrs[i].get();
        }

      std::pair<std::vector<diff_t>, bool> ret;

      // Execute in parallel
      partial_thread_pool->operator()(
            [&partial_shared, begin, &task, &ret](int partial_id,
                                                  int partial_num_threads)
        {
        Sorter sorter(*partial_shared.local[partial_id]);
        sorter.setShared(&partial_shared);
        if (partial_id == 0)
          {
          ret = sorter.parallelPartitionPrimary(
          begin + task.begin, begin + task.end, partial_num_threads);
          }
        else
          sorter.parallelPartitionSecondary(begin + task.begin,
                                            begin + task.end, partial_id,
                                            partial_num_threads);
        },
        partial_thread_pool->numThreads());

      const auto& offsets = ret.first;
      const auto equal_buckets = ret.second;
      const int num_buckets = offsets.size() - 1;

      // Move my thread pool to the trash as I might create a new one.
      tp_trash.emplace_back(std::move(shared_->thread_pools[id]));

      queueTasks(stripe, id, partial_thread_pool->numThreads(), task.end - task.begin,
                 task.begin, offsets.data(), num_buckets, equal_buckets);

      partial_thread_pool->release_threads();
      }

    void processBigTasksSecondary(const int id)
      {
      BigTask& task = shared_->big_tasks[id];
      auto partial_thread_pool = shared_->thread_pools[task.root_thread];
      partial_thread_pool->join(task.task_thread_id);
      }

    void queueTasks(const diff_t stripe, const int id, const int task_num_threads,
                    const diff_t parent_task_size, const diff_t offset,
                    const diff_t* bucket_start, int num_buckets, bool equal_buckets)
      {
      // create a new task sorter on subsequent levels

      const diff_t parent_task_stripe =
              (parent_task_size + task_num_threads - 1) / task_num_threads;

      const auto queueTask = [&](const diff_t task_begin, const diff_t task_end)
        {
        const int thread_begin = (offset + task_begin + stripe / 2) / stripe;
        const int thread_end = (offset + task_end + stripe / 2) / stripe;

        const auto task_size = task_end - task_begin;

        if (thread_end - thread_begin <= 1
            || task_end - task_begin <= Cfg::kBaseCaseSize)
          {
          const auto thread = (task_begin + task_size / 2) / parent_task_stripe;

          shared_->local[id + thread]->seq_task_queue.emplace(offset + task_begin,
                                                              offset + task_end);
          }
        else
          {
          shared_->thread_pools[thread_begin] =
                  std::make_shared<SubThreadPool>(thread_end - thread_begin);

          for (auto t = thread_begin; t != thread_end; ++t)
            {
            auto& bt = shared_->big_tasks[t];

            bt.begin = offset + task_begin;
            bt.end = offset + task_end;
            bt.task_thread_id = t - thread_begin;
            bt.root_thread = thread_begin;
            bt.has_task = true;
            }
          }
        };

      for (auto t = id; t != id + task_num_threads; ++t)
        shared_->big_tasks[t].has_task = false;

      // Queue subtasks if we didn't reach the last level yet
      const bool is_last_level = parent_task_size <= Cfg::kSingleLevelThreshold;
      if (!is_last_level)
        {
        if (equal_buckets)
          {
          const auto start = bucket_start[num_buckets - 1];
          const auto stop = bucket_start[num_buckets];
          if (start < stop) queueTask(start, stop);
          }

        // Skip equality buckets
        for (int i = num_buckets - 1 - equal_buckets; i >= 0; i -= 1 + equal_buckets)
          {
          const auto start = bucket_start[i];
          const auto stop = bucket_start[i + 1];
          if (start < stop) queueTask(start, stop);
          }
        }
      }
  };

/** Branch-free classifier. */
template <typename Cfg> class Sorter<Cfg>::Classifier
  {
    using iterator = typename Cfg::iterator;
    using value_type = typename Cfg::value_type;
    using bucket_type = typename Cfg::bucket_type;
    using less = typename Cfg::less;

  public:
    Classifier(less comp) : comp_(std::move(comp)) {}

    ~Classifier()
      { if (log_buckets_) cleanup(); }

    /** Calls destructors on splitter elements. */
    void reset()
      { if (log_buckets_) cleanup(); }

    /** The sorted array of splitters, to be filled externally. */
    value_type* getSortedSplitters()
      { return static_cast<value_type*>(static_cast<void*>(sorted_storage_)); }

    /** The comparison operator. */
    less getComparator() const { return comp_; }

    /** Builds the tree from the sorted splitters. */
    void build(int log_buckets)
      {
      log_buckets_ = log_buckets;
      num_buckets_ = 1 << log_buckets;
      const auto num_splitters = (1 << log_buckets) - 1;
      IPS4OML_ASSUME_NOT(getSortedSplitters() + num_splitters == nullptr);
      new (getSortedSplitters() + num_splitters)
        value_type(getSortedSplitters()[num_splitters - 1]);
      build(getSortedSplitters(), getSortedSplitters() + num_splitters, 1);
      }

    /** Classifies a single element. */
    template <bool kEqualBuckets> bucket_type classify(const value_type& value) const {
      const int log_buckets = log_buckets_;
      const bucket_type num_buckets = num_buckets_;
      IPS4OML_ASSUME_NOT(log_buckets < 1);
      IPS4OML_ASSUME_NOT(log_buckets > Cfg::kLogBuckets + 1);

      bucket_type b = 1;
      for (int l = 0; l < log_buckets; ++l)
        b = 2 * b + comp_(splitter(b), value);
      if (kEqualBuckets)
        b = 2 * b + !comp_(value, sortedSplitter(b - num_buckets));
      return b - (kEqualBuckets ? 2 * num_buckets : num_buckets);
      }

    /** Classifies all elements using a callback. */
    template <bool kEqualBuckets, typename Yield> 
    void classify(iterator begin, iterator end, Yield&& yield) const {
      classifySwitch<kEqualBuckets>(begin, end, std::forward<Yield>(yield),
	std::make_integer_sequence<int, Cfg::kLogBuckets + 1>{});
    }

    /** Classifies all elements using a callback. */
  template <bool kEqualBuckets, typename Yield, int...Args>
  void classifySwitch(iterator begin, iterator end, Yield&& yield,
		      std::integer_sequence<int, Args...>) const {
    IPS4OML_ASSUME_NOT(log_buckets_ <= 0 && log_buckets_ >= static_cast<int>(sizeof...(Args)));
    ((Args == log_buckets_ &&
      classifyUnrolled<kEqualBuckets, Args>(begin, end, std::forward<Yield>(yield)))
     || ...);
    }

    /** Classifies all elements using a callback. */
    template <bool kEqualBuckets, int kLogBuckets, typename Yield>
    bool classifyUnrolled(iterator begin, const iterator end, Yield&& yield) const {

        constexpr const bucket_type kNumBuckets = 1l << (kLogBuckets + kEqualBuckets);
        constexpr const int kUnroll = Cfg::kUnrollClassifier;
        IPS4OML_ASSUME_NOT(begin >= end);
        IPS4OML_ASSUME_NOT(begin > (end - kUnroll));

        bucket_type b[kUnroll];
        for (auto cutoff = end - kUnroll; begin <= cutoff; begin += kUnroll) {
            for (int i = 0; i < kUnroll; ++i)
                b[i] = 1;

            for (int l = 0; l < kLogBuckets; ++l)
                for (int i = 0; i < kUnroll; ++i)
                    b[i] = 2 * b[i] + comp_(splitter(b[i]), begin[i]);

            if (kEqualBuckets)
                for (int i = 0; i < kUnroll; ++i)
                    b[i] = 2 * b[i]
                           + !comp_(begin[i], sortedSplitter(b[i] - kNumBuckets / 2));

            for (int i = 0; i < kUnroll; ++i)
                yield(b[i] - kNumBuckets, begin + i);
        }

        IPS4OML_ASSUME_NOT(begin > end);
        for (; begin != end; ++begin) {
            bucket_type b = 1;
            for (int l = 0; l < kLogBuckets; ++l)
                b = 2 * b + comp_(splitter(b), *begin);
            if (kEqualBuckets)
                b = 2 * b + !comp_(*begin, sortedSplitter(b - kNumBuckets / 2));
            yield(b - kNumBuckets, begin);
        }
	return true;
    }

  private:
    const value_type& splitter(bucket_type i) const {
        return static_cast<const value_type*>(static_cast<const void*>(storage_))[i];
    }

    const value_type& sortedSplitter(bucket_type i) const {
        return static_cast<const value_type*>(
                static_cast<const void*>(sorted_storage_))[i];
    }

    value_type* data()
      { return static_cast<value_type*>(static_cast<void*>(storage_)); }

    /** Recursively builds the tree. */
    void build(const value_type* const left, const value_type* const right,
               const bucket_type pos) {
        const auto mid = left + (right - left) / 2;
        IPS4OML_ASSUME_NOT(data() + pos == nullptr);
        new (data() + pos) value_type(*mid);
        if (2 * pos < num_buckets_) {
            build(left, mid, 2 * pos);
            build(mid, right, 2 * pos + 1);
        }
    }

    /** Destructs splitters. */
    void cleanup() {
        auto p = data() + 1;
        auto q = getSortedSplitters();
        for (int i = num_buckets_ - 1; i; --i) {
            p++->~value_type();
            q++->~value_type();
        }
        q->~value_type();
        log_buckets_ = 0;
    }

    // Filled from 1 to num_buckets_
    std::aligned_storage_t<sizeof(value_type), alignof(value_type)>
            storage_[Cfg::kMaxBuckets / 2];
    // Filled from 0 to num_buckets_, last one is duplicated
    std::aligned_storage_t<sizeof(value_type), alignof(value_type)>
            sorted_storage_[Cfg::kMaxBuckets / 2];
    int log_buckets_ = 0;
    bucket_type num_buckets_ = 0;
    less comp_;
};

/** A single buffer block. */
template <typename Cfg> class Sorter<Cfg>::Block {
    using iterator = typename Cfg::iterator;
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;

  public:
    static constexpr const bool kInitializedStorage =
            std::is_trivially_default_constructible<value_type>::value;
    static constexpr const bool kDestruct =
            !kInitializedStorage && !std::is_trivially_destructible<value_type>::value;

    /** Pointer to data. */
    value_type* data() {
        return static_cast<value_type*>(static_cast<void*>(storage_));
    }

    /** First element. */
    const value_type& head() { return *data(); }

    /** Reads a full block from input. */
    void readFrom(iterator src) {
        if (kInitializedStorage) {
            std::move(src, src + Cfg::kBlockSize, data());
        } else {
            for (auto p = data(), end = p + Cfg::kBlockSize; p < end; ++p) {
                IPS4OML_ASSUME_NOT(p == nullptr);
                new (p) value_type(std::move(*src++));
            }
        }
    }

    /** Reads a partial block from input. */
    void readFrom(iterator src, const diff_t n)
      {
      if (kInitializedStorage)
        std::move(src, src + n, data());
      else
        for (auto p = data(), end = p + n; p < end; ++p)
          {
          IPS4OML_ASSUME_NOT(p == nullptr);
          new (p) value_type(std::move(*src++));
          }
      }

    /** Resets a partial block. */
    void reset(const diff_t n)
      {
      if (kDestruct)
        for (auto p = data(), end = p + n; p < end; ++p)
          p->~value_type();
      }

    /** Writes a full block to other block. */
    void writeTo(Block& block)
      {
      if (kInitializedStorage)
        std::move(data(), data() + Cfg::kBlockSize, block.data());
      else
        for (auto src = data(), dst = block.data(), end = src + Cfg::kBlockSize;
                 src < end; ++src, ++dst)
          {
          IPS4OML_ASSUME_NOT(dst == nullptr);
          new (dst) value_type(std::move(*src));
          }
      if (kDestruct)
        for (auto p = data(), end = p + Cfg::kBlockSize; p < end; ++p)
          p->~value_type();
      }

    /** Writes a full block to input. */
    void writeTo(iterator dest)
      {
      std::move(data(), data() + Cfg::kBlockSize, std::move(dest));
      if (kDestruct)
        for (auto p = data(), end = p + Cfg::kBlockSize; p < end; ++p)
          p->~value_type();
      }

  private:
    using storage_type = std::conditional_t<
            kInitializedStorage, value_type,
            std::aligned_storage_t<sizeof(value_type), alignof(value_type)>>;
    storage_type storage_[Cfg::kBlockSize];
  };

/** Per-thread buffers for each bucket. */
template <typename Cfg> class Sorter<Cfg>::Buffers {
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;

  public:
    Buffers(char* storage) : storage_(static_cast<Block*>(static_cast<void*>(storage))) {
        for (diff_t i = 0; i < Cfg::kMaxBuckets; ++i) {
            resetBuffer(i);
            buffer_[i].end = buffer_[i].ptr + Cfg::kBlockSize;
        }
    }

    /** Checks if buffer is full. */
    bool isFull(const int i) const
      { return buffer_[i].ptr == buffer_[i].end; }

    /** Pointer to buffer data. */
    value_type* data(const int i) {
        return static_cast<value_type*>(static_cast<void*>(storage_))
               + i * Cfg::kBlockSize;
    }

    /** Number of elements in buffer. */
    diff_t size(const int i) const
      { return Cfg::kBlockSize - (buffer_[i].end - buffer_[i].ptr); }

    /** Resets buffer. */
    void reset(const int i) {
        if (Block::kDestruct)
            for (auto p = data(i), end = p + size(i); p < end; ++p)
                p->~value_type();
        resetBuffer(i);
    }

    /** Pushes new element to buffer. */
    void push(const int i, value_type&& value)
      {
      if (Block::kInitializedStorage)
        *buffer_[i].ptr++ = std::move(value);
      else
        {
        IPS4OML_ASSUME_NOT(buffer_[i].ptr == nullptr);
        new (buffer_[i].ptr++) value_type(std::move(value));
        }
      }

    /** Flushes buffer to input. */
    void writeTo(const int i, typename Cfg::iterator dest) {
        resetBuffer(i);
        auto ptr = buffer_[i].ptr;
        std::move(ptr, ptr + Cfg::kBlockSize, std::move(dest));

        if (Block::kDestruct)
            for (const auto end = buffer_[i].end; ptr < end; ++ptr)
                ptr->~value_type();
    }

  private:
    struct Info {
        value_type* ptr;
        const value_type* end;
    };

    void resetBuffer(const int i) {
        buffer_[i].ptr = static_cast<value_type*>(static_cast<void*>(storage_))
                         + i * Cfg::kBlockSize;
    }

    Info buffer_[Cfg::kMaxBuckets];
    Block* storage_;
    // Blocks should have no extra elements or padding
    static_assert(sizeof(Block) == sizeof(typename Cfg::value_type) * Cfg::kBlockSize,
                  "Block size mismatch.");
    static_assert(std::is_trivially_default_constructible<Block>::value,
                  "Block must be trivially default constructible.");
    static_assert(std::is_trivially_destructible<Block>::value,
                  "Block must be trivially destructible.");
  };

template <typename Cfg> class Sorter<Cfg>::BucketPointers
  {
    using diff_t = typename Cfg::difference_type;

    class Uint128 {
     public:
        void set(diff_t l, diff_t m) {
            m_ = m;
            l_ = l;
        }

        diff_t getLeastSignificant() const { return l_; }

        template <bool kAtomic>
        std::pair<diff_t, diff_t> fetchSubMostSignificant(diff_t m) {
            if (kAtomic) {
                std::lock_guard<std::mutex> lock(mtx_);
                std::pair<diff_t, diff_t> p{l_, m_};
                m_ -= m;
                return p;
            } else {
                const auto tmp = m_;
                m_ -= m;
                return {l_, tmp};
            }
        }

        template <bool kAtomic>
        std::pair<diff_t, diff_t> fetchAddLeastSignificant(diff_t l) {
            if (kAtomic) {
                std::lock_guard<std::mutex> lock(mtx_);
                std::pair<diff_t, diff_t> p{l_, m_};
                l_ += l;
                return p;
            } else {
                const auto tmp = l_;
                l_ += l;
                return {tmp, m_};
            }
        }

      private:
        diff_t m_, l_;
        std::mutex mtx_;
    };

  public:
    /** Sets write/read pointers. */
    void set(diff_t w, diff_t r) {
        ptr_.set(w, r);
        num_reading_.store(0, std::memory_order_relaxed);
    }

    /** Gets the write pointer. */
    diff_t getWrite() const {
        return ptr_.getLeastSignificant();
    }

    /** Gets write/read pointers and increases the write pointer. */
    template <bool kAtomic>
    std::pair<diff_t, diff_t> incWrite() {
        return ptr_.template fetchAddLeastSignificant<kAtomic>(Cfg::kBlockSize);
    }

    /**
     * Gets write/read pointers, decreases the read pointer, and increases the read
     * counter.
     */
    template <bool kAtomic>
    std::pair<diff_t, diff_t> decRead() {
        if (kAtomic) {
            // Must not be moved after the following fetch_sub, as that could lead to
            // another thread writing to our block, because isReading() returns false.
            num_reading_.fetch_add(1, std::memory_order_acquire);
            const auto p =
                    ptr_.template fetchSubMostSignificant<kAtomic>(Cfg::kBlockSize);
            return {p.first, p.second & ~(Cfg::kBlockSize - 1)};
        } else {
            return ptr_.template fetchSubMostSignificant<kAtomic>(Cfg::kBlockSize);
        }
    }

    /** Decreases the read counter. */
    void stopRead() {
        // Synchronizes with threads wanting to write to this bucket
        num_reading_.fetch_sub(1, std::memory_order_release);
    }

    /** Returns true if any thread is currently reading from here. */
    bool isReading() {
        // Synchronizes with threads currently reading from this bucket
        return num_reading_.load(std::memory_order_acquire) != 0;
    }

  private:
    Uint128 ptr_;
    std::atomic_int num_reading_;
  };

/** Aligned storage for use in buffers. */
template <typename Cfg> class Sorter<Cfg>::BufferStorage : public AlignedPtr<void>
  {
  public:
    static constexpr const auto kPerThread =
            Cfg::kBlockSizeInBytes * Cfg::kMaxBuckets * (1 + Cfg::kAllowEqualBuckets);

    BufferStorage() {}

    explicit BufferStorage(int num_threads)
        : AlignedPtr<void>(Cfg::kDataAlignment, num_threads * kPerThread) {}

    char* forThread(int id) { return this->get() + id * kPerThread; }
  };

/** Data local to each thread. */
template <typename Cfg> struct Sorter<Cfg>::LocalData
  {
  using diff_t = typename Cfg::difference_type;
  // Buffers
  diff_t bucket_size[Cfg::kMaxBuckets];
  Buffers buffers;
  Block swap[2];
  Block overflow;

  PrivateQueue<Task> seq_task_queue;

  // Bucket information
  BucketPointers bucket_pointers[Cfg::kMaxBuckets];

  // Classifier
  Classifier classifier;

  // Information used during empty block movement
  diff_t first_block;
  diff_t first_empty_block;

  // Random bit generator for sampling
  // LCG using constants by Knuth (for 64 bit) or Numerical Recipes (for 32 bit)
  std::linear_congruential_engine<
          std::uintptr_t, Cfg::kIs64Bit ? 6364136223846793005u : 1664525u,
          Cfg::kIs64Bit ? 1442695040888963407u : 1013904223u, 0u>
          random_generator;

  LocalData(typename Cfg::less comp, char* buffer_storage)
    : buffers(buffer_storage), classifier(std::move(comp))
    {
    std::random_device rdev;
    std::ptrdiff_t seed = rdev();
    if (Cfg::kIs64Bit) seed = (seed << (Cfg::kIs64Bit * 32)) | rdev();
    random_generator.seed(seed);
    reset();
    }

  /** Resets local data after partitioning is done. */
  void reset()
    {
    classifier.reset();
    std::fill_n(bucket_size, Cfg::kMaxBuckets, 0);
    }
  };

/** Data shared between all threads. */
template <typename Cfg> struct Sorter<Cfg>::SharedData
  {
  // Bucket information
  typename Cfg::difference_type bucket_start[Cfg::kMaxBuckets + 1];
  BucketPointers bucket_pointers[Cfg::kMaxBuckets];
  Block* overflow;
  int num_buckets;
  bool use_equal_buckets;

  // Classifier for parallel partitioning
  Classifier classifier;

  // Synchronisation support
  typename Cfg::Sync sync;

  // Local thread data
  std::vector<LocalData*> local;

  // Thread pools for bigtasks. One entry for each thread.
  std::vector<std::shared_ptr<SubThreadPool>> thread_pools;

  // Bigtasks. One entry per thread.
  std::vector<BigTask> big_tasks;

  // Scheduler of small tasks.
  Scheduler<Task> scheduler;

  SharedData(typename Cfg::less comp, typename Cfg::Sync sync, int num_threads)
      : classifier(std::move(comp))
      , sync(std::forward<typename Cfg::Sync>(sync))
      , local(num_threads)
      , thread_pools(num_threads)
      , big_tasks(num_threads)
      , scheduler(num_threads)
    { reset(); }

  /** Resets shared data after partitioning is done. */
  void reset()
    {
    classifier.reset();
    std::fill_n(bucket_start, Cfg::kMaxBuckets + 1, 0);
    overflow = nullptr;
    scheduler.reset();
    }
  };

}  // namespace detail

/** Reusable sequential sorter. */
template <typename Cfg> class SequentialSorter
  {
  using Sorter = detail::Sorter<Cfg>;
  using iterator = typename Cfg::iterator;

  public:
    explicit SequentialSorter(bool check_sorted, typename Cfg::less comp)
        : check_sorted_(check_sorted)
        , buffer_storage_(1)
        , local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage_.get()) {}

    explicit SequentialSorter(bool check_sorted, typename Cfg::less comp,
                              char* buffer_storage)
        : check_sorted_(check_sorted)
        , local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage) {}

    void operator()(iterator begin, iterator end) {
        if (check_sorted_) {
            const bool sorted = detail::sortSimpleCases(
                    begin, end, local_ptr_.get().classifier.getComparator());
            if (sorted) return;
        }

        Sorter(local_ptr_.get()).sequential(std::move(begin), std::move(end));
    }

  private:
    const bool check_sorted_;
    typename Sorter::BufferStorage buffer_storage_;
    detail::AlignedPtr<typename Sorter::LocalData> local_ptr_;
  };

#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))

/** Reusable parallel sorter. */
template <typename Cfg>
class ParallelSorter {
    using Sorter = detail::Sorter<Cfg>;
    using iterator = typename Cfg::iterator;

 public:
    /** Construct the sorter. Thread pool may be passed by reference. */
    ParallelSorter(typename Cfg::less comp, typename Cfg::ThreadPool thread_pool,
                   bool check_sorted)
        : check_sorted_(check_sorted)
        , thread_pool_(std::forward<typename Cfg::ThreadPool>(thread_pool))
        , shared_ptr_(Cfg::kDataAlignment, std::move(comp), thread_pool_.sync(),
                      thread_pool_.numThreads())
        , buffer_storage_(thread_pool_.numThreads())
        , local_ptrs_(new detail::AlignedPtr<
                      typename Sorter::LocalData>[thread_pool_.numThreads()])
    {
        // Allocate local data and reuse memory of the previous recursion level
        thread_pool_([this](int my_id, int) {
            auto& shared = this->shared_ptr_.get();
            this->local_ptrs_[my_id] = detail::AlignedPtr<typename Sorter::LocalData>(
                    Cfg::kDataAlignment, shared.classifier.getComparator(),
                    buffer_storage_.forThread(my_id));
            shared.local[my_id] = &this->local_ptrs_[my_id].get();
        });
    }

    /** Sort in parallel. */
    void operator()(iterator begin, iterator end) {
        // Sort small input sequentially
        const int num_threads = Cfg::numThreadsFor(begin, end, thread_pool_.numThreads());
        if (num_threads < 2 || end - begin <= 2 * Cfg::kBaseCaseSize) {
            Sorter(local_ptrs_[0].get()).sequential(std::move(begin), std::move(end));
            return;
        }

        if (check_sorted_
            && detail::isSorted(begin, end,
                                local_ptrs_[0].get().classifier.getComparator(),
                                thread_pool_)) {
            return;
        }

        // Set up base data before switching to parallel mode
        // auto& shared = shared_ptr_.get();

        // Execute in parallel
        thread_pool_(
                [this, begin, end](int my_id, int num_threads) {
                    std::vector<std::shared_ptr<typename Sorter::SubThreadPool>> tp_trash;
                    auto& shared = this->shared_ptr_.get();
                    Sorter sorter(*shared.local[my_id]);
                    sorter.setShared(&shared);
                    if (my_id == 0)
                        sorter.parallelSortPrimary(begin, end, num_threads,
                                                   buffer_storage_, tp_trash);
                    else
                        sorter.parallelSortSecondary(begin, end, my_id, num_threads,
                                                     buffer_storage_, tp_trash);
                },
                num_threads);
    }

 private:
    const bool check_sorted_;
    typename Cfg::ThreadPool thread_pool_;
    detail::AlignedPtr<typename Sorter::SharedData> shared_ptr_;
    typename Sorter::BufferStorage buffer_storage_;
    std::unique_ptr<detail::AlignedPtr<typename Sorter::LocalData>[]> local_ptrs_;
};
#endif  // threading

/** Helper function for creating a reusable sequential sorter. */
template <typename It, typename Cfg = Config<>, typename Comp = std::less<>>
SequentialSorter<ExtendedConfig<It, Comp, Cfg>> make_sorter(Comp comp = Comp()) {
  return SequentialSorter<ExtendedConfig<It, Comp, Cfg>>{true, std::move(comp)};
}

/** Configurable interface. */
template <typename Cfg, typename It, typename Comp = std::less<>>
void sort(It begin, It end, Comp comp = Comp()) {

  if (detail::sortSimpleCases(begin, end, comp)) return;

  if ((end-begin) <= Cfg::kBaseCaseMultiplier*Cfg::kBaseCaseSize) {
    detail::baseCaseSort(std::move(begin), std::move(end), std::move(comp));
  } else {
    ips4o::SequentialSorter<ips4o::ExtendedConfig<It, Comp, Cfg>> sorter{
                false, std::move(comp)};
    sorter(std::move(begin), std::move(end));
  }
}

/** Standard interface. */
template <typename It, typename Comp>
void sort(It begin, It end, Comp comp) {
    ips4o::sort<Config<>>(std::move(begin), std::move(end), std::move(comp));
}

template <typename It>
void sort(It begin, It end) {
  ips4o::sort<Config<>>(std::move(begin), std::move(end), std::less<>());
}

#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))

namespace parallel {

/** Helper functions for creating a reusable parallel sorter. */
template <typename It, typename Cfg = Config<>, typename ThreadPool, typename Comp = std::less<>>
std::enable_if_t<std::is_class<std::remove_reference_t<ThreadPool>>::value,
                 ParallelSorter<ExtendedConfig<It, Comp, Cfg, ThreadPool>>>
make_sorter(ThreadPool&& thread_pool, Comp comp = Comp(), bool check_sorted = true) {
    return ParallelSorter<ExtendedConfig<It, Comp, Cfg, ThreadPool>>(
            std::move(comp), std::forward<ThreadPool>(thread_pool), check_sorted);
}

template <typename It, typename Cfg = Config<>, typename Comp = std::less<>>
ParallelSorter<ExtendedConfig<It, Comp, Cfg>> make_sorter(
        int num_threads = DefaultThreadPool::maxNumThreads(), Comp comp = Comp(),
        bool check_sorted = true) {
    return make_sorter<It, Cfg>(DefaultThreadPool(num_threads), std::move(comp),
                                check_sorted);
}

/** Configurable interface. */
template <typename Cfg = Config<>, typename It, typename Comp, typename ThreadPool>
std::enable_if_t<std::is_class<std::remove_reference_t<ThreadPool>>::value> sort(
        It begin, It end, Comp comp, ThreadPool&& thread_pool) {

    if (Cfg::numThreadsFor(begin, end, thread_pool.numThreads()) < 2) {
        ips4o::sort<Cfg>(std::move(begin), std::move(end), std::move(comp));
    } else if (!detail::isSorted(begin, end, comp, thread_pool)) {
        auto sorter = ips4o::parallel::make_sorter<It, Cfg>(
                std::forward<ThreadPool>(thread_pool), std::move(comp), false);
        sorter(std::move(begin), std::move(end));
    }

}

template <typename Cfg = Config<>, typename It, typename Comp>
void sort(It begin, It end, Comp comp, int num_threads) {
  num_threads = Cfg::numThreadsFor(begin, end, num_threads);
  (num_threads<2) ? ips4o::sort<Cfg>(std::move(begin), std::move(end), std::move(comp))
                  : ips4o::parallel::sort<Cfg>(begin, end, comp, DefaultThreadPool(num_threads));
}

/** Standard interface. */
template <typename It, typename Comp>
void sort(It begin, It end, Comp comp) {
  ips4o::parallel::sort<Config<>>(std::move(begin), std::move(end), std::move(comp),
                                  DefaultThreadPool::maxNumThreads());
}

template <typename It>
void sort(It begin, It end) {
  ips4o::parallel::sort(std::move(begin), std::move(end), std::less<>());
}

}  // namespace parallel
#endif  // threading
}  // namespace ips4o

template <typename It, typename Comp>
void duccsort(It begin, It end, Comp comp, size_t num_threads) {
#if (!defined(DUCC0_NO_LOWLEVEL_THREADING))
  ips4o::parallel::sort(std::move(begin), std::move(end), comp, int(num_threads));
#else
  ips4o::sort(std::move(begin), std::move(end), comp);
#endif
}

template <typename It>
void duccsort(It begin, It end, size_t nthreads) {
  duccsort(std::move(begin), std::move(end), std::less<>(), nthreads);
}

}

using detail_sort::duccsort;

}

#endif
