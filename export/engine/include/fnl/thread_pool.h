// Fixed set of worker threads for fork-join loops: For(n, fn) runs fn(0..n-1) on the caller and the workers.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fnl {

class ThreadPool {
 public:
  // `threads` counts the caller: 1 runs everything inline, n starts n-1 workers.
  explicit ThreadPool(int threads) {
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { Work(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
  }

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // One caller at a time. Indices are handed out dynamically, so uneven items balance.
  void For(int n, const std::function<void(int)>& fn) {
    if (n <= 1 || workers_.empty()) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> l(m_);
      fn_ = &fn, n_ = n, next_ = 0, pending_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    wake_.notify_all();
    for (int i; (i = next_.fetch_add(1)) < n;) fn(i);
    std::unique_lock<std::mutex> l(m_);
    done_.wait(l, [this] { return pending_ == 0; });
  }

 private:
  void Work() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(m_);
        wake_.wait(l, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      for (int i; (i = next_.fetch_add(1)) < n_;) (*fn_)(i);
      std::lock_guard<std::mutex> l(m_);
      if (--pending_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* fn_ = nullptr;
  int n_ = 0, pending_ = 0;
  std::atomic<int> next_{0};
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace fnl
