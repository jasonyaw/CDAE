#ifndef _LIBCF_THREADSAFE_QUEUE_HPP_
#define _LIBCF_THREADSAFE_QUEUE_HPP_

#include <queue>

#include <base/parallel.hpp>

namespace libcf {

template <class T>
class ThreadsafeQueue {
 public:

  ThreadsafeQueue() {
  }

  void push(T v) {
    {
      std::unique_lock<std::mutex> lock(m);
      q.push(v);
    }
    cv.notify_all();
  }

  T wait_and_pop() {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this]{ return !q.empty(); });
    T ret = std::move(p.front());
    p.pop();
  }

 private:
  std::mutex m;
  std::condition_variable cv;
  std::queue<T> q;
};

} // namespace
#endif // _LIBCF_THREADSAFE_QUEUE_HPP_
