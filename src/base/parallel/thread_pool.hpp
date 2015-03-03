#ifndef _LIBCF_THREAD_POOL_HPP_
#define _LIBCF_THREAD_POOL_HPP_

#include <vector>
#include <list>
#include <functional>

#include <base/parallel.hpp>

namespace libcf {

/**
 *  Thread pool for dynamicly scheduling tasks for multiple threads
 *
 *  Example:
 *
 *    libcf::ThreadPool pl(libcf::num_hardware_threads());
 *    for (size_t idx = 0; idx < 100; idx++) {
 *      pl.add([&, idx]()  { 
 *          // do something with idx
 *        });
 *    }
 *    pl.run();
 *   
 */
class ThreadPool {
 public:
  // task type 
  typedef std::function<void ()> task_t;

 public:
  
  ThreadPool() : ThreadPool(num_hardware_threads()) {} 
  explicit ThreadPool(size_t num_workers); 

  ~ThreadPool();

  // add a task to the pool
  void add(const task_t& t);

  // run all the tasks in the pool
  void run();

 private:
  
  std::mutex mut_;        /*!< locker */
  std::condition_variable cond_; /*!< conditional variable for locker*/
  size_t num_workers_; /*!< num of parallel workers */
  std::vector<std::thread> workers_; /*!< workers */
  std::list<task_t> tasks_;  /*!< task list */
  bool has_started_;
};


} // namespace

#include <base/parallel/thread_pool-inl.hpp>

#endif // _LIBCF_THREAD_POOL_HPP_
