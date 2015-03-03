#include <base/parallel/thread_pool.hpp> 

namespace libcf {

ThreadPool::ThreadPool(size_t num_workers) 
    : num_workers_(num_workers), has_started_(false)
{
  // define job for each thread
  auto worker_job = [&] () {
    // keep running, until some conditions changed 
    for (; ;) { 
      std::unique_lock<std::mutex> lock(mut_);
      // if we are not done and the queue is empty, 
      // just wait until condition changes
      while(!has_started_ && tasks_.empty()) {
        cond_.wait(lock);
      }

      // ok, we are done
      if(has_started_ && tasks_.empty()) {
        return;
      }
      
      // get the task at the front of the task list, and run it
      task_t task = std::move(tasks_.front());
      tasks_.pop_front();
      lock.unlock();
      task();
    }
  };

  workers_.reserve(num_workers_);
  for (size_t idx = 0; idx < num_workers_; idx++) {
    workers_.push_back(std::thread(worker_job));
  }
}

void ThreadPool::add(const task_t& t) {
  std::unique_lock<std::mutex> lock(mut_);
  tasks_.push_back(t);
  if(has_started_) cond_.notify_all();
}

void ThreadPool::run() {
  { // notify all threads that we has started 
    std::unique_lock<std::mutex> lock(mut_);
    has_started_ = true;
    cond_.notify_all();
  }
  for (auto& worker : workers_) 
    worker.join();
}

ThreadPool::~ThreadPool() {
  if (!has_started_) {
    run();
  }
}

} // namespace

