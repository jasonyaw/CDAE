#ifndef _LIBCF_PARALLEL_HPP_
#define _LIBCF_PARALLEL_HPP_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

#include <gflags/gflags.h>

DEFINE_int32(num_thread, 1, "NUM OF THREADS");

namespace libcf {

//////////////////////////////////////////////
// some global functions

inline size_t num_hardware_threads() {
  if (FLAGS_num_thread) {
    return FLAGS_num_thread;
  }
  return std::thread::hardware_concurrency();
}

}

#include <base/parallel/thread_pool.hpp>
#include <base/parallel/parallel_lambda.hpp>

#endif 
