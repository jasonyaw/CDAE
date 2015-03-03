#ifndef _LIBCF_PARALLEL_HPP_
#define _LIBCF_PARALLEL_HPP_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

namespace libcf {

//////////////////////////////////////////////
// some global functions

inline size_t num_hardware_threads() {
  return std::thread::hardware_concurrency();
}

}

#include <base/parallel/thread_pool.hpp>
#include <base/parallel/parallel_lambda.hpp>

#endif 
