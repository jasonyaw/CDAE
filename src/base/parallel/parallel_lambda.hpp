#ifndef _LIBCF_PARALLEL_LAMBDA_HPP_
#define _LIBCF_PARALLEL_LAMBDA_HPP_

#include <vector>
#include <future>
#include <algorithm>
#include <functional>

#include <base/parallel.hpp>

namespace libcf { 

/** in_parallel using C++11 multi-thread programming
 *  
 *  \param fn is a lambda function which has two arguments as input.
 *  \return fn
 *
 *  Example:
 *  =======
 *  
 *  std::vector<size_t> vec(1e3);
 *  std::iota(vec.begin(), vec.end(), 0);
 *  
 *  size_t n_threads = libcf::num_hardware_threads();
 *  std::vector<size_t> multi_counter(n_threads, 0);
 *  libcf::in_parallel([&](size_t thread_idx, size_t n_threads){
 *    size_t begin = thread_idx * vec.size() / num_threads;
 *    size_t end = (thread_idx + 1) * vec.size() / num_threads;
 *    multi_counter[thread_idx] = std::accumulate(vec.begin() + begin,
 *                                vec.begin() + end, 0);
 *  });
 *
 *  size_t sum = std::accumulate(multi_counter.begin(), multi_counter.end(), 0);
 */

inline void in_parallel(const std::function<void (size_t, size_t)>& fn) {

  size_t num_threads = num_hardware_threads();
  //std::cout << "Num of threads : " << num_threads << std::endl;   
  
  //std::vector<std::future<void>> workers(num_threads - 1);
  std::vector<std::thread> workers(num_threads);
  size_t thread_id = 0;
  // set async workers
  for (auto& worker : workers) {
    worker = std::move(std::thread(std::cref(fn), thread_id, num_threads));
    thread_id++;
  }

  // deal with the last block by current thread 
  //fn(thread_id, num_threads);

  // wait all the async workers done
  for (auto& worker : workers) {
    //worker.get();
    worker.join();
  }
}


/** parallel_for using C++11 multi-thread programming
 *
 *  Example:
 *  =======
 *
 *  std::vector<size_t> vec(100, 1);
 *  libcf::parallel_for(0, 100,
 *            [&](size_t x) { vec[x] += 1; });
 */
inline void parallel_for(const size_t first, const size_t last, 
                  const std::function<void (size_t)>& fn) {

  size_t length = last - first;
  
  in_parallel([&](size_t thread_id, size_t num_threads){
    size_t begin = first + (thread_id * length) / num_threads;
    size_t end = first + ((thread_id + 1) * length) / num_threads;
    for (size_t idx = begin; idx < end; idx++) {
      fn(idx);
    }
  });
}

/** parallel_for_each using C++11 multi-thread programming
 *
 *  Example:
 *  =======
 *
 *  std::vector<size_t> vec(100, 1);
 *  libcf::parallel_for_each(vec.begin(), vec.end(),
 *            [&](size_t& x) { x += 1; });
 */
template<typename Iterator>
inline void parallel_for_each(const Iterator& first, const Iterator& last, 
                  const std::function<void (size_t&)>& fn) {
   
  size_t length= std::distance(first, last);
  
  in_parallel([&](size_t thread_id, size_t num_threads){
    Iterator begin = first + (thread_id * length) / num_threads;
    Iterator end = first + ((thread_id + 1) * length) / num_threads;
    std::for_each<Iterator>(begin, end, std::cref(fn));
  });
}


/** Parallel accumulate function using C++ 11 multi-thread programming
 *  
 *  Example:
 *  =======
 *  - Partial sums
 *    std::vector<size_t> partial_sums libcf::parallel_accumulate<size_t>(0, 
 *        vec.size(), 0, [&](size_t idx, size_t& ret){
 *          return ret += vec[idx];
 *        });
 *    size_t total_sum = std::accumulate(partial_sums.begin(),
 *                  partial_sums.end(), 0, std::plus<size_t>());
 *   - Dot product
 *    std::vector<size_t> partial_dotprods = libcf::parallel_accumulate(
 *          0, vec1.size(), 0, [&](size_t idx, size_t& ret) {
 *            ret += vec1[idx] * vec2[idx];
 *          });
 *    size_t total_dotprod = std::accumulate(partial_dotprods.begin(),
 *         partial_dotprods.end(), 0, std::plus<size_t>());
 */
template <typename T>
inline std::vector<T> parallel_accumulate(const size_t first,
                            const size_t last,
                            const T& init,
                            const std::function<void (size_t, T&)>& fn) {

  size_t n_threads = num_hardware_threads();
  size_t length = last - first;
  
  std::vector<T> results(n_threads);
   
  in_parallel([&](size_t thread_id, size_t num_threads){
    size_t begin = first + (thread_id * length) / num_threads;
    size_t end = first + ((thread_id + 1) * length) / num_threads;
    auto ret = init;
    for (size_t idx = begin; idx < end; idx++) {
      fn(idx, ret);
    }
    results[thread_id] = std::move(ret);
  });
  
  return std::move(results);
}

/** parallel_accumulate_and_reduce using C++ multi thread programming
 *  
 *  size_t total_counts = parallel_accumulate_and_reduce<size_t>(
 *      0, vec.size(), 0, [&](size_t& ret, size_t idx) { 
 *              ret += vec[idx];
 *      }, 0, [](size_t& a, size_t b) { a += b;});
 */
template <typename T>
inline T parallel_accumulate_and_reduce(const size_t first,
                            const size_t last,
                            const T& init,
                            const std::function<void (T&, size_t)>& fn,
                            const T& reduce_init,
                            const std::function<void (T&, T)>& rfn) {

  size_t n_threads = num_hardware_threads();
  size_t length = last - first;
  
  std::vector<T> results(n_threads);
   
  in_parallel([&](size_t thread_id, size_t num_threads){
    size_t begin = first + (thread_id * length) / num_threads;
    size_t end = first + ((thread_id + 1) * length) / num_threads;
    auto ret = init;
    for (size_t idx = begin; idx < end; idx++) {
      fn(ret, idx);
    }
    results[thread_id] = std::move(ret);
  });
  
  T ret_val(reduce_init);
  
  for (auto& v : results) {
    rfn(ret_val, v);
  }

  return std::move(ret_val);
}

void dynamic_parallel_for(const size_t first, const size_t last, 
                  const std::function<void (size_t)>& fn) {
   
  //ThreadPool tp(num_hardware_threads());
  ThreadPool tp;
  for (size_t idx = first; idx < last; idx++) {
    tp.add([&, idx]() {
            fn(idx);
           });
  }
  tp.run();
}

template<typename Iterator>
void dynamic_parallel_for_each(const Iterator& first, const Iterator& last, 
                  const std::function<void (size_t&)>& fn) {
   
  //ThreadPool tp(num_hardware_threads());
  ThreadPool tp;
  for (auto iter = first; iter != last; iter++) {
    tp.add([&, iter]() { fn(*iter);});
  }
  tp.run();
}

} // namespace 
#endif // _LIBCF_PARALLEL_LAMBDA_HPP_
