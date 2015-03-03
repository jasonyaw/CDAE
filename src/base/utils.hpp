#ifndef _LIBCF_UTILS_HPP_
#define _LIBCF_UTILS_HPP_

#include <iostream>
#include <iterator>
#include <algorithm>
#include <cassert>

#include <glog/logging.h>

#include <base/timer.hpp>

namespace libcf {
  
template<typename K, typename V>
inline bool sort_by_second_desc(const std::pair<K,V>& a,
             const std::pair<K,V>& b) {
  return a.second > b.second;
}
 
template<typename K, typename V>
inline bool sort_by_second_asc(const std::pair<K,V>& a,
             const std::pair<K,V>& b) {
  return a.second < b.second;
}

template<class First, class Second>
std::ostream& operator<<(std::ostream& out,
                         const std::pair<First, Second>& p) {
  out << '(' << p.first << "," << p.second << ')';
  return out;
}

template<class T>
std::ostream& operator<<(std::ostream& out,
                         const std::vector<T>& vec) {
  out << "[";
  size_t idx = 0;
  size_t max_out = 10;
  for (; idx < std::min(max_out, vec.size()); ++idx) {
    if (idx > 0) out << ",";
    out << vec[idx];
  }
  if (idx < vec.size()) out << ",...";
  if (vec.size() > 2 * max_out) {
    for (idx = vec.size() - max_out; idx < vec.size(); ++idx) {
      out << "," << vec[idx];
    }
  }
  out << "]";
  return out;
}



/** Print a range of iterators to output
 *  
 *  Example:
 *  
 *  std::vector<size_t> vec(100, 0);
 *  print_range(vec.begin(), vec.end(), std::cout);
 */
template <typename Iterator>
inline void print_range(const Iterator& first, 
                        const Iterator& last, 
                        std::ostream& s, 
                        const std::string& delimiter = ", ", 
                        const std::string& name = "") {
  typedef decltype(*first) T;
  std::ostream_iterator<T> out_iter(s, delimiter.c_str());
  if (name.length() > 0)  s << name << " : ";
  s << "[";
  std::copy(first, last, out_iter);
  s << "]" << std::endl;
}

/**
 *  Time how much a (lambda) function costs
 *
 *  Example:
 *  auto f = [&]() { std::cout << "hello world" << std::endl; }
 *  time_function(f, "f");
 *  
 */
inline void time_function(const std::function<void ()>& fn, const std::string& msg = "") {
  Timer t;
  t.start();
  fn();
  LOG(INFO) << "Message : (" << msg << ")" << std::endl;
  LOG(INFO) << "Function costs " << t << std::endl; 
}


}

#endif // _LIBCF_UTILS_HPP_
