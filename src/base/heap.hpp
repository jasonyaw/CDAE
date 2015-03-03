#ifndef _LIBCF_HEAP_HPP_
#define _LIBCF_HEAP_HPP_ 

#include <vector>
#include <functional>

namespace libcf {

/**
 *  An implementation of Heap. 
 */
template <class T>
class Heap {
 public:
  typedef typename std::function<bool (const T&, const T&)> func_type;
  
  Heap(const func_type& comp, size_t reserve_size = 0) : 
      comp_(comp) { 
    data_.reserve(reserve_size);
  }
  
  template <class Iterator>
  Heap(Iterator begin, Iterator end, const func_type& comp) :
    comp_(comp) {
      data_.assign(begin, end);
      std::make_heap(data_.begin(), data_.end(), comp_);
    }

  void push(const T& t) {
    data_.emplace_back(t);
    std::push_heap(data_.begin(), data_.end(), comp_);
  }

  T pop() {
    if (size() == 0) {
      LOG(FATAL) << "Heap size is 0! Can not pop any more!";
    }
    std::pop_heap(data_.begin(), data_.end(), comp_);
    T ret = std::move(data_.back());
    data_.pop_back();
    return std::move(ret);
  }

  T push_and_pop(const T& t) {
    if (comp_(t, data_.front())) {
      T ret = pop();
      push(t);
      return std::move(ret);
    } else {
      return t;
    }
  }
  
  T& front() {
    return data_.front();
  }

  void sort() {
    std::sort_heap(data_.begin(), data_.end(), comp_);
  }

  std::vector<T> get_data() {
    return std::move(data_);
  }
 
  std::vector<T> get_sorted_data() {
    sort();
    return std::move(data_);
  }

  std::vector<T> get_data_copy() const {
    return data_;
  }
 
  std::vector<T> get_sorted_data_copy() const {
    auto ret = data_;
    std::sort_heap(ret.begin(), ret.end(), comp_);
    return std::move(ret);
  }
 
  size_t size() const {
    return data_.size();
  }

 private:
  std::vector<T> data_;
  func_type comp_;
};

} // namespace

#endif // _LIBCF_HEAP_HPP_
