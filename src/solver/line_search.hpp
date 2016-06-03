#ifndef _LIBCF_LINE_SEARCH_HPP_
#define _LIBCF_LINE_SEARCH_HPP_

#include <functional>

#include <base/mat.hpp>

namespace libcf {

// line search to get step size
template <typename T = DVector>
inline double line_search(const T& t, 
                          const std::function<double (const T&)>& f_func, // loss function
                          const T& grad,  // gradient  
                          double alpha = 0.3,
                          double beta = 0.3,
                          size_t max_iters = 10,
                          double max_value = 100.) {

  double step_size = 1.;
  double init_error = f_func(t);

  for (size_t iter = 0; iter < max_iters; ++iter) {

    T new_val = t - step_size * grad;
  
    // clip values
    new_val = new_val.cwiseMin(std::abs(max_value)).cwiseMax(-std::abs(max_value)); 

    if (f_func(new_val) > init_error - alpha * step_size * grad.dot(grad)) {
      step_size *= beta; 
    } else {
      break;
    }

    if (iter == max_iters - 1) {
      step_size = 0.;
      break;
    }
  }
  return step_size;
}


} // namespace 
#endif // _LIBCF_LINE_SEARCH_HPP_
