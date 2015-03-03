#ifndef _LIBCF_RANDOM_HPP_
#define _LIBCF_RANDOM_HPP_

#include <cmath>
#include <cstdlib>

namespace libcf {

inline void seed(size_t seed) {
  srand(seed);
}

inline double uniform() {
  return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.);
}

inline double uniform(double min, double max) {
  return min + (max - min) * uniform();
}

inline size_t uniform(size_t min, size_t max) {
  return min + static_cast<size_t>((max - min) * uniform());
} 

} // namespace


#endif // _LIBCF_RANDOM_HPP_
