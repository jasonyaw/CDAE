#ifndef _LIBCF_RANDOM_HPP_
#define _LIBCF_RANDOM_HPP_

#include <random>
#include <initializer_list>
#include <time.h>  

namespace libcf {

/**
 *  Random number generator
 */
class Random {
 public:
  typedef std::mt19937_64 rng_type;
  
  static inline void seed() {
    std::random_device rd;
    rng.seed(rd());
  }
   
  /* set seed */
  static inline void timed_seed()  {
    rng.seed(time(NULL));
  }

 
  /* set seed */
  static inline void seed(size_t number)  {
    rng.seed(number);
  }

  /* Generate a random number in [min, max) */
  static inline double uniform(double min = 0., double max = 1.) {
    std::uniform_real_distribution<> dist(min, max);
    return dist(rng);
  }
  
  /* Generate a random number from N(mean, stddev) */
  static inline double normal(double mean = 0., double stddev = 1.) {
    std::normal_distribution<> dist(mean, stddev);
    return dist(rng);
  }

  /* Generate a random size_t from [begin, end) */
  static inline size_t uniform(size_t begin, size_t end) {
    CHECK((end - begin) >= 1);
    std::uniform_int_distribution<> dist(static_cast<int>(begin), static_cast<int>(end-1)); 
    return static_cast<size_t>(dist(rng));
  }
  
  /* Randomly shuffle a container */
  template<typename Iter>
  static inline void shuffle(Iter begin, Iter end) {
    return std::shuffle(begin, end, rng);
  }
  
  /* Wrapper of discrete distribution */
  template<typename T>
  class discrete_distribution {
   public:
    template<typename Iter>
    discrete_distribution(Iter begin, Iter end) : distribution(begin, end) {}
    
    discrete_distribution(std::initializer_list<double> l) : distribution(l) {}

    T sample() {
      return distribution(rng);
    }

    private:
     std::discrete_distribution<T> distribution;
  };
  

 public:
  // random number generator
  static rng_type rng;
};

// set static member
Random::rng_type Random::rng;

} // namespace

#endif // _LIBCF_RANDOM_HPP_
