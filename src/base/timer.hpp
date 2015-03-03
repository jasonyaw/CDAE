#ifndef _LIBCF_TIMER_HPP_
#define _LIBCF_TIMER_HPP_

#include <chrono>
#include <ostream>

namespace libcf {

class Timer {

 public:

  typedef std::chrono::high_resolution_clock high_resolution_clock;
  typedef std::chrono::milliseconds milliseconds;

  Timer() {
    reset();
  }

  void start() { reset(); }

  void reset() { _start = high_resolution_clock::now(); }

  double elapsed() const {
    auto eslaped_msecs = std::chrono::duration_cast<milliseconds>(
        high_resolution_clock::now() - _start);
    return static_cast<double>(eslaped_msecs.count()) / 1000.;
  }

  friend std::ostream& operator << (std::ostream& out, const Timer& t) {
    return out << t.elapsed() << " secs";
  }

 private:

  high_resolution_clock::time_point _start;

};

} // namespace

#endif // _LIBCF_TIMER_HPP_
