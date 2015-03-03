#ifndef _LIBCF_PENALTY_HPP_
#define _LIBCF_PENALTY_HPP_

#include <cmath>
#include <memory>

#include <base/mat.hpp>

namespace libcf {

enum PenaltyType {
  L1 = 0,
  L2
};

class Penalty {

 public:
  static std::shared_ptr<Penalty> create(const PenaltyType& pt); 

  virtual std::string penalty_type() const = 0;
  virtual bool is_smooth() const = 0;
  virtual double evaluate(const DMatrix& mat) = 0;
};

class L2Penalty : public Penalty {
  
  std::string penalty_type() const {
    return "L2";
  }
  
  bool is_smooth() const {
    return true;
  }

  double evaluate(const DMatrix& mat) {
    if (mat.size() == 0) return 0.;
    return mat.squaredNorm();// / mat.size(); 
  }
  
};

class L1Penalty : public Penalty {

  std::string penalty_type() const {
    return "L1";
  }
  
  bool is_smooth() const {
    return false;
  }

  double evaluate(const DMatrix& mat) {
    if (mat.size() == 0) return 0.;
    return mat.lpNorm<1>();// / mat.size();
  }
};

std::shared_ptr<Penalty> Penalty::create(const PenaltyType& pt) {
  switch (pt) {
    case L1:
      return std::shared_ptr<Penalty>(new L1Penalty());
    case L2:
    default:
      return std::shared_ptr<Penalty>(new L2Penalty());
  }
}

}


#endif // _LIBCF_PENALTY_HPP_
