#ifndef _LIBCF_LOSS_HPP_
#define _LIBCF_LOSS_HPP_

#include <cmath>
#include <memory>
#include <algorithm>

namespace libcf {

enum LossType {
  SQUARE = 0,
  LOGISTIC,
  LOG,
  HINGE,
  SQUARED_HINGE,
  CROSS_ENTROPY,
  LOGM // GCR
};


class Loss {

 public:
  static std::shared_ptr<Loss> create(const LossType& lt);

  virtual LossType loss() const = 0;
  virtual std::string loss_type() const = 0;
  virtual double evaluate(double pred, double truth) const = 0;
  virtual double gradient(double pred, double truth) const = 0;
  virtual double predict(double x) const = 0;
  virtual double positive_label() const = 0;
  virtual double negative_label() const = 0;
};

/**
 *  square loss: l(a, y) = (y - a)^2
 */
class SquareLoss : public Loss {
  
  LossType loss() const {
    return SQUARE;
  }

  std::string loss_type() const {
    return std::string("Square");
  }

  double evaluate(double pred, double truth) const {
    double err = truth - pred;
    return err * err;
  }

  double gradient(double pred, double truth) const {
    return - 2. * (truth - pred);
  }

  double predict(double x) const {
    return x;
  }

  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return 0.;
  }
};

/**
 *  logistic_loss : 
 *    l(p,y) = - y log(p) - (1 - y) log(1 - p)
 */
class LogisticLoss : public Loss {
   
  LossType loss() const {
    return LOGISTIC;
  }

  std::string loss_type() const {
    return std::string("Logistic");
  }

  double evaluate(double pred, double truth) const {
    CHECK(pred >= 0. && pred <= 1.);
    CHECK(truth == 1. || truth == 0.);
    if (truth == 0.) {
      return - std::log(std::max(0.0001, 1. - pred));
    } else if (truth == 1.) {
      return - std::log(std::max(0.0001, pred));
    } 
    return 0.;
  }

  double gradient(double pred, double truth) const {
    CHECK(pred > 0. && pred < 1.);
    CHECK(truth == 1. || truth == 0.);
    return (pred - truth) / (pred * (1. - pred));
  }

  double predict(double x) const {
    return x;
  }

  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return 0.;
  }
};

/**
 *  CE : 
 *    p = 1 / (1 + exp(-a))
 *    l(a,y) = - y log(p) - (1 - y) log(1 - p)
 *           = (1 - y) * a + log(1 + exp(-a))
 *    d/da l(a,y) = (1 - y) - 1/ (1 + exp(a))
 */
class CrossEntropyLoss : public Loss {
   
  LossType loss() const {
    return CROSS_ENTROPY;
  }

  std::string loss_type() const {
    return std::string("CrossEntropy");
  }

  // TODO: check the numeric stability
  double evaluate(double pred, double truth) const {
    double ret = (1 - truth) * pred;
    if (pred > 18) 
      return ret + exp(-pred);
    if (pred < -18)
      return ret - pred;
    return ret + std::log1p(std::exp(-pred)); 
  }

  double gradient(double pred, double truth) const {
    if (pred < -18) 
      return std::exp(pred) - truth;
    if (pred > 18) 
      return 1 - truth; 
    return 1. / (1. + std::exp(-pred)) - truth ;
  }

  double predict(double x) const {
    return 1. / (1. + std::exp(-x));
  }
  
  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return 0.;
  }
};


/**
 *  log_loss : 
 *    l(a,y) =  log(1 + exp(-a * y))
 *
 *    dl/da = - y / (1 + exp(a*y)) 
 */
class LogLoss : public Loss {
  
  LossType loss() const {
    return LOG;
  }

  std::string loss_type() const {
    return std::string("Log");
  }

  // todo: check the numeric stability
  double evaluate(double pred, double truth) const {
    double z = pred * truth;
    if (z > 18)
      return std::exp(-z);
    if (z < -18)
      return -z;
    return std::log1p(std::exp(-z));
  }

  double gradient(double pred, double truth) const {
    double z = pred * truth;
    if (z > 18)
      return - truth * std::exp(-z);
    if (z < -18)
      return - truth;
    return -truth / (1. + std::exp(z));
    //double expnz = std::exp(-z);
    //return - truth * expnz / (1. + expnz);
  }

  double predict(double x) const {
    return x;
  }

  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return -1.;
  }
};

/**
 *  multiplicative log_loss : 
 *    l(a,y) =  y * log(1 + exp(-a))
 *
 *    dl/da = - y / (1 + exp(a)) 
 */
class LogMLoss : public Loss {
    
  LossType loss() const {
    return LOGM;
  }
  
  std::string loss_type() const {
    return std::string("LogM");
  }

  // todo: check the numeric stability
  double evaluate(double pred, double truth) const {
    double z = pred;
    if (z > 18)
      return truth * std::exp(-z);
    if (z < -18)
      return -z * truth;
    return truth * std::log1p(std::exp(-pred));
  }

  double gradient(double pred, double truth) const {
    double z = pred;
    if (z > 18)
      return - truth * std::exp(-z);
    if (z < -18)
      return - truth;
    return -truth / (1. + std::exp(z));
  }

  double predict(double x) const {
    return x;
  }
  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return -1.;
  }
};

/**
 *  hinge_loss : 
 *    l(a,y) = max(0, 1 - a*y)
 *
 *    dl/da = 
 *
 */
class HingeLoss : public Loss {  
  
  LossType loss() const {
    return HINGE;
  }

  std::string loss_type() const {
    return std::string("Hinge");
  }
  // todo: check the numeric stability
  double evaluate(double pred, double truth) const {
    double z = pred * truth;
    if (z > 1)
      return 0;
    return 1 - z;
  }

  double gradient(double pred, double truth) const {
    double z = pred * truth;
    if (z > 1)
      return 0;
    return -truth;
  }

  double predict(double x) const {
    return x;
  }

  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return -1.;
  }
};


/**
 *  squared_hinge_loss : 
 *    l(a,y) = 1/2 * max(0, 1 - a*y)^2
 *
 */
class SquaredHingeLoss : public Loss {
 
  LossType loss() const {
    return SQUARED_HINGE;
  }

  std::string loss_type() const {
    return std::string("SquaredHinge");
  }
  // todo: check the numeric stability
  double evaluate(double pred, double truth) const {
    double z = pred * truth;
    if (z > 1)
      return 0;
    double d = 1 - z;
    return 0.5 * d * d;
  }

  double gradient(double pred, double truth) const {
    double z = pred * truth;
    if (z > 1)
      return 0;
    return -truth * (1 - z);
  }

  double predict(double x) const {
    return x;
  }

  double positive_label() const { 
    return 1.;
  }

  double negative_label() const {
    return -1.;
  }
};

std::shared_ptr<Loss> Loss::create(const LossType& lt) {
  switch (lt) {
    case SQUARE:
      return std::shared_ptr<Loss>(new SquareLoss());
    case LOGISTIC:
      return std::shared_ptr<Loss>(new LogisticLoss());
    case CROSS_ENTROPY:
      return std::shared_ptr<Loss>(new CrossEntropyLoss());
    case LOG:
      return std::shared_ptr<Loss>(new LogLoss());
    case LOGM:
      return std::shared_ptr<Loss>(new LogMLoss());
    case HINGE:
      return std::shared_ptr<Loss>(new HingeLoss());
    case SQUARED_HINGE:
      return std::shared_ptr<Loss>(new SquaredHingeLoss());
    default:
      return std::shared_ptr<Loss>(new SquareLoss());
  }
}

} // namespace

#endif // _LIBCF_LOSS_HPP_
