#ifndef _LIBCF_LINEAR_MODEL_HPP_
#define _LIBCF_LINEAR_MODEL_HPP_

#include <base/mat.hpp>
#include <base/data.hpp>
#include <model/loss.hpp>
#include <model/penalty.hpp>
#include <model/model_base.hpp>

namespace libcf {

class LinearModelConfig {
 public:
  LinearModelConfig() = default;

  double lambda = 0.001;  // regularization coefficient 
  LossType lt = SQUARE; // loss type
  PenaltyType pt = L2;  // penalty type
  bool using_global_mean = true;
  bool using_adagrad = true;
};

class LinearModel : public ModelBase, public SGDBase{
 public:
  LinearModel(const LinearModelConfig& mcfg) 
      : ModelBase(), 
      lambda_(mcfg.lambda), using_global_mean_(mcfg.using_global_mean),
      using_adagrad_(mcfg.using_adagrad)
  {
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);
    LOG(INFO) << "Linear Model Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Using Global Mean: " << using_global_mean_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}";
  }
  
  LinearModel() : LinearModel(LinearModelConfig()) {}

  // initialize the model
  void reset(const Data& data_set);

  virtual double data_loss(const Data& data_set, size_t sample_size=0) const {
    double total_loss = 0.0;
    if (data_set.size() == 0) 
      return total_loss;
    if (loss_ == nullptr) 
      return total_loss;
    if (sample_size == 0) sample_size = data_set.size();
    size_t idx = 0;
    for (auto iter = data_set.begin(); 
         idx < sample_size && iter != data_set.end(); ++idx, ++iter) {
      total_loss += loss_->evaluate(predict(*iter), iter->label());
    }
    //total_loss /= data_set.size();
    return total_loss;
  }


  double penalty_loss() const {
    return 0.5 * lambda_ * penalty_->evaluate(coefficients_);
  }

  void update_one_sgd_step(const Instance& ins, double step_size); 

  double predict(const Instance& ins) const;

  double regularization_coefficent() const {
    return lambda_;
  }

 private:
  DVector coefficients_;
  DVector gradient_square_; // for AdaGrad
  double lambda_ = 0.;
  double global_mean_ = 0.;
  bool using_global_mean_ = true; 
  bool using_adagrad_ = false; // using_AdaGrad
};

} // namespace

#include <model/linear_model-inl.hpp>

#endif // _LIBCF_LINEAR_MODEL_HPP_
