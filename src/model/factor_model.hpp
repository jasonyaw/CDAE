#ifndef _LIBCF_FACTOR_MODEL_HPP_
#define _LIBCF_FACTOR_MODEL_HPP_

#include <base/mat.hpp>
#include <base/data.hpp>
#include <model/loss.hpp>
#include <model/penalty.hpp>
#include <model/model_base.hpp>

namespace libcf {

struct FactorModelConfig {
  FactorModelConfig() = default;
  double lambda = 0.01;  // regularization coefficient 
  LossType lt = SQUARE; // loss type
  PenaltyType pt = L2;  // penalty type
  size_t num_dim = 5;
  bool using_bias_term = true;
  bool using_factor_term = true;
  bool using_global_mean = true;
  bool using_adagrad = true;
};

class FactorModel : public ModelBase, public SGDBase {
 public:
  FactorModel(const FactorModelConfig& mcfg) 
      : ModelBase(),
      lambda_(mcfg.lambda), num_dim_(mcfg.num_dim), 
      using_bias_term_(mcfg.using_bias_term),
      using_factor_term_(mcfg.using_factor_term),
      using_global_mean_(mcfg.using_global_mean)
  {  
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "Factor Model Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{BiasTerm: " << using_bias_term_ << "}, "
        << "{FactorTerm: " << using_factor_term_ << "}\n"
        << "\t{Using Global Mean: " << using_global_mean_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}";
  }
  
  FactorModel() :FactorModel(FactorModelConfig()) {}

  // initialize the model
  virtual void reset(const Data& data_set);

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

  virtual double penalty_loss() const {
    return 0.5 * lambda_ * (penalty_->evaluate(coefficients_)  
                            + penalty_->evaluate(factors_));
  }

  virtual void update_one_instance(const Instance& ins, double step_size); 
  virtual void update_one_sgd_step(const Instance& ins, double step_size); 
  
  virtual double predict(const Instance& ins) const;

  virtual double regularization_coefficent() const {
    return lambda_;
  }

 protected:

  DVector coefficients_;
  DMatrix factors_;
  DVector coeff_grad_square_;
  DMatrix factor_grad_square_;

  double lambda_ = 0;
  size_t num_dim_ = 0;
  bool using_bias_term_ = true;
  bool using_factor_term_ = true;  
  double global_mean_ = 0.;
  bool using_global_mean_ = true; 
  bool using_adagrad_ = true;
};

} // namespace

#include <model/factor_model-inl.hpp>

#endif // _LIBCF_FACTOR_MODEL_HPP_
