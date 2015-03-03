#ifndef _LIBCF_BPR_HPP_
#define _LIBCF_BPR_HPP_

#include <algorithm>
#include <base/heap.hpp>
#include <base/utils.hpp>
#include <model/loss.hpp>
#include <model/factor_model.hpp>

namespace libcf {

struct BPRModelConfig {
  BPRModelConfig() = default;
  double lambda = 0.01;  // regularization coefficient 
  LossType lt = LOG; // loss type
  PenaltyType pt = L2;  // penalty type
  size_t num_dim = 10;
  size_t num_neg = 5;
  bool using_bias_term = true;
  bool using_factor_term = true;
  bool using_global_mean = false;
  bool using_adagrad = true;
};

class BPR_MF : public FactorModel {

 public:
  BPR_MF(const BPRModelConfig& mcfg) : FactorModel() {  
    lambda_ = mcfg.lambda;
    num_dim_ = mcfg.num_dim;
    num_neg_ = mcfg.num_neg;
    using_bias_term_ = mcfg.using_bias_term;
    using_factor_term_ = mcfg.using_factor_term;
    using_global_mean_ = mcfg.using_global_mean;
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "BPR Model Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{BiasTerm: " << using_bias_term_ << "}, "
        << "{FactorTerm: " << using_factor_term_ << "}\n"
        << "\t{Using Global Mean: " << using_global_mean_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}, "
        << "{Num Negative: " << num_neg_ << "}";
  }

  BPR_MF() : BPR_MF(BPRModelConfig()) {}

  void reset(const Data& data_set) {
    FactorModel::reset(data_set);
    user_rated_items_ = data_set.get_feature_to_set_hashtable(0, 1);
    num_users_ = data_->feature_group_total_dimension(0);
    num_items_ = data_->feature_group_total_dimension(1);
  }

  virtual double data_loss(const Data& data_set, size_t sample_size = 0) const {
    return 0.;
  }

  virtual double penalty_loss() const {
    return 0.;
  }

  virtual void update_one_sgd_step(const Instance& ins, double step_size) {

    double pred = predict(ins);

    size_t user_id, item_id, item_neg_id;
    user_id = ins.get_feature_group_index(0, 0);
    item_id = ins.get_feature_group_index(1, 0) + data_->feature_group_start_idx(1);

    for (size_t idx = 0; idx < num_neg_; idx++) {
      ///////////////////////////////////////////
      // sample negative instance
      item_neg_id = sample_negative_item(user_id);

      //TODO
      double pred_neg = predict_user_item_rating(user_id, item_neg_id - num_users_);

      double pair_pred = pred - pred_neg;
      double gradient = loss_->gradient(pair_pred, 1.);

      DMatrix user_grad = factors_.row(user_id) * lambda_;
      DMatrix item_grad = factors_.row(item_id) * lambda_;
      DMatrix item_neg_grad = factors_.row(item_neg_id) * lambda_;

      user_grad += gradient * (factors_.row(item_id) - factors_.row(item_neg_id));
      item_grad += gradient * factors_.row(user_id);
      item_neg_grad += gradient * -1. * factors_.row(user_id);

      ////////////////////////////////////////
      // Update 
      double grad; 

      grad = coefficients_(item_id) * lambda_ + gradient;
      if (using_adagrad_) {
        coeff_grad_square_(item_id) += grad * grad;
        grad /= std::sqrt(coeff_grad_square_(item_id));
      }
      coefficients_(item_id) -= step_size * grad;

      grad = coefficients_(item_neg_id) * lambda_ + gradient;
      if (using_adagrad_) {
        coeff_grad_square_(item_neg_id) += grad * grad;
        grad /= std::sqrt(coeff_grad_square_(item_neg_id));
      }
      coefficients_(item_neg_id) -= step_size * grad;

      if (using_adagrad_) {
        factor_grad_square_.row(user_id) += user_grad.cwiseProduct(user_grad);
        user_grad = user_grad.cwiseQuotient(factor_grad_square_.row(user_id).cwiseSqrt());
      }
      factors_.row(user_id) -= step_size * user_grad;

      if (using_adagrad_) {
        factor_grad_square_.row(item_id) += item_grad.cwiseProduct(item_grad);
        item_grad = item_grad.cwiseQuotient(factor_grad_square_.row(item_id).cwiseSqrt());
      }
      factors_.row(item_id) -= step_size * item_grad;

      if (using_adagrad_) {
        factor_grad_square_.row(item_neg_id) += item_neg_grad.cwiseProduct(item_neg_grad);
        item_neg_grad = item_neg_grad.cwiseQuotient(factor_grad_square_.row(item_neg_id).cwiseSqrt());
      }
      factors_.row(item_neg_id) -= step_size * item_neg_grad;
    }

  }

  virtual size_t sample_negative_item(size_t uid) const {
    auto fit = user_rated_items_.find(uid);
    CHECK(fit != user_rated_items_.end());
    auto& user_rated_items_set = fit->second;
    size_t random_item;
    while(true) {
      random_item = rand() % num_items_;
      if (user_rated_items_set.count(random_item)) {
        continue;
      } else {
        break;
      }
    }
    return random_item + num_users_;
  }


 protected:
  size_t num_users_;
  size_t num_items_;
  size_t num_neg_;
  std::unordered_map<size_t, std::unordered_set<size_t>> user_rated_items_;
};

} // namespace

#endif // _LIBCF_BPR_HPP_
