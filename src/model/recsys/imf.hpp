#ifndef _LIBCF_IMF_HPP_
#define _LIBCF_IMF_HPP_

#include <algorithm>
#include <base/heap.hpp>
#include <base/utils.hpp>
#include <model/loss.hpp>
#include <model/recsys/recsys_model_base.hpp>

namespace libcf {

struct IMFConfig {
  IMFConfig() = default;
  double learn_rate = 0.1;
  double beta = 1.;
  double lambda = 0.01;   
  LossType lt = SQUARE; 
  PenaltyType pt = L2;  
  size_t num_dim = 10;
  size_t num_neg = 5;
  bool using_bias_term = true;
  bool using_adagrad = true;
};

/** Matrix Factorization with Implicit Feedback
 */ 

class IMF : public RecsysModelBase {

 public:
  IMF(const IMFConfig& mcfg) {  
    learn_rate_ = mcfg.learn_rate;
    beta_ = mcfg.beta;
    lambda_ = mcfg.lambda;
    num_dim_ = mcfg.num_dim;
    num_neg_ = mcfg.num_neg;
    using_bias_term_ = mcfg.using_bias_term;
    using_adagrad_ = mcfg.using_adagrad;
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "IMF Model Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Learn Rate: " << learn_rate_ << "}, "
        << "{Beta: " << beta_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{BiasTerm: " << using_bias_term_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}, "
        << "{Num Negative: " << num_neg_ << "}";
  }

  IMF() = default;

  virtual void reset(const Data& data_set) {
    RecsysModelBase::reset(data_set);

    uv_ = DMatrix::Random(num_users_, num_dim_) * 0.01;
    iv_ = DMatrix::Random(num_items_, num_dim_) * 0.01;
    uv_ag_ = DMatrix::Ones(num_users_, num_dim_) * 0.0001;
    iv_ag_ = DMatrix::Ones(num_items_, num_dim_) * 0.0001; 

    ub_ = DVector::Zero(num_users_);
    ib_ = DVector::Zero(num_items_);
    ub_ag_ = DVector::Ones(num_users_) * 0.0001;
    ib_ag_ = DVector::Ones(num_items_) * 0.0001;
  }

  virtual void train_one_iteration(const Data& train_data) {
    for (size_t uid = 0; uid < num_users_; ++uid) {
      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      auto& item_map = fit->second;
      for (auto& p : item_map) {
        auto& iid = p.first;
        train_one_instance(uid, iid, loss_->positive_label());
        for (size_t idx = 0; idx < num_neg_; ++idx) {
          size_t jid = sample_negative_item(item_map);
          train_one_instance(uid, jid, loss_->negative_label());
        }
      }
    }
  }

  virtual void train_one_instance(size_t uid, size_t iid, double rui) {
    double pred = predict_user_item_rating(uid, iid);
    double gradient = loss_->gradient(pred, rui);

    double ub_grad = gradient + 2. * lambda_ * ub_(uid);
    double ib_grad = gradient + 2. * lambda_ * ib_(iid);
    DVector uv_grad = gradient * iv_.row(iid) + 2. * lambda_ * uv_.row(uid);
    DVector iv_grad = gradient * uv_.row(uid) + 2. * lambda_ * iv_.row(iid);

    if (using_adagrad_) {
      if (using_bias_term_) {
        ub_ag_(uid) += ub_grad * ub_grad;
        ib_ag_(iid) += ib_grad * ib_grad;
        ub_grad /= (beta_ + std::sqrt(ub_ag_(uid)));
        ib_grad /= (beta_ + std::sqrt(ib_ag_(iid)));
      }
      uv_ag_.row(uid) += uv_grad.cwiseProduct(uv_grad);
      iv_ag_.row(iid) += iv_grad.cwiseProduct(iv_grad);
      uv_grad = uv_grad.cwiseQuotient((uv_ag_.row(uid).cwiseSqrt().transpose().array() + beta_).matrix());
      iv_grad = iv_grad.cwiseQuotient((iv_ag_.row(iid).cwiseSqrt().transpose().array() + beta_).matrix());
    }

    if (using_bias_term_) {
      ub_(uid) -= learn_rate_ * ub_grad;
      ib_(iid) -= learn_rate_ * ib_grad;
    }

    uv_.row(uid) -= learn_rate_ * uv_grad;
    iv_.row(iid) -= learn_rate_ * iv_grad;
  }

  double predict_user_item_rating(size_t uid, size_t iid) const {
    return ub_(uid) + ib_(iid) + uv_.row(uid).dot(iv_.row(iid));
  }

  DMatrix get_user_vecs() {
    return uv_;
  }

  DMatrix get_item_vecs() {
    return iv_;
  }

 protected:

  DMatrix uv_, iv_, uv_ag_, iv_ag_;
  DVector ub_, ib_, ub_ag_, ib_ag_;

  double learn_rate_ = 0.1;
  double beta_ = 1.;
  double lambda_ = 0;
  size_t num_dim_ = 0;
  bool using_bias_term_ = true;
  bool using_factor_term_ = true;  
  bool using_adagrad_ = true;
  size_t num_neg_;
};

} // namespace


#endif // _LIBCF_IMF_HPP_
