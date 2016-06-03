#ifndef _LIBCF_FISMP_HPP_
#define _LIBCF_FISMP_HPP_

#include <model/recsys_model_base.hpp>

namespace libcf {

struct FISMPConfig {
  FISMPConfig() = default;
  double lambda = 0.01;  // regularization coefficient 
  LossType lt = SQUARE; // loss type
  PenaltyType pt = L2;  // penalty type
  size_t num_dim = 10;
  size_t num_neg = 5;
  double alpha = 1.;
  bool using_bias_term = true;
  bool using_factor_term = true;
  bool using_global_mean = false;
  bool using_adagrad = true;
};

/** Implementation of the paper 
 *  
 *  FISMP: Factored Item Similarity Models for Top-N Recommender
 *  Systems. KDD13
 *  
 */

class FISMP : public RecsysModelBase, public SGDBase {
 public:
  FISMP(const FISMPConfig& mcfg) 
      : lambda_(mcfg.lambda), num_dim_(mcfg.num_dim), num_neg_(mcfg.num_neg), 
      alpha_(mcfg.alpha),
      using_bias_term_(mcfg.using_bias_term),
      using_factor_term_(mcfg.using_factor_term),
      using_global_mean_(mcfg.using_global_mean),
      using_adagrad_(mcfg.using_adagrad)
  {  
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "FISMP Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{BiasTerm: " << using_bias_term_ << "}, "
        << "{FactorTerm: " << using_factor_term_ << "}\n"
        << "\t{Using Global Mean: " << using_global_mean_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}, "
        << "{Num Negative: " << num_neg_ << "}, "
        << "{alpha: " << alpha_ << "}";

  }

  virtual void reset(const Data& data_set) {
    RecsysModelBase::reset(data_set);

    if (using_bias_term_) {
      bu_ = DVector::Zero(num_users_);
      bi_ = DVector::Zero(num_items_);
      bu_grad_ = DVector::Zero(num_users_);
      bi_grad_ = DVector::Zero(num_items_);
    }

    if (using_factor_term_) {
      p_ = DMatrix::Random(num_items_, num_dim_) * 0.001;
      p_grad_ = DMatrix::Zero(num_items_, num_dim_);
      q_ = DMatrix::Random(num_items_, num_dim_) * 0.001;
      q_grad_ = DMatrix::Zero(num_items_, num_dim_);
      x_ = DMatrix::Zero(num_users_, num_dim_);
      for (size_t uid = 0; uid < num_users_; uid++) {
        auto fit = user_rated_items_.find(uid);
        CHECK(fit != user_rated_items_.end());
        for (auto& item_id : fit->second) {
          x_.row(uid) += p_.row(item_id);
        }
      }
    }

    global_mean_ = 0;

    if (using_global_mean_ && data_set.size() > 0) {
      for (auto iter = data_set.begin(); iter != data_set.end(); ++iter) {
        global_mean_ += iter->label();
      }
      global_mean_ /= data_set.size();
      LOG(INFO) << "Global mean score is " << global_mean_;
    }
  }
  
  virtual double penalty_loss() const {
    return 0.0;
  }

  virtual void update_one_sgd_step(const Instance& ins, double step_size) {
    update_one_instance(ins, step_size);
  }

  void update_one_instance(const Instance& ins, double step_size)  {
    size_t uid = ins.get_feature_group_index(0, 0);
    size_t iid = ins.get_feature_group_index(1, 0);
    size_t jid;

    double pred = predict(ins);
    
    auto fit = user_rated_items_.find(uid);
    CHECK(fit != user_rated_items_.end());

    for (size_t idx = 0; idx < num_neg_; ++ idx) {
      jid = sample_negative_item(fit->second);
      
      double pred_neg = predict_user_item_rating(uid, jid);

      double grad = loss_->gradient(pred - pred_neg, 1.);

      auto bigrad = grad + lambda_ * bi_(iid);
      auto bjgrad = -grad + lambda_ * bi_(jid);

      if (using_adagrad_) {
        bi_grad_(iid) += bigrad * bigrad;
        bi_grad_(jid) += bjgrad * bjgrad;
        bigrad /= std::sqrt(bi_grad_(iid)); 
        bjgrad /= std::sqrt(bi_grad_(jid)); 
      }

      bi_(iid) -= step_size * bigrad;
      bi_(jid) -= step_size * bjgrad;

      DVector x_grad = DVector::Zero(num_dim_);

      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      double user_size = static_cast<double>(fit->second.size());
      for (auto& kid : fit->second) {
        if (kid == iid) continue;
        DVector pj_grad = grad * (q_.row(iid) - q_.row(jid)) / std::pow((user_size - 1.), alpha_) + lambda_ * p_.row(kid);
        if (using_adagrad_) {
          p_grad_.row(kid) += pj_grad.cwiseProduct(pj_grad); 
          for (size_t k = 0; k < num_dim_; ++k) 
            pj_grad(k) = pj_grad(k) / std::sqrt(p_grad_(kid, k));
        }
        p_.row(kid) -= step_size * pj_grad;
        x_grad += pj_grad;
      }

      DVector qi_grad = grad * (x_.row(uid) - p_.row(iid)) / std::pow((user_size - 1.), alpha_) + lambda_ * q_.row(iid);
      DVector qj_grad = - grad * (x_.row(uid) - p_.row(iid)) / std::pow((user_size - 1.), alpha_) + lambda_ * q_.row(jid);
      if (using_adagrad_) {
        q_grad_.row(iid) += qi_grad.cwiseProduct(qi_grad); 
        q_grad_.row(jid) += qj_grad.cwiseProduct(qj_grad); 
        for (size_t k = 0; k < num_dim_; ++k) { 
          qi_grad(k) = qi_grad(k) / std::sqrt(q_grad_(iid, k));
          qj_grad(k) = qj_grad(k) / std::sqrt(q_grad_(jid, k));
        }
      }
      q_.row(iid) -= step_size * qi_grad;
      q_.row(jid) -= step_size * qj_grad;
      x_.row(uid) -= step_size * x_grad; 
    }
  }

  virtual double predict(const Instance& ins) const {
    double ret = 0.;
    size_t user_id = ins.get_feature_group_index(0, 0);
    size_t item_id = ins.get_feature_group_index(1, 0);

    auto fit = user_rated_items_.find(user_id);
    CHECK(fit != user_rated_items_.end());
    auto& user_rated_items = fit->second;

    double user_size = static_cast<double>(fit->second.size());
    if (user_rated_items.count(item_id)) {
      ret += bu_(user_id) + bi_(item_id) + (x_.row(user_id) - p_.row(item_id)).dot(q_.row(item_id)) / std::pow((user_size - 1.), alpha_);
    } else {
      ret += bu_(user_id) + bi_(item_id) + x_.row(user_id).dot(q_.row(item_id)) / std::pow(user_size, alpha_);
    }

    return ret;
  }

 protected:

  DVector bu_, bi_;
  DVector bu_grad_, bi_grad_;
  DMatrix p_; // p in the paper
  DMatrix p_grad_; // p in the paper
  DMatrix q_; // q in the paper
  DMatrix q_grad_; // q in the paper
  DMatrix x_; // x in the paper

  double lambda_ = 0;
  size_t num_dim_ = 0;
  size_t num_neg_ = 5;
  double alpha_  = 1.;
  bool using_bias_term_ = true;
  bool using_factor_term_ = true;  
  double global_mean_ = 0.;
  bool using_global_mean_ = true; 
  bool using_adagrad_ = true;
};

} // namespace

#endif // _LIBCF_FISMP_HPP_
