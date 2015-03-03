#ifndef _LIBCF_NEG_MF_HPP_
#define _LIBCF_NEG_MF_HPP_

#include <algorithm>
#include <base/heap.hpp>
#include <base/utils.hpp>
#include <model/loss.hpp>
#include <model/factor_model.hpp>

namespace libcf {

struct NegMFConfig {
  NegMFConfig() = default;
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

/** Matrix Factorization with Negative sampling
 *
 */ 

class NegMF : public FactorModel {

 public:
  NegMF(const NegMFConfig& mcfg) : FactorModel() {  
    lambda_ = mcfg.lambda;
    num_dim_ = mcfg.num_dim;
    num_neg_ = mcfg.num_neg;
    using_bias_term_ = mcfg.using_bias_term;
    using_factor_term_ = mcfg.using_factor_term;
    using_global_mean_ = mcfg.using_global_mean;
    using_adagrad_ = mcfg.using_adagrad;
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "NegMF Model Configure: \n" 
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

  NegMF() : NegMF(NegMFConfig()) {}

  void reset(const Data& data_set) {
    FactorModel::reset(data_set);
    user_rated_items_ = data_set.get_feature_to_set_hashtable(0, 1);
    num_users_ = data_->feature_group_total_dimension(0);
    num_items_ = data_->feature_group_total_dimension(1);
  }

  virtual double data_loss(const Data& data_set, size_t sample_size = 0) const {
    return 0;
  }

  virtual double penalty_loss() const {
    return 0;
  }

  virtual void update_one_sgd_step(const Instance& ins, double step_size) {
    update_one_instance(ins, step_size); 
    size_t user_id = ins.get_feature_group_index(0, 0);
    for (size_t idx = 0; idx < num_neg_ ; ++idx) {
      size_t item_id = sample_negative_item(user_id);
      Instance neg_ins;
      neg_ins.add_feat_group(std::vector<size_t>{user_id});
      neg_ins.add_feat_group(std::vector<size_t>{item_id});
      if (loss_->loss_type() == "Log" || 
          loss_->loss_type() == "Hinge") {
        neg_ins.set_label(-1.);
      } else {
        neg_ins.set_label(0.);
      }
      update_one_instance(neg_ins, step_size);
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
    return random_item;
  }

  //double predict(const Instance& ins) const;

 protected:
  size_t num_users_;
  size_t num_items_;
  size_t num_neg_;
  std::unordered_map<size_t, std::unordered_set<size_t>> user_rated_items_;
};

} // namespace


#endif // _LIBCF_NEG_MF_HPP_
