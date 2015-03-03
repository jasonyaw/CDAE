#ifndef _LIBCF_FISM_HPP_
#define _LIBCF_FISM_HPP_

#include <model/model_base.hpp>

namespace libcf {

struct FISMConfig {
  FISMConfig() = default;
  double lambda = 0.01;   
  LossType lt = SQUARE; 
  PenaltyType pt = L2;  
  size_t num_dim = 10;
  size_t num_neg = 5;
  int alpha = 1;
  bool using_bias_term = true;
  bool using_factor_term = true;
  bool using_global_mean = false;
  bool using_adagrad = true;
};

/** Implementation of the paper 
 *  
 *  FISM: Factored Item Similarity Models for Top-N Recommender
 *  Systems. KDD13
 *  
 */

class FISM : public ModelBase, public SGDBase{
 public:
  FISM(const FISMConfig& mcfg) 
      : ModelBase(),
      lambda_(mcfg.lambda), num_dim_(mcfg.num_dim), num_neg_(mcfg.num_neg), 
      alpha_(mcfg.alpha),
      using_bias_term_(mcfg.using_bias_term),
      using_factor_term_(mcfg.using_factor_term),
      using_global_mean_(mcfg.using_global_mean),
      using_adagrad_(mcfg.using_adagrad)
  {  
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "FISM Configure: \n" 
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
    data_ = std::make_shared<const Data>(data_set);
    num_users_ = data_set.feature_group_total_dimension(0);
    num_items_ = data_set.feature_group_total_dimension(1);
    user_rated_items_ = data_set.get_feature_to_set_hashtable(0, 1);

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
    size_t uid = ins.get_feature_group_index(0, 0);

    for (size_t idx = 0; idx < num_neg_; idx++) {
      size_t iid = sample_negative_item(uid);
      Instance neg_ins;
      neg_ins.add_feat_group(std::vector<size_t>{uid});
      neg_ins.add_feat_group(std::vector<size_t>{iid});
      neg_ins.set_label(0.);
      update_one_instance(neg_ins, step_size); 
    }
  }

  void update_one_instance(const Instance& ins, double step_size)  {
    size_t uid = ins.get_feature_group_index(0, 0);
    size_t iid = ins.get_feature_group_index(1, 0);

    double pred = predict(ins);
    double grad = loss_->gradient(pred, ins.label());

    auto bugrad = grad + lambda_ * bu_(uid);
    auto bigrad = grad + lambda_ * bi_(iid);

    if (using_adagrad_) {
      bu_grad_(uid) += bugrad * bugrad;
      bi_grad_(iid) += bigrad * bigrad;
      bugrad /= (std::sqrt(bu_grad_(uid))); 
      bigrad /= (std::sqrt(bi_grad_(iid))); 
    }

    bu_(uid) -= step_size * bugrad;
    bi_(iid) -= step_size * bigrad;

    DVector x_grad = DVector::Zero(num_dim_);

    auto fit = user_rated_items_.find(uid);
    CHECK(fit != user_rated_items_.end());
    double user_size = static_cast<double>(fit->second.size());
    bool rated = fit->second.count(iid);
    double scale = 0;
    
    if (rated) {
      scale = 1 / static_cast<double>(std::pow((user_size - 1), alpha_));
    } else {
      scale = 1 / static_cast<double>(std::pow(user_size, alpha_));
    }

    for (auto& jid : fit->second) {
      if (jid == iid) continue;
      DVector pj_grad = grad * q_.row(iid) * scale + lambda_ * p_.row(jid);
      if (using_adagrad_) {
        p_grad_.row(jid) += pj_grad.cwiseProduct(pj_grad); 
        pj_grad = pj_grad.cwiseQuotient(p_grad_.row(jid).transpose().cwiseSqrt());
      }
      p_.row(jid) -= step_size * pj_grad;
      x_grad += pj_grad;
    }

    DVector qi_grad = DVector::Zero(num_dim_);
    
    if (rated) {
      qi_grad = grad * (x_.row(uid) - p_.row(iid)) * scale + lambda_ * q_.row(iid);
    } else {
      qi_grad = grad * x_.row(uid) * scale + lambda_ * q_.row(iid);
    }

    if (using_adagrad_) {
      q_grad_.row(iid) += qi_grad.cwiseProduct(qi_grad); 
      qi_grad = qi_grad.cwiseQuotient(q_grad_.row(iid).transpose().cwiseSqrt());
    }

    q_.row(iid) -= step_size * qi_grad;
    x_.row(uid) -= step_size * x_grad; 
  }


  // required by evaluation measure TOPN
  virtual std::vector<size_t> recommend(size_t uid, size_t topk,
                                        const std::unordered_set<size_t>& rated_item_set) const {
    size_t item_id = 0;
    size_t item_id_end = item_id + data_->feature_group_total_dimension(1);
    
    double scale = 1. / static_cast<double>(std::pow(rated_item_set.size(), alpha_));

    Heap<std::pair<size_t, double>> topk_heap(sort_by_second_desc<size_t, double>, topk);
    double pred;
    for (; item_id != item_id_end; ++item_id) {
      if (rated_item_set.count(item_id)) {
        continue;
      }
      pred = bu_(uid) + bi_(item_id) + scale * x_.row(uid).dot(q_.row(item_id));
      if (topk_heap.size() < topk) {
        topk_heap.push({item_id, pred});
      } else {
        topk_heap.push_and_pop({item_id, pred});
      }
    }
    CHECK_EQ(topk_heap.size(), topk);
    auto topk_heap_vec = topk_heap.get_sorted_data();
    std::vector<size_t> ret(topk);
    std::transform(topk_heap_vec.begin(), topk_heap_vec.end(),
                   ret.begin(),
                   [](const std::pair<size_t, double>& p) {
                   return p.first;
                   });
    return std::move(ret);
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
      ret += bu_(user_id) + bi_(item_id) + (x_.row(user_id) - p_.row(item_id)).dot(q_.row(item_id)) 
          / static_cast<double>(std::pow((user_size - 1), alpha_));
    } else {
      ret += bu_(user_id) + bi_(item_id) + x_.row(user_id).dot(q_.row(item_id)) 
          / static_cast<double>(std::pow(user_size, alpha_));
    }
    return ret;
  }

  virtual double regularization_coefficent() const {
    return lambda_;
  }

  size_t sample_negative_item(size_t uid) const {
    auto fit = user_rated_items_.find(uid);
    CHECK(fit != user_rated_items_.end());
    auto& user_rated_items = fit->second;
    size_t random_item;
    while(true) {
      random_item = rand() % num_items_;
      if (user_rated_items.count(random_item)) {
        continue;
      } else {
        break;
      }
    }
    return random_item;
  }

 protected:

  DVector bu_, bi_;
  DVector bu_grad_, bi_grad_;
  DMatrix p_; // p in the paper
  DMatrix p_grad_; // p in the paper
  DMatrix q_; // q in the paper
  DMatrix q_grad_; // q in the paper
  DMatrix x_; // x in the paper

  std::unordered_map<size_t, std::unordered_set<size_t>> user_rated_items_;  

  size_t num_users_ = 0, num_items_ = 0;

  double lambda_ = 0;
  size_t num_dim_ = 0;
  size_t num_neg_ = 5;
  int alpha_ = 1.;
  bool using_bias_term_ = true;
  bool using_factor_term_ = true;  
  double global_mean_ = 0.;
  bool using_global_mean_ = true; 
  bool using_adagrad_ = true;
};

} // namespace

#endif // _LIBCF_FISM_HPP_
