#ifndef _LIBCF_ALS_HPP_
#define _LIBCF_ALS_HPP_ 

#include <base/parallel.hpp>
#include <model/model_base.hpp>


namespace libcf {

struct ALSConfig {
  ALSConfig() = default;
  double lambda = 0.01;  // regularization coefficient 
  LossType lt = SQUARE; // loss type
  PenaltyType pt = L2;  // penalty type
  size_t num_dim = 10;
};

/**
 *  Implementation of the paper :
 *  Collaborative filtering for implicit feedback datasets, ICDM'11
 */
class ALS : public ModelBase {
 public:
  ALS(const ALSConfig& mcfg) 
      : ModelBase(),
      lambda_(mcfg.lambda), num_dim_(mcfg.num_dim)
  {  
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);
    
    LOG(INFO) << "ALS Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}\n";
  }

  ALS() : ALS(ALSConfig()) {}

  virtual void reset(const Data& data_set) {
    data_ = std::make_shared<const Data>(data_set);
       
    num_users_ = data_->feature_group_total_dimension(0);
    num_items_ = data_->feature_group_total_dimension(1);
  
    p_ = DMatrix::Random(num_users_, num_dim_) * 0.001;
    q_ = DMatrix::Random(num_items_, num_dim_) * 0.001;
      
    user_idx_map_ = data_->get_feature_pair_label_hashtable(0, 1);
    item_idx_map_ = data_->get_feature_pair_label_hashtable(1, 0);
  }

  virtual double penalty_loss() const {
    return lambda_ * (penalty_->evaluate(p_) + penalty_->evaluate(q_)); 
  };

  virtual double predict(const Instance& ins) const {
    double ret = 0;
    size_t user_id = ins.get_feature_group_index(0, 0);
    size_t item_id = ins.get_feature_group_index(1, 0);
    ret = p_.row(user_id).dot(q_.row(item_id));
    return ret;
  }

  void train_one_index(size_t idx, 
                       const std::unordered_map<size_t, double>& index_vec, 
                       const DMatrix& Y, DMatrix& X) {

    size_t vec_size = index_vec.size();
    DMatrix YCY = DMatrix::Zero(num_dim_, num_dim_);
    for (size_t k = 0; k < num_dim_; ++k) {
      YCY(k, k) += lambda_;
    }

    for (auto& p : index_vec) {
      size_t oth_idx = p.first;
      CHECK_LT(oth_idx, Y.rows());
      CHECK_GE(oth_idx, 0);
      //double rating = p.second;
      //CHECK_EQ(rating, 1.0);
      for (size_t i = 0; i < num_dim_; ++i)
        for (size_t j = 0; j < num_dim_; ++j)
          YCY(i,j) += Y(oth_idx, i) * Y(oth_idx, j);
    }
    YCY = YCY.inverse();
    
    X.row(idx) = DVector::Zero(num_dim_);

    size_t i = 0;
    for (auto& p : index_vec) {
      size_t oth_idx = p.first;
      double rating = p.second;
      for (size_t k = 0; k < num_dim_; ++k)
        X(idx, k) += YCY.col(k).dot(Y.row(oth_idx) * rating);
      i++;
    }
    CHECK_EQ(i, vec_size);
  }
  
  virtual void train_one_iteration(const Data& trian_data) {
    dynamic_parallel_for(0, num_users_, [&](size_t user_id) {
                          train_one_user(user_id);                          
                         });
    dynamic_parallel_for(0, num_items_, [&](size_t item_id) {
                          train_one_item(item_id);                          
                         });
  }

  void train_one_user(size_t uid) {
    auto fit = user_idx_map_.find(uid);
    if (fit == user_idx_map_.end()) 
      return;
    train_one_index(uid, fit->second, q_, p_);
  }

  void train_one_item(size_t iid) {
    auto fit = item_idx_map_.find(iid);
    if (fit == item_idx_map_.end()) 
      return;
    train_one_index(iid, fit->second, p_, q_);
  }

 private:
  std::unordered_map<size_t, std::unordered_map<size_t, double>> user_idx_map_, item_idx_map_;
  double lambda_ = 0.;
  size_t num_users_ = 0;
  size_t num_items_ = 0;
  size_t num_dim_ = 0;
  DMatrix p_;
  DMatrix q_;
};

} // namespace

#endif // _LIBCF_ALS_HPP_
