#ifndef _LIBCF_MODEL_BASE_HPP_
#define _LIBCF_MODEL_BASE_HPP_

#include <unordered_set>

#include <base/mat.hpp>
#include <base/data.hpp>
#include <base/heap.hpp>
#include <model/loss.hpp>
#include <model/penalty.hpp>


namespace libcf {


class ModelBase {
 public:
  ModelBase() = default;
  
  /** Reset the model parameters 
   */
  virtual void reset(const Data& data_set) {
    data_ = std::make_shared<const Data>(data_set);
  }

  /** Current Loss
   */
  virtual double current_loss(const Data& data_set, size_t sample_size=0) const {
    return data_loss(data_set, sample_size) + penalty_loss();
  }

  /** Prediction error on training data
   */
  virtual double data_loss(const Data& data_set, size_t sample_size=0) const {
    return 0.0;
  }
  
  /** Regularization Loss
   */
  virtual double penalty_loss() const {
    return 0.0; 
  }
  
  // required by evaluation measures RMSE/MAE
  virtual double predict(const Instance& ins) const {
    LOG(FATAL) << "Unimplemented!";
    return 0.;
  }
  
  virtual double predict_user_item_rating(size_t uid, size_t iid) const {
    Instance ins;
    ins.add_feat_group(std::vector<size_t>{uid});
    ins.add_feat_group(std::vector<size_t>{iid});
    
    CHECK_EQ(ins.get_feature_group_index(0, 0), uid);
    CHECK_EQ(ins.get_feature_group_index(1, 0), iid);
    return predict(ins);
  }

  // required by evaluation measure TOPN
  virtual std::vector<size_t> recommend(size_t uid, size_t topk,
                                        const std::unordered_set<size_t>& rated_item_set) const {
    size_t item_id = 0;
    size_t item_id_end = item_id + data_->feature_group_total_dimension(1);
  
    Heap<std::pair<size_t, double>> topk_heap(sort_by_second_desc<size_t, double>, topk);
    double pred;
    for (; item_id != item_id_end; ++item_id) {
      if (rated_item_set.count(item_id)) {
        continue;
      }
      pred = predict_user_item_rating(uid, item_id);
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

  virtual double regularization_coefficent() const {
    LOG(FATAL) << "Unimplemented!";
    return 0;
  }
  
  // required for SolverBase
  virtual void train_one_iteration(const Data& train_data) {
    LOG(FATAL) << "Unimplemented!";
  }

 protected:
  std::shared_ptr<const Data> data_ = nullptr ;
  std::shared_ptr<Loss> loss_ = nullptr;
  std::shared_ptr<Penalty> penalty_ = nullptr;  
};

// required for SGD solver
class SGDBase {
  virtual void update_one_sgd_step(const Instance& ins, double step_size) {
    LOG(FATAL) << "update_one_sgd_step not implemented!";
  }
};

//required for ALS solver

class ALSBase {
  virtual void train_one_index(size_t idx, 
                       const std::vector<std::pair<size_t, double>>& index_vec,
                       const DMatrix& Y, DMatrix& X) {
    LOG(FATAL) << "train_one_index not implemented!";
  }
                  
};

}


#endif // _LIBCF_MODEL_BASE_HPP_
