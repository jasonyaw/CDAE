#ifndef _LIBCF_RECSYS_MODEL_BASE_HPP_
#define _LIBCF_RECSYS_MODEL_BASE_HPP_

#include <unordered_map>

#include <base/mat.hpp>
#include <base/data.hpp>
#include <base/heap.hpp>
#include <model/loss.hpp>
#include <model/penalty.hpp>
#include <model/model_base.hpp>

namespace libcf {

/**
 * Recsys Model base
 */
class RecsysModelBase : public ModelBase {
 public: 
  
  virtual bool is_implicit() const {
    return true;
  }
  
  virtual double rating_converter(double x) const {
    return (x > 3.0) ? 1. : 0.;
  }

  virtual void reset(const Data& data_set) {
    ModelBase::reset(data_set);
    user_rated_items_ = data_->get_feature_pair_label_hashtable(0, 1);
    num_users_ = data_->feature_group_total_dimension(0);
    num_items_ = data_->feature_group_total_dimension(1);
  }

  virtual double predict_user_item_rating(size_t uid, size_t iid) const {
    return 0.;
  }

  virtual double predict(const Instance& ins) const {
    size_t uid = ins.get_feature_group_index(0, 0);
    size_t iid = ins.get_feature_group_index(1, 0);
    return predict_user_item_rating(uid, iid);
  }
  
  virtual size_t sample_negative_item(const std::unordered_map<size_t, double>& user_map) const {
    size_t random_item;
    while(true) {
      random_item = rand() % num_items_;
      if (user_map.count(random_item)) {
        continue;
      } else {
        break;
      }
    }
    return random_item;
  }
    
  virtual size_t sample_negative_item(const std::unordered_set<size_t>& user_set) const {
    size_t random_item;
    while(true) {
      random_item = rand() % num_items_;
      if (user_set.count(random_item)) {
        continue;
      } else {
        break;
      }
    }
    return random_item;
  }

  virtual void pre_recommend() {
    // do nothing
  }

  // required by evaluation measure TOPN
  virtual std::vector<size_t> recommend(size_t uid, size_t topk,
                                        const std::unordered_map<size_t, double>& rated_item_map) const {
    size_t item_id = 0;
    size_t item_id_end = data_->feature_group_total_dimension(1);
  
    Heap<std::pair<size_t, double>> topk_heap(sort_by_second_desc<size_t, double>, topk);
    double pred;
    for (; item_id != item_id_end; ++item_id) {
      if (rated_item_map.count(item_id)) {
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

 protected:
  size_t num_users_, num_items_;
  std::unordered_map<size_t, std::unordered_map<size_t, double>> user_rated_items_;
};

} // namespace


#endif // _LIBCF_RECSYS_MODEL_BASE_HPP_
