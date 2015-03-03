#ifndef _LIBCF_POPULARITY_HPP_
#define _LIBCF_POPULARITY_HPP_

#include <unordered_map>

#include <base/data.hpp>
#include <base/utils.hpp>
#include <model/model_base.hpp>
#include <model/evaluation.hpp>

namespace libcf {

class Popularity : public ModelBase{

 public:
  Popularity() : ModelBase() {
    LOG(INFO) << "Popularity model";
  }

  virtual void train_one_iteration(const Data& train_data) {
    // do nothing
  }
    
  virtual std::vector<size_t> recommend(size_t user_id, size_t topk, 
                                        const std::unordered_set<size_t>& rated_items_set) const {
    std::vector<size_t> ret;
    ret.reserve(topk);

    auto iter = item_popularity.begin();
    auto iter_end = item_popularity.end();
    size_t iid, cnt = 0;
    for (; iter != iter_end; ++iter) {
      if (cnt == topk) break;
      iid = iter->first;
      if (rated_items_set.find(iid) == rated_items_set.end()) {
        ret.push_back(iid);
        ++cnt;
      }
    }
    CHECK(cnt == topk || rated_items_set.size() > num_items - topk);
    return std::move(ret);
  }
  
  /*
  virtual double predict(const Instance& ins) const {
    size_t item_id = ins.get_feature_group_index(1, 0);
    CHECK_LT(item_id, num_items);
    CHECK_EQ(item_popularity[item_id].first, item_id);
    return item_popularity[item_id].second;
  }
  */

  void reset(const Data& data_set) {
    ModelBase::reset(data_set); 
    num_users = data_set.feature_group_total_dimension(0);
    num_items = data_set.feature_group_total_dimension(1);

    item_popularity.resize(num_items);
    for (size_t iid = 0; iid < num_items; ++iid) {
      item_popularity[iid].first = iid;
      item_popularity[iid].second = 0.;
    }

    size_t iid;
    for (auto iter = data_set.begin(); iter != data_set.end(); ++iter) {
      iid = iter->get_feature_group_index(1, 0);
      CHECK_LT(iid, num_items);
      item_popularity[iid].second += 1.;
    }
    std::sort(item_popularity.begin(), item_popularity.end(), 
              sort_by_second_desc<size_t, double>);
    LOG(INFO) << "Item Popularity vector" << item_popularity; 
  }

 protected:
  std::vector<std::pair<size_t, double>> item_popularity;
  size_t num_users, num_items;
};

}


#endif // _LIBCF_POPULARITY_HPP_
