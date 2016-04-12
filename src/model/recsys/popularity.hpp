#ifndef _LIBCF_POPULARITY_HPP_
#define _LIBCF_POPULARITY_HPP_

#include <unordered_map>

#include <base/data.hpp>
#include <base/utils.hpp>
#include <model/recsys/recsys_model_base.hpp>

namespace libcf {

class Popularity : public RecsysModelBase{

 public:
  Popularity() : RecsysModelBase() {
    LOG(INFO) << "Popularity model";
  }

  virtual void train_one_iteration(const Data& train_data) {
    // do nothing
  }
    
  virtual std::vector<size_t> recommend(size_t user_id, size_t topk, 
                                        const std::unordered_map<size_t, double>& rated_items_map) const {
    std::vector<size_t> ret;
    ret.reserve(topk);

    auto iter = item_popularity.begin();
    auto iter_end = item_popularity.end();
    size_t iid, cnt = 0;
    for (; iter != iter_end; ++iter) {
      if (cnt == topk) break;
      iid = iter->first;
      if (rated_items_map.find(iid) == rated_items_map.end()) {
        ret.push_back(iid);
        ++cnt;
      }
    }
    CHECK(cnt == topk || rated_items_map.size() > num_items_ - topk);
    return std::move(ret);
  }
  
  void reset(const Data& data_set) {
    RecsysModelBase::reset(data_set); 

    item_popularity.resize(num_items_);
    for (size_t iid = 0; iid < num_items_; ++iid) {
      item_popularity[iid].first = iid;
      item_popularity[iid].second = 0.;
    }

    size_t iid;
    for (auto iter = data_set.begin(); iter != data_set.end(); ++iter) {
      iid = iter->get_feature_group_index(1, 0);
      CHECK_LT(iid, num_items_);
      item_popularity[iid].second += 1.;
    }
    std::sort(item_popularity.begin(), item_popularity.end(), 
              sort_by_second_desc<size_t, double>);
    LOG(INFO) << "Item Popularity vector" << item_popularity; 
  }

 protected:
  std::vector<std::pair<size_t, double>> item_popularity;
};

}


#endif // _LIBCF_POPULARITY_HPP_
