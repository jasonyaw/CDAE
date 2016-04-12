#ifndef _LIBCF_USERCF_HPP_
#define _LIBCF_USERCF_HPP_ 

#include <unordered_map>

#include <model/recsys/similarity_base.hpp>

namespace libcf {

class UserCF : public SimilarityBase {
  
 public :
  UserCF(SimilarityType sim_type = Jaccard, size_t topk = 50) :
      SimilarityBase(0, 1, sim_type, topk) 
  {
    LOG(INFO) << "User Similarity Model";
    LOG(INFO) << "\t{SimType: " << sim_type_ << "}, "
        << "{TOPK: " << topk << "}";
  }

  virtual std::vector<size_t> recommend(size_t uid, size_t topk,
                                        const std::unordered_map<size_t, double>& rated_map) const {
  
    
    CHECK_LT(uid, topk_neighbors_.size());
    auto& similar_users = topk_neighbors_[uid];
    std::unordered_map<size_t, double> topk_rets;
    
    for (auto& user_sim_pair : similar_users) {
      CHECK(index_data_pair.count(user_sim_pair.first));
      auto& sim_index_data_pair = index_data_pair.at(user_sim_pair.first);    
      for (auto& item_id : sim_index_data_pair) {
        if (rated_map.count(item_id)) 
          continue;
        if (topk_rets.count(item_id)) {
          topk_rets[item_id] += user_sim_pair.second;
        } else {
          topk_rets[item_id] = user_sim_pair.second;
        }
      }
    }

    std::vector<std::pair<size_t, double>> ret_pairs(topk_rets.begin(), 
                                                     topk_rets.end());
    size_t topk_star = std::min(topk, ret_pairs.size());
    std::partial_sort(ret_pairs.begin(), ret_pairs.begin() + topk_star, ret_pairs.end(),
                      sort_by_second_desc<size_t, double>);
    std::vector<size_t> rets(topk_star);
    std::transform(ret_pairs.begin(), ret_pairs.begin() + topk_star,
                   rets.begin(), [](const std::pair<size_t, double>& p){
                   return p.first;
                   });
    return std::move(rets);
  }


};


} // namespace

#endif // _LIBCF_USERCF_HPP_
