#ifndef _LIBCF_ITEMCF_HPP_
#define _LIBCF_ITEMCF_HPP_

#include <unordered_set>

#include <model/recsys/similarity_base.hpp>

namespace libcf {

class ItemCF : public SimilarityBase {
  
 public :
  ItemCF(SimilarityType sim_type = Jaccard, size_t topk = 50) :
      SimilarityBase(1, 0, sim_type, topk) 
  {
    LOG(INFO) << "Item Similarity Model";
    LOG(INFO) << "\t{SimType: " << sim_type_ << "}, "
        << "{TOPK: " << topk << "}";
  }
  

  virtual std::vector<size_t> recommend(size_t uid, size_t topk,
                                        const std::unordered_set<size_t>& rated_set) const {

    std::unordered_map<size_t, double> topk_rets;

    for (auto& rated_iid : rated_set) {
      for (auto& item_sim_pair : topk_neighbors_[rated_iid]) {
        if (rated_set.count(item_sim_pair.first)) 
          continue;
        if (topk_rets.count(item_sim_pair.first)) {
          topk_rets[item_sim_pair.first] += item_sim_pair.second;
        } else {
          topk_rets[item_sim_pair.first] = item_sim_pair.second;
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

#endif // _LIBCF_ITEMCF_HPP_
