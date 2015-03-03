#ifndef _LIBCF_SIMILARITY_BASE_HPP_
#define _LIBCF_SIMILARITY_BASE_HPP_

#include <base/parallel.hpp>
#include <model/model_base.hpp>

namespace libcf {

enum SimilarityType {
  Jaccard,
  Cosine
};

std::ostream& operator<< (std::ostream& out, const SimilarityType& st) {
  if (st == Jaccard) {
    out << "Jaccard"; 
  } else if (st == Cosine) {
    out << "Cosine";
  } else {
    LOG(FATAL) << "Undefined similarity type!";
  }
  return out;
}

/**
 *
 *  TODO:
 *   - [] support Ratings
 */
class SimilarityBase : public ModelBase {

 public:

  SimilarityBase(size_t index_feature_group, size_t data_feature_group, 
                 SimilarityType sim_type, size_t topk) {
    index_feature_group_ = index_feature_group;
    data_feature_group_ = data_feature_group;
    sim_type_ = sim_type;
    topk_ = topk;
  }

  virtual void reset(const Data& data_set) {
    data_ = std::make_shared<const Data>(data_set);
    Timer timer;
    CHECK_LT(index_feature_group_, data_set.num_feature_groups());
    CHECK_LT(data_feature_group_, data_set.num_feature_groups());
    index_data_pair = data_set.get_feature_to_vec_hashtable(index_feature_group_, data_feature_group_); 
    data_index_pair = data_set.get_feature_to_vec_hashtable(data_feature_group_, index_feature_group_);

    //CHECK_EQ(index_data_pair.size(), data_->feature_group_total_dimension(index_feature_group_));
    std::vector<double> index_ind_stats(data_set.feature_group_total_dimension(index_feature_group_), 0.);
    for (auto& p : index_data_pair) {
      CHECK_LT(p.first, index_ind_stats.size());
      index_ind_stats[p.first] = p.second.size();  
    } 
    topk_neighbors_.resize(index_ind_stats.size());
    //for (size_t idx = 0; idx < index_ind_stats.size(); idx++) {
    dynamic_parallel_for (0, index_ind_stats.size(), [&](size_t idx) {
        auto fit = index_data_pair.find(idx);
        //CHECK(fit != index_data_pair.end());
        if (fit == index_data_pair.end()) 
          //continue;
          return;
        std::unordered_map<size_t, double> candidates;
        for (auto& data_idx : fit->second) {
          auto iter = data_index_pair.find(data_idx);
          CHECK(iter != data_index_pair.end());
          for (auto& other_index : iter->second) {
            if (other_index == idx) {
              continue;
            }
            if (candidates.count(other_index)) {
              candidates[other_index] += 1.;
            } else {
              candidates[other_index] = 1.;
            }
          }
        }
        std::vector<std::pair<size_t, double>> cand_vec(candidates.begin(), candidates.end());
        for (auto& p : cand_vec) {
          if (sim_type_ == Jaccard) {
            p.second  /= (index_ind_stats[p.first] + index_ind_stats[idx]
                          - p.second);
          } else if (sim_type_ == Cosine) {
            p.second /= std::sqrt(index_ind_stats[p.first] * index_ind_stats[idx]);
          }
        }
        size_t topk_star = std::min(topk_, cand_vec.size());
        std::partial_sort(cand_vec.begin(), cand_vec.begin() + topk_star, cand_vec.end(),
                          sort_by_second_desc<size_t, double>);
        topk_neighbors_[idx].assign(cand_vec.begin(), cand_vec.begin() + topk_star); 
      });
    //}
    LOG(INFO) << "Finished getting nearest neighbors in " << timer;
  }

  // data loss in base class
  virtual double data_loss(const Data& data_set, size_t sample_size = 0) const {
    return 0.0;
  }

  virtual std::vector<size_t> recommend(size_t uid, size_t topk,
                                        const std::unordered_set<size_t>& rated_items) const {

    LOG(FATAL) << "UnImplemented!";
    return std::vector<size_t>{};
  }

  virtual void train_one_iteration(const Data& train_data) {
    // do nothing
  }
  
  std::vector<std::vector<std::pair<size_t, double>>> get_neighbors() const {
    return topk_neighbors_;
  } 

 protected:
  std::vector<std::vector<std::pair<size_t, double>>> topk_neighbors_;  
  std::unordered_map<size_t, std::vector<size_t>> data_index_pair;
  std::unordered_map<size_t, std::vector<size_t>> index_data_pair;
  enum SimilarityType sim_type_;
  size_t topk_;
  size_t index_feature_group_;
  size_t data_feature_group_;

};

} // namespace

#endif // _LIBCF_SIMILARITY_BASE_HPP_
