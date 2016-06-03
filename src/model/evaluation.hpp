#ifndef _LIBCF_EVALUATION_HPP_
#define _LIBCF_EVALUATION_HPP_

#include <unordered_set>
#include <iomanip>
#include <algorithm>

#include <base/parallel.hpp>
#include <base/data.hpp>

namespace libcf {

enum EvalType {
  RMSE = 0,
  MAE,
  TOPN, // Precision, Recall, MAP for implicit data
  RANKING // NDCG, AP for explicit data
};


template<class Model>  
class Evaluation {
 public:
  static std::shared_ptr<Evaluation> create(const EvalType& et);
  virtual std::string evaluation_type() const = 0;

  virtual std::string evaluate(Model& model, 
                               const Data& validation_data,
                               const Data& train_data = Data()) const {
    LOG(FATAL) << "Unimplemented !"; 
    return std::string();
  }

};

template<class Model>
class RMSE_Evaluation : public Evaluation<Model> {

  std::string evaluation_type() const {
    std::stringstream ss;
    ss << std::setw(8) << "RMSE";
    return ss.str(); 
  }
  
  //TODO
  std::string evaluate(Model& model, const Data& validation_data,
                       const Data& train_data = Data()) const {
    double ret = 0;
    double err;
    for (auto iter = validation_data.begin(); 
         iter != validation_data.end(); ++iter) {
      auto& ins = *iter;
      err = model.predict(ins) - ins.label();
      ret += err * err; 
    }
    if (validation_data.size() > 0)
      ret = std::sqrt(ret / static_cast<double>(validation_data.size()));
    std::stringstream ss;
    ss << std::setw(8) << std::setprecision(5) << ret;
    return ss.str();
  }

};

template<class Model>
class MAE_Evaluation : public Evaluation<Model> {

  std::string evaluation_type() const {
    std::stringstream ss;
    ss << std::setw(8) << "MAE";
    return ss.str(); 
  }
  
  std::string evaluate(Model& model, const Data& validation_data,
                       const Data& train_data = Data()) const {
    double ret = 0;
    double err;
    for (auto iter = validation_data.begin(); 
         iter != validation_data.end(); ++iter) {
      auto& ins = *iter;
      err = model.predict(ins) - ins.label();
      ret += std::fabs(err); 
    }
    if (validation_data.size() > 0)
      ret = ret / static_cast<double>(validation_data.size());
    std::stringstream ss;
    ss << std::setw(8) << std::setprecision(5) << ret;
    return ss.str();
  }

};


template<class Model>
class TOPN_Evaluation : public Evaluation<Model> {

  std::string evaluation_type() const {
    std::stringstream ss;
    ss << std::setw(8) << "P@1" << "|"
        << std::setw(8) << "P@5" << "|"
        << std::setw(8) << "P@10" << "|"
        //<< std::setw(8) << "P@20" << "|"
        << std::setw(8) << "R@1" << "|"
        << std::setw(8) << "R@5" << "|"
        << std::setw(8) << "R@10" << "|"
        //<< std::setw(8) << "R@20" << "|"
        << std::setw(8) << "MAP@5" << "|"
        << std::setw(8) << "MAP@10" << "|"
        << std::setw(8) << "TestTime"; 
    return ss.str(); 
  }

  std::string evaluate(Model& model, 
                       const Data& validation_data,
                       const Data& train_data = Data()) const {

    CHECK_GT(validation_data.size(), 0);
    auto validation_user_itemset = validation_data.get_feature_pair_label_hashtable(0, 1);

    std::unordered_map<size_t, std::unordered_map<size_t, double>> train_user_itemset;
    if (train_data.size() != 0) {
      train_user_itemset = train_data.get_feature_pair_label_hashtable(0, 1);
    }
    
    size_t num_users = train_data.feature_group_total_dimension(0);
    CHECK_EQ(num_users, train_user_itemset.size());
    
    Timer t;

    std::vector<std::vector<double>> user_rets(num_users);
    parallel_for(0, num_users, [&](size_t uid) {
                  user_rets[uid] = std::vector<double>(8, 0.);
                });
    
    model.pre_recommend();

    dynamic_parallel_for(0, num_users, [&](size_t uid) {
    //for (size_t uid = 0; uid < num_users; ++uid) {
      auto iter = validation_user_itemset.find(uid);
      if (iter == validation_user_itemset.end()) return;
      auto train_it = train_user_itemset.find(iter->first);
      CHECK(train_it != train_user_itemset.end());
      auto& validation_set = iter->second;
      // Models are required to have this function
      auto rec_list = model.recommend(iter->first, 10, train_it->second);
      
      for (auto& rec_iid : rec_list) {
        CHECK_LT(rec_iid, train_data.feature_group_total_dimension(1));
      }
      for (auto& p : validation_set){
        auto& iid = p.first;
        CHECK_LT(iid, train_data.feature_group_total_dimension(1));
      }
      auto eval_rets = evaluate_rec_list(rec_list, validation_set);
      //std::transform(rets.begin(), rets.end(), eval_rets.begin(), rets.begin(),
      //               std::plus<double>());
      user_rets[uid].assign(eval_rets.begin(), eval_rets.end()); 
    });
    //}
    double num_users_for_test = static_cast<double>(validation_user_itemset.size());
    std::vector<double> rets(8, 0.);
    parallel_for(0, 8, [&](size_t colid) {
              for (size_t uid = 0; uid < num_users; ++uid) {
                rets[colid] += user_rets[uid][colid] / num_users_for_test;
              }
    });

    std::stringstream ss;
    ss << std::setw(8) << std::setprecision(5) << rets[0] << "|"
        << std::setw(8) << std::setprecision(5) << rets[1]  << "|"
        << std::setw(8) << std::setprecision(5) << rets[2] << "|"
        << std::setw(8) << std::setprecision(5) << rets[3] << "|"
        << std::setw(8) << std::setprecision(5) << rets[4] << "|"
        << std::setw(8) << std::setprecision(5) << rets[5] << "|"
        << std::setw(8) << std::setprecision(5) << rets[6] << "|"
        << std::setw(8) << std::setprecision(5) << rets[7] << "|"
        << std::setw(8) << std::setprecision(3) << t.elapsed();
        //<< std::setw(8) << std::setprecision(5) << rets[8] << "|"
        //<< std::setw(8) << std::setprecision(5) << rets[9]; 
    return ss.str(); 
  } 

  std::vector<double> evaluate_rec_list(const std::vector<size_t>& list,
                                        const std::unordered_map<size_t, double>& map) const {
    std::vector<double> rets(8, 0.);
    size_t TOPK = 20;
    double hit = 0.;
    double map5 = 0;
    double map10 = 0;
    TOPK = std::min(TOPK, list.size());
    for (size_t idx = 0; idx < TOPK; ++idx) {
      if (map.find(list[idx]) != map.end()) {
        hit += 1.;
        if (idx < 5) {
          map5 += hit / (idx + 1);
        }
        if (idx < 10) {
          map10 += hit / (idx + 1);
        }
      }
      if (idx == 0) {
        rets[0] = hit / 1.;
        rets[3] = hit / map.size();
      } else if (idx == 4) {
        rets[1] = hit / 5.; 
        rets[4] = hit / map.size(); 
      } else if (idx == 9) {
        rets[2] = hit / 10.;
        rets[5] = hit / map.size();
      }// else if (idx == 19) {
       // rets[3] = hit / 20.;
       // rets[7] = hit / map.size();
     // }
    }
    rets[6] = map5 / static_cast<double>(std::min(size_t(5), map.size()));
    rets[7] = map10 / static_cast<double>(std::min(size_t(10), map.size()));
    return std::move(rets);

  }
};

template<class Model>
class RANKING_Evaluation : public Evaluation<Model> {

  std::string evaluation_type() const {
    std::stringstream ss;
    ss << std::setw(8) << "NDCG@5" << "|"
        << std::setw(8) << "NDCG@10" << "|"
        << std::setw(8) << "Prec@5" << "|"
        << std::setw(8) << "Prec@10" << "|"
        << std::setw(8) << "Recall@5" << "|"
        << std::setw(8) << "Recall@10" << "|"
        << std::setw(8) << "MAP@5" << "|"
        << std::setw(8) << "MAP@10" << "|"
        << std::setw(8) << "TestTime"; 
    return ss.str(); 
  }

  std::string evaluate(Model& model, 
                       const Data& validation_data,
                       const Data& train_data = Data()) const {

    CHECK_GT(validation_data.size(), 0);
    auto validation_user_itemset = validation_data.get_feature_pair_label_hashtable(0, 1);

    std::unordered_map<size_t, std::unordered_map<size_t, double>> train_user_itemset;
    if (train_data.size() != 0) {
      train_user_itemset = train_data.get_feature_pair_label_hashtable(0, 1);
    }
    
    size_t num_users = train_data.feature_group_total_dimension(0);
    CHECK_EQ(num_users, train_user_itemset.size());
    
    Timer t;

    std::vector<std::vector<double>> user_rets(num_users);
    parallel_for(0, num_users, [&](size_t uid) {
                  user_rets[uid] = std::vector<double>(8, 0.);
                });
    
    model.pre_recommend();

    dynamic_parallel_for(0, num_users, [&](size_t uid) {
    //for (size_t uid = 0; uid < num_users; ++uid) {
      auto iter = validation_user_itemset.find(uid);
      if (iter == validation_user_itemset.end()) return;
      auto train_it = train_user_itemset.find(iter->first);
      CHECK(train_it != train_user_itemset.end());
      auto& validation_set = iter->second;
      // Models are required to have this function
      auto rec_list = model.recommend(iter->first, 10, train_it->second);
      
      for (auto& rec_iid : rec_list) {
        CHECK_LT(rec_iid, train_data.feature_group_total_dimension(1));
      }
      for (auto& p : validation_set){
        auto& iid = p.first;
        CHECK_LT(iid, train_data.feature_group_total_dimension(1));
      }
      auto eval_rets = evaluate_rec_list(rec_list, validation_set);
      //std::transform(rets.begin(), rets.end(), eval_rets.begin(), rets.begin(),
      //               std::plus<double>());
      user_rets[uid].assign(eval_rets.begin(), eval_rets.end()); 
    });
    //}
    double num_users_for_test = static_cast<double>(validation_user_itemset.size());
    std::vector<double> rets(8, 0.);
    parallel_for(0, 8, [&](size_t colid) {
              for (size_t uid = 0; uid < num_users; ++uid) {
                rets[colid] += user_rets[uid][colid] / num_users_for_test;
              }
    });

    std::stringstream ss;
    ss << std::setw(8) << std::setprecision(5) << rets[0] << "|"
        << std::setw(8) << std::setprecision(5) << rets[1]  << "|"
        << std::setw(8) << std::setprecision(5) << rets[2] << "|"
        << std::setw(8) << std::setprecision(5) << rets[3] << "|"
        << std::setw(8) << std::setprecision(5) << rets[4] << "|"
        << std::setw(8) << std::setprecision(5) << rets[5] << "|"
        << std::setw(8) << std::setprecision(5) << rets[6] << "|"
        << std::setw(8) << std::setprecision(5) << rets[7] << "|"
        << std::setw(8) << std::setprecision(3) << t.elapsed();
        //<< std::setw(8) << std::setprecision(5) << rets[8] << "|"
        //<< std::setw(8) << std::setprecision(5) << rets[9]; 
    return ss.str(); 
  } 

  std::vector<double> evaluate_rec_list(const std::vector<size_t>& list,
                                        const std::unordered_map<size_t, double>& map) const {
    std::vector<double> rets(8, 0.);
    std::vector<std::pair<size_t, double>> ground_truth(map.begin(), map.end());
    std::sort(ground_truth.begin(), ground_truth.end(), sort_by_second_desc<size_t, double>);
  
    double DCG5 = 0., DCG10 = 0.;
    double IDCG5 = 0., IDCG10 = 0.;
    double hit5 = 0., hit10 = 0.;
    double map5 = 0., map10 = 0.;

    for (size_t idx = 0; idx < 10; ++idx) {
      if (idx < map.size()) { 
       if(idx < 5) {
          IDCG5 += (std::pow(2, ground_truth[idx].second) - 1.) / std::log(idx + 2.);  
        }
        IDCG10 += (std::pow(2, ground_truth[idx].second) - 1.) / std::log(idx + 2.);  
      }
      auto& iid = list[idx];
      if (map.count(iid)) {
        if(idx < 5) {
          DCG5 += (std::pow(2, map.at(iid)) - 1.) / std::log(idx + 2.);  
        }
        DCG10 += (std::pow(2, map.at(iid)) - 1.) / std::log(idx + 2.);
        if (map.at(iid) >= 4.) {
          if (idx < 5) {  
            hit5 += 1.;
            map5 += hit5 / (idx + 1.);
          }
          hit10 += 1.;
          map10 += hit10 / (idx + 1.);
        }
      }
    }

    rets[0] = DCG5 / IDCG5;
    rets[1] = DCG10 / IDCG10;
    
    rets[2] = hit5 / 5.;
    rets[3] = hit10 / 10.;

    int num_rels = std::count_if(map.begin(), map.end(), [](const std::pair<size_t, double>& v) {return v.second >= 4.;});
    if (num_rels > 0) {
      rets[4] = hit5 / num_rels;
      rets[5] = hit10 / num_rels;

      rets[6] = map5 / std::min(5., static_cast<double>(map.size()));
      rets[7] = map10 / std::min(10., static_cast<double>(map.size()));
    }
  
    
    return std::move(rets);
  }
};



template<class Model>
std::shared_ptr<Evaluation<Model>> Evaluation<Model>::create(const EvalType& rt) {
  switch (rt) {
    case RMSE:
      return std::shared_ptr<Evaluation<Model>>(new RMSE_Evaluation<Model>());
    case MAE:
      return std::shared_ptr<Evaluation<Model>>(new MAE_Evaluation<Model>());
    case TOPN:
      return std::shared_ptr<Evaluation<Model>>(new TOPN_Evaluation<Model>());
    case RANKING:
      return std::shared_ptr<Evaluation<Model>>(new RANKING_Evaluation<Model>());
    default:
      return std::shared_ptr<Evaluation<Model>>(new RMSE_Evaluation<Model>());
  }
}

} // namespace

#endif // _LIBCF_EVALUATION_HPP_
