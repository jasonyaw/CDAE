#include <base/data.hpp>

#include <random>
#include <algorithm>

#include <base/io.hpp>
#include <base/timer.hpp>
#include <base/utils.hpp>
#include <base/random.hpp>

namespace libcf {

void Data::load(const std::string& filename, 
                const DataFormat& df, 
                const LineParser& parser,
                bool skip_header) {
  if (data_info_ == nullptr) {
    data_info_ = std::make_shared<DataInfo>(new DataInfo());
  }

  switch (df) {
    case VECTOR : {
      FileLineReader f(filename);
      add_feature_group(DENSE);
      set_label_type(EMPTY);
      f.set_line_callback(
         [&](const std::string& line, size_t line_num) {
          if (skip_header && line_num == 0) return;
          auto rets = parser(line);
          if (rets.size() == 0) return;
          Instance ins;
          std::vector<double> vec(rets.size());
          std::transform(rets.begin(), rets.end(), vec.begin(),
            [&](const std::string& str) { return std::stod(str); });
          ins.add_feat_group(data_info_->feature_group_infos_[0], vec);
          instances_.push_back(ins);
        });
      f.load();
      break; 
    }
    case LIBSVM : {
      //TODO
      break;
    }
    case RECSYS : {
      FileLineReader f(filename);
      // user item rating
      add_feature_group(SPARSE_BINARY);
      add_feature_group(SPARSE_BINARY);
      set_label_type(CONTINUOUS);
      f.set_line_callback(
          [&](const std::string& line, size_t line_num) {
          if (skip_header && line_num == 0) return;
          auto rets = parser(line);
          if (rets.size() == 0) return;
          Instance ins;
          ins.add_feat_group(data_info_->feature_group_infos_[0], rets[0]); 
          ins.add_feat_group(data_info_->feature_group_infos_[1], rets[1]); 
          ins.set_label(std::stod(rets[2]));
          instances_.push_back(ins);
          });
      f.load();
      break;
    }
    default : {
      break;
    }
  }

  data_info_->total_dimensions_ = 0;
  data_info_->feature_group_global_idx_.assign(num_feature_groups(), 0);
  size_t idx = 0;
  for (auto& fg_info : data_info_->feature_group_infos_) {
    data_info_->feature_group_global_idx_[idx++] = data_info_->total_dimensions_;
    data_info_->total_dimensions_ += fg_info.size();
  }
  LOG(INFO) << "Data loaded successfully.\n";
  LOG(INFO) << *this;

}

std::ostream& operator<< (std::ostream& stream, const Data& data) {

  stream << "\nData set summary : \n";
  stream << "\tNum of Instance: " << data.instances_.size() << std::endl;
  stream << "\tNum of feature groups: " << data.data_info_->feature_group_infos_.size() << std::endl;
  stream << "\tTotal feature dimensions: " << data.data_info_->total_dimensions_ << std::endl;
  stream << "\tFeature group idx scope: [";
  for (size_t idx = 0; idx < data.data_info_->feature_group_global_idx_.size(); ++idx) {
    if (idx > 0) stream << " ";
    stream << data.data_info_->feature_group_global_idx_[idx];   
  }
  stream << "]\n";
  size_t idx = 0;
  for (auto& fg_info : data.data_info_->feature_group_infos_) {
    stream << "\tFeature group " << idx++ << " -> " << fg_info << std::endl;
  }
  stream << "Head of the data set:\n"; 
  size_t num_lines = std::min(size_t{10}, data.instances_.size());
  for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
    auto& ins = data.instances_[line_idx];
    stream << "  " << ins << std::endl;
  }
  return stream;
}

class Data::instance_iterator {
 public:
  instance_iterator(const Data& data, const Instance& ins) :
      data_cref_(&data), instance_cref_(&ins),
      fg_idx_(0), feat_idx_(0) 
  {}

  instance_iterator(const Data& data, const Instance& ins,
                    size_t fg_idx, size_t feat_idx) : 
      data_cref_(&data), instance_cref_(&ins),
      fg_idx_(fg_idx), feat_idx_(feat_idx) 
  {}

  instance_iterator(const instance_iterator&) = default;
  instance_iterator(instance_iterator&&) = default;

  size_t feature_group_idx() const { return fg_idx_; }

  size_t index() const { 
    CHECK_LT(fg_idx_, instance_cref_->num_feature_groups());
    CHECK_LT(feat_idx_, instance_cref_->feature_group_size(fg_idx_));
    return data_cref_->feature_group_start_idx(fg_idx_) 
        + instance_cref_->get_feature_group_index(fg_idx_, feat_idx_); 
  }

  double value() const {
    CHECK_LT(fg_idx_, instance_cref_->num_feature_groups());
    CHECK_LT(feat_idx_, instance_cref_->feature_group_size(fg_idx_));
    return instance_cref_->get_feature_group_value(fg_idx_, feat_idx_); 
  }

  bool operator == (const instance_iterator& oth) {
    return (data_cref_ == oth.data_cref_) && 
        (instance_cref_ == oth.instance_cref_) && 
        (fg_idx_ == oth.fg_idx_) && 
        (feat_idx_ == oth.feat_idx_);
  }

  bool operator != (const instance_iterator& oth) {
    return ! (*this == oth);
  }

  instance_iterator& operator = (const instance_iterator& oth) {
    data_cref_ = oth.data_cref_;
    instance_cref_ = oth.instance_cref_;
    fg_idx_ = oth.fg_idx_;
    feat_idx_ = oth.feat_idx_;
    return *this;
  }

  instance_iterator& operator ++ () {
    if (fg_idx_ < instance_cref_->num_feature_groups() ) {
      if (feat_idx_ < instance_cref_->feature_group_size(fg_idx_) - 1) {
        ++feat_idx_;
      } else {
        ++fg_idx_;
        feat_idx_ = 0;
      }
    }
    return *this;
  }

  instance_iterator operator ++ (int) {
    instance_iterator tmp = *this;
    ++*this;
    return std::move(tmp);
  }

 private: 
  const Data* data_cref_;
  const Instance* instance_cref_; 
  size_t fg_idx_;
  size_t feat_idx_;

};

Data::instance_iterator Data::begin(size_t idx) const {
  CHECK_LT(idx, size());
  return instance_iterator(*this, instances_[idx], 0, 0);
}

Data::instance_iterator Data::end(size_t idx) const {
  CHECK_LT(idx, size());
  return instance_iterator(*this, instances_[idx], instances_[idx].size(), 0);
}

Data::instance_iterator Data::begin(const Instance& ins) const {
  return instance_iterator(*this, ins, 0, 0);
}

Data::instance_iterator Data::end(const Instance& ins) const {
  return instance_iterator(*this, ins, ins.size(), 0);
}

void Data::shuffle_data() {
  Random::shuffle(std::begin(instances_), std::end(instances_));
}


void Data::random_split(Data& train, Data& test, double test_ratio) const {

  CHECK_LT(test_ratio, 1.0);
  // shuffle_data();
  size_t num_train = static_cast<size_t>((1. - test_ratio) * size());
  size_t num_test = size() - num_train;

  std::vector<size_t> index_vec(size(), 0);
  std::iota(index_vec.begin(), index_vec.end(), 0);
  Random::shuffle(std::begin(index_vec), std::end(index_vec)); 

  std::vector<Instance> train_ins_vec(num_train);
  for(size_t idx = 0; idx < num_train; ++idx) {
    train_ins_vec[idx] = instances_[index_vec[idx]];
  }

  std::vector<Instance> test_ins_vec(this->size() - num_train);
  for(size_t idx = 0; idx < num_test; ++idx) {
    test_ins_vec[idx] = instances_[index_vec[num_train + idx]];
  }

  train = Data(std::move(train_ins_vec), data_info_);
  test = Data(std::move(test_ins_vec), data_info_);
}

void Data::random_split_by_feature_group(Data& train, Data& test,
                                         size_t feature_group_idx, double test_ratio) const {

  Timer timer;

  size_t est_num_test = static_cast<size_t>(test_ratio * size());
  size_t est_num_train = size() - est_num_test;

  std::vector<Instance> train_ins_vec;
  train_ins_vec.reserve(est_num_train + size() * 0.01);
  std::vector<Instance> test_ins_vec;
  test_ins_vec.reserve(est_num_test + size() * 0.01);

  auto fg_idx_ins_idx_hashtable = get_feature_ins_idx_hashtable(feature_group_idx);

  size_t cnt = 0;
  size_t num_test;

  for (auto iter = fg_idx_ins_idx_hashtable.begin(); iter != fg_idx_ins_idx_hashtable.end(); ++iter) {
    auto& tmp_vec = iter->second;
    Random::shuffle(std::begin(tmp_vec), std::end(tmp_vec));
    num_test = static_cast<size_t>(tmp_vec.size() * test_ratio);
    for(size_t idx = 0; idx < tmp_vec.size(); ++idx) {
      if (idx < num_test) {
        test_ins_vec.push_back(instances_[tmp_vec[idx]]);
      } else {
        train_ins_vec.push_back(instances_[tmp_vec[idx]]);
      }
    }
    ++cnt;
  }
  CHECK_EQ(cnt, feature_group_total_dimension(feature_group_idx));
  CHECK_EQ(test_ins_vec.size() + train_ins_vec.size(), size());

  Random::shuffle(std::begin(train_ins_vec), std::end(train_ins_vec));
  Random::shuffle(std::begin(test_ins_vec), std::end(test_ins_vec));

  train = Data(std::move(train_ins_vec), data_info_);
  test = Data(std::move(test_ins_vec), data_info_);

  LOG(INFO) << "Finished splitting data set in " << timer;
}

void Data::inplace_random_split_by_feature_group(Data& train, Data& test,
                                         size_t feature_group_idx, double test_ratio)  {

  Timer timer;

  size_t est_num_test = static_cast<size_t>(test_ratio * size());
  size_t est_num_train = size() - est_num_test;

  std::vector<Instance> train_ins_vec;
  train_ins_vec.reserve(est_num_train + size() * 0.01);
  std::vector<Instance> test_ins_vec;
  test_ins_vec.reserve(est_num_test + size() * 0.01);

  auto fg_idx_ins_idx_hashtable = get_feature_ins_idx_hashtable(feature_group_idx);
  size_t cnt = 0;
  size_t num_test;

  for (auto iter = fg_idx_ins_idx_hashtable.begin(); iter != fg_idx_ins_idx_hashtable.end(); ++iter) {
    auto& tmp_vec = iter->second;
    Random::shuffle(std::begin(tmp_vec), std::end(tmp_vec));
    num_test = static_cast<size_t>(tmp_vec.size() * test_ratio);
    for(size_t idx = 0; idx < tmp_vec.size(); ++idx) {
      if (idx < num_test) {
        test_ins_vec.push_back(std::move(instances_[tmp_vec[idx]]));
      } else {
        train_ins_vec.push_back(std::move(instances_[tmp_vec[idx]]));
      }
    }
    ++cnt;
  }
  CHECK_EQ(cnt, feature_group_total_dimension(feature_group_idx));
  CHECK_EQ(test_ins_vec.size() + train_ins_vec.size(), size());

  Random::shuffle(std::begin(train_ins_vec), std::end(train_ins_vec));
  Random::shuffle(std::begin(test_ins_vec), std::end(test_ins_vec));

  train = Data(std::move(train_ins_vec), data_info_);
  test = Data(std::move(test_ins_vec), data_info_);

  LOG(INFO) << "Finished splitting data set in " << timer;
}



std::unordered_map<size_t, std::vector<size_t>> 
Data::get_feature_ins_idx_hashtable(size_t feature_group_idx) const {

  CHECK_LT(feature_group_idx, num_feature_groups());
  std::vector<std::pair<size_t, size_t>> fg_idx_ins_id_pair_vec;
  fg_idx_ins_id_pair_vec.reserve(size());
  size_t idx = 0;
  size_t ft_idx;
  for(auto iter = begin(); iter != end(); ++iter) {
    CHECK_EQ(iter->feature_group_size(feature_group_idx), 1);
    ft_idx = iter->get_feature_group_index(feature_group_idx, 0) 
        + feature_group_start_idx(feature_group_idx);
    CHECK_GE(ft_idx, feature_group_start_idx(feature_group_idx));
    if (feature_group_idx < num_feature_groups() - 1) {
      CHECK_LT(ft_idx, feature_group_start_idx(feature_group_idx + 1));
    } else {
      CHECK_LT(ft_idx, total_dimensions());
    }
    fg_idx_ins_id_pair_vec.emplace_back(ft_idx, idx++); 
  }
  CHECK_EQ(idx, size());
  CHECK_EQ(fg_idx_ins_id_pair_vec.size(), size()); 

  std::sort(fg_idx_ins_id_pair_vec.begin(),
            fg_idx_ins_id_pair_vec.end());

  auto pair_vec_iter = fg_idx_ins_id_pair_vec.begin();
  auto pair_vec_internal_iter = fg_idx_ins_id_pair_vec.begin();
  auto pair_vec_iter_end = fg_idx_ins_id_pair_vec.end();
  size_t cnt = 0;

  std::unordered_map<size_t, std::vector<size_t>> rets;
  rets.reserve(feature_group_total_dimension(feature_group_idx));
  std::vector<size_t> tmp_vec;
  while (pair_vec_iter != pair_vec_iter_end) {
    pair_vec_internal_iter = pair_vec_iter;
    while(++pair_vec_internal_iter != pair_vec_iter_end) {
      if (pair_vec_iter->first != pair_vec_internal_iter->first) {
        break;
      } 
    }
    tmp_vec.resize(std::distance(pair_vec_iter, pair_vec_internal_iter));
    std::transform(pair_vec_iter, pair_vec_internal_iter,
                   tmp_vec.begin(),
                   [](const std::pair<size_t, size_t>& v) {
                   return v.second;
                   });
    CHECK(std::is_sorted(tmp_vec.begin(), tmp_vec.end()));
    rets[pair_vec_iter->first - feature_group_start_idx(feature_group_idx)] = std::move(tmp_vec);
    pair_vec_iter = pair_vec_internal_iter;
    ++cnt;
  }
  //CHECK_EQ(cnt, feature_group_total_dimension(feature_group_idx));
  return std::move(rets);
}


std::unordered_map<size_t, std::vector<size_t>> 
Data::get_feature_to_vec_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const {
  auto rets = get_feature_ins_idx_hashtable(feature_group_idx_a);
  std::vector<size_t> tmp_vec;
  for (auto outer_iter = rets.begin(); outer_iter != rets.end(); ++outer_iter) {
    tmp_vec.assign(outer_iter->second.size(), 0);
    size_t idx = 0;
    for (auto& v : outer_iter->second) {
      tmp_vec[idx++] = instances_[v].get_feature_group_index(feature_group_idx_b, 0);// + feature_group_start_idx(feature_group_idx_b);
    }
    std::sort(tmp_vec.begin(), tmp_vec.end());
    outer_iter->second = std::move(tmp_vec);
  }
  return std::move(rets);
}

std::unordered_map<size_t, std::unordered_set<size_t>> 
Data::get_feature_to_set_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const {
  std::unordered_map<size_t, std::unordered_set<size_t>> rets;
  std::unordered_set<size_t> tmp_set;
  
  auto feature_ins_table = get_feature_ins_idx_hashtable(feature_group_idx_a);
  rets.reserve(feature_ins_table.size());

  for (auto outer_iter = feature_ins_table.begin(); outer_iter != feature_ins_table.end(); ++outer_iter) {
    tmp_set.clear();
    tmp_set.reserve(outer_iter->second.size());
    for (auto& v : outer_iter->second) {
      tmp_set.insert(instances_[v].get_feature_group_index(feature_group_idx_b, 0));
    }
    rets[outer_iter->first] = std::move(tmp_set);
  }
  CHECK_EQ(rets.size(), feature_ins_table.size());
  return std::move(rets);
}

std::unordered_map<size_t, std::unordered_map<size_t, double>> 
Data::get_feature_pair_label_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const {
  auto feat_ins_hashtable = get_feature_ins_idx_hashtable(feature_group_idx_a);
  std::unordered_map<size_t, std::unordered_map<size_t, double>> rets; 
  rets.reserve(feat_ins_hashtable.size());
  std::unordered_map<size_t, double> tmp_map;
  for (auto outer_iter = feat_ins_hashtable.begin(); outer_iter != feat_ins_hashtable.end(); ++outer_iter) {
    tmp_map.clear();
    for (auto& v : outer_iter->second) {
      //tmp_vec[idx++] = std::make_pair(instances_[v].get_feature_group_index(feature_group_idx_b, 0) + feature_group_start_idx(feature_group_idx_b), instances_[v].label());
      tmp_map.insert(std::make_pair(instances_[v].get_feature_group_index(feature_group_idx_b, 0), instances_[v].label()));
    }
    rets[outer_iter->first] = std::move(tmp_map);
  }
  return std::move(rets);
}

} // namesapce 

