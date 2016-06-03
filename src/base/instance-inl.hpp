#include <base/instance.hpp>

namespace libcf {

std::ostream& operator<< (std::ostream& stream, 
                          const FeatureGroupInfo& fg_info) {
  switch (fg_info.feat_type_) {
    case DENSE : 
      stream << "{type: DENSE}, {size: " << fg_info.size() << "}";
      break;
    case SPARSE_BINARY:
    case SPARSE : 
      stream << "{type : SPARSE_BINARY}, {size: " << fg_info.size() << "}";
      break;
    default:
      LOG(FATAL) << "Undefined Feature Type";
  }
  return stream;
}


size_t FeatureGroupInfo::get_index(const std::string& key, 
                                   bool allow_new_value) {
  auto fit = idx_map_.find(key);
  if(fit != idx_map_.end()) {
    return fit->second;
  } else {
    if (allow_new_value) {
      size_t idx = idx_map_.size();
      idx_map_[key] = idx;
      raw_str_map_.push_back(key);
      return idx;
    } else {
      return size_t(-1);
    }
  }
}

size_t FeatureGroupInfo::size() const {
  if (feat_type_ == DENSE) {
    return length_;
  }
  return idx_map_.size();
}


FeatureGroup::FeatureGroup(FeatureGroupInfo& fg_info,
                           const std::string& str) {

  ft_ = fg_info.feature_type();
  std::vector<std::string> values = split_line(str, " ");
  if (fg_info.feature_type() == DENSE) {
    // format: 1 2 4 5 6 7 
    feat_vals.resize(values.size());
    if (fg_info.size() == 0) {
      fg_info.set_length(values.size());
    } else {
      CHECK_EQ(fg_info.size(), values.size());
    }

    std::transform(values.begin(), 
                   values.end(),
                   feat_vals.begin(),
                   [](const std::string& str){
                   return std::stod(str);
                   });
  } else if (fg_info.feature_type() == SPARSE_BINARY) {
    // format: 1 2 4 5 6 7 
    feat_ids.resize(values.size());
    std::transform(values.begin(), 
                   values.end(),
                   feat_ids.begin(),
                   [&](const std::string& str){
                   return fg_info.get_index(str, true);
                   }
                  );

  } else {
    // format, 1:2 2:3 3:4
    feat_ids.resize(values.size());
    feat_vals.resize(values.size());
    size_t idx = 0;
    for (auto& p : values) {
      std::vector<std::string> kv_pairs = split_line(p, ":");
      CHECK_EQ(kv_pairs.size(), 2);
      feat_ids[idx] = fg_info.get_index(kv_pairs[0], true);
      feat_vals[idx] = std::stod(kv_pairs[1]);
      ++idx;
    }
  }
}

FeatureGroup::FeatureGroup(FeatureGroupInfo& fg_info, const std::vector<double>& vec) {
  CHECK_EQ(fg_info.feature_type(), DENSE);
  if (fg_info.size() == 0) {
    fg_info.set_length(vec.size());
  } else {
    CHECK_EQ(fg_info.size(), vec.size());
  }
  ft_ = DENSE;
  feat_vals.assign(vec.begin(), vec.end());
}

FeatureGroup::FeatureGroup(FeatureGroupInfo& fg_info, const std::vector<size_t>& vec) {
  CHECK_EQ(fg_info.feature_type(), SPARSE_BINARY);
  ft_ = SPARSE_BINARY;
  feat_ids.assign(vec.begin(), vec.end());
}

FeatureGroup::FeatureGroup(FeatureGroupInfo& fg_info, 
                           const std::vector<std::pair<size_t, double>>& vec) {
  CHECK_EQ(fg_info.feature_type(), SPARSE);
  ft_ = SPARSE;
  feat_ids.resize(vec.size());
  feat_vals.resize(vec.size());
  for (size_t idx = 0; idx < vec.size(); ++idx) {
    feat_ids[idx] = vec[idx].first;
    feat_vals[idx] = vec[idx].second;
  }
}



size_t FeatureGroup::size() const { 
  if (ft_ == DENSE) { 
    return feat_vals.size();
  } else if (ft_ == SPARSE_BINARY) {
    return feat_ids.size();
  } else {
    return feat_ids.size();
  }
}

std::ostream& operator<< (std::ostream& stream, 
                          const FeatureGroup& fg) {

  if (fg.ft_ == DENSE) {
    stream << "[";
    for (size_t idx = 0; idx < fg.feat_vals.size(); ++idx) {
      if (idx > 0) stream << " ";
      stream << fg.feat_vals[idx];
    }
    stream << "]";
  } else if (fg.ft_ == SPARSE_BINARY) {
    stream << "[";
    for (size_t idx = 0; idx < fg.feat_ids.size(); ++idx) {
      if (idx > 0) stream << " ";
      stream << "(" << fg.feat_ids[idx] << ":1" << ")"; 
    }
    stream << "]";
  } else {
    stream << "[";
    for (size_t idx = 0; idx < fg.feat_ids.size(); ++idx) {
      if (idx > 0) stream << " ";
      stream << "(" << fg.feat_ids[idx] << ":"<< fg.feat_vals[idx] << ")"; 
    }
    stream << "]";
  }
  return stream;
}

size_t FeatureGroup::index(size_t idx) const {
  CHECK_LT(idx, size());
  if (ft_ == DENSE) { 
    return idx;
  } else if (ft_ == SPARSE_BINARY) {
    return feat_ids[idx];
  } else {
    return feat_ids[idx];
  }
}


double FeatureGroup::value(size_t idx) const {
  CHECK_LT(idx, size());
  if (ft_ == DENSE) { 
    return feat_vals[idx];
  } else if (ft_ == SPARSE_BINARY) {
    return 1.;
  } else {
    return feat_vals[idx];
  }
}


std::ostream& operator<< (std::ostream& stream,
                          const Instance& ins) {
  size_t fg_idx = 0;
  stream << "{Label: " << ins.label_ << "}, " << "{Feature Groups: ["; 
  for(auto& fg : ins.feat_groups_) {
    stream << "{" << fg_idx << ": " << fg << "}";
    if (fg_idx < ins.feat_groups_.size() - 1) stream << ", ";
    ++fg_idx;
  }
  stream << "]";
  return stream;
}


}
