#ifndef _LIBCF_INSTANCE_HPP_
#define _LIBCF_INSTANCE_HPP_

#include <vector>
#include <unordered_map>
#include <iterator>
#include <algorithm>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <base/io.hpp>
#include <base/utils.hpp>

namespace libcf {

enum LabelType {
  EMPTY = 0,
  BINARY,
  MULTICLASS,
  CONTINUOUS
};

enum FeatureType {
  DENSE = 0,
  SPARSE,
  SPARSE_BINARY
};

/**
 *  Feature group information
 */
class FeatureGroupInfo {

  /////////////////////////////////////////////////////////////////
  // Boost serialization
  // 
  friend class boost::serialization::access;
  template<class Archive>
      void save(Archive& ar, const unsigned int version) const {
        ar & length_;
        ar & feat_type_;
        ar & raw_str_map_;
        std::vector<std::pair<std::string, size_t>> data(
            idx_map_.begin(), idx_map_.end());
        ar & data;
      }

  template<class Archive>
      void load(Archive& ar, const unsigned int version) {
        ar & length_;
        ar & feat_type_;
        ar & raw_str_map_;
        std::vector<std::pair<std::string, size_t>> data;
        ar & data;
        idx_map_ = std::unordered_map<std::string, size_t>(
            data.begin(), data.end());
      }

  template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        boost::serialization::split_member(ar, *this, version);
      }

  friend std::ostream& operator<< (std::ostream& stream, 
                                   const FeatureGroupInfo& fg_info);

 public:
  FeatureGroupInfo() = default;
  FeatureGroupInfo(const FeatureGroupInfo&) = default;
  FeatureGroupInfo(FeatureGroupInfo &&) = default;
  FeatureGroupInfo& operator= (const FeatureGroupInfo&) = default; 

  explicit FeatureGroupInfo(const FeatureType& ft) 
      : feat_type_(ft) {}

  size_t get_index(const std::string& key, 
                   bool allow_new_value = true);

  size_t size() const;

  void set_length(size_t length) { length_ = length; }

  FeatureType feature_type() const { return feat_type_; }

 private:

  std::unordered_map<std::string, size_t> idx_map_;
  std::vector<std::string> raw_str_map_;
  size_t length_ = 0;
  enum FeatureType feat_type_;
};

class FeatureGroup {

  friend class boost::serialization::access;
  template<class Archive> 
      void serialize(Archive& ar, const unsigned int version) {
        ar & ft_;
        ar & feat_ids;
        ar & feat_vals;
      }

  friend std::ostream& operator<< (std::ostream& stream, 
                                   const FeatureGroup& fg);
 public:
  FeatureGroup() = default;
  FeatureGroup(const FeatureGroup&) = default;
  FeatureGroup(FeatureGroup&&) = default;
  FeatureGroup& operator=(const FeatureGroup&) = default;

  FeatureGroup(FeatureGroupInfo& fg_info, const std::string& str);
  FeatureGroup(FeatureGroupInfo& fg_info, const std::vector<double>& vec);
  FeatureGroup(FeatureGroupInfo& fg_info, const std::vector<size_t>& vec);
  FeatureGroup(FeatureGroupInfo& fg_info, const std::vector<std::pair<size_t, double>>& vec);

  size_t size() const; 

  size_t index(size_t idx) const;
  double value(size_t idx) const;

 private:
  FeatureType ft_;
  std::vector<size_t> feat_ids;
  std::vector<double> feat_vals;
};

class Instance {

  friend class boost::serialization::access;
  template<class Archive> 
      void serialize(Archive& ar, const unsigned int version) {
        ar & feat_groups_;
        ar & label_;
        ar & size_;
      }

  friend std::ostream& operator<< (std::ostream& stream,
                                   const Instance& ins);
 public:

  Instance() = default;
  Instance (const Instance&) = default;
  Instance (Instance&&) = default;
  Instance& operator=(const Instance&) = default;

  void add_feat_group(FeatureGroupInfo& fg_info,
                      const std::string& str) {
    feat_groups_.push_back(FeatureGroup(fg_info, str));   
    size_ += feat_groups_.back().size();
  }

  void add_feat_group(const std::vector<double>& vec) {
    FeatureGroupInfo fg_info(DENSE); 
    feat_groups_.push_back(FeatureGroup(fg_info, vec));   
    size_ += feat_groups_.back().size();
  }

  void add_feat_group(const std::vector<size_t>& vec) {
    FeatureGroupInfo fg_info(SPARSE_BINARY); 
    feat_groups_.push_back(FeatureGroup(fg_info, vec));   
    size_ += feat_groups_.back().size();
  }

  void add_feat_group(const std::vector<std::pair<size_t, double>>& vec) {
    FeatureGroupInfo fg_info(SPARSE); 
    feat_groups_.push_back(FeatureGroup(fg_info, vec));   
    size_ += feat_groups_.back().size();
  }

  void add_feat_group(FeatureGroupInfo& fg_info,
                      const std::vector<double>& vec) {
    feat_groups_.push_back(FeatureGroup(fg_info, vec));   
    size_ += feat_groups_.back().size();
  }

  void add_feat_group(FeatureGroupInfo& fg_info,
                      const std::vector<size_t>& vec) {
    feat_groups_.push_back(FeatureGroup(fg_info, vec));   
    size_ += feat_groups_.back().size();
  }

  void add_feat_group(FeatureGroupInfo& fg_info,
                      const std::vector<std::pair<size_t, double>>& vec) {
    feat_groups_.push_back(FeatureGroup(fg_info, vec));   
    size_ += feat_groups_.back().size();
  }

  //size_t get_id() const { return Instance_id_; }
  //void set_id(size_t ins_id) { Instance_id_ = ins_id; }

  double label() const { return label_; }
  void set_label(double label) { label_ = label; }

  friend void swap(Instance& a, Instance& b) {
    std::swap(a.feat_groups_, b.feat_groups_);
    std::swap(a.label_, b.label_);
    std::swap(a.size_, b.size_);
  }

  size_t size() const { return size_; }

  size_t num_feature_groups() const {
    return feat_groups_.size();
  }

  size_t feature_group_size(size_t fg_idx) const {
    return feat_groups_[fg_idx].size();
  }

  size_t get_feature_group_index(size_t fg_idx, size_t idx) const {
    return feat_groups_[fg_idx].index(idx);
  }

  double get_feature_group_value(size_t fg_idx, size_t idx) const {
    return feat_groups_[fg_idx].value(idx);
  }

 private:

  std::vector<FeatureGroup> feat_groups_;
  double label_ = 0;
  size_t size_ = 0;
  //size_t Instance_id_;
};

} // namespace

#include<base/instance-inl.hpp>

#endif // _LIBCF_INSTANCE_HPP_
