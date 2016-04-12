#ifndef _LIBCF_DATA_HPP_
#define _LIBCF_DATA_HPP_

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include <boost/serialization/vector.hpp>

#include <base/mat.hpp>
#include <base/instance.hpp>

namespace libcf {

enum DataFormat {
  VECTOR,
  LIBSVM,
  RECSYS
};

class Data;

class DataInfo { 
  
  friend class boost::serialization::access;
  template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar & feature_group_infos_;
        ar & total_dimensions_;
        ar & feature_group_global_idx_;
        ar & label_info_;
        ar & label_type_;
      }

 public:
  DataInfo() {}
  
  DataInfo(const DataInfo& oth) = default;
  DataInfo(DataInfo&&) = default;
  
  DataInfo(DataInfo* oth) : DataInfo(*oth) {}

  std::vector<FeatureGroupInfo> feature_group_infos_;
  size_t total_dimensions_ = 0;
  std::vector<size_t> feature_group_global_idx_;
  FeatureGroupInfo label_info_;
  enum LabelType label_type_ = CONTINUOUS;
};

class Data {
  
  friend class boost::serialization::access;
  template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar & instances_;
        if (data_info_ == nullptr) {
          data_info_ = std::make_shared<DataInfo>(new DataInfo());
        }
        ar & *data_info_;      
      }

  friend std::ostream& operator<< (std::ostream& stream, 
                                   const Data& data);
 public:
  
  Data() = default;
  Data(const Data&) = default;

  Data(const std::vector<Instance>& ins_vec,
       const std::shared_ptr<DataInfo>& data_info) :
      instances_(ins_vec), data_info_(data_info) {}
  
  Data(std::vector<Instance>&& ins_vec,
       const std::shared_ptr<DataInfo>& data_info) :
      instances_(std::move(ins_vec)), data_info_(data_info) {}

  Data(const std::shared_ptr<DataInfo>& data_info) : data_info_(data_info) {}

  Data& operator= (const Data&) = default;

  typedef std::function<std::vector<std::string> (const std::string&)> LineParser;
  void load(const std::string& filename, 
            const DataFormat& df, 
            const LineParser& parser,
            bool skip_header = false);

  void set_label_type(const LabelType& lt) {
    //CHECK_EQ(lt, CONTINUOUS);
    data_info_->label_info_ = FeatureGroupInfo(DENSE);
  }

  void add_feature_group(const FeatureType& ft) {
    data_info_->feature_group_infos_.push_back(FeatureGroupInfo(ft));  
  }

  template<class Func>
      void add_line_to_instance(const std::string& line,
                                const Func& f) {
        Instance ins = f(line);
        instances_.push_back(std::move(ins));
      }

  size_t size() const { return instances_.size(); }

  size_t num_feature_groups() const {
    CHECK(data_info_ != nullptr);
    return data_info_->feature_group_infos_.size();
  }

  size_t total_dimensions() const {
    CHECK(data_info_ != nullptr);
    return data_info_->total_dimensions_;
  }
  
  size_t feature_group_total_dimension(size_t fg_idx) const {
    CHECK_LT(fg_idx, num_feature_groups());
    return data_info_->feature_group_infos_[fg_idx].size();
  }

  size_t feature_group_start_idx(size_t fg_idx) const {
    CHECK(data_info_ != nullptr);
    return data_info_->feature_group_global_idx_[fg_idx];
  }
 
  std::shared_ptr<DataInfo> get_data_info() const {
    return std::shared_ptr<DataInfo>(data_info_);
  }

  // iterator
  const Instance* data() const { return instances_.data(); }
  Instance* data() { return instances_.data(); }
  
  Instance* begin() { return data(); } 
  const Instance* begin() const { return data(); } 
  
  Instance* end() { return data() + size(); }
  const Instance* end() const { return data() + size(); }

  void shuffle_data();
  
  class instance_iterator;
 
  // iterator for the idx-th instance
  instance_iterator begin(size_t idx) const;
  instance_iterator end(size_t idx) const;

  // return instance iterator for 
  instance_iterator begin(const Instance& ins) const;
  instance_iterator end(const Instance& ins) const;

  void random_split(Data& train, Data& test,
                    double test_ratio = 0.2) const; 

  void random_split_by_feature_group(Data& train, Data& test,
                                     size_t feature_group_idx, 
                                     double test_ratio) const;
  
  void inplace_random_split_by_feature_group(Data& train, Data& test,
                                     size_t feature_group_idx, 
                                     double test_ratio);

  std::unordered_map<size_t, std::vector<size_t>> 
      get_feature_ins_idx_hashtable(size_t feature_group_idx) const; 

  std::unordered_map<size_t, std::vector<size_t>> 
      get_feature_to_vec_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const;
  
  std::unordered_map<size_t, std::unordered_set<size_t>> 
      get_feature_to_set_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const;

  std::unordered_map<size_t, std::unordered_map<size_t, double>> 
      get_feature_pair_label_hashtable(size_t feature_group_idx_a, 
                                       size_t feature_group_idx_b) const;


 private:
  std::vector<Instance> instances_;
  std::shared_ptr<DataInfo> data_info_ = nullptr;
};

} // namespace

#include <base/data-inl.hpp>

#endif // _LIBCF_DATA_HPP_
