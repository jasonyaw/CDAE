#include <iostream>
#include <numeric>
#include <algorithm>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "gtest/gtest.h"

#include <base/io.hpp>
#include <base/utils.hpp>
#include <base/data.hpp>

TEST(file, test_line_split) {
  
  std::string test_line = "12%&123124#$%&*,asdj#lwei#$";
  auto rets = libcf::split_line(test_line, "#$");
  std::vector<std::string> truth{"12%&123124", "%&*,asdj", "lwei"};
  EXPECT_EQ(rets.size(), truth.size());
  for(size_t idx = 0; idx < rets.size(); idx++) {
    EXPECT_TRUE(rets[idx] == truth[idx]);
  }
}

TEST(file, test_file_line_reader) {
  
  libcf::File test_out("test_data/test_flr.txt", "wb");
  test_out.write_line("1");
  test_out.write_line("2");
  test_out.write_line("3");
  test_out.write_line("4");
  test_out.write("5");
  test_out.close();
  
  std::vector<std::string> rets;
  libcf::FileLineReader flr("test_data/test_flr.txt");
  flr.set_line_callback([&](const std::string& line, size_t line_num){
                        if(line_num % 2 == 0) 
                          rets.push_back(line);
                        });
  flr.load();
  std::vector<std::string> truth{"1", "3", "5"};
  EXPECT_EQ(rets.size(), truth.size());
  for(size_t idx = 0; idx < rets.size(); idx++) {
    //EXPECT_TRUE(rets[idx] == truth[idx]);
  }

}

TEST(file, test_file_open) {

  std::vector<int> vec(10);
  std::iota(vec.begin(), vec.end(), 0);
  libcf::File test_if("test_data/test_if.txt", "wb");
  test_if.write_vector<int>(vec);  
  test_if.write_line("hello world");
  test_if.write("test world");
  test_if.close();

  libcf::File test_of("test_data/test_if.txt", "rb");
  std::string line;
  std::vector<int> vec1;
  test_of.read_vector<int>(vec1);
  EXPECT_EQ(vec1.size(), vec.size());
  for(size_t idx = 0; idx < vec.size(); idx++){
    EXPECT_EQ(vec[idx], vec1[idx]);
  }
  std::vector<std::string> rets;
  while (test_of.good()){
    line = test_of.read_line();
    rets.push_back(line);
  }
  test_of.close();
  EXPECT_TRUE(rets[0] == "hello world");
  EXPECT_TRUE(rets[1] == "test world");
}

TEST(data, test_recsys_data) {
  libcf::File test_out("test_data/test_recsys_data.txt", "w");
  test_out.write_line("1 2 3");
  test_out.write_line("2 4 7");
  test_out.write_line("2 2 4");
  test_out.write_line("4 5 7");
  test_out.write("5 3 66");
  test_out.close();
  
  libcf::Data data_set;
  
  auto line_parser = [&](const std::string& line) {
    auto rets = libcf::split_line(line, " ");
    EXPECT_EQ(rets.size(), 3);
    return rets;
  };

  data_set.load("test_data/test_recsys_data.txt", libcf::RECSYS, line_parser);
  std::stringstream ss;
  boost::archive::text_oarchive oa(ss);  
  oa << data_set;
  
  libcf::Data data_set1;
  boost::archive::text_iarchive ia(ss);
  ia >> data_set1;
  
  LOG(INFO) << data_set1;
  auto u_idx_map = data_set.get_feature_ins_idx_hashtable(0);
  auto u_item_map = data_set.get_feature_to_vec_hashtable(0, 1);
  auto u_item_rating_map = data_set.get_feature_pair_label_hashtable(0, 1); 
  {
    std::stringstream ss;
    for (auto& p : u_idx_map) {
      ss << "(" << p.first << ",[";
      for (auto& v : p.second) {
        ss << v << ",";
      }
      ss << "])\n";
    }
    LOG(INFO) << ss.str();
  }
   
  {
    std::stringstream ss;
    for (auto& p : u_item_map) {
      ss << "(" << p.first << ",[";
      for (auto& v : p.second) {
        ss << v << ",";
      }
      ss << "])\n";
    }
    LOG(INFO) << ss.str();
  }
  
  {
    std::stringstream ss;
    for (auto& p : u_item_rating_map) {
      ss << "(" << p.first << ",[";
      for (auto& v : p.second) {
        ss << "(" << v.first << "," << v.second << "),";
      }
      ss << "])\n";
    }
    LOG(INFO) << ss.str();
  }

}

TEST(file, test_config) {

  std::map<std::string, std::string> opts;
  opts.emplace("a", "1");
  opts.emplace("b", "2");
  opts.emplace("c", "3");
  opts.emplace("d", "4");
  opts.emplace("e", "5");
  
  libcf::write_config_file(opts, "test_data/test_config.txt");
    
  auto opts_out = libcf::read_config_file("test_data/test_config.txt");
  
  for(auto& p : opts_out) {
    EXPECT_EQ(opts[p.first], p.second);
  }
}


