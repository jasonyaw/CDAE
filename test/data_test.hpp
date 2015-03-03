#include <iostream>
#include <numeric>
#include <algorithm>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "gtest/gtest.h"

#include <base/utils.hpp>
#include <base/io.hpp>
#include <base/data.hpp>


TEST(dataset, test_recsys_data) {
  using namespace libcf;
  std::string sample_data("./test_data/sample_movielens_data.txt");
  
  auto line_parser = [&](const std::string& line) {
    auto rets = split_line(line, ": ");
    CHECK_EQ(rets.size(), 4);
    return std::vector<std::string>(std::make_move_iterator(rets.begin()),
                                    std::make_move_iterator(rets.begin() + 3));
  };

  Data data;
  data.load(sample_data, RECSYS, line_parser);
  save(data, "./test_data/sample_movielens_data.txt.bin");
  
  Data data1;
  load("./test_data/sample_movielens_data.txt.bin", data1);
  LOG(INFO) << data1;
  
  auto data_iter = data.begin();
  auto data_iter_end = data.end();
  size_t cnt = 0;
  for (; data_iter != data_iter_end; ++data_iter) {
    auto ins_iter = data.begin(cnt); 
    auto ins_iter_end = data.end(cnt); 
    size_t in_cnt = 0;
    for (; ins_iter != ins_iter_end; ++ins_iter) {
      //LOG(INFO) << ins_iter.index() << ": " << ins_iter.value();
       ++in_cnt;
    }
    EXPECT_EQ(in_cnt, 2);
    ++cnt;
  }
  EXPECT_EQ(cnt, 200);
  LOG(INFO) << "iterate over " << cnt << " instances.";

  LOG(INFO) << "random split : ";
  Data train, test;
  data.random_split_by_feature_group(train, test, 0, 0.3);
  data.random_split(train, test, 0.3);
  EXPECT_EQ(train.size(), data.size() * 0.7);
  EXPECT_EQ(test.size(), data.size() * 0.3);
  LOG(INFO) << train;
  LOG(INFO) << test;
}


