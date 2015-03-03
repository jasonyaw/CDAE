#include <iostream>
#include <numeric>
#include <algorithm>

#include <base/heap.hpp>
#include <base/utils.hpp>

#include "gtest/gtest.h"

TEST(heap, test_heap) {
  using namespace libcf;
  {
    std::vector<int>  v{10, 20, 30, 5, 15};
    Heap<int> h(v.begin(), v.end(), std::less<int>());
    EXPECT_EQ(h.front(), 30);
    EXPECT_EQ(h.size(), 5);

    //LOG(INFO) << h.get_data_copy();
    h.pop();
    EXPECT_EQ(h.front(), 20);
    EXPECT_EQ(h.size(), 4);

    //LOG(INFO) << h.get_data_copy();
    h.push(99);
    EXPECT_EQ(h.front(), 99);
    EXPECT_EQ(h.size(), 5);

    //LOG(INFO) << h.get_data_copy();
    h.push_and_pop(77);
    EXPECT_EQ(h.front(), 77);
    EXPECT_EQ(h.size(), 5);

    //LOG(INFO) << h.get_data_copy();
    h.push_and_pop(88);
    EXPECT_EQ(h.front(), 77);
    EXPECT_EQ(h.size(), 5);
  }
  {
    std::vector<std::pair<size_t, double>>  v{{10, 10.}, {20, 20.}, {30, 30.}, {5, 5.}, {15, 15.}};
    Heap<std::pair<size_t, double>> h(v.begin(), v.end(), sort_by_second_desc<size_t, double>);

    EXPECT_EQ(h.front().first, 5);
    EXPECT_EQ(h.size(), 5);
    
    //LOG(INFO) << h.get_data_copy();

    EXPECT_EQ(h.pop().first, 5);
    EXPECT_EQ(h.front().first, 10);
    EXPECT_EQ(h.size(), 4);

    //LOG(INFO) << h.get_data_copy();
    h.push({8, 8.});
    EXPECT_EQ(h.front().first, 8);
    EXPECT_EQ(h.size(), 5);

    //LOG(INFO) << h.get_data_copy();
    EXPECT_EQ(h.push_and_pop({7, 7.}).first, 7);
    EXPECT_EQ(h.front().first, 8);
    EXPECT_EQ(h.size(), 5);

    //LOG(INFO) << h.get_data_copy();
    EXPECT_EQ(h.push_and_pop({9, 9.}).first, 8);
    EXPECT_EQ(h.front().first, 9);
    EXPECT_EQ(h.size(), 5);
  }
  {
    Heap<std::pair<size_t, double>> h(sort_by_second_desc<size_t, double>);
    std::vector<std::pair<size_t, double>>  v{{10, 10.}, {20, 20.}, {30, 30.}, {5, 5.}, {15, 15.}};
    for (auto& p : v) {
      if (h.size() < 3) {
        h.push(p);
      } else {
        h.push_and_pop(p);
      }
    }
    EXPECT_EQ(h.size(), 3);
    auto sorted_copy = h.get_sorted_data_copy();
    EXPECT_EQ(sorted_copy[0].first, 30);
    EXPECT_EQ(sorted_copy[1].first, 20);
    EXPECT_EQ(sorted_copy[2].first, 15);
    EXPECT_EQ(h.pop().first, 15);
    EXPECT_EQ(h.pop().first, 20);
    EXPECT_EQ(h.pop().first, 30);
    EXPECT_EQ(h.size(), 0);
  }


}





