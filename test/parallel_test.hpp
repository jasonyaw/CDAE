#include <iostream>
#include <cassert>
#include <numeric>
#include <cstdlib>
#include <ctime>

#include "gtest/gtest.h"
#include "glog/logging.h"

#include <base/parallel.hpp>
#include <base/timer.hpp>
#include <base/utils.hpp>

TEST(test_parallel, num_threads) {
  LOG(INFO) << "Num of threads: " << libcf::num_hardware_threads();
}

TEST(test_parallel, lambda) {

  std::srand(time(NULL));

  std::vector<size_t> vec(100, 0);
  std::generate(vec.begin(), vec.end(), std::rand);
  std::vector<size_t> vec2(vec.begin(), vec.end());
  std::vector<size_t> vec3(vec.begin(), vec.end());

  auto plus1 = [] (size_t& x) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); x += 1; };
  auto time2 = [] (size_t& x) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); x *= 2; };

  libcf::time_function( [&]() {
                       std::for_each(vec.begin(), vec.end(), plus1);
                       std::for_each(vec.begin(), vec.end(), time2);
                       }, "single thread");

  libcf::time_function([&]() {
                       libcf::parallel_for(0, vec2.size(), [&](size_t x)  {std::this_thread::sleep_for(std::chrono::milliseconds(1)); vec2[x] += 1;});
                       libcf::parallel_for(0, vec2.size(), [&](size_t x)  {std::this_thread::sleep_for(std::chrono::milliseconds(1)); vec2[x] *= 2;});
                       }, "parallel_for");

  libcf::time_function([&]() {
                       libcf::parallel_for_each(vec3.begin(), vec3.end(), plus1);
                       libcf::parallel_for_each(vec3.begin(), vec3.end(), time2);
                       }, "parallel_for_each");

  for (size_t idx = 0; idx < vec.size(); ++idx){
    EXPECT_EQ(vec[idx], vec2[idx]);
    EXPECT_EQ(vec[idx], vec3[idx]);
  }

  vec.resize(1e6, 0);
  std::generate(vec.begin(), vec.end(), [] () {return std::rand() % 20;} );

  size_t single_thread_total_counts = 0; 
  libcf::time_function([&]() {
                       single_thread_total_counts = std::accumulate(vec.begin(), vec.end(), 0);
                       }, "single thread!");

  size_t parallel_for_total_counts = 0;
  libcf::time_function( [&]() {
                       size_t num_threads = std::thread::hardware_concurrency();
                       std::vector<size_t> stat_vec(num_threads, 0);
                       libcf::in_parallel([&](size_t thread_id, size_t num_threads){
                                          size_t begin = thread_id * vec.size() / num_threads;
                                          size_t end = (thread_id + 1) * vec.size() / num_threads;
                                          stat_vec[thread_id] = std::accumulate(vec.begin() + begin, vec.begin() + end, 0);
                                          });
                       parallel_for_total_counts = std::accumulate(stat_vec.begin(), stat_vec.end(), size_t(0));
                       }, "in_parallel");
  EXPECT_EQ(single_thread_total_counts, parallel_for_total_counts);

  size_t parallel_accumulate_reduce_total_counts = 0; 

  libcf::time_function([&]() {
                       parallel_accumulate_reduce_total_counts = 
                       libcf::parallel_accumulate_and_reduce<size_t>(0, vec.size(), 0, 
                                                                     [&](size_t& ret, size_t idx){
                                                                     ret += vec[idx];
                                                                     }, 0, [] (size_t& a, size_t b) {
                                                                     a += b;
                                                                     }
                                                                    );
                       }, "parallel_accumulate_and_reduce");

  EXPECT_EQ(single_thread_total_counts, parallel_accumulate_reduce_total_counts);

  size_t parallel_accumulate_total_counts = 0; 

  libcf::time_function( [&]() {
                       std::vector<size_t> partial_counts =
                       libcf::parallel_accumulate<size_t>(0, vec.size(), 0,
                                                          [&](size_t idx, size_t& ret){
                                                          ret += vec[idx];
                                                          });
                       parallel_accumulate_total_counts = std::accumulate(partial_counts.begin(), partial_counts.end(), 0, std::plus<size_t>());
                       }, "parallel_accumulate");

  EXPECT_EQ(single_thread_total_counts, parallel_accumulate_total_counts);

}

TEST(test_parallel, test_thread_pool) {  

  std::vector<size_t> vec(100, 0);
  std::generate(vec.begin(), vec.end(), std::rand);
  std::iota(vec.begin(), vec.end(), 0);
  std::vector<size_t> vec2(vec.begin(), vec.end());
  std::vector<size_t> vec3(vec.begin(), vec.end());
  std::vector<size_t> vec4(vec.begin(), vec.end());

  auto f = [&] (size_t& x) { 
    std::this_thread::sleep_for(std::chrono::milliseconds(x/10)); 
    x += 1; };

    libcf::time_function([&]() {
                         std::for_each(vec.begin(), vec.end(), f);                     
                         }, "serial for_each");

    libcf::time_function([&](){
                         libcf::ThreadPool tp;
                         for (auto& v : vec2) {
                         tp.add([&]() {f(v);});
                         }
                         tp.run();
                         }, "thread_pool");

    libcf::time_function([&]() {
                         libcf::parallel_for_each(vec3.begin(), vec3.end(), f);                     
                         }, "serial parallel_for_each");

       libcf::time_function([&]() {
       libcf::dynamic_parallel_for_each(vec4.begin(),
       vec4.end(), f);
       }, "dynamic_parallel_for_each");
    for (size_t idx = 0; idx < vec.size(); idx++) {
      EXPECT_EQ(vec[idx], vec2[idx]);
      EXPECT_EQ(vec[idx], vec3[idx]);
      EXPECT_EQ(vec[idx], vec4[idx]);
    }
}

TEST(test_parallel, test_more_thread_pool) {

  std::vector<std::vector<int>> mat(100);
  auto f = [] () { return std::rand() % 100;};
  for(size_t idx = 0; idx < mat.size(); idx++){
    mat[idx].resize(idx + 1);
    std::generate(mat[idx].begin(), mat[idx].end(), f);
  }

  std::vector<int> partial_sum(mat.size());
  std::vector<int> static_partial_sum(mat.size());
  std::vector<int> dynamic_partial_sum(mat.size());

  //libcf::time_function([&]() {
  for(size_t idx = 0; idx < mat.size(); idx++) {
    partial_sum[idx] = std::accumulate(mat[idx].begin(),
                                       mat[idx].end(), size_t(0));
  }
  //                     }, "serial partial_sum");

  //libcf::time_function([&](){
  libcf::parallel_for(0, mat.size(), [&](size_t idx){
                      static_partial_sum[idx] = std::accumulate(mat[idx].begin(), mat[idx].end(), 0);                    
                      });
  //                     }, "static_partial_sum");

  auto f1 = [&] (size_t idx) {
    dynamic_partial_sum[idx] = std::accumulate(mat[idx].begin(), mat[idx].end(), size_t(0)); 
  };

  //libcf::time_function([&](){
  libcf::ThreadPool tp;
  for (size_t idx = 0; idx < mat.size(); ++idx){
    tp.add(std::bind(f1, idx));
  }
  tp.run();
  //                     }, "dynamic_partial_sum");

  for (size_t idx = 0; idx < mat.size(); ++idx){
    EXPECT_EQ(static_partial_sum[idx], partial_sum[idx]);
    EXPECT_EQ(partial_sum[idx], dynamic_partial_sum[idx]);
  }
}

