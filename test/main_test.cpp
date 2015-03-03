#include <glog/logging.h>
#include <gtest/gtest.h>

#include "parallel_test.hpp"
#include "mat_test.hpp"
#include "file_test.hpp"
#include "data_test.hpp"
#include "model_test.hpp"
#include "loss_test.hpp"
#include "heap_test.hpp"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
 
  FLAGS_log_dir = ".";
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
