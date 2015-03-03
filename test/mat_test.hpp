#include <iostream>
#include <numeric>
#include <algorithm>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <base/utils.hpp>
#include <base/mat.hpp>
#include <base/mat_io.hpp>
#include <base/data.hpp>

#include "gtest/gtest.h"

TEST(mat, io) {

  libcf::IMatrix test_int_mat;
  test_int_mat.resize(3,4);
  test_int_mat << 1, 2, 3, 10,
               4, 5, 6, 11,
               7, 8, 9, 12;
  std::stringstream ss;
  {
    boost::archive::text_oarchive oa(ss);
    oa << test_int_mat;
  }
  {
    boost::archive::text_iarchive ia(ss);
    libcf::IMatrix test_int_mat_out;
    ia >> test_int_mat_out;
    for (size_t i = 0; i < 3; i++) 
      for (size_t j = 0; j < 4; j++) 
        EXPECT_EQ(test_int_mat(i,j), test_int_mat_out(i,j));
  }
}

TEST(mat, compuation) {
  using namespace libcf;

  DMatrix mat = DMatrix::Random(10, 10);
  DVector vec = DVector::Ones(10);
  auto r1 = vec.cwiseQuotient(mat.row(1).transpose());
  
  for (size_t idx = 0; idx < 10; idx++) {
    EXPECT_EQ(r1(idx), vec(idx) / mat(1, idx));
  }
  
  mat.row(2) += vec.cwiseProduct(vec);

  DVector r2 = vec + mat.row(3).transpose();

}



