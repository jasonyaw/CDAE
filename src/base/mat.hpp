#ifndef _LIBCF_MAT_HPP_
#define _LIBCF_MAT_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace libcf {

template<typename T> using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T> using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

typedef ColVector<double> DVector;
typedef RowVector<double> DRowVector;
typedef ColVector<int>    IVector;
typedef RowVector<int>    IRowVector;
typedef ColVector<size_t> SVector;
typedef RowVector<size_t> SRowVector;

template<typename T> using Matrix 
    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Matrix<double> DMatrix;
typedef Matrix<int>    IMatrix;
typedef Matrix<size_t> SMatrix;

// sparse row matrix
template<typename T> using SRMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;
template<typename T> using SCMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor>;

template<typename T> using SRVector = Eigen::SparseVector<T, Eigen::RowMajor>;
template<typename T> using SCVector = Eigen::SparseVector<T, Eigen::ColMajor>;

typedef SRMatrix<double>  DSRMatrix;
typedef SCMatrix<double>  DSCMatrix;
typedef SRMatrix<int>     ISRMatrix;
typedef SCMatrix<int>     ISCMatrix;
typedef SRMatrix<bool>    BSRMatrix;
typedef SCMatrix<bool>    BSCMatrix;

typedef SRVector<double>  DSRVector;

} // namespace

#endif // _LIBCF_MAT_HPP_
