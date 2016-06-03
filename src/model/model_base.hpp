#ifndef _LIBCF_MODEL_BASE_HPP_
#define _LIBCF_MODEL_BASE_HPP_

#include <unordered_set>

#include <base/mat.hpp>
#include <base/data.hpp>
#include <base/heap.hpp>
#include <model/loss.hpp>
#include <model/penalty.hpp>


namespace libcf {

/** Model Base
 */
class ModelBase {
 public:
  ModelBase() = default;
  
  /** Reset the model parameters 
   */
  virtual void reset(const Data& data_set) {
    data_ = &data_set;
  }

  /** Current Loss
   */
  virtual double current_loss(const Data& data_set, 
                              size_t sample_size=0) const {
    return data_loss(data_set, sample_size) + penalty_loss();
  }

  /** Prediction error on training data
   */
  virtual double data_loss(const Data& data_set, 
                           size_t sample_size=0) const {
    return 0.0;
  }
  
  /** Regularization Loss
   */
  virtual double penalty_loss() const {
    return 0.0; 
  }
  
  // required by evaluation measures RMSE/MAE
  virtual double predict(const Instance& ins) const {
    LOG(FATAL) << "Unimplemented!";
    return 0.;
  }
 
  virtual double regularization_coefficent() const {
    return 0.;
  }
  
  // required for SolverBase
  virtual void train_one_iteration(const Data& train_data) {
    LOG(FATAL) << "Unimplemented!";
  }

 protected:
  const Data* data_ = nullptr;
  std::shared_ptr<Loss> loss_ = nullptr;
  std::shared_ptr<Penalty> penalty_ = nullptr;  
};

// required for SGD solver
class SGDBase {
  virtual void update_one_sgd_step(const Instance& ins, double step_size) {
    LOG(FATAL) << "update_one_sgd_step not implemented!";
  }
};

//required for ALS solver

class ALSBase {
  virtual void train_one_index(size_t idx, 
                       const std::vector<std::pair<size_t, double>>& index_vec,
                       const DMatrix& Y, DMatrix& X) {
    LOG(FATAL) << "train_one_index not implemented!";
  }
                  
};

}


#endif // _LIBCF_MODEL_BASE_HPP_
