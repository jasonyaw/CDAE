#ifndef _LIBCF_SOLVER_HPP_
#define _LIBCF_SOLVER_HPP_

#include <memory>

#include <base/data.hpp>
#include <model/evaluation.hpp>

namespace libcf {

template<class Model>
class Solver {
 public:
  
  Solver(Model& model, size_t max_iteration, size_t eval_iterations=1) :
      max_iteration_(max_iteration), eval_iterations(eval_iterations),
      model_(std::make_shared<Model>(model))
  {}

  Solver(Model& model) : Solver(model, 1)
  {}
  
  std::shared_ptr<Model> get_model() {
    return model_;
  }

  virtual void pre_train(const Data&, const Data&) {
    // do nothing 
  }

  virtual void train_one_iteration(const Data& train_data) {
    model_->train_one_iteration(train_data); 
  }; 

  virtual void train(const Data& train_data, 
             const Data& validation_data = Data(),
             const std::vector<EvalType>& eval_types = {});

  virtual void test(const Data& test_data,
            const std::vector<EvalType>& eval_types = {});

 protected:
  size_t max_iteration_ = 1;
  size_t eval_iterations = 1;
  std::shared_ptr<Model> model_;
};

} // namespace 

#include <solver/solver-inl.hpp>

#endif // _LIBCF_SOLVER_HPP_
