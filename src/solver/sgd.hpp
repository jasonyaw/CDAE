#ifndef _LIBCF_SGD_HPP_
#define _LIBCF_SGD_HPP_

#include <memory>

#include <base/data.hpp>
#include <base/mat.hpp>
#include <model/evaluation.hpp>
#include <solver/solver.hpp>

namespace libcf {

struct SGDConfig {
  SGDConfig() = default;
  SGDConfig(const SGDConfig& oth) = default;
  
  size_t max_iteration = 50;
  double learn_rate = 0.1;
  bool automatic_intialization = false; 
  bool update_learn_rate = false;
};


template <class Model>
class SGD : public Solver<Model> {

 public:

  SGD(Model& model, SGDConfig& scfg) :
      Solver<Model>(model, scfg.max_iteration), 
      config_(std::make_shared<SGDConfig>(scfg)) 
  {
       LOG(INFO) << "SGD Configure: \n" 
        << "\t{max_iters: " << config_->max_iteration << "}, "
        << "{learn_rate: " << config_->learn_rate << "}\n\t"
        << "{auto_init: " << config_->automatic_intialization << "}, "
        << "{update_rate: " << config_->update_learn_rate << "}";
  }
  
  void pre_train(const Data& train_data,
                 const Data& validataion_data);

  virtual void train_one_iteration(const Data& train_data) {
    update(train_data);
  }

 protected:
  /** initialize the step_size using a small set of data
  */
  void initialize(const Data& data_set,
                  const Data& validation_set,
                  size_t sample_size = 10000);

  /** Update the model using data_set.
   *  If sample_size = 0, it uses all data in the data_set.
   *  Otherwise, it only use the first <sample_size> data points.
   */
  void update(const Data& data_set, size_t sample_size = 0);

  void update_with_one_instance(const Instance& ins);
  void update_learn_rate();

 private:
  std::shared_ptr<SGDConfig> config_;

 private:
  double initialized_learn_rate_;
  double learn_rate_;
  size_t steps_ = 0;
};

} // namespace

#include <solver/sgd-inl.hpp>

#endif // _LIBCF_SGD_HPP_
