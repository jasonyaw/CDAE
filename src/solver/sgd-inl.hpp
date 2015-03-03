#include <solver/sgd.hpp>

#include <cmath>
#include <iomanip>
#include <algorithm>

#include <base/timer.hpp>
#include <base/utils.hpp>

namespace libcf {

template<class Model>
    void SGD<Model>::initialize(const Data& train_data,
                                const Data& validation_data,
                                size_t sample_size) {

      /*
         sample_size = std::min(sample_size, train_data.size());
         LOG(INFO) << "Picking SGD step_size using "
         << sample_size << " samples" << std::endl;

         LOG(INFO) << std::setw(80) << std::setfill('-') 
         << "-" << std::endl;

         LOG(INFO) << std::setfill(' ') << std::setw(20) << "step_size" 
         << std::setw(20) << "loss" 
         << std::endl;

      /////////////////////////////////////////////////////
      // search best step size 
      double step_size = 0, loss = 0;
      std::vector<std::pair<double, double>> trace;
      for(int i = 0; i < 20; i+=2) {
      step_size = std::pow(2., -i);
      learn_rate_ = step_size;
      steps_ = 0;

      model_->reset(train_data);

      for (size_t j = 0; j < 10; j++)
      update(train_data, sample_size); 

      if (validation_data.size() == 0) {
      loss = model_->current_loss(train_data, sample_size);
      } else {
      loss = model_->data_loss(validation_data);
      }

      trace.emplace_back(step_size, loss);
      LOG(INFO) << std::setfill(' ') << std::setw(20) << step_size 
      << std::setw(20) << std::setprecision(2) << loss 
      << std::endl;
      }

      learn_rate_ = 
      std::min_element(trace.begin(),
      trace.end(),
      sort_by_second_asc<double, double>) -> first;
      initialized_learn_rate_ = learn_rate_;
      LOG(INFO) << "Initialized step_size: " << initialized_learn_rate_;
      steps_ = 0;
      */
    }

template<class Model>
    void SGD<Model>::update(const Data& data_set, size_t sample_size) {
      if (sample_size == 0) 
        sample_size = data_set.size();
      size_t count = 0;
      for (auto iter = data_set.begin(); 
           iter != data_set.end() && ++count != sample_size; iter++) {
        auto& ins = *iter;
        update_with_one_instance(ins);
      }
    }

template<class Model>
    void SGD<Model>::update_with_one_instance(const Instance& ins) {
      Solver<Model>::model_->update_one_sgd_step(ins, learn_rate_);
      update_learn_rate();
      steps_++;
    }

template<class Model>
    void SGD<Model>::update_learn_rate() {
      // TODO
      if (!config_->update_learn_rate) 
        return;
      learn_rate_ = initialized_learn_rate_ / 
          (1. + initialized_learn_rate_ * Solver<Model>::model_-> regularization_coefficent() * steps_);
    }

template<class Model>
    void SGD<Model>::pre_train(const Data& train_data, const Data& validation_data) {
      if (config_->automatic_intialization) {
        initialize(train_data, validation_data);
      } else {
        learn_rate_ = config_->learn_rate;
        initialized_learn_rate_ = config_->learn_rate;
      }
      steps_ = 0; 
    }

} // namespace
