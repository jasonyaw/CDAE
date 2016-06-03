#include <solver/solver.hpp>

namespace libcf {
  
template<class Model>
void Solver<Model>::train(const Data& train_data, 
                              const Data& validation_data,
                              const std::vector<EvalType>& eval_types) {


  double train_loss = 0;

  std::vector<std::shared_ptr<Evaluation<Model>>> evaluations(eval_types.size());
  for (size_t idx = 0; idx < eval_types.size(); ++idx) {
    evaluations[idx] = Evaluation<Model>::create(eval_types[idx]);
  }

  size_t iteration = 0;
  model_->reset(train_data);
  pre_train(train_data, validation_data);

  Timer t;
  
  LOG(INFO) << std::string(110, '-') << std::endl;
  {
    std::stringstream ss;
    ss << std::setfill(' ') << std::setw(5) << "Iters" << "|"
        << std::setw(8) << "Time"  << "|" 
        << std::setw(10) << "Train Loss" << "|";
    if(validation_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluation_type() << "|";
    } 
    LOG(INFO) << ss.str();
  }

  if (iteration % eval_iterations == 0)
  {
    std::stringstream ss;
    ss << std::setw(5) << iteration << "|"
        << std::setw(8) << std::setprecision(3) << t.elapsed() << "|"
        << std::setw(10) << std::setprecision(5) << train_loss << "|";
    if (validation_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluate(*model_, validation_data, train_data) << "|";
    }
    LOG(INFO) << ss.str();
  }

  bool stop = false;
  while(!stop) {

    train_one_iteration(train_data);

    train_loss = model_->current_loss(train_data);

    iteration ++;
    if (iteration % eval_iterations == 0)
    {
      std::stringstream ss;
      ss << std::setw(5) << iteration << "|"
          << std::setw(8) << std::setprecision(3) << t.elapsed() << "|"
          << std::setw(10) << std::setprecision(5) << train_loss << "|";
      if (validation_data.size() > 0) {
        for (size_t idx = 0; idx < eval_types.size(); ++idx) 
          ss << evaluations[idx]->evaluate(*model_, validation_data, train_data) << "|";
      }
      LOG(INFO) << ss.str();
    }

    // check conditions
    if (iteration >= max_iteration_) {
      stop = true;
    }
    // other conditions
  }

  LOG(INFO) << std::string(110, '-') << std::endl;
}

template<class Model>
void Solver<Model>::test(const Data& test_data,
                         const std::vector<EvalType>& eval_types) {

  Timer t;
  std::vector<std::shared_ptr<Evaluation<Model>>> evaluations(eval_types.size());
  for (size_t idx = 0; idx < eval_types.size(); ++idx) {
    evaluations[idx] = Evaluation<Model>::create(eval_types[idx]);
  }

  LOG(INFO) << std::string(100, '-') << std::endl;
  {
    std::stringstream ss;
    ss << std::setfill(' ') 
        << std::setw(8) << "Time"  << "|";
    if(test_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluation_type() << "|";
    }
    LOG(INFO) << ss.str();
  }

  {
    std::stringstream ss;
    ss << std::setw(8) << std::setprecision(3) << t.elapsed() << "|";
    if (test_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluate(*model_, test_data) << "|";
    }
    LOG(INFO) << ss.str();
  }
}

} // namespace
