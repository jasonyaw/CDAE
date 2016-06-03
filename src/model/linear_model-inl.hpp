#include <base/timer.hpp>
#include <model/linear_model.hpp>

namespace libcf {

void LinearModel::reset(const Data& data_set) {
  ModelBase::reset(data_set);

  coefficients_ = DMatrix::Random(data_->total_dimensions(), 1) * 0.01;

  if (using_adagrad_) { 
    gradient_square_ = DMatrix::Zero(data_->total_dimensions(), 1);
  }

  global_mean_ = 0;
  for (auto iter = data_->begin(); iter != data_->end(); ++iter) {
    global_mean_ += iter->label();
  }

  global_mean_ /= data_->size();
  LOG(INFO) << "Global mean score is " << global_mean_;
}


double LinearModel::predict(const Instance& ins) const {
  double ret = 0;
  
  if (using_global_mean_) 
    ret += global_mean_;

  auto ins_iter = data_->begin(ins);
  auto ins_iter_end = data_->end(ins);
  
  int feat_idx = 0;
  double feat_value = 0;

  for (; ins_iter != ins_iter_end; ++ins_iter) {
    feat_idx = ins_iter.index();
    feat_value = ins_iter.value();
    ret += coefficients_(feat_idx) * feat_value;
  }

  return ret;
}

void LinearModel::update_one_sgd_step(const Instance& ins, double step_size) {

  double pred = predict(ins);
  double gradient = loss_->gradient(pred, ins.label());

  auto ins_iter = data_->begin(ins);
  auto ins_iter_end = data_->end(ins);

  int feat_idx = 0;
  double feat_value = 0;
  double grad = 0;
  for (; ins_iter != ins_iter_end; ++ins_iter) {
    feat_idx = ins_iter.index();
    feat_value = ins_iter.value();
    grad = lambda_ * coefficients_(feat_idx) + gradient * feat_value;
    if (using_adagrad_) {
      gradient_square_(feat_idx) += grad * grad;   
      grad /= std::sqrt(gradient_square_(feat_idx));
    }
    coefficients_(feat_idx) -= step_size * grad;
  }

}

} // namespace
