#include <model/factor_model.hpp>

namespace libcf {

void FactorModel::reset(const Data& data_set) {
  
  ModelBase::reset(data_set);

  if (using_bias_term_) {
    coefficients_ = DMatrix::Random(data_set.total_dimensions(), 1) * 0.01;
    if (using_adagrad_) {
      coeff_grad_square_ = DMatrix::Zero(data_set.total_dimensions(), 1); 
    }
  }

  if (using_factor_term_) {
    factors_ = DMatrix::Random(data_set.total_dimensions(), num_dim_) * 0.01;
    if (using_adagrad_) {
      factor_grad_square_ = DMatrix::Zero(data_set.total_dimensions(), num_dim_); 
    }
  }

  global_mean_ = 0;

  if (using_global_mean_ && data_set.size() > 0) {
    for (auto iter = data_set.begin(); iter != data_set.end(); ++iter) {
      global_mean_ += iter->label();
    }
    global_mean_ /= data_set.size();
    LOG(INFO) << "Global mean score is " << global_mean_;
  }
}

double FactorModel::predict(const Instance& ins) const {
  double ret = 0;

  if (using_global_mean_) 
    ret += global_mean_;

  auto ins_iter_a = data_->begin(ins);
  auto ins_iter_b = data_->begin(ins);
  auto ins_iter_end = data_->end(ins);

  int feat_idx_a = 0, feat_idx_b = 0;
  double feat_value_a = 0, feat_value_b = 0;

  for (; ins_iter_a != ins_iter_end; ++ins_iter_a) {
    feat_idx_a = ins_iter_a.index();
    feat_value_a = ins_iter_a.value();
    if (using_bias_term_) 
      ret += coefficients_(feat_idx_a) * feat_value_a;
    if (using_factor_term_) {
      for (ins_iter_b = ++ins_iter_a; 
           ins_iter_b != ins_iter_end; ++ins_iter_b) {
        feat_idx_b = ins_iter_b.index();
        feat_value_b = ins_iter_b.value();
        ret += feat_value_a * feat_value_b 
            * factors_.row(feat_idx_a).dot(factors_.row(feat_idx_b));
      }
    }
  }
  return ret;
}

void FactorModel::update_one_instance(const Instance& ins, double step_size) {

  double pred = predict(ins);
  double gradient = loss_->gradient(pred, ins.label());

  auto ins_iter_begin = data_->begin(ins);
  auto ins_iter_end = data_->end(ins);

  auto ins_iter_a = ins_iter_begin;
  auto ins_iter_b = ins_iter_begin;

  size_t feat_idx_a = 0, feat_idx_b = 0, fg_idx;
  double feat_value_a = 0, feat_value_b = 0;

  DMatrix factors_grad(ins.num_feature_groups(), num_dim_);

  for (; ins_iter_a != ins_iter_end; ++ins_iter_a) {
    fg_idx = ins_iter_a.feature_group_idx();
    CHECK_LT(fg_idx, factors_grad.rows());
    feat_idx_a = ins_iter_a.index();
    factors_grad.row(fg_idx) = lambda_ * factors_.row(feat_idx_a); 
  }

  double grad = 0; 

  for (ins_iter_a = ins_iter_begin; ins_iter_a != ins_iter_end; ++ins_iter_a) {
    fg_idx = ins_iter_a.feature_group_idx();
    feat_idx_a = ins_iter_a.index();
    feat_value_a = ins_iter_a.value();
    if (using_bias_term_) {
      grad = lambda_ * coefficients_(feat_idx_a) + gradient * feat_value_a;
      if (using_adagrad_) {
        coeff_grad_square_(feat_idx_a) += grad * grad;   
        grad /= std::sqrt(coeff_grad_square_(feat_idx_a));
      }
      coefficients_(feat_idx_a) -= step_size * grad;
    }

    // compute gradients for factors
    if (using_factor_term_) {
      for (ins_iter_b = ins_iter_begin; ins_iter_b != ins_iter_end; ++ins_iter_b) {
        if (ins_iter_a.feature_group_idx() == ins_iter_b.feature_group_idx()) 
          continue;
        feat_idx_b = ins_iter_b.index();
        feat_value_b = ins_iter_b.value();
        factors_grad.row(fg_idx) += gradient 
            * feat_value_a * feat_value_b * factors_.row(feat_idx_b);
      }
    }
  }
  // update factors
  if (using_factor_term_) {
    for (ins_iter_a = ins_iter_begin; ins_iter_a != ins_iter_end; ++ins_iter_a) {
      fg_idx = ins_iter_a.feature_group_idx();
      feat_idx_a = ins_iter_a.index();
      if (using_adagrad_) {
        factor_grad_square_.row(feat_idx_a) += factors_grad.row(fg_idx).cwiseProduct(factors_grad.row(fg_idx));   
        factors_grad.row(fg_idx) = factors_grad.row(fg_idx).cwiseQuotient(factor_grad_square_.row(feat_idx_a).cwiseSqrt());
      }
      factors_.row(feat_idx_a) -= step_size * factors_grad.row(fg_idx);
    }
  }

}

void FactorModel::update_one_sgd_step(const Instance& ins, double step_size) {
  update_one_instance(ins, step_size);
}



} // namespace
