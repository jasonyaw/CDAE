#ifndef _LIBCF_CDAE_HPP_
#define _LIBCF_CDAE_HPP_

#include <base/random.hpp>
#include <base/mat.hpp>
#include <base/instance.hpp>
#include <base/data.hpp>
#include <base/parallel.hpp>
#include <model/recsys/recsys_model_base.hpp>

namespace libcf {

struct CDAEConfig {
  CDAEConfig() = default;
  double lambda = 0.01;    
  double learn_rate = 0.1;   
  LossType lt = LOGISTIC; 
  PenaltyType pt = L2;  
  size_t num_dim = 10;
  bool using_adagrad = true;
  double corruption_ratio = 0.5;
  size_t num_corruptions = 1;
  bool asymmetric = false;
  bool user_factor = true;
  bool linear = false;
  size_t num_neg = 5;
  bool scaled = true;
  double beta = 0.;
  bool linear_function = false;
  bool tanh = false;
};

/* Denoising Auto-Encoder
 *
 */
class CDAE : public RecsysModelBase {

 public:
  CDAE(const CDAEConfig& mcfg) {  
    lambda_ = mcfg.lambda;
    learn_rate_ = mcfg.learn_rate;
    num_dim_ = mcfg.num_dim; 
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);
    using_adagrad_ = mcfg.using_adagrad;
    corruption_ratio_ = mcfg.corruption_ratio;
    num_corruptions_ = mcfg.num_corruptions;
    asymmetric_ = mcfg.asymmetric;
    user_factor_ = mcfg.user_factor;
    linear_ = mcfg.linear;
    num_neg_ = mcfg.num_neg;
    scaled_ = mcfg.scaled;
    beta_ = mcfg.beta;
    linear_function_ = mcfg.linear_function;
    tanh_ = mcfg.tanh;

    LOG(INFO) << "CDAE Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{LearnRate: " << learn_rate_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}\n"
        << "\t{Corruption Ratio: " << corruption_ratio_ << "}, "
        << "{Num Corruptions: " << num_corruptions_ << "}, "
        << "{Asymmetric: " << asymmetric_ << "}\n"
        << "\t{UserFactor: " << user_factor_ << "}, "
        << "{Linear: " << linear_ << "}, " 
        << "{Num Negative: " << num_neg_ << "}, "
        << "{Scaled: " << scaled_ << "}\n"
        << "\t{Beta: " << beta_ << "}, "
        << "{LinearFunction: " << linear_function_ << "}, "
        << "{tanh: " << tanh_ << "}"; 
  }

  CDAE() : CDAE(CDAEConfig()) {}
  
  double data_loss(const Data& data_set, size_t sample_size=0) const {
    std::atomic<double> rets(0.);
    
    parallel_for (0, num_users_, [&](size_t uid) {
      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      auto& item_set = fit->second;
      double user_rets = 0;
      for (size_t jid = 0; jid < num_corruptions_; ++jid) {
        auto corrpted_item_set = get_corrputed_input(item_set, corruption_ratio_);
        double scale = 1;
        if (scaled_) {
         scale /=  (1. - corruption_ratio_) ;
        }
        auto z = get_hidden_values(uid, corrpted_item_set, scale);
        for (auto& p : item_set) {
          size_t iid = p.first;
          user_rets += loss_->evaluate(get_output_values(z, iid), 1.);
        }
      }
      rets = rets + user_rets / num_corruptions_;
    });
    return rets;
  }
   
  double penalty_loss() const {
    return 0.5 * lambda_ * (penalty_->evaluate(W) + penalty_->evaluate(V)
                            + penalty_->evaluate(Wu) + penalty_->evaluate(b) 
                            + penalty_->evaluate(b_prime)); 
  }

  void reset(const Data& data_set) {
    RecsysModelBase::reset(data_set);

    double init_scale = 4. * std::sqrt(6. / static_cast<double>(num_items_ + num_dim_));
    W = DMatrix::Random(num_items_, num_dim_) * init_scale;
    W_ag = DMatrix::Constant(num_items_, num_dim_, 0.0001);
    if (asymmetric_) {
      V = DMatrix::Random(num_items_, num_dim_) * init_scale;
      V_ag = DMatrix::Constant(num_items_, num_dim_, 0.0001);
    } 
    if (user_factor_) {
      Wu = DMatrix::Random(num_users_, num_dim_) * init_scale;
      Wu_ag = DMatrix::Constant(num_users_, num_dim_, 0.0001);
    }
    b = DVector::Zero(num_dim_);
    b_ag = DVector::Ones(num_dim_) * 0.0001;
    b_prime = DVector::Zero(num_items_);
    b_prime_ag = DVector::Ones(num_items_) * 0.0001;
    bu = DVector::Zero(num_users_);
    bu_ag = DVector::Ones(num_users_) * 0.0001;

    if (linear_function_) { 
      Uu = DMatrix::Constant(num_users_, num_dim_, 1.);
      Uu_ag = DMatrix::Constant(num_users_, num_dim_, 0.0001);
    }
  } 

  void train_one_iteration(const Data& train_data) {
    for (size_t uid = 0; uid < num_users_; ++uid) {
      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      auto& item_set = fit->second;
      for (size_t idx = 0; idx < num_corruptions_; ++idx) {
        auto corrpted_item_set = get_corrputed_input(item_set, corruption_ratio_);
        train_one_user_corruption(uid, corrpted_item_set, item_set);
      }
    }
  }
  
  DMatrix get_user_representations() {
    
    DMatrix user_vec(num_users_, num_dim_);

    for (size_t uid = 0; uid < num_users_; ++uid) {
      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      user_vec.row(uid) = get_hidden_values(uid, fit->second);
    }
  
    return std::move(user_vec);
  }

  // required by evaluation measure TOPN
  std::vector<size_t> recommend(size_t uid, size_t topk,
                                const std::unordered_map<size_t, double>& rated_item_set) const {
    size_t item_id = 0;
    size_t item_id_end = item_id + data_->feature_group_total_dimension(1);
     
    DVector z = DVector::Zero(num_dim_);
    if (corruption_ratio_ != 1.) { 
      z = get_hidden_values(uid, rated_item_set);
    } else { 
      z = get_hidden_values(uid, std::unordered_map<size_t, double>{});
    }

    Heap<std::pair<size_t, double>> topk_heap(sort_by_second_desc<size_t, double>, topk);
    double pred;
    for (; item_id != item_id_end; ++item_id) {
      if (rated_item_set.count(item_id)) {
        continue;
      }
      pred = get_output_values(z, item_id);
      if (topk_heap.size() < topk) {
        topk_heap.push({item_id, pred});
      } else {
        topk_heap.push_and_pop({item_id, pred});
      }
    }
    CHECK_EQ(topk_heap.size(), topk);
    auto topk_heap_vec = topk_heap.get_sorted_data();
    std::vector<size_t> ret(topk);
    std::transform(topk_heap_vec.begin(), topk_heap_vec.end(),
                   ret.begin(),
                   [](const std::pair<size_t, double>& p) {
                   return p.first;
                   });
    return std::move(ret);
  }

  void train_one_user_corruption(size_t uid, 
                                 const std::unordered_map<size_t, double>& input_set, 
                                 const std::unordered_map<size_t, double>& output_set) {
    
    double scale = 1.;
    if (scaled_) {
      scale /= (1. - corruption_ratio_);
    }

    DVector z = get_hidden_values(uid, input_set, scale);
    DVector z_1_z =  DVector::Ones(num_dim_);
    if (! linear_) {
      if (! tanh_) {
        z_1_z = z - z.cwiseProduct(z);
      } else {
        z_1_z = DVector::Ones(num_dim_) - z.cwiseProduct(z); 
      }
    }
    
    std::vector<size_t> negative_sampels(output_set.size() * num_neg_);
    for (size_t idx = 0; idx < negative_sampels.size(); ++idx) {
      negative_sampels[idx] = sample_negative_item(output_set);
    }
    
    std::unordered_map<size_t, DVector> input_gradient;
    DVector hidden_gradient = DVector::Zero(num_dim_);

    for (auto& p : output_set) {
      size_t iid = p.first;
      double y = get_output_values(z, iid);
      double gradient = loss_->gradient(y, 1.);
      
      {
        double grad = gradient + lambda_ * b_prime(iid);
        if (using_adagrad_) {
          b_prime_ag(iid) += grad * grad;
          grad /= (beta_ + std::sqrt(b_prime_ag(iid)));
        }
        b_prime(iid) -= learn_rate_ * grad;
      }

      if (asymmetric_) {
        hidden_gradient += gradient * V.row(iid);
        DVector grad = gradient * z + lambda_ * V.row(iid).transpose();
        if (using_adagrad_) {
          V_ag.row(iid) += grad.cwiseProduct(grad);
          grad = grad.cwiseQuotient((V_ag.row(iid).transpose().cwiseSqrt().array() + beta_).matrix());
        }
        V.row(iid) -= learn_rate_ * grad;
      } else {
        hidden_gradient += gradient * W.row(iid);
        if (input_set.count(iid)) {
          input_gradient[iid] = gradient * z;
        } else {
          DVector grad = gradient * z + lambda_ * W.row(iid).transpose();
          if (using_adagrad_) {
            W_ag.row(iid) += grad.cwiseProduct(grad);
            grad = grad.cwiseQuotient((W_ag.row(iid).transpose().cwiseSqrt().array()+ beta_).matrix());
          }
          W.row(iid) -= learn_rate_ * grad;
        }
      }
    }

    for (auto& iid : negative_sampels) {
      double y = get_output_values(z, iid);
      
      double gradient = loss_->gradient(y, 0.);

      {
        double grad = gradient + lambda_ * b_prime(iid);
        if (using_adagrad_) {
          b_prime_ag(iid) += grad * grad;
          grad /= (beta_ + std::sqrt(b_prime_ag(iid)));
        }
        b_prime(iid) -= learn_rate_ * grad;
      }

      if (asymmetric_) {
        hidden_gradient += gradient * V.row(iid);
        DVector grad = gradient * z + lambda_ * V.row(iid).transpose();
        if (using_adagrad_) {
          V_ag.row(iid) += grad.cwiseProduct(grad);
          grad = grad.cwiseQuotient((V_ag.row(iid).transpose().cwiseSqrt().array() + beta_).matrix());
        }
        V.row(iid) -= learn_rate_ * grad;
      } else {
        hidden_gradient += gradient * W.row(iid);
        DVector grad = gradient * z + lambda_ * W.row(iid).transpose();
        if (using_adagrad_) {
          W_ag.row(iid) += grad.cwiseProduct(grad);
          grad = grad.cwiseQuotient((W_ag.row(iid).transpose().cwiseSqrt().array() + beta_).matrix());
        }
        W.row(iid) -= learn_rate_ * grad;
      }
    }
 
    DVector Uu_grad;
    if (linear_function_) {
      Uu_grad = DVector::Zero(num_dim_);
      Uu_grad += Uu.row(uid).transpose() * lambda_;
    }

    // b
    {
      DVector grad = DVector::Zero(num_dim_);
      //if (!linear_function_) {
        grad = hidden_gradient.cwiseProduct(z_1_z) + lambda_ * b;
      //} else {
        //grad = Uu[uid].tranpose() * hidden_gradient.cwiseProduct(z_1_z) + lambda_ * b;
        //Uu_grad += hidden_gradient.cwiseProduct(z_1_z) * b.transpose();
      //}
      if (using_adagrad_) {
        b_ag += grad.cwiseProduct(grad);
        grad = grad.cwiseQuotient((b_ag.cwiseSqrt().array() + beta_).matrix());
      }
      b -= learn_rate_ * grad;
    }
   
    if (user_factor_)
    {   
      DVector grad = DVector::Zero(num_dim_);
      //if (!linear_function_) {
        grad = hidden_gradient.cwiseProduct(z_1_z) + lambda_ * Wu.row(uid).transpose();
      //} else {
      //  grad = Uu[uid].transpose() * hidden_gradient.cwiseProduct(z_1_z) + lambda_ * Wu.row(uid).transpose();
      //  Uu_grad += hidden_gradient.cwiseProduct(z_1_z) * Wu.row(uid);
      //}
      if (using_adagrad_) {
        Wu_ag.row(uid) += grad.cwiseProduct(grad);
        grad = grad.cwiseQuotient((Wu_ag.row(uid).transpose().cwiseSqrt().array() + beta_).matrix());
      }
      Wu.row(uid) -= learn_rate_ * grad;
    }

    for (auto& p : input_set) {
      size_t jid = p.first;
      DVector grad = DVector::Zero(num_dim_);
      if (!linear_function_) {
        grad = hidden_gradient.cwiseProduct(z_1_z) * scale + lambda_ * W.row(jid).transpose();
      } else {
        grad = Uu.row(uid).transpose().cwiseProduct(hidden_gradient.cwiseProduct(z_1_z)) * scale + lambda_ * W.row(jid).transpose();
        Uu_grad += hidden_gradient.cwiseProduct(z_1_z).cwiseProduct(W.row(jid).transpose());
      }
      if (input_gradient.count(jid))
        grad += input_gradient[jid];
      if (using_adagrad_) {
        W_ag.row(jid) += grad.cwiseProduct(grad);
        grad = grad.cwiseQuotient((W_ag.row(jid).transpose().cwiseSqrt().array() + beta_).matrix());
      }
      W.row(jid) -= learn_rate_ * grad;
    }
    
    if (linear_function_) {
      if (using_adagrad_) {
        Uu_ag.row(uid) += Uu_grad.cwiseProduct(Uu_grad).transpose(); 
        Uu_grad = Uu_grad.cwiseQuotient((Uu_ag.row(uid).transpose().cwiseSqrt().array() + beta_).matrix());
      }
      Uu.row(uid) -= learn_rate_ * Uu_grad;
    }
  }


  std::unordered_map<size_t, double> get_corrputed_input(const std::unordered_map<size_t, double>& input_set, 
                                          double corruption_ratio) const {
    std::unordered_map<size_t, double> rets;
    rets.reserve(static_cast<size_t>(input_set.size() * (1. - corruption_ratio)));
    for (auto& p : input_set) {
      if (Random::uniform() > corruption_ratio) {
        rets.insert(p);
      }
    }
    return rets;
  }

  DVector get_hidden_values(size_t uid, const std::unordered_map<size_t, double>& item_set,
                            double scale = 1.0) const {
    DVector h1 = DVector::Zero(num_dim_);
    
    for (auto& p : item_set) {
      size_t iid = p.first;
      h1 += W.row(iid) * scale;
    }
    
    if (linear_function_) {
      h1 = Uu.row(uid).transpose().cwiseProduct(h1);
    }

    h1 += b; 
    if (user_factor_) {
      h1 += Wu.row(uid);
    }

    if (! linear_) {
      if (! tanh_) {
      h1 = h1.unaryExpr([](double x) {
                        if (x > 18.) {
                        return 1.;
                        } 
                        if (x < -18.) {
                        return 0.;
                        }
                        return 1. / (1. + std::exp(-x));
                        });
      } else {
        h1 = h1.unaryExpr([](double x) {
                          if (x > 9.) {
                            return 1.;
                          }
                          if (x < -9.) {
                            return -1.;
                          }  
                          double r = std::exp(-2. * x);
                          return (1. - r) / (1. + r); 
                        });
      }
    }
    return h1;
  }

  double get_output_values(const DVector& z, size_t idx) const {
    double h2 = 0; 
    if (asymmetric_) {
      h2 += V.row(idx).dot(z) + b_prime(idx);
    } else {
      h2 += W.row(idx).dot(z) + b_prime(idx);
    }
    return h2;
  }

 private:

  DMatrix W;
  DMatrix V;
  DMatrix W_ag;
  DMatrix V_ag;
  DMatrix Wu;
  DMatrix Wu_ag;
  DVector b, b_prime, bu;
  DVector b_ag, b_prime_ag, bu_ag;
  DMatrix Uu;
  DMatrix Uu_ag;
  size_t num_dim_ = 0.;
  double learn_rate_ = 0.;
  double lambda_ = 0.;  
  double corruption_ratio_ = 0.5;
  size_t num_corruptions_ = 10;
  size_t num_neg_ = 5;
  bool using_adagrad_ = true;
  bool asymmetric_ = false;
  bool user_factor_ = true;
  bool linear_ = false;
  bool scaled_ = true;
  double beta_ = 0.; 
  bool linear_function_ = false;
  bool tanh_ = false;
};

} // namespace

#endif // _LIBCF_CDAE_HPP_
