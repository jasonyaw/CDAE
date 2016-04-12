#ifndef _LIBCF_WARP_HPP_
#define _LIBCF_WARP_HPP_

#include <algorithm>
#include <base/heap.hpp>
#include <base/utils.hpp>
#include <model/loss.hpp>
#include <model/recsys/imf.hpp>

namespace libcf {

struct WARPConfig {
  WARPConfig() = default;
  double learn_rate = 0.1;
  double beta = 0.;
  double lambda = 0.1;  // regularization coefficient 
  LossType lt = HINGE; // loss type
  PenaltyType pt = L2;  // penalty type
  size_t num_dim = 10;
  size_t num_neg = 5;
  bool using_bias_term = true;
  bool using_adagrad = true;
};

class WARP : public IMF {

 public:
  WARP(const WARPConfig& mcfg) {  
    learn_rate_ = mcfg.learn_rate;
    beta_ = mcfg.beta;
    lambda_ = mcfg.lambda;
    num_dim_ = mcfg.num_dim;
    num_neg_ = mcfg.num_neg;
    using_bias_term_ = mcfg.using_bias_term;
    using_adagrad_ = mcfg.using_adagrad;
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);

    LOG(INFO) << "WARP  Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Learn Rate: " << learn_rate_ << "}, "
        << "{Beta " << beta_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{BiasTerm: " << using_bias_term_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}, "
        << "{Num Negative: " << num_neg_ << "}";
  }

  //WARP() : WARP(WARPConfig()) {}

  void reset(const Data& data_set) {
  
    IMF::reset(data_set);

    l_.assign(num_items_, 1.);
    for (size_t idx = 1; idx < num_items_; ++idx) {
      l_[idx] = l_[idx - 1] + 1. / static_cast<double>(idx + 1);
    }
  }

  virtual void train_one_iteration(const Data& train_data) {
    for (size_t uid = 0; uid < num_users_; ++uid) {
      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      auto& item_map = fit->second;
      size_t items_left = num_items_ - item_map.size();
      for (auto& p : item_map) {
        auto& iid = p.first;
        double yui = predict_user_item_rating(uid, iid);
        double yuj;
        for (size_t idx = 0; idx < num_neg_; ++idx) {
          size_t jid = -1, cnt = 0;
          while (true) {
            jid = sample_negative_item(item_map);
            yuj = predict_user_item_rating(uid, jid);
            ++cnt;
            if (yuj > yui - 1. || cnt >= 500) {
              break;
            }
          }
          if (cnt >= 500) continue;
          train_one_pair(uid, iid, jid, yui, yuj, l_[items_left / cnt]);
        }
      }
    }
  }

  virtual void train_one_pair(size_t uid, size_t iid, size_t jid, double yui, double yuj, double l) {
    double pred_ij = yui - yuj;
    double gradient = loss_->gradient(pred_ij, 1.) * l;
    //double ib_grad = gradient + 2. * lambda_ * ib_(iid);
    //double jb_grad = - gradient + 2. * lambda_ * ib_(jid);
    DVector uv_grad = gradient * (iv_.row(iid) - iv_.row(jid)) + 2. * lambda_ * uv_.row(uid);
    DVector iv_grad = gradient * uv_.row(uid) + 2. * lambda_ * iv_.row(iid);
    DVector jv_grad = - gradient * uv_.row(uid) + 2. * lambda_ * iv_.row(jid);

    if (using_adagrad_) {
      //ib_ag_(iid) += ib_grad * ib_grad;
      //ib_ag_(jid) += jb_grad * jb_grad;
      //ib_grad /= (beta_ + std::sqrt(ib_ag_(iid)));
      //jb_grad /= (beta_ + std::sqrt(ib_ag_(jid)));
      uv_ag_.row(uid) += uv_grad.cwiseProduct(uv_grad);
      iv_ag_.row(iid) += iv_grad.cwiseProduct(iv_grad);
      iv_ag_.row(jid) += jv_grad.cwiseProduct(jv_grad);
      uv_grad = uv_grad.cwiseQuotient(uv_ag_.row(uid).cwiseSqrt().transpose());
      iv_grad = iv_grad.cwiseQuotient(iv_ag_.row(iid).cwiseSqrt().transpose());
      jv_grad = jv_grad.cwiseQuotient(iv_ag_.row(jid).cwiseSqrt().transpose());
    }

    //ib_(iid) -= learn_rate_ * ib_grad;
    //ib_(jid) -= learn_rate_ * jb_grad;
    uv_.row(uid) -= learn_rate_ * uv_grad;
    iv_.row(iid) -= learn_rate_ * iv_grad;
    iv_.row(jid) -= learn_rate_ * jv_grad;
  }

 private:

  std::vector<double> l_;
};

} // namespace

#endif // _LIBCF_WARP_HPP_
