#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <base/data.hpp>
#include <base/io.hpp>
#include <base/io/file.hpp>
#include <base/timer.hpp>
#include <base/random.hpp>
#include <model/linear_model.hpp>
#include <model/factor_model.hpp>
#include <model/recsys/popularity.hpp>
#include <model/recsys/itemcf.hpp>
#include <model/recsys/usercf.hpp>
#include <solver/sgd.hpp>
#include <solver/solver.hpp>
#include <model/recsys/bpr.hpp>
#include <model/recsys/neg_mf.hpp>
#include <model/recsys/fism.hpp>
#include <model/dae/cdae.hpp>

DEFINE_string(input_file, "./yelp_10core.txt", "input data");
DEFINE_string(cache_file, "./yelp.bin", "cache file");
DEFINE_string(task, "train", "Task type");

DEFINE_int32(seed, 20141119, "Random Seed");
DEFINE_string(method, "NONE", "Which Method to use");

DEFINE_int32(num_dim, 10, "Num of latent dimensions");
DEFINE_int32(num_neg, 5, "Num of negative samples");
DEFINE_double(learn_rate, 0.1, "Learning Rate");
DEFINE_bool(adagrad, true, "Use AdaGrad");
DEFINE_bool(bias, true, "Use bias term");
DEFINE_bool(linear_function, false, "Using Linear Mapping Function");
DEFINE_bool(tanh, false, "Using tanh NonLinear Function");
DEFINE_bool(asym, false, "Asymmetric DAE");
DEFINE_bool(linear, false, "Linear DAE");
DEFINE_bool(linear_output, false, "Linear output DAE");
DEFINE_bool(scaled, false, "scaled input");
DEFINE_bool(user_factor, true, "using user factor");
DEFINE_int32(cnum, 1, "Num of Corruptions");
DEFINE_double(cratio, 0, "Corruption Ratio");
DEFINE_string(loss_type, "SQUARE", "Loss function type");
DEFINE_double(beta, 1., "Beta for adagrad");

int main(int argc, char* argv[]) {
  
  using namespace libcf;
  
  FLAGS_log_dir = "./log";
  google::SetLogDestination(google::GLOG_INFO, "log/yelp_implicit.log");
  google::InitGoogleLogging(argv[0]);
  
  gflags::SetUsageMessage("yelp");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  auto line_parser = [&](const std::string& line) {
    auto rets = split_line(line, " ");
    CHECK_EQ(rets.size(), 2);
    //if (std::stod(rets[2]) < 4) 
    //  return std::vector<std::string>{};
    return std::vector<std::string>{rets[0], rets[1], "1"};
  };

  if (FLAGS_task == "prepare") {
    Data data;
    data.load(FLAGS_input_file, RECSYS, line_parser, true);
    save(data, FLAGS_cache_file);
  }
  
  if (FLAGS_task == "train") {
    seed(FLAGS_seed);  
    Data data;
    load(FLAGS_cache_file, data);
    LOG(INFO) << data; 
    Data train, test;
    data.random_split_by_feature_group(train, test, 0, 0.2);
    LOG(INFO) << train;
    LOG(INFO) << test;
    {
      Popularity pop_model;
      Solver<Popularity> solver(pop_model);
      solver.train(train, test, {TOPN});
    }

    if (FLAGS_method == "ITEMCF") {
      {
        ItemCF model(Jaccard, 50);
        Solver<ItemCF> solver(model);
        solver.train(train, test, {TOPN});
      }
    }

       
    if (FLAGS_method == "MF") {
      {
        NegMFConfig config;
        config.num_dim = FLAGS_num_dim;
        config.num_neg = FLAGS_num_neg;
        config.using_adagrad = FLAGS_adagrad;
        config.using_bias_term = FLAGS_bias;
        if (FLAGS_loss_type == "SQUARE") {
          config.lt = SQUARE;
        } else if (FLAGS_loss_type == "HINGE") {
          config.lt = HINGE;
        } else if (FLAGS_loss_type == "LOG") {
          config.lt = LOG;
        } else if (FLAGS_loss_type == "CE") {
          config.lt = CROSS_ENTROPY;
        } else {
          LOG(FATAL) << "UNKNOWN LOSS";
        }
        NegMF bpr_model(config);
        SGDConfig sgd_config;
        sgd_config.max_iteration = 300;
        sgd_config.learn_rate = FLAGS_learn_rate;
        SGD<NegMF> sgd(bpr_model, sgd_config);
        sgd.train(train, test, {TOPN});
      }
    }

    if (FLAGS_method == "BPR") {
      {
        BPRModelConfig config;
        config.num_dim = FLAGS_num_dim;
        config.num_neg = FLAGS_num_neg;
        config.using_adagrad = FLAGS_adagrad;
        if (FLAGS_loss_type == "SQUARE") {
          config.lt = SQUARE;
        } else if (FLAGS_loss_type == "HINGE") {
          config.lt = HINGE;
        } else if (FLAGS_loss_type == "LOG") {
          config.lt = LOG;
        } else if (FLAGS_loss_type == "LOGISTIC") {
          config.lt = LOGISTIC;
        } else {
          LOG(FATAL) << "UNKNOWN LOSS";
        }
        BPR_MF bpr_model(config);
        SGDConfig sgd_config;
        sgd_config.max_iteration = 300;
        sgd_config.learn_rate = FLAGS_learn_rate;
        SGD<BPR_MF> sgd(bpr_model, sgd_config);
        sgd.train(train, test, {TOPN});
      }
    }

  

    if (FLAGS_method == "FISM") {
      {
        FISMConfig config;
        config.num_dim = FLAGS_num_dim;
        config.num_neg = FLAGS_num_neg;
        config.using_adagrad = FLAGS_adagrad;
        config.lt = SQUARE; 
        FISM bpr_model(config);
        SGDConfig sgd_config;
        sgd_config.max_iteration = 300;
        sgd_config.learn_rate = FLAGS_learn_rate;
        SGD<FISM> sgd(bpr_model, sgd_config);
        sgd.train(train, test, {TOPN});
      }
    }

     
    if (FLAGS_method == "CDAE") {
      {
        DAEPSConfig cfg;
        cfg.learn_rate = FLAGS_learn_rate;
        cfg.num_dim = FLAGS_num_dim;
        cfg.using_adagrad = FLAGS_adagrad;
        cfg.asymmetric = FLAGS_asym;
        cfg.num_corruptions = FLAGS_cnum;
        cfg.corruption_ratio = FLAGS_cratio;
        cfg.linear = FLAGS_linear;
        cfg.linear_prime = FLAGS_linear_output;
        cfg.scaled = FLAGS_scaled;
        cfg.num_neg = FLAGS_num_neg;
        cfg.user_factor = FLAGS_user_factor;
        cfg.beta = FLAGS_beta; 
        cfg.linear_function = FLAGS_linear_function;
        cfg.tanh = FLAGS_tanh;
        if (FLAGS_loss_type == "SQUARE") {
          cfg.lt = SQUARE;
        } else if (FLAGS_loss_type == "LOG") {
          cfg.lt = LOG;
        } else if (FLAGS_loss_type == "HINGE") {
          cfg.lt = HINGE;
        } else if (FLAGS_loss_type == "LOGISTIC") {
          cfg.lt = LOGISTIC;
        } else if (FLAGS_loss_type == "CE") {
          cfg.lt = CROSS_ENTROPY;
        } else {
          LOG(FATAL) << "UNKNOWN LOSS";
        }
        DAEPS model(cfg);
        Solver<DAEPS> solver(model, 300);
        solver.train(train, test, {TOPN});
      }
    }
   
  }
  return 0;
}
