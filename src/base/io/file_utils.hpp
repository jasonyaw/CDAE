#ifndef _LIBCF_FILE_UTILS_HPP_
#define _LIBCF_FILE_UTILS_HPP_

#include <map>

#include <boost/tokenizer.hpp>
#include <glog/logging.h>

#include <base/timer.hpp>
#include <base/io/file.hpp>
#include <base/io/file_line_reader.hpp>

namespace libcf {

inline std::vector<std::string> split_line(const std::string& line, 
                                    const std::string& delimiters = " ") {
  std::vector<std::string> rets;  
  boost::char_separator<char> sep(delimiters.c_str());
  boost::tokenizer<boost::char_separator<char>> tokens(line, sep);
  for (auto it = tokens.begin(); it != tokens.end(); ++it){
    rets.push_back(*it);
  }
  return std::move(rets);
}

inline void write_config_file(const std::map<std::string, std::string>& opts, 
                              const std::string& filename) {

  File f(filename, "w");
  size_t opts_size = opts.size();
  for(auto& p : opts) {
    std::string output = p.first + std::string(" : ") + p.second;  
    if (--opts_size > 0) {
      output += "\n";
    }
    f.write(output);
  }
  f.close();
}

inline std::map<std::string, std::string> read_config_file(const std::string& filename) {

  std::map<std::string, std::string> opts;
  FileLineReader flr(filename);
  flr.set_line_callback([&](const std::string& line, 
                            size_t line_num){
                        auto splits = split_line(line, " :");
                        CHECK(splits.size() == 2) << "Expect the size is 2";
                        opts.emplace(splits[0], splits[1]);
                        });
  flr.load();
  return std::move(opts);
}

} // namespace


#endif // _LIBCF_FILE_UTILS_HPP_
