#include <base/io/file_line_reader.hpp>

#include <glog/logging.h>

#include <base/timer.hpp>

namespace libcf {

void FileLineReader::load() {
  Timer t;
  File f(filename_, "r");
  size_t line_num = 0;
  size_t line_skipped = 0;
  std::string line;
  while (f.good()) {
    f.read_line(line);
    if (line.size() == 0) {
      ++line_skipped;
      continue;
    }
    line_callback_(line, line_num);
    ++line_num;
  }
  f.close();
  loaded_successfully_ = true;
  LOG(INFO) << line_num << " lines loaded from file " << filename_ 
      << " in " << t ; 
  LOG(INFO) << line_skipped << " lines skipped" << std::endl; 
}

} // namespace

