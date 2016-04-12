#include <base/io/file.hpp>

#include <glog/logging.h>

namespace libcf {

File::File(const std::string& filename, const std::string& flag) : f_(new std::fstream) {

  std::ios::openmode mode;
  if (flag == "r") {
    mode = std::ios::in;
  } else if (flag == "w") {
    read_only = false;
    mode = std::ios::out;
  } else if (flag == "rb") {
    is_binary = true;
    mode = std::ios::in | std::ios::binary;
  } else if (flag == "wb") {
    is_binary = true;
    read_only = false;
    mode = std::ios::out | std::ios::binary;
  } else {
    LOG(FATAL) << "Cannot open file " << filename << " ! "
        << "Invalid flag!" << flag << std::endl;
  }

  f_->open(filename.c_str(), mode);

  if (f_ == NULL || !f_->is_open()) {
    LOG(FATAL) << "Failed to open file " << filename << std::endl;
  }

}

size_t File::size() const {
  f_->seekg(0, f_->end);
  size_t length = f_->tellg();
  f_->seekg(0, f_->beg);
  return length;
}

bool File::read_line(std::string& line, char delim) const {
  CHECK_EQ(read_only, true);
  if (good()) {
    std::getline(*f_, line, delim);
  }
  return ok();
}

std::string File::read_line(char delim) const {
  CHECK_EQ(read_only, true);
  std::string line;
  read_line(line, delim);
  return std::move(line);
}

bool File::write_str(const std::string& line) const {
  CHECK_EQ(read_only, false);
  if(good()){
    f_->write(line.c_str(), line.size());
  }
  return ok();
}

bool File::write_line(const std::string& line) const {
  CHECK_EQ(read_only, false);
  if(good()){
    f_->write(line.c_str(), line.size());
    if (line.back() != '\n') 
      f_->write("\n", 1);
  }
  return ok();
}
} // namespace 

