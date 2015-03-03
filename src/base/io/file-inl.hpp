#include <base/io/file.hpp>

#include <glog/logging.h>

namespace libcf {

File::File(const std::string& filename, const std::string& flag) : f_(new std::fstream) {

  std::ios::openmode mode;
  if (flag == "r") {
    mode = std::ios::in;
  } else if (flag == "w") {
    mode = std::ios::out;
  } else if (flag == "rb") {
    mode = std::ios::in | std::ios::binary;
  } else if (flag == "wb") {
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

void File::read_line(std::string& line, char delim) const {
  if (good()) {
    std::getline(*f_, line, delim);
  }
}

std::string File::read_line(char delim) const {
  std::string line;
  read_line(line, delim);
  return std::move(line);
}

void File::write(const std::string& line) const {
  if(good()){
    f_->write(line.c_str(), line.size());
  }
}

void File::write_line(const std::string& line) const {
  if(good()){
    f_->write(line.c_str(), line.size());
    if (line.back() != '\n') 
      f_->write("\n", 1);
  }
}
} // namespace 

