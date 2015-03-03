#ifndef _LIBCF_FILE_LINE_READER_HPP_
#define _LIBCF_FILE_LINE_READER_HPP_


#include <base/io/file.hpp>

namespace libcf {

typedef std::function<void (const std::string&, size_t)> line_callback_t;

class FileLineReader {

 public:
  explicit FileLineReader(const std::string& filename) 
      : filename_(filename), loaded_successfully_(false) {}

  ~FileLineReader() {}

  void set_line_callback(const line_callback_t& callback) {
    line_callback_ = callback;
  }

  void load();
  
  bool loaded_successfully() const { return loaded_successfully_; }

 private:
  const std::string& filename_;
  line_callback_t line_callback_;
  bool loaded_successfully_;
};


} // namespace

#include <base/io/file_line_reader-inl.hpp>

#endif // _LIBCF_FILE_LINE_READER_HPP_
