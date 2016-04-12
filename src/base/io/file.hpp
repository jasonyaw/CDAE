#ifndef _LIBCF_FILE_HPP_
#define _LIBCF_FILE_HPP_

#include <vector>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace libcf {

class File {
 public:
  File(const std::string& filename, const std::string& flag);
  
  ~File() { if (is_open()) { close(); } }

  bool is_open() const { return f_->is_open(); } 
  bool good() const { return f_->good(); }
  bool ok() const { 
    if (*f_) { return true; }
    else { return false; }
  } 

  size_t size() const;
  void restart() const { f_->seekg(0, f_->beg); }

  bool read_line(std::string& line, char delim = '\n') const;
  std::string read_line(char delim = '\n') const;

  bool write_str(const std::string& line) const;
  bool write_line(const std::string& line) const;
  
  template <class T>
    bool read(T* t, size_t n = 1) const;

  template <class T> 
    bool write(T* t, size_t n = 1) const;


  template<class T>
      bool write_vector(const std::vector<T>& vec) const;

  template<class T>
      bool read_vector(std::vector<T>& vec) const;

  template<class T>
      std::vector<T> read_vector() const;

  template<class Iterator>
      bool write_iterator(const Iterator& first, 
                          const Iterator& last) const;
  
  template<class Archive, class T>
      void serialize_save(const T& t) ;
  
  template<class Archive, class T>
      void serialize_load(T& t);

  void close() const { f_->close(); }

 private:
  bool is_binary = false;
  bool read_only = true;
  std::unique_ptr<std::fstream> f_;
  std::string name_;
};

template<class T>
bool File::read(T* t, size_t n) const {
  CHECK_EQ(is_binary, true);
  CHECK_EQ(read_only, true);
  if (good()) {
    f_->read(reinterpret_cast<char*>(t), n * sizeof(T));
  }
  return ok();
}

template<class T> 
bool File::write(T* t, size_t n) const {
  CHECK_EQ(is_binary, true);
  CHECK_EQ(read_only, false);
  if (good()) {
    f_->write(reinterpret_cast<char*>(t), n * sizeof(T));
  }
  return ok();
}

template<class T>
bool File::write_vector(const std::vector<T>& vec) const {
  CHECK_EQ(is_binary, true);
  CHECK_EQ(read_only, false);
  write_iterator(vec.begin(), vec.end());
  return ok();
}

template<class T>
bool File::read_vector(std::vector<T>& vec) const {
  CHECK_EQ(is_binary, true);
  CHECK_EQ(read_only, true);
  size_t vec_size;
  f_->read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
  vec.resize(vec_size);
  f_->read(reinterpret_cast<char*>(&vec[0]), vec.size() * sizeof(T)); 
  return ok();
}

template<class T>
std::vector<T> File::read_vector() const {
  CHECK_EQ(is_binary, true);
  CHECK_EQ(read_only, true);
  std::vector<T> ret;
  read_vector<T>(ret);
  return std::move(ret);
}

template<class Iterator>
bool File::write_iterator(const Iterator& first, 
                          const Iterator& last) const {
  CHECK_EQ(is_binary, true);
  CHECK_EQ(read_only, false);
  size_t length = std::distance(first, last);
  f_->write(reinterpret_cast<const char*>(&length), sizeof(size_t));
  f_->write(reinterpret_cast<const char*>(&(*first)), length * sizeof(*first));
  return ok();
}


template<class Archive, class T>
void File::serialize_save(const T& t) {
  boost::iostreams::filtering_stream<boost::iostreams::output> out;
  out.push(boost::iostreams::gzip_compressor());
  out.push(*f_);
  Archive ar(out);
  ar << t;
}

template<class Archive, class T>
void File::serialize_load(T& t) {
  boost::iostreams::filtering_stream<boost::iostreams::input> input;
  input.push(boost::iostreams::gzip_decompressor());
  input.push(*f_);
  Archive ar(input);
  ar >> t;
}

} // namespace

#include <base/io/file-inl.hpp>

#endif // _LIBCF_FILE_HPP_
