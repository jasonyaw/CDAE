#ifndef _LIBCF_SERIALIZE_HPP_
#define _LIBCF_SERIALIZE_HPP_

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <glog/logging.h>

#include <base/timer.hpp>
#include <base/io/file.hpp>

namespace libcf {

template<typename T>
    inline void save(const T& t, const std::string& filename, bool binary_format = true) {
      Timer timer;
      std::string openmode = "w";
      if (binary_format) openmode += "b";

      File f(filename, openmode);
      if (binary_format) {  
        f.serialize_save<boost::archive::binary_oarchive, T>(t);
      } else {
        f.serialize_save<boost::archive::text_oarchive, T>(t);
      }
      f.close();
      LOG(INFO) << "Save data to " << filename << " in " << timer;
    }

template<typename T>
    inline void load(const std::string& filename, T& t, bool binary_format = true) {
      Timer timer;
      std::string openmode = "r";
      if (binary_format) openmode += "b";

      File f(filename, openmode);
      if (binary_format) {  
        f.serialize_load<boost::archive::binary_iarchive, T>(t);
      } else {
        f.serialize_load<boost::archive::text_iarchive, T>(t);
      }
      f.close();
      LOG(INFO) << "Load data from " << filename << " in " << timer;
    }

} // namespace

#endif // _LIBCF_SERIALIZE_HPP_
