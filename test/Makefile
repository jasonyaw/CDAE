GTEST_DIR = ../../gtest-1.7.0
BOOST_DIR = /usr/local
GLOG_GFLAGS_DIR = /usr/local

SRC_DIR = ../src

CXX = g++
CFLAGS = -O3 -g -std=c++11 #-shared -fPIC
LDFLAGS= -lpthread -lboost_serialization -lboost_iostreams -lglog 
INCLUDE = -I$(SRC_DIR) -I$(BOOST_DIR)/include -I$(GTEST_DIR)/include -I$(GLOG_GFLAGS_DIR)/include 
LIBS = $(GTEST_DIR)/lib/libgtest.a -L$(GLOG_GFLAGS_DIR)/lib -L$(BOOST_DIR)/lib 

BIN = run_all_tests
SOURCES = main_test.cpp 
 		
OBJ = $(SOURCES:.cpp=.o)

all: $(SOURCES)  $(BIN) 

$(BIN) : $(OBJ) 
	$(CXX) $(CFLAGS) $(INCLUDE) $(OBJ) -o $@ $(LIBS) $(LDFLAGS) 

.cpp.o: 
	$(CXX) $(INCLUDE) -c $(CFLAGS) $< -o $@ 

clean:
	$(RM) $(BIN)* $(OBJ) *~ *.dSYM test_data/*.bin

test:
	make clean && make && ./run_all_tests

