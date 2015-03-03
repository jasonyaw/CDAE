BOOST_DIR = /usr/local

# Where to find src code.
SRC_DIR = ../../src

CXX = g++
CFLAGS = -O3 -g -std=c++11 #-shared -fPIC
LDFLAGS= -lpthread -lboost_serialization -lboost_iostreams -lglog -lgflags 
INCLUDE = -I$(SRC_DIR) -I$(BOOST_DIR)/include 
LIBS = -L$(BOOST_DIR)/lib -Wl,-rpath $(BOOST_DIR)/lib 

BIN =  yelp_implicit 
SOURCES = yelp_implicit.cpp
OBJ = $(SOURCES:.cpp=.o)

all:  $(BIN) 

yelp_implicit : yelp_implicit.o 
	$(CXX) $(CFLAGS) $(INCLUDE) $(LIBS) yelp_implicit.o -o $@  $(LDFLAGS) 

.cpp.o: 
	$(CXX) $(INCLUDE) -c $(CFLAGS) $< -o $@ 

clean:
	$(RM) $(BIN) $(OBJ) 


