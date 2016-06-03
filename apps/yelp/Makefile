BOOST_DIR = /usr/local

# Where to find src code.
SRC_DIR = ../../src

CXX = g++
CFLAGS = -O3 -g -std=c++11 #-shared -fPIC
LDFLAGS= -lpthread -lboost_serialization-mt -lboost_iostreams-mt -lglog -lgflags 
INCLUDE = -I$(SRC_DIR) -I$(BOOST_DIR)/include 
LIBS = -L$(BOOST_DIR)/lib -Wl,-rpath $(BOOST_DIR)/lib 

BIN =  yelp 
SOURCES = yelp.cpp
OBJ = $(SOURCES:.cpp=.o)

all:  $(BIN) 

yelp : yelp.o 
	$(CXX) $(CFLAGS) $(INCLUDE) $(LIBS) yelp.o -o $@  $(LDFLAGS) 

.cpp.o: 
	$(CXX) $(INCLUDE) -c $(CFLAGS) $< -o $@ 

clean:
	$(RM) $(BIN) $(OBJ) 


