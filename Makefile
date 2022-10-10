CC = gcc
CXX = g++

CFLAGS = -g -std=gnu99 -Wall -Wno-unused-parameter -Wno-unused-function -O3
CXXFLAGS = -g -Wall -O3
CPPFLAGS = $$(pkg-config --includes opencv4)
LDFLAGS = $$(pkg-config --libs opencv4) -lpthread -lm


opencv_test : opencv_test.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I/usr/include/opencv4 -o $@ $^ $(LDFLAGS)
