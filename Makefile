CC=g++
CXXFLAGS=-std=c++0x -Wall `pkg-config opencv --cflags`  
# -gdwarf-3 for different format of debugging info
# -Wall to enable all compiler's warning messages
# c++0x partially support c++11 features

LDFLAGS=-g `pkg-config opencv --libs`  # -g for debugging symbols

all: clean headTracking

headTracking: cvLib.o Tracker.o
	${CC} -o headTracking *.o headTracking.cpp ${CXXFLAGS} ${LDFLAGS}
#  use *.o to link with all .o files, otherwise will ignore all .o files

cvLib.o: cvLib.hpp

Tracker.o: Tracker.hpp

clean: 
	rm -f headTracking *.o
