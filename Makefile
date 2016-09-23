CC=g++
CXXFLAGS=-g -std=c++0x -Wall `pkg-config opencv --cflags`  
# -gdwarf-3 for different format of debugging info
# -Wall to enable all compiler's warning messages
# c++0x partially support c++11 features

LDFLAGS=`pkg-config opencv --libs`  # -g for debugging symbols

all: clean headTracking

headTracking: global.o cvLib.o Tracker.o imgSVM.o cmpLib.o
	${CC} -o headTracking *.o headTracking.cpp ${CXXFLAGS} ${LDFLAGS}
#  use *.o to link with all .o files, otherwise will ignore all .o files

global.o: global.hpp

cvLib.o: cvLib.hpp

cmpLib.o: cmpLib.hpp

Tracker.o: Tracker.hpp

imgSVM.o: imgSVM.hpp

clean: 
	rm -f headTracking *.o
