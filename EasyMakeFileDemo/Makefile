######################################################################
## Filename:      Makefile
## Author:        Kevin Mack-Fisher <kevin.d.mackfischer@ou.edu>
## Version:       
## Description:   Makefile for Example for Group
######################################################################
######################################################
# common (trivial) flags
######################################################

AR := ar
RANLIB := ranlib
DIR1 := -I./Gauss 
#
# Flags for debugging:
#
FC := gfortran
FFLAGS := -g -O3 

CC := gcc
CFLAGS := -g

CXX := g++
CXXFLAGS := -std=c++17 -O3 


#LINKER FLAGS

LFLAGS := 


#Source files:


OBJ1 := HelloWorld.o HelloWorldNoHeader.o
OBJ2 := main.o
TARGET1 := HelloWorldExample.out

$(TARGET1): $(OBJ2) $(OBJ1)
	$(CXX) $(CXXFLAG) $(LFLAGS) $^ -o $@

TARGETS := $(TARGET1)
default: TARGET1

all: $(TARGETS)


.PHONY: clean

clean:
	\rm -f *.o *.mod $(TARGETS) 


#
# here's how we translate the files:
#
%.o: %.f90 Makefile
	$(FC) $(FFLAGS) -c $< 

%.o: %.f Makefile
	$(FC) $(FFLAGS) -c $< 
%.o: %.c Makefile
	$(CC) $(CFLAGS) -c $<

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -c $<

