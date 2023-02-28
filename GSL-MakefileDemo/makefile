######################################################################
## Filename:      Makefile
## Author:        Kevin Mack-Fisher <kevin.d.mackfischer@ou.edu>
## Version:       
## Description:   Makefile for Example for Group
######################################################################
######################################################
# common (trivial) flags
######################################################
SHELL=/bin/sh
AR := ar
RANLIB := ranlib
DIR1 := -I./Gauss 
#
# Flags for debugging:
#
FC := gfortran
FFLAGS := -g -O3 

CC := gcc
CFLAGS := -std=c99

CXX := g++
CXXFLAGS := -std=c++17 -O3 


#LINKER FLAGS

LFLAGS = -Lgsl/lib -lgsl -lgslcblas -lm
INCLDIRS = -Igsl/include
#Source files:

#Just add subroutine_name.o to OBJ1 to add more
OBJ1 := mysubroutine.o
OBJ2 := main.o
TARGET1 := test.out

$(TARGET1): $(OBJ2) $(OBJ1)
	$(CC) $(CFLAGS) $(INCLDIRS) -o $@ $^ $(LFLAGS)

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
