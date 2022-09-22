######################################################################
## Filename:      Makefile
## Author:        Kevin M (Blume Group)
## Version:       
## Description:   Makefile for ScaLapack Banded Solver
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
FC := ifort
FFLAGS := -g -O3 -fopenmp

CC := mpiicc
CFLAGS := -g -O3 -fopenmp

CXX := mpiicpc
CXXS := icpc

CXXFLAGS := -g -O3 -qopenmp


#LINKER FLAGS

LFLAGS := -qmkl=parallel -L${MKLROOT}/lib/intel64 -I/${MKLROOT}/include -DMKL_ILP64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lpthread -ldl
LFLAGSSerial := -qmkl=parallel -L${MKLROOT}/lib/intel64 -I/${MKLROOT}/include -DMKL_ILP64 -lpthread -ldl

#Source files:

OBJ1 := BandedExampleCode.o
TARGET1 := BandedExampleCode.mpi
OBJ2 := BandedExampleCodeBasic.o
TARGET2 := BandedExampleCodeBasic.mpi
OBJ3 := MKLSerialBandedCode.o
TARGET3 := MKLSerialBandedCode.sout


default: $(TARGET1) $(TARGET2) $(TARGET3)
$(TARGET1): $(OBJ1)
	$(CXX) $(CXXFLAGS) $(LFLAGS) $^ -o $@
$(TARGET2): $(OBJ2)
	$(CXX) $(CXXFLAGS) $(LFLAGS) $^ -o $@
$(TARGET3): $(OBJ3)
	$(CXX) $(CXXFLAGS) $(LFLAGSSerial) $^ -o $@
TARGETS := $(TARGET1) $(TARGET2) $(TARGET3)


all: $(TARGETS)

.PHONY: clean

clean:
	\rm -f *.o *.mod $(TARGETS) *.lst Gauss/*.csv

cleancsv:
	\rm -f Gauss/*.csv
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

