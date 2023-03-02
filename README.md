# BLUMEGROUPSAMPLECODE
This repository is a set of basic example codes for the Blume Group at OU. It is maintained by Kevin M.
The list of examples are below
1) ScaLapackBandedSolverExample: Getting started with MKL's ScaLapack.
  a)There are three files. 1) A basic explicit example of distributing your matrix in the case of a banded matrix. 2) A basic method of distributing your matrix in the case of a larger band matrix. This one includes timings so it doesn't show the results. Please note the size of distributed matrix must be a above some minimum or it will toss a error. (deprecated) 3) A serial version of example 2. 
  Compile with make.
  You must run "source {intel oneapi location}/setvars.sh intel64"
  before compiling / and or running to say where the libraries and compilers are.
  E.g., on giraffe, run:
  source source /myhome1/opt/intel/oneapi/setvars.sh intel64
  Should see,
  "
    :: initializing oneAPI environment ...
      bash: BASH_VERSION = 4.4.20(1)-release
      args: Using "$@" for setvars.sh arguments: intel64
    :: advisor -- latest
    :: ccl -- latest
    :: clck -- latest
    :: compiler -- latest
    :: dal -- latest
    :: debugger -- latest
    :: dev-utilities -- latest
    :: dnnl -- latest
    :: dpcpp-ct -- latest
    :: dpl -- latest
    :: inspector -- latest
    :: intelpython -- latest
    :: ipp -- latest
    :: ippcp -- latest
    :: ipp -- latest
    :: itac -- latest
    :: mkl -- latest
    :: mpi -- latest
    :: tbb -- latest
    :: vpl -- latest
    :: vtune -- latest
    :: oneAPI environment initialized ::
  "
  The mpi basic example runs with"
  mpirun -np 3 ./2DCARTSETEMainScala.lmpi 
2) EasyMakeFileDemo
  a) A simple demo on using Makefiles
  Run make to compile the files. It also shows what happens when you have a header file and if you don't want to use one. Best practice is to use one.

3) GSL-MakefileDemo
In example 2, it gives a simple make file demo that should help in most cases. For GSL, you have to be strict in the order you link the gsl libraries in general. This folder gives an example makefile for gsl that is robust for compiling on most (being safe in language) systems. It has a main file that calls a gsl function and a subroutine file that calls a gsl function (plus the header for linking). Sample output is shown in the SampleOutput.md file. 
4) A simple 2D array example file. It is a an example co-created by Xylo.
