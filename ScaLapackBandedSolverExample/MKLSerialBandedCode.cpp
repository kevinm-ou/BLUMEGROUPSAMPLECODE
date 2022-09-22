#include <iostream>
#include <chrono>
#include <thread>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include "mkl.h"
#include "mkl_lapacke.h"

using namespace std::chrono;
using namespace std;
//  run with
// ./MKLSerialBandedCode.sout

void GetStats(int numiterations, double* timings, double* statvalues){
	//statvalues is a 2 values array
	//statvalues[0] = average statvalues[1] = variance
	int N = numiterations -1; //we skip the first timing
	//we loop over each value to get average
	double tempavg=0; //local average
	double tempvar = 0; //local variance
	for(int i = 1; i<numiterations;i++){
		tempavg = tempavg + timings[i];
	}
	tempavg = tempavg/N; //get our average
	
	//we now calculate the std
	for(int i = 1; i<numiterations;i++){
		tempvar = pow(timings[i]-tempavg,2)+tempvar;
	}
	tempvar = tempvar/(N-1); //get our variance
	//We now assign to our variables
	statvalues[0] = tempavg;
	statvalues[1] = tempvar;
	return;
	
}


int main(int argc, char **argv) {

	//We get our mpi variables
	int myrank_mpi, nprocs_mpi, provided;

	MKL_INT N=20;
	int n_int = N;
	MKL_INT i;
	MKL_INT ipiv[n_int];
	MKL_INT NRHS =1;
	MKL_INT ku = 3; //how many upper diagonals
	MKL_INT kl = 3; //how many lower diagonals
	MKL_INT BWU = ku;
	MKL_INT BWL = kl;
	MKL_INT M = 2*kl+ku+1;
	MKL_INT ldb =N*M;
	//double tempb[] =  {4.42, 27.13, -6.14, 10.5};
	double *b;
	b = (double *)mkl_calloc(n_int,sizeof(double),64) ; //initialize local arrays ... here we skip global   
	//std::cout << ldb;
	//double ab[ldb];
	double *A; //Matrix A
	A = (double *)mkl_calloc(ldb,sizeof(double),64) ; //initialize local arrays ... here we skip global   
	//Define our diagonal elements
	double bandA = -1;
	double bandB = -3;
	double bandC = -5;
	double bandD = 8;
	double bandE = 7;
	double bandF = 6;
	double bandG = 2;
	int indexglobal, iglobal, jglobal;
	char trans = 'N';
	int numberofiterations = 25;
	MKL_INT info; //for lapack calls
	double *timingsforcall; //holds times for calls
	timingsforcall = (double *)calloc(numberofiterations,sizeof(double)) ;
	auto startglobal = high_resolution_clock::now();
	for(int iterationnumber = 0; iterationnumber<numberofiterations; iterationnumber++){
	if(iterationnumber>0){auto startglobal = high_resolution_clock::now();}
	for(jglobal =0;jglobal<N;jglobal++){ //loop over columns
		for(iglobal =0;iglobal<M;iglobal++){ //loop over rows
			indexglobal  = jglobal*M+iglobal;
			if(iglobal <(BWL)){ //Empty first rows
				A[indexglobal] = 0;
				continue;
			}
			if(jglobal+iglobal-BWL<BWU && iglobal>= BWU){ //Empty upper left triangle from sparse matrix
				A[indexglobal] = 0;
				continue;	
			}
			if(iglobal+jglobal-M-N+2>-BWL && N-BWL <= jglobal){ //Empty lower right triangle from sparse matrix
				A[indexglobal] = 0;
				continue;	
			}
				//We now use a switch to handle each band
				switch(iglobal){
					case 3: //Band A assignment
						A[indexglobal] = bandA*(iterationnumber+1);
						break;
					case 4: //Band B assignment
						A[indexglobal] = bandB*(iterationnumber+1);
						break;
					case 5: //Band C assignment
						A[indexglobal] = bandC*(iterationnumber+1);
						break;
					case 6: //Band D assignment
						A[indexglobal] = bandD*(iterationnumber+1);
						break;
					case 7: //Band E assignment
						A[indexglobal] = bandE*(iterationnumber+1);
						break;
					case 8: //Band F assignment
						A[indexglobal] = bandF*(iterationnumber+1);
						break;
					case 9: //Band G assignment
						A[indexglobal] = bandG*(iterationnumber+1);
						break;
					default:
						cout<<"We have an invalid band case!!! Aborting!"<<endl;
						exit(0);
					}
		}
	}
	for(i=0;i<N;i++){b[i]=i+1; }
	//for(i=0;i<ldb;i++) printf("%f\n", A[i]);
	//exit(0);
	// Use auto keyword to avoid typing long
	// type definitions to get the timepoint
	// at this instant use function now()
	auto start = high_resolution_clock::now();
 	//info = dgbsv(LAPACK_COL_MAJOR,N,kl,ku,NRHS,A,M,ipiv,b,N);
 	info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, N,kl,ku,NRHS,A,M,ipiv,b,N);
 	if(info != 0){
 	cout<<"\n Error code on DGBSV call: "<<info<<endl;
 	exit(0);
 	}
 	auto stop = high_resolution_clock::now();
	 // Subtract stop and start timepoints and
	// cast it to required unit. Predefined units
	// are nanoseconds, microseconds, milliseconds,
	// seconds, minutes, hours. Use duration_cast()
	// function.
 	auto duration = duration_cast<milliseconds>(stop - start); //we look in seconds
	 // To get the value of duration use the count()
	// member function on the duration object
	timingsforcall[iterationnumber] = duration.count();
  	}
  	auto stopglobal = high_resolution_clock::now();
  	auto durationglobal = duration_cast<milliseconds>(stopglobal - startglobal); //we look in ms	
        
  	
  	///print out our timings
  	cout <<"\n The timings for the serial version are : (";
  	for(int z=0;z<numberofiterations;z++){
  		cout<<timingsforcall[z]<<",";
  	}
  	cout<<") in ms for N="<<N<<endl;
  	
  	//we get statistics
	double stats[4]; //stat[0] = avg stat[1] =var stat[2] = max stat[3] = min
  	GetStats(numberofiterations, timingsforcall, stats); //normal
  	stats[2] = *std::max_element(timingsforcall+1, timingsforcall + numberofiterations);
   	stats[3] = *std::min_element(timingsforcall+1, timingsforcall + numberofiterations);
  	cout<<"N Meshpts: "<< n_int <<"\t Avg: "<<stats[0]<<"\t Var: "<<stats[1]<<"\t Max: "<<stats[2]<<"\t Min: "<<stats[3]<<"\t Global: "<<durationglobal.count()<<"Trials"<<numberofiterations<<endl;
  	//write results to file
  	//"N Meshpts","t_avg(ms)","t_var(ms)","t_max(ms)","t_min(ms)","t_total(ms)","Trials"
  	string timefilename = "./Timings/TimingSerialData.csv";
  	//
  	std::ofstream filetiming(timefilename, ios::app);
  				filetiming<<n_int<<","<<stats[0]<<","<<stats[1]<<","<<stats[2]<<","<<stats[3]<<","<<durationglobal.count()<<","<<numberofiterations<<endl;
  	//int info;
	//dgetrf_(&N, &N, A, &ldb, ipiv, &info);
	// dgetrs_(&trans, &N, &NRHS, A, &ldb, ipiv, b, &ldb, &info);
	// MKL_INT info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, N, 1, 2, 1, A, N, ipiv, b, 1);
    	//for(i=0;i<N;i++) {printf("%f\t",b[i]);}
       //for(i=0;i<N;i++) {printf("%f\t",b[i]);}
        //for(i=0;i<N;i++) cout<<b[i]<<","<<endl;
	mkl_free(A);
	mkl_free(b);
    free(timingsforcall);


      }