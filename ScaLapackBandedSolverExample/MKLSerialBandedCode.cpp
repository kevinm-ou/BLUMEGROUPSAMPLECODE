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

using namespace std;
//  run with
// ./MKLSerialBandedCode.sout



int main(int argc, char **argv) {
	//We solve the following
/*

*
*     Generate matrices A and B and distribute them to the process grid
*
*           P0          Pi                                              P(Nprocs-1)          X            B
*       <--   P0    -->
*      / D   C   B   A .           .          	                       .              \  / x0      \    / b0   \  -
*      | E   D   C   B . A         .               	                   .               | | x1      |    | b1   |
*      | F   E   D   C . B   A     .          	                       .               | | x2      |    | b2   |
*      | G   F   E   D . C   B   A .          	                       .               | | x3      |    | b3   |
*      |     G   F   E . D   C   B . A        	                       .               | | x4      | =  | b4   |
*      |         G   F . E   D   C . B   A    	                       .               | | x5      |    | b5   |
*      |             G . F . E   D . C . B   A	                       .               | | x6      |    | b6   |
*      |               . *   *   * . *   *   *   *                     .               | | .       |    | *    |
*      |               .     *   * . *   *   *   *   *                 .               | | .       |    | *    | 
*      |               .         * . *   *   *   *   *   *             .               | | .       |    | *    .
*      |               .           . *   *   *   *   *   *   *         .               | | .       |    | *    |
*      |               .           .     *   *   *   *   *   *   *     .               | | .       |    | *    |
*      |               .           .         *   *   *   *   *   *   * .               | | .       |    | *    |
*      |               .           .             *   *   *   *   *   * . *             | | .       |    | *    |
*      |               .           .                 *   *   *   *   * . *   *         | | .       |    | *    |
*      |               .           .                     G   F   E   D . C   B   A     | | x(N-4)  |    | bN-4 |  |
*      |               .           .                         G   F   E . D   C   B   A | | x(N-3)  |    | bN-3 |  |
*      |               .           .                             G   F . E   D   C   B | | x(N-2)  |    | bN-2 |  |
*      \               .           .                                 G . F   E   D   C/  \ x(N-1)  /    \ bN-1 /  -
*
*/
	


	MKL_INT N=8; //Length of our b-vector
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
	MKL_INT info; //for lapack calls
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
						A[indexglobal] = bandA;
						break;
					case 4: //Band B assignment
						A[indexglobal] = bandB;
						break;
					case 5: //Band C assignment
						A[indexglobal] = bandC;
						break;
					case 6: //Band D assignment
						A[indexglobal] = bandD;
						break;
					case 7: //Band E assignment
						A[indexglobal] = bandE;
						break;
					case 8: //Band F assignment
						A[indexglobal] = bandF;
						break;
					case 9: //Band G assignment
						A[indexglobal] = bandG;
						break;
					default:
						cout<<"We have an invalid band case!!! Aborting!"<<endl;
						exit(0);
					}
		}
	}
	//Fill b vector
	for(i=0;i<N;i++){
		b[i]=i+1; }
	//Print bi]
	printf("Right hand side \n");
	for(i=0;i<N;i++){
	    printf("b[%i]=%f \n",i,b[i]); }
	//for(i=0;i<ldb;i++) printf("%f\n", A[i]);
	//exit(0);

 	info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, N,kl,ku,NRHS,A,M,ipiv,b,N);
 	if(info != 0){
 	cout<<"\n Error code on DGBSV call: "<<info<<endl;
 	exit(0);
 	}
	//print results
	printf("Left hand side we solved for \n");

		for(i=0;i<N;i++){
		printf("b[%i]=%f \n",i,b[i]); }

	mkl_free(A);
	mkl_free(b);


      }