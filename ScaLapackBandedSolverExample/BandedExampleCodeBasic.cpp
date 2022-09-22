#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include "mpi.h"
#include "mkl.h"
#include "mkl_scalapack.h"
#include "mkl_blacs.h"
using namespace std;


//run with 
//  mpirun -np 3 ./BandedExampleCodeBasic.mpi
//We will use a block-cyclic type declaration here

/*Type define*/
typedef MKL_INT MDESC[ 9 ];

/*Parameters*/
const MKL_INT iZERO = 0, iONE =1 , iNEONE = -1;
const char layout='R'; // Row major processor mapping
const char TRANS = 'N'; //For pdgbsv call

int main(int argc, char **argv) {
//     This is an example of using PDGBTRF and PDGBTRS.
//     A matrix of size 9x9 is distributed on a 1x3 process
//     grid, factored, and solved in parallel.
	//We get our mpi variables
	int myrank_mpi, nprocs_mpi;
	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);
	
	//Define helping variables
	MKL_INT INFO,NPROCS, ICTXT, MYCOL, MYROW, NPCOL, NPROW,IAM,iam, nprocs, INFO2,NRHS;
	INFO =0;
	INFO2 = 0;
	//Define matrix specificaions
	MKL_INT BWL, BWU, N,M, NB, MB, LAF, LWORK, i_index, j_index;
	int n_int, nb_int;
	int LAF_int, LWORK_int;
	//Assign values to above variables
	BWL = 1; //How many lower bands
	BWU = 1; //How many upper bands
	M = 2*(BWL+BWU)+1; //How many rows in sparse matrix
	N = 10; //How many entries along a row
	NB = 4; //Blocking factor for columns
	MB = M; //Blocking factor for rows
	NRHS =1; //number of rhs
	//LAF=NB*(BWL+BWU)+6*max(BWL,BWU)*max(BWL,BWU);
    LAF= (int) (NB+BWU)*(BWL+BWU)+6*(BWL+BWU)*(BWL+2*BWU)+max(NRHS*(NB+2*BWL+4*BWU), 1);
	LWORK_int = LAF;
	//cout<<LAF<<endl;
	//exit(0);
	//Size of user-input Auxiliary Fillin space AF. Must be >=
 //*          NB*(bwl+bwu)+6*max(bwl,bwu)*max(bwl,bwu)
 //*          If LAF is not large enough, an error code will be returned
 //*          and the minimum acceptable size will be returned in AF( 1 )
	LWORK = LAF;
	LAF_int = LAF;
	LWORK_int = LAF;
	//we now declare more helping variables
	MDESC DESCA,DESCB, DESCA_local,DESCB_local, IPIV;
	int FILL_IN;

	//local arays
	double *WORK;//[LWORK];
	//AF = (double *)calloc(LAF,sizeof(double)) ;
	WORK = (double *) mkl_calloc(LWORK_int,sizeof(double),64) ;
	//we now initialize the process grid
	NPROW =1; //one row
	NPCOL =3; //three columns
	blacs_pinfo(&iam, &nprocs) ; // BLACS rank and world size
	blacs_get(&iNEONE, &iZERO, &ICTXT ); // -> Create context
	blacs_gridinit(&ICTXT, &layout, &NPROW, &NPCOL ); // Context -> Initialize the grid
	blacs_gridinfo(&ICTXT, &NPROW, &NPCOL, &MYROW, &MYCOL ); // Context -> Context grid info 
	                                                          //(# procs row/col, current procs row/col)
	//We now allocate our locate arrays
    MKL_INT mpA = M;//numroc_(&N, &NB, &MYROW, &iZERO, &NPROW); //get local rows
    MKL_INT nqA = numroc(&N, &NB, &MYCOL, &iZERO, &NPCOL); //get local columns
    //Test print

        printf("Hi. Proc %d/%d for MPI, proc %d/%d for BLACS in position (%d,%d)/(%d,%d) with local matrix %dx%d, global matrix %d, block size %d\n by %d\n ",myrank_mpi,nprocs_mpi,iam,nprocs,MYROW,MYCOL,NPROW,NPCOL,mpA,nqA,N,NB,MB);
        blacs_barrier(&ICTXT, "All");


    for (int id = 0; id < nprocs; ++id) { //looping over each id?
    					     // And putting up a barrier?
        blacs_barrier(&ICTXT, "All");
    }  
   // goto stop;
        //We now initialize our local arrays
	double *A; //Matrix A
	A = (double *) mkl_calloc(mpA*nqA,sizeof(double),64) ; //initialize local arrays ... here we skip global   
	double *B; //Matrix B
	B = (double *)mkl_calloc(nqA,sizeof(double),64) ; //initialize local arrays ... here we skip global   
	//We add a safety check on A
	if (A==NULL || B==NULL){
	 //printf("Error of memory allocation A on proc %dx%d\n",MYROW,MYCOL); exit(0); 
	 cout<<"\nError of memory allocation A\t"<<MYROW<<" "<<MYCOL<<endl;
	 goto stop;
	 }
	//We now distribute the global array into the local arrays
/*

*
*     Generate matrices A and B and distribute them to the process grid
*
*           P1          P2           P3         X        B
*
*      / 1   1     .           .           \  / 1 \    / 2 \
*      | 1   1   1 .           .            | | 1 |    | 3 |
*      |     1   1 . 2         .            | | 1 |    | 4 |
*      |         1 . 2   2     .            | | 1 |    | 5 |
*      |           . 2   2   2 .            | | 1 | =  | 6 |
*      |           .     2   2 . 3          | | 1 |    | 7 |
*      |           .         2 . 3   3      | | 1 |    | 8 |
*      |           .           . 3   3   3  | | 1 |    | 9 |
*      \           .           .     3   3 /  \ 1 /    \ 6 /
*
*/
	FILL_IN = (BWU+BWL)*nqA; //The fill in rows
	cout<<MYCOL<<"\t"<<nqA<<"\t"<<mpA<<"\t"<<FILL_IN<<endl;
	//goto stop;
	//This is for NB=4
	if(MYCOL == 0){
		//Processor 0
		A[2] = 0.0; //A[0] = *
		A[3] = 1.0; //A[1]
		A[4] = 1.0; //A[2]
		A[7] = 1.0; //A[3]
		A[8] = 1.0; //A[4]
		A[9] = 1.0; //A[5]
		A[12] = 1.0; //A[6]
		A[13] = 1.0; //A[7]
		A[14] = 1.0; //A[8]
		A[17] = 2.0; //A[9]
		A[18] = 2.0; //A[10]
		A[19] = 2.0; //A[11]
		B[0] = 2;
		B[1] = 3;
		B[2] = 4;
		B[3] = 5;
		}
	else if(MYCOL == 1){
		//Processor 1
		A[2] = 2.0; //A[0] = *
		A[3] = 2.0; //A[1]
		A[4] = 2.0; //A[2]
		A[7] = 2.0; //A[3]
		A[8] = 2.0; //A[4]
		A[9] = 2.0; //A[5]
		A[12] = 2.0; //A[6]
		A[13] = 2.0; //A[7]
		A[14] = 2.0; //A[8]
		A[17] = 3.0; //A[9]
		A[18] = 3.0; //A[10]
		A[19] = 3.0; //A[11]		
		B[0] = 6;
		B[1] = 6;
		B[2] = 7;
		B[3] = 8;
		}
	else if(MYCOL == 2){
		//Processor 2
		A[2] = 3.0; //A[0] 
		A[3] = 3.0; //A[1]
		A[4] = 3.0; //A[2]
		A[7] = 3.0; //A[3]
		A[8] = 3.0; //A[4] = *
		B[0] = 9;
		B[1] = 6;
		}
	else{
		cout<<"Not on grid!!"<<endl;
		goto stop;
	}
	//Test our input distributed arrays
	cout<<"A Before=> Row: "<<MYROW<<", Col: "<<MYCOL<<":\t(";
	for(int k = 0; k < mpA*nqA ;k++){
	cout<<A[k] <<",\t";
	}
	cout<<")"<<endl;     
	cout<<"B Before=> Row: "<<MYROW<<", Col: "<<MYCOL<<" , NB:"<<NB<<" (";
	for(int k = 0; k < nqA ;k++){
	cout<<B[k] <<",\t";
	}
	cout<<")"<<endl;  
	blacs_barrier(&ICTXT, "All");
	//test b input
	//cout<<"Initial "<<MYROW<<",\t"<<MYCOL<<"\t("<<B[0]<<","<<B[1]<<","<<B[2]<<")"<<endl;
	
	//goto stop;

	//We finally are ready to give our descriptor arrays
	//goto stop;
		                                                        
//
//     DISTRIBUTE THE MATRIX ON THE PROCESS GRID
//     Initialize the array descriptors for the matrices A and B
//

      DESCA[ 0 ] = 501;                   // descriptor type
      DESCA[ 1 ] = ICTXT;                 // BLACS process grid handle
      DESCA[ 2 ] = N;                     // number of columns in global A
      DESCA[ 3 ] = NB;                    // Blocking factor of the distribution Column Block  Size (Global)
      DESCA[ 4 ] = 0;                     // Process column over which first column of the global matrix A is distributed (Global)
      DESCA[ 5 ] = M;         // leading dimension of A (local)
      DESCA[ 6 ] = 0;                     // process row for 1st row of A
      DESCA[ 7 ] = 0;                     // Not used
      DESCA[ 8 ] = 0;                     // Not used

      DESCB[ 0 ] = 502;                   // descriptor type
      DESCB[ 1 ] = ICTXT;                 // BLACS process grid handle
      DESCB[ 2 ] = N;                     // number of rows in B
      DESCB[ 3 ] = NB;                    // Blocking factor of the distribution
      DESCB[ 4 ] = 0;                     // size of block rows
      DESCB[ 5 ] = NB;                    // leading dimension of B
      DESCB[ 6 ] = 0;                     // process row for 1st row of B
      DESCB[ 7 ] = 0;                     // Not used
      DESCB[ 8 ] = 0;                     // Not used
	//A (501,0,10,4,0,5,0)
	//B (502,0,10,4,0,4,0)
      //DESCA[ 0 ] = 501;                   // descriptor type
      //DESCB[ 0 ] = 502;                   // descriptor type

//Now the actual Scalapack calls are done
/*
**
*     Perform LU factorization
*
*/
//void	pdgbtrf(const MKL_INT* n, const MKL_INT* bwl, const MKL_INT* bwu, double* a, const MKL_INT* ja, const MKL_INT* desca, MKL_INT* ipiv, double* af, const MKL_INT* laf, double* work, const MKL_INT* lwork, MKL_INT* info);
       /*
       pdgbtrf( &N, &BWL, &BWU, A, &iONE, DESCA, IPIV,AF, &LAF, WORK, &LWORK, &INFO );
       if(INFO!=0){
       std::cout << endl<<"Info flag from PDGBTRF = "<<INFO<< ", Col = "<<MYCOL<<endl;
	goto stop;
       }  

	//test A LU factorization
	cout<<"A AFTER LU=> Row: "<<MYROW<<", Col: "<<MYCOL<<":\t(";
	for(int k = 0; k < mpA*nqA ;k++){
	cout<<A[k] <<",\t";
	}
	cout<<")"<<endl; 
	*/
	//We now run the sparse matrix solver
/*
*
*     Solve using the LU factorization from PDGBTRF
*
*/	    

	/*
	pdgbtrs(&TRANS, &N, &BWL, &BWU, &iONE, A, &iONE, DESCA, IPIV, B, &iONE, DESCB, AF, &LAF, WORK, &LWORK, &INFO);
       
       if(INFO!=0){
       std::cout << endl<<"Info flag from PDGBTRS = "<<INFO<< ", Col = "<<MYCOL<<endl;
	goto stop;
       }  
       */
        blacs_barrier(&ICTXT, "All");
	cout <<"\nA PC:"<< MYCOL<<"\t Error:"<<INFO<<"\t("<<DESCA[0]<<","<<DESCA[1]<<","<<DESCA[2]<<","<<DESCA[3]<<","<<DESCA[4]<<","<<DESCA[5]<<","<<DESCA[6]<<","<<DESCA[7]<<","<<DESCA[8]<<")\n";
	cout <<"\nB PC:"<< MYCOL<<"\t Error:"<<INFO2<<"\t("<<DESCB[0]<<","<<DESCB[1]<<","<<DESCB[2]<<","<<DESCB[3]<<","<<DESCB[4]<<","<<DESCB[5]<<","<<DESCB[6]<<","<<DESCB[7]<<","<<DESCB[8]<<")\n";
        blacs_barrier(&ICTXT, "All");
       //Perform driver call
       pdgbsv(&N,&BWL, &BWU, &iONE, A, &iONE, DESCA, IPIV, B, &iONE, DESCB, WORK, &LWORK, &INFO);	
       if(INFO!=0){
       std::cout << endl<<"Info flag from pdgbsv = "<<INFO<< ", Col = "<<MYCOL<<endl;
	goto stop;
       } 
       
       
           for (int id = 0; id < nprocs; ++id) { //looping over each id?
    					     // And putting up a barrier?
        blacs_barrier(&ICTXT, "All");
    }  
	//We test our final results
	cout<<"B final: MyCOL "<<MYCOL<<"MY ROW "<<MYROW<<"\t(";
	for(int k = 0; k < nqA ;k++){
	cout<<B[k] <<",\t";
	}
	cout<<")"<<endl;


	goto stop;
	stop:
	mkl_free(A);
	mkl_free(B);
	blacs_gridexit_(&ICTXT);
	MPI_Finalize();
	cout<<"!!Program Done!!"<<endl;
	return 0;	
}
