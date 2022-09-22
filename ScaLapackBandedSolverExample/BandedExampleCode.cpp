#include <stdio.h>
#include <cmath>
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
#include <omp.h>
using namespace std;

//run with
// mpirun -np YOURNUMOFPROCESSORSHERE BandedExampleCode.mpi

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


/*Type define*/
typedef MKL_INT MDESC[ 9 ];


/*Parameters*/
const MKL_INT iZERO = 0, iONE =1 , iNEONE = -1;
const char layout='R'; // Row major processor mapping
const char TRANS = 'N'; //For pdgbsv call
const char bALL = 'A'; //All barrier
const char bCol = 'C'; //Column barrier
const char bRow = 'R'; //Row barrier
const int MKLINTEGERALIGNMENT = 64; //what alignment for MKL_calloc integers
const int MKLDOUBLEALIGNMENT = 64; //what alignment for MKL_calloc doubles


int main(int argc, char **argv) {
//     This is an example of using PDGBTRF and PDGBTRS.
//     A matrix of size NxN is distributed on a 1xN process
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
    NRHS = 1; //the number of rhs we are solving for
	//Define matrix specificaions
	MKL_INT BWL, BWU, N,M, NB, MB, LAF, LWORK, i_index, j_index;
	int n_int, nb_int;
	int LAF_int, LWORK_int;
	//Assign values to above variables
	BWL = 3; //How many lower bands
	BWU = 3; //How many upper bands
	M = 2*(BWL+BWU)+1; //How many rows in sparse matrix
	N = 20; //How many entries along a row
    n_int = N;
	if(n_int%nprocs_mpi==0){
		NB = n_int/nprocs_mpi; //Blocking factor for columns
	}
	else{
		NB = n_int/nprocs_mpi+1; //Blocking factor for columns
	}
	MB = M; //Blocking factor for rows
	//LAF=NB*(BWL+BWU)+6*max(BWL,BWU)*max(BWL,BWU);
    LAF= (int) (NB+BWU)*(BWL+BWU)+6*(BWL+BWU)*(BWL+2*BWU)+max(NRHS*(NB+2*BWL+4*BWU), 1);
	//cout<<LAF<<endl;
	//exit(0);
	//Size of user-input Auxiliary Fillin space AF. Must be >=
 //*          NB*(bwl+bwu)+6*max(bwl,bwu)*max(bwl,bwu)
 //*          If LAF is not large enough, an error code will be returned
 //*          and the minimum acceptable size will be returned in AF( 1 )
	LWORK = LAF;
	//LAF_int = LAF;
	LWORK_int = LAF;
	//we now declare more helping variables
    	MDESC DESCA,DESCB, DESCA_local,DESCB_local;
	//int DESCA[9],DESCB[9], IPIV[NB], FILL_IN;
    	MKL_INT FILL_IN;
    	MKL_INT *IPIV = (MKL_INT *)mkl_calloc(NB, sizeof(MKL_INT) , MKLINTEGERALIGNMENT);
	//double A[(1+2*BWL+2*BWU)*NB];
	//double B[NB];
	double *WORK;//[LWORK];
    	WORK = (double *) mkl_calloc(LWORK,sizeof(double),MKLDOUBLEALIGNMENT) ;
	//we now initialize the process grid
	NPROW =1; //one row
	NPCOL = nprocs_mpi; //nprocs_mpi columns
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
        blacs_barrier(&ICTXT, &bALL); //wait until all processors report in
 
	bool BLACSroot = (MYCOL == 0); //Am I the root process?
  	 // goto stop;
        //We now initialize our local arrays
	double *A; //Matrix A
	A = (double *) mkl_calloc(mpA*nqA,sizeof(double),64) ; //initialize local arrays ... here we skip global   
	double *B; //Matrix B
	B = (double *)mkl_calloc(nqA,sizeof(double),64) ; //initialize local arrays ... here we skip global   
	//We add a safety check on A
	if (A==NULL || B==NULL){ printf("Error of memory allocation A on proc %dx%d\n",MYROW,MYCOL); exit(0); }
	//We will also initialize a local array on root that holds all the values
	MKL_INT Nroot;
	if(MYCOL == 0){
		Nroot = N;	
	}
	else{
		Nroot = 1;
	}
	double *Xroot; //This is used to collect the values onto the root process
	Xroot = (double *)mkl_calloc(Nroot,sizeof(double),64) ; //initialize local arrays ... here we skip global 
	  
	//We now distribute the global array into the local arrays
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
	FILL_IN = (BWU+BWL)*nqA; //The fill in rows
	cout<<"ID:"<<MYCOL<<"\tnqA:"<<nqA<<"\tmpA:"<<mpA<<"\tFIllIN:"<<FILL_IN<<"\tINFO:"<<INFO<<endl;
	//goto stop;
	
	//Define our diagonal elements
	double bandA = -1;
	double bandB = -3;
	double bandC = -5;
	double bandD = 8;
	double bandE = 7;
	double bandF = 6;
	double bandG = 2;
	//We now fill in our matrix
	//Define our indexing variables
	int jglobal, iglobal, indexglobal;
	int jlocal, ilocal, indexlocal;
	int startindex = MYCOL*NB*M; //local subarray starting index
	//index variables for B-vector
	//define indexing variables
	int bstartindex = MYCOL*NB; //starting index of the sub array
	int bindexlocal; //local index for b
	int bindexglobal; //global index for b
	//Define our root combine variables
        int sendN, destr; //combine on root variables
        int sendr = MYCOL*NB; //combine on root variables
        //Data for saving
        string realfilename = "./Data.csv";
        //We initialize our variables for timing
        int numberofiterations = 5; //how many times will we iterate
        double *timerresults; //array to hold timer results
        timerresults = (double *)mkl_calloc(numberofiterations,sizeof(double),64) ; //array to hold timer results
        double *timerresultsroot; //array to hold timer results to send to root process
        timerresultsroot = (double *)mkl_calloc(numberofiterations,sizeof(double),64) ; //array to hold timer results
        //we also allocate an array to hold statistical values
        double *timingstats; //normal timing stats
        timingstats = (double *)calloc(2,sizeof(double)) ; //normal timing stats
        double *timingstatsroot; //root timing stats
        timingstatsroot = (double *)calloc(2,sizeof(double)) ; //root timing stats
        double MPIt1; //time stamp 1
        double MPIt2; //time stamp 2

	//We will also initialize a local array on root that holds all the timing values
	int Ntimingsroot;
	if(MYCOL == 0){
		Ntimingsroot = nprocs_mpi;	
	}
	else{
		Ntimingsroot = 1;
	}
	double *myroottimings; //This is used to collect the average values onto the root process
	myroottimings = (double *)calloc(Ntimingsroot,sizeof(double)) ; //initialize local arrays ... here we skip global 
	double *myroottimingsvar; //This is used to collect the var values onto the root process
	myroottimingsvar = (double *)calloc(Ntimingsroot,sizeof(double)) ; //initialize local arrays ... here we skip global 

	double *myroottimingsroot; //This is used to collect the average values onto the root process
	myroottimingsroot = (double *)calloc(Ntimingsroot,sizeof(double)) ; //initialize local arrays ... here we skip global 
	double *myroottimingsvarroot; //This is used to collect the var values onto the root process
	myroottimingsvarroot = (double *)calloc(Ntimingsroot,sizeof(double)) ; //initialize local arrays ... here we skip global 
	//define average values now
	double timeaveragemin;
	double timeaveragemax;
	double timeaveragevar;
	double timeaveragesum;
	
	double timeaverageminroot;
	double timeaveragemaxroot;
	double timeaveragevarroot;
	double timeaveragesumroot;
	string timefilename = "./Timings/TimingData.csv";	
        //we loop over the solver multiple times
        blacs_barrier(&ICTXT, &bALL); //synchronize
        double runtimeglobalstart, runtimeglobalend; 
        runtimeglobalstart= MPI_Wtime();
        for(int solveriteration = 0; solveriteration<numberofiterations; solveriteration++){
		
		//We now
		//Check if we are on the process grid
		if(MYCOL<nprocs_mpi && MYCOL>=0){
			//Loop over each submatrix index
			for(jlocal =0;jlocal<nqA;jlocal++){ //loop over columns
				for(ilocal =0;ilocal<mpA;ilocal++){ //loop over rows
					//we calculate our indexing
					indexlocal  = jlocal*M+ilocal; //local index
					indexglobal = startindex+indexlocal; //global index
					if(MYCOL!=0){
					jglobal     = indexglobal/M; //global j index
					iglobal     = indexglobal%(jglobal*M); //global i index
					}
					else{
					iglobal     = ilocal; //global i index
					jglobal     = jlocal; //global j index
					}
					//we now assign our values to our subarrays
					//first we state the condition for a zero value entry
					if(iglobal <(BWU+BWL)){ //Empty first rows
						A[indexlocal] = 0;
						continue;
					}
					if(jglobal+iglobal-BWU-BWL<BWU && iglobal>= BWU+BWL){ //Empty upper left triangle from sparse matrix
						A[indexlocal] = 0;
						continue;	
					}
					if(iglobal+jglobal-M-N+2>-BWL && N-BWL <= jglobal){ //Empty lower right triangle from sparse matrix
						A[indexlocal] = 0;
						continue;	
					}
					//We now use a switch to handle each band
					switch(iglobal){
						case 6: //Band A assignment
							A[indexlocal] = bandA*(solveriteration+1);
							break;
						case 7: //Band B assignment
							A[indexlocal] = bandB*(solveriteration+1);
							break;
						case 8: //Band C assignment
							A[indexlocal] = bandC*(solveriteration+1);
							break;
						case 9: //Band D assignment
							A[indexlocal] = bandD*(solveriteration+1);
							break;
						case 10: //Band E assignment
							A[indexlocal] = bandE*(solveriteration+1);
							break;
						case 11: //Band F assignment
							A[indexlocal] = bandF*(solveriteration+1);
							break;
						case 12: //Band G assignment
							A[indexlocal] = bandG*(solveriteration+1);
							break;
						default:
							cout<<"We have an invalid band case!!! Aborting!"<<endl;
							goto stop;
					
					}
				
				}
			
			}
			
			
			}
		else{
			cout<<"Not on grid!!"<<endl;
			goto stop;
		}
		//goto stop;
		//We now fill our b vector

		//We now populate our b-vector
		for(bindexlocal=0;bindexlocal<nqA;bindexlocal++){
			B[bindexlocal] = bindexlocal+NB*MYCOL+1;
		}

		//Test our input distributed arrays
		/*
		cout<<"A Before=> Row: "<<MYROW<<", Col: "<<MYCOL<<":\t(";
		for(int k = 0; k < mpA*nqA ;k++){
		cout<<A[k] <<",\t";
		}
		cout<<")"<<endl;     
		*/
		/*
		cout<<"B Before=> Row: "<<MYROW<<", Col: "<<MYCOL<<":\t(";
		for(int k = 0; k < nqA ;k++){
		cout<<B[k] <<",\t";
		}
		cout<<")"<<endl;  
		*/
		//blacs_barrier(&ICTXT, &bALL);
		//test b input
		//cout<<"Initial "<<MYROW<<",\t"<<MYCOL<<"\t("<<B[0]<<","<<B[1]<<","<<B[2]<<")"<<endl;
		
		//goto stop;

		//We finally are ready to give our descriptor arrays
		//goto stop;
				                                                
	//
	//     Initialize the array descriptors for the matrices A and B
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

		//Now the actual Scalapack calls are done
		/*
		**
		*     Perform LU factorization
		*
		*/	
		//We wait until all arrays are initialized

	       blacs_barrier(&ICTXT, "All");
		//start time
		MPIt1 = MPI_Wtime();
		//we call the factorization routine
	       //pdgbtrf( &N, &BWL, &BWU, A, &iONE, DESCA, IPIV,AF, &LAF, WORK, &LWORK, &INFO );
	     //  if(INFO!=0){
	     //  std::cout << endl<<"Info flag from PDGBTRF = "<<INFO<< ", Col = "<<MYCOL<<endl;
		//goto stop;
	     //  }  

		//test A LU factorization
		/*
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


		//pdgbtrs_(&TRANS, &N, &BWL, &BWU, &iONE, A, &iONE, DESCA, IPIV, B, &iONE, DESCB, AF, &LAF, WORK, &LWORK, &INFO);
	    //  if(INFO!=0){
	    //   std::cout << endl<<"Info flag from PDGBTRS = "<<INFO<< ", Col = "<<MYCOL<<endl;
		//goto stop;
	    //   }

       //Perform driver call that combines the two above routines
        //blacs_barrier(&ICTXT, "All");

       pdgbsv(&N,&BWL, &BWU, &iONE, A, &iONE, DESCA, IPIV, B, &iONE, DESCB, WORK, &LWORK, &INFO);	
        if(INFO!=0){
            std::cout << endl<<"Info flag from pdgbsv = "<<INFO<< ", Col = "<<MYCOL<<endl;
            goto stop;
       } 
        
            MPIt2 = MPI_Wtime();
  
		//stop time
		MPIt2 = MPI_Wtime(); ///timerresultsroot
		//save stop time to array
		timerresults[solveriteration] = MPIt2-MPIt1; //save timer results for solver process
		blacs_barrier(&ICTXT, &bALL); //Barrier to synchronize results before sending them to the root process
		
		//we now see how long to push to root process
		//We combine our results on the root process
		
		destr = 0;
		sendN = (int) nqA; //How many rows we send
		dgesd2d(&ICTXT, &sendN, &iONE, B,&N, &iZERO, &iZERO); //send values
		blacs_barrier(&ICTXT, &bALL); //wait for all sends
		if(BLACSroot){ //root recieves
		for(int sendr = 0; sendr<N;sendr += NB, destr ++ ){
				sendN = NB; //How many rows we send
				if(N-sendr<NB){
			sendN = N-sendr;
		} 
		dgerv2d(&ICTXT, &sendN, &iONE, Xroot+sendr,&N, &iZERO, &destr); //recieve values
		}
		}
		blacs_barrier(&ICTXT, &bALL); 
		MPIt2 = MPI_Wtime();
		timerresultsroot[solveriteration] = MPIt2-MPIt1; //save timer results for sending to root process
       }
       //resynchronize
       blacs_barrier(&ICTXT, &bALL); 
       runtimeglobalend= MPI_Wtime(); //get ending global time
       
       //calculate timing averages
       //timingstats
       GetStats(numberofiterations, timerresults, timingstats); //normal
       GetStats(numberofiterations, timerresultsroot, timingstatsroot); //send to root

       blacs_barrier(&ICTXT, &bALL); 
       //we now gather the timings on the root process
	//get averages for root
        destr = 0;
        sendN = 1;
	dgesd2d(&ICTXT, &sendN, &iONE, timingstats,&sendN, &iZERO, &iZERO); //send values
	blacs_barrier(&ICTXT, &bALL); //wait for all sends
	if(BLACSroot){ //root recieves
	//dgerv2d(&ICTXT, &sendN, &iONE, myroottimings+sendr,&sendN, &iZERO, &destr); //recieve values
	for(int sendr = 0; sendr<Ntimingsroot;sendr ++, destr ++ ){
	//	dgerv2d(&ICTXT, &Ntimingsroot, &iONE, myroottimings+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	dgerv2d(&ICTXT, &sendN, &iONE, myroottimings+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	}
	}
	blacs_barrier(&ICTXT, &bALL); 
	//we now gather the timings on the root process
	//get variance for root
        destr = 0;
        sendN = 1;
	dgesd2d(&ICTXT, &sendN, &iONE, timingstats+1,&sendN, &iZERO, &iZERO); //send values
	blacs_barrier(&ICTXT, &bALL); //wait for all sends
	if(BLACSroot){ //root recieves
	//dgerv2d(&ICTXT, &sendN, &iONE, myroottimings+sendr,&sendN, &iZERO, &destr); //recieve values
	for(int sendr = 0; sendr<Ntimingsroot;sendr ++, destr ++ ){
	//	dgerv2d(&ICTXT, &Ntimingsroot, &iONE, myroottimings+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	dgerv2d(&ICTXT, &sendN, &iONE, myroottimingsvar+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	}
	}
	blacs_barrier(&ICTXT, &bALL); 
	
	
	       //we now gather the timings on the root process for send to root
	//get averages for root
        destr = 0;
        sendN = 1;
	dgesd2d(&ICTXT, &sendN, &iONE, timingstatsroot,&sendN, &iZERO, &iZERO); //send values
	blacs_barrier(&ICTXT, &bALL); //wait for all sends
	if(BLACSroot){ //root recieves
	//dgerv2d(&ICTXT, &sendN, &iONE, myroottimings+sendr,&sendN, &iZERO, &destr); //recieve values
	for(int sendr = 0; sendr<Ntimingsroot;sendr ++, destr ++ ){
	//	dgerv2d(&ICTXT, &Ntimingsroot, &iONE, myroottimings+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	dgerv2d(&ICTXT, &sendN, &iONE, myroottimingsroot+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	}
	}
	blacs_barrier(&ICTXT, &bALL); 
	//we now gather the timings on the root process
	//get variance for root
        destr = 0;
        sendN = 1;
	dgesd2d(&ICTXT, &sendN, &iONE, timingstatsroot+1,&sendN, &iZERO, &iZERO); //send values
	blacs_barrier(&ICTXT, &bALL); //wait for all sends
	if(BLACSroot){ //root recieves
	//dgerv2d(&ICTXT, &sendN, &iONE, myroottimings+sendr,&sendN, &iZERO, &destr); //recieve values
	for(int sendr = 0; sendr<Ntimingsroot;sendr ++, destr ++ ){
	//	dgerv2d(&ICTXT, &Ntimingsroot, &iONE, myroottimings+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	dgerv2d(&ICTXT, &sendN, &iONE, myroottimingsvarroot+sendr,&Ntimingsroot, &iZERO, &destr); //recieve values
	}
	}
	blacs_barrier(&ICTXT, &bALL); 
	
	
	
	
	
       //print out function calll timings
	/*
    blacs_barrier(&ICTXT, &bALL); 

	cout<<"\nProcessor "<<MYCOL<<" function call timings are: Avg "<<timingstats[0]<<" Var "<<timingstats[1]<< " All(";
	for(int z = 0; z<numberofiterations; z++){
		cout<<timerresults[z]<<",";
	}
	cout<<") secs"<<endl;
	blacs_barrier(&ICTXT, &bALL); 
	//print out send to root timings
	cout<<"\nProcessor "<<MYCOL<<" function call and send to root timings are: Avg "<<timingstatsroot[0]<<" Var "<<timingstatsroot[1]<< " All(";
	for(int z = 0; z<numberofiterations; z++){
		cout<<timerresultsroot[z]<<",";
	}
	cout<<") secs"<<endl;	
    	blacs_barrier(&ICTXT, &bALL);

	*/
	
	//we now print out our average values
	if(BLACSroot){
	//realfilename
        cout<<"\nRoot results"<<endl;
		for(int k = 0; k < Ntimingsroot ;k++){
			cout<<"P"<<k<<" Avg: "<<myroottimings[k]<<"\t Var:" <<myroottimingsvar[k]<<" STR Avg: "<<myroottimingsroot[k]<<"\t STRVar:" <<myroottimingsvarroot[k]<<",\n";
		}
	cout<<"\n Max average: "<< *std::max_element(myroottimings, myroottimings + Ntimingsroot)<<" Min average: "<<*std::min_element(myroottimings, myroottimings + Ntimingsroot)<<endl;
	}
	
	if(BLACSroot){
	timeaveragemin =*std::min_element(myroottimings, myroottimings + Ntimingsroot);
	timeaveragemax = *std::max_element(myroottimings, myroottimings + Ntimingsroot);
	timeaveragevar =*std::max_element(myroottimingsvar, myroottimingsvar + Ntimingsroot);
	timeaveragesum=0;
	
	timeaverageminroot =*std::min_element(myroottimingsroot, myroottimingsroot + Ntimingsroot);
	timeaveragemaxroot = *std::max_element(myroottimingsroot, myroottimingsroot + Ntimingsroot);
	timeaveragevarroot =*std::max_element(myroottimingsvarroot, myroottimingsvarroot + Ntimingsroot);
	timeaveragesumroot=0;
	for(int k = 0; k < Ntimingsroot ;k++){
	timeaveragesum = timeaveragesum + myroottimings[k];
	timeaveragesumroot = timeaveragesumroot + myroottimingsroot[k];
	}
	
	cout<<"\tCompiled: "<<timeaveragemin<<"\t"<<timeaveragemax<<"\t"<<timeaveragevar<<"\t"<<timeaveragesum<<"\t"<<timeaverageminroot<<"\t"<<timeaveragemaxroot<<"\t"<<timeaveragevarroot<<"\t"<<timeaveragesumroot<<runtimeglobalend-runtimeglobalstart<<"\n";
//"Nprocs","NMeshPts","t_avg_min(s)","t_avg_max(s)","t_avg_var(s)","t_avg_sum(s)","t_avg_min_root(s)","t_avg_max_root(s)","t_avg_var_root(s)","t_avg_sum_root(s)","Trials","t_total(s)"
	std::ofstream filetiming(timefilename, ios::app);
	filetiming<<nprocs_mpi<<","<<n_int<<","<<timeaveragemin<<","<<timeaveragemax<<","<<timeaveragevar<<","<<timeaveragesum<<","<<timeaverageminroot<<","<<timeaveragemaxroot<<","<<timeaveragevarroot<<","<<timeaveragesumroot<<","<<numberofiterations<<","<<runtimeglobalend-runtimeglobalstart<<endl;
	filetiming.close();
	}
	blacs_barrier(&ICTXT, &bALL);
	


	//We now write out our root results

	/*
	if(BLACSroot){
	//realfilename
        cout<<endl;
		for(int k = 0; k < N ;k++){
			cout<<Xroot[k] <<",\n";
		}	
	}
	*/

	
	//We now write out our root results
	/*
	if(BLACSroot){
	//realfilename
	std::ofstream file(realfilename);
		for(int k = 0; k < N ;k++){
			file<<Xroot[k] <<",\n";
		}
	file.close();
	
	}
	*/
	
        //blacs_barrier(&ICTXT, &bALL);
	//We test our final results
	/*
	cout<<"B final: MyCOL "<<MYCOL<<"MY ROW "<<MYROW<<"\t(";
	for(int k = 0; k < nqA ;k++){
	cout<<B[k] <<",\t";
	}
	cout<<")"<<endl;
	*/
	blacs_barrier(&ICTXT, &bALL);
	//if(BLACSroot){
	//	cout<<"X final: MyCOL "<<MYCOL<<"MY ROW "<<MYROW<<"\t(";
	//	for(int k = 0; k < N ;k++){
	//		cout<<Xroot[k] <<",\t";
	//	}
	//	cout<<")"<<endl;	
	//}
	//we write root results to a file
	
	stop:
	mkl_free(A);
	mkl_free(B);
    mkl_free(Xroot);
	mkl_free(timerresults);
    mkl_free(timerresultsroot);
    mkl_free(IPIV);
	blacs_gridexit_(&ICTXT);
	MPI_Finalize();
	cout<<"!!Program Done!!"<<endl;
	return 0;	
}
