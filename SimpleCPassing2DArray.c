/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/
//compile with gcc as usual
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>

//Adapted from
//https://stackoverflow.com/questions/3911400/how-to-pass-2d-array-matrix-in-a-function-in-c

//If in the same code, you must have your subroutines and variables declared before main
//Your code was not running pass the do_stuff call before.


void do_stuff(int dimension,int dimension2, complex double **matrix) 
{
    int i;
    int j;
    
    printf("\n Hello Xylo, the matrix elements in the subroutine call are below.\n");

    for(i = 0; i < dimension; i++)
        for(j = 0; j < dimension2; j++){
            {
                printf("\n Before, For (i,j) = (%i,%i), we have %lf\t + I %lf\n",i,j, creal(matrix[i][j]), cimag(matrix[i][j]));

                matrix[i][j] = i + I * (j+i*2);
                printf("\n After, For (i,j) = (%i,%i), we have %lf\t + I %lf\n",i,j, creal(matrix[i][j]), cimag(matrix[i][j]));

            }
        }
} 

void main()
{
    int i,j;
    int dimension = 3;
    int dimension2 = 4;
      /* obtain values for rows & cols */

  /* allocate the array */
  complex double **matrix; 
      printf("\n Hello Xylo, Allocating memory.\n");

    matrix = malloc(dimension * sizeof *matrix);

   for (i=0; i<dimension; i++)
      {
        matrix[i] = malloc(dimension2 * sizeof *matrix[i]);
      }   
    
  
    
    printf("\n Hello Xylo, the initial matrix elements in the main function are below.\n");

    for(i = 0; i < dimension; i++)
    {
        for(j = 0; j < dimension2; j++)
        {
            matrix[i][j] = i + I * j;
            printf("\n For (i,j) = (%i,%i), we have %lf\t + I %lf\n",i,j, creal(matrix[i][j]), cimag(matrix[i][j]));

        }
    }

    //Pass the pointer.. yes a reference decays to a pointer in this case
    //but it is a pointer here not an a reference
    do_stuff(dimension,dimension2, matrix);
    printf("\n Hello Xylo, the matrix elements in the main after the subroutine call are below.\n");
        for(i = 0; i < dimension; i++)
    {
        for(j = 0; j < dimension2; j++)
        {
            printf("\n For (i,j) = (%i,%i), we have %lf\t + I %lf\n",i,j, creal(matrix[i][j]), cimag(matrix[i][j]));
        }
    }
    
    //Free memory
       for (i=0; i<dimension; i++)
      {
        free(matrix[i]);
      }
        free(matrix);


}

