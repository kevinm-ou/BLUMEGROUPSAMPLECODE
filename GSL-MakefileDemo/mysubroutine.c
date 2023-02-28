#include<math.h>
#include<stdio.h>
#include<gsl/gsl_rng.h>
#include <gsl/gsl_sf_bessel.h>
#include "mysubroutine.h"

void examplesubroutine(double x){
	//source 
	//https://www.gnu.org/software/gsl/doc/html/usage.html
  double y = gsl_sf_bessel_J0 (x);
  printf ("J0(%g) = %.18e\n", x, y);
}
