#include<math.h>
#include<stdio.h>
#include<gsl/gsl_rng.h>
#include "mysubroutine.h"
gsl_rng * r;  /* global generator */

int main()
{
  //GSL main test
  const gsl_rng_type * T;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  printf ("generator type: %s\n", gsl_rng_name (r));
  printf ("seed = %lu\n", gsl_rng_default_seed);
  printf ("first value = %lu\n", gsl_rng_get (r));
  gsl_rng_free (r);
  //Math library check
  sqrt(4);
  //GSL subroutine test

  examplesubroutine(5);
  return 0;

}
