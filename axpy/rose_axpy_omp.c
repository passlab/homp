#include "axpy.h"
#include "libxomp.h" 

struct OUT__1__4666___data 
{
  void *x_p;
  void *y_p;
  void *n_p;
  void *a_p;
}
;
static void OUT__1__4666__(void *__out_argv);

void axpy_omp(double *x,double *y,int n,double a)
{
  int i;
  struct OUT__1__4666___data __out_argv1__4666__;
  __out_argv1__4666__.a_p = ((void *)(&a));
  __out_argv1__4666__.n_p = ((void *)(&n));
  __out_argv1__4666__.y_p = ((void *)(&y));
  __out_argv1__4666__.x_p = ((void *)(&x));
  XOMP_parallel_start(OUT__1__4666__,&__out_argv1__4666__,1,0);
  XOMP_parallel_end();
}

static void OUT__1__4666__(void *__out_argv)
{
  double **x = (double **)(((struct OUT__1__4666___data *)__out_argv) -> x_p);
  double **y = (double **)(((struct OUT__1__4666___data *)__out_argv) -> y_p);
  int *n = (int *)(((struct OUT__1__4666___data *)__out_argv) -> n_p);
  double *a = (double *)(((struct OUT__1__4666___data *)__out_argv) -> a_p);
  int _p_i;
  long p_index_;
  long p_lower_;
  long p_upper_;
  XOMP_loop_default(0, *n - 1,1,&p_lower_,&p_upper_);
  for (p_index_ = p_lower_; p_index_ <= p_upper_; p_index_ += 1) {
    ( *y)[p_index_] += ( *a * ( *x)[p_index_]);
  }
  XOMP_barrier();
}
