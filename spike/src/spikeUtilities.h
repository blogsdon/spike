//spikeUtilities C library declarations
//Copyright 2016, Benjamin A Logsdon, PhD
#include <R.h>
#include <Rmath.h>
#include <stdio.h>
#include <R_ext/BLAS.h>

//ddot_w -> innerProduct
void innerProduct(int n,double *vector1,double *vector2,double * result);

//daxpy_w -> scaledVectorAddition
void scaledVectorAddition(int n,double *vector1,double *vector2,double alpha);

//dnrm2_w -> l2Norm
void l2Norm(int n,double *vector,double *result);

//dscal_w -> scaleVector
void scaleVector(int n,double * vector, double alpha);

//scale_vector -> standardizeVector
void standardizeVector(double * vector,double * onesVector,int n);

//cor -> correlation
void correlation(double * vector1, double * vector2, double * onesVector,double * result,int n);

//compute_ssq -> vectorSumOfSquares
double vectorSumOfSquares(double * vector,int n);

//define all wrapper functions that will do other matrix/vector operations.
