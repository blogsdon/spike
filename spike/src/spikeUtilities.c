//spikeUtilities C library
//Copyright 2016 Benjamin A Logsdon, PhD
#include "spikeUtilities.h"

void innerProduct(int n,double *vector1,double *vector2,double * result){
  const int incxy = 1;
  //result <- vector'vector
	(*result)=F77_NAME(ddot)(&n,vector1,&incxy,vector2,&incxy);
}

void scaledVectorAddition(int n,double *vector1,double *vector2,double alpha){
	//vector2 <- alpha * vector1 + vector2;
	const int incxy =1;
	F77_NAME(daxpy)(&n,&alpha,vector1,&incxy,vector2,&incxy);
}

void l2Norm(int n,double *vector,double *result){
  //result <- sqrt(vector'vector)
  const int incxy=1;
  (*result)=F77_NAME(dnrm2)(&n,vector,&incxy);
}

void scaleVector(int n,double * vector, double alpha){
  const int incxy=1;
	F77_NAME(dscal)(&n,&alpha,vector,&incxy);
}

void standardizeVector(double * vector,double * onesVector,int n){
  double mean,sd;
	double nd = (double) n;
  //compute mean
	innerProduct(n,vec,onesVector,&mean);
	mean = mean/nd;

  //substract mean from vector
	scaledVectorAddition(n,onesVector,vec,-mean);

  //compute standard deviation
	l2Norm(n,vec,&sd);
	sd = pow(sd,2);
	nd = nd-1;
	sd=sqrt(nd/sd);

  //scale the vector
	scaleVector(n,vec,sd);
}

void correlation(double * vector1, double * vector2, double * onesVector,double * result,int n){
  double mean1,mean2,sd1,sd2,cov;
	double nd = (double) n;

	//compute means:
	innerProduct(n,vector1,onesVector,&mean1);
	mean1 = mean1/nd;

	innerProduct(n,vector2,onesVector,&mean2);
	mean2 = mean2/nd;

	//compute standard deviations:
	//substract means from vectors
	scaledVectorAddition(n,onesVector,vector1,-mean1);
	l2Norm(n,vector1,&sd1);

	scaledVectorAddition(n,onesVector,vector2,-mean2);
	l2Norm(n,vector2,&sd2);

	//rescale
	//sd1 = sd1/sq;
	//sd2 = sd2/sqrt(nd-1);

	innerProduct(n,vector1,vector2,&cov);
	//rescale
	//cov = cov/(;
	//Rprintf("cov: %g\n",cov);
	corv[0] = cov/(sd1*sd2);

	//add means back to vectors
	scaledVectorAddition(n,onesVector,vector1,mean1);
	scaledVectorAddition(n,onesVector,vector2,mean2);
}

double vectorSumOfSquares(double * vector,int n){
  double a;
	innerProduct(n,vector,vector,&a);
	return a;
}
