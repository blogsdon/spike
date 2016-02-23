//gaussian variational Bayes spike regression (vbsr) R package C library
//Copyright 2016 Benjamin A Logsdon
#include "gaussianVariationalBayesSpikeRegression.h"

double * extractPenalizedFeatureMatrixColumn(struct gaussianModelRealization * model,
	int columnIndex){
		//return the pointer to the columnIndex^th column of the penalizedDataMatrix
	return (&(model->data.penalizedDataMatrix[columnIndex]))->column;
}


int * extractRealizationMatrixColumn(struct gaussianModelRealization * model,
	int columnIndex){
		//return the pointer to the columnIndex^th column of the realizationMatrix update realizationMatrix matrix
	return (&(model->data.realizationMatrix[columnIndex]))->column;
}

struct gaussianModelRealization * getParameters(struct gaussianModelRealization * model,
	 int realizationIndex,
	 int penaltyIndex){
		 //return the pointer to the struct for the realiationIndex^th, penaltyIndex^th model parameters
	return (&((&(model->modelParameterRealization[realizationIndex]))->modelParameters[penaltyIndex]));
}

void initializeGaussianModelParameters(int numberSamples,
				int numberPenalizedFeatures,
				int realizationIndex,
				int penaltyIndex,
				struct gaussianModelRealization * model,
				double * responseVariable,
				double responseVariance){
	//initialize the Gaussian spike model parameters
	//integer k: update iterator
	int k;

	//malloc space for beta mu parameters aka I cast a spell of malloc
	getParameters(model,realizationIndex,penaltyIndex)->betaMu = (double *) malloc(sizeof(double)*numberPenalizedFeatures);

	//malloc space for beta sigmat squared parameters
	getParameters(model,realizationIndex,penaltyIndex)->betaSigmaSquared = (double *) malloc(sizeof(double)*numberPenalizedFeatures);

	//malloc space for the test statistic parameters
	getParameters(model,realizationIndex,penaltyIndex)->betaChi = (double *) malloc(sizeof(double)*numberPenalizedFeatures);

	//malloc space for the posterior probability parameter
	getParameters(model,realizationIndex,penaltyIndex)->betaPosteriorProbability = (double *) malloc(sizeof(double)*numberPenalizedFeatures);

	//malloc space for the expectation parameters
	getParameters(model,realizationIndex,penaltyIndex)->expectationBeta = (double *) malloc(sizeof(double)*numberPenalizedFeatures);

	//malloc space for the expectation of beta^2 parameters
	getParameters(model,realizationIndex,penaltyIndex)->expectationBetaSquared = (double *) malloc(sizeof(double)*numberPenalizedFeatures);

	//initialize all the relevant model parameters to 0.0
	for(k=0;k<numberPenalizedFeatures;k++){
		getParameters(model,realizationIndex,penaltyIndex)->betaMu[k] = 0;
		getParameters(model,realizationIndex,penaltyIndex)->betaSigmaSquared[k] = 0;
		getParameters(model,realizationIndex,penaltyIndex)->betaChi[k]= 0;
		getParameters(model,realizationIndex,penaltyIndex)->betaPosteriorProbability[k] = 0;
		getParameters(model,realizationIndex,penaltyIndex)->expectationBeta[k] = 0;
		getParameters(model,realizationIndex,penaltyIndex)->expectationBetaSquared[k] = 0;
	}

	//initialize error variance parameter
	getParameters(model,realizationIndex,penaltyIndex)->sigmaSquaredError = responseVariance;

	//initialize lower bound
	getParameters(model,realizationIndex,penaltyIndex)->lowerBound = 0;

	//initialize sum of the beta posterior probabilities
	getParameters(model,realizationIndex,penaltyIndex)->sumOfBetaPosteriorProbabilities = 0;

	//initialize the posterior probability entropy
	getParameters(model,realizationIndex,penaltyIndex)->posteriorProbabilityEntropy = 0;

	//initialize the correction for the beta squared error expectations
	getParameters(model,realizationIndex,penaltyIndex)->betaSquaredExpectationCorrection = 0;

	//malloc space for the residualVector
	getParameters(model,realizationIndex,penaltyIndex)->residualVector = (double *) malloc(sizeof(double)*numberSamples);

	//initialize the residualVector to the model where all effects are zero
	for(k=0;k<numberSamples;k++){
		getParameters(model,realizationIndex,penaltyIndex)->residualVector[k] = responseVariable[k];
	}

	//set the realization index of the current model parameter realization
	getParameters(model,realizationIndex,penaltyIndex)->realizationIndex = realizationIndex;

	//set the penalty index of the current model parameter realization
	getParameters(model,realizationIndex,penaltyIndex)->penaltyIndex = penaltyIndex;
}

void freeGaussianModelParameters(struct gaussianModelRealization * model, int realizationIndex, int penaltyIndex){
	//free the memory associated with the beta mu parameters
	free(getParameters(model,realizationIndex,penaltyIndex)->betaMu);

	//free the memory associated with the beta sigma squared parameters
	free(getParameters(model,realizationIndex,penaltyIndex)->betaSigmaSquared);

	//free the memory associated with the betachi parameters
	free(getParameters(model,realizationIndex,penaltyIndex)->betaChi);

	//free the memory associated with the posterior probability of beta parameters
	free(getParameters(model,realizationIndex,penaltyIndex)->betaPosteriorProbability);

	//free the memory associated with the expectation of beta parameters
	free(getParameters(model,realizationIndex,penaltyIndex)->expectationBeta);

	//free the memory associated with the expectation of beta squared parameters
	free(getParameters(model,realizationIndex,penaltyIndex)->expectationBetaSquared);

	//free the memory associatedw tith the residual vector state
	free(getParameters(model,realizationIndex,penaltyIndex)->residualVector);
}

void dataStandardization(struct gaussianModelRealization * model){
	//standardize the data if necessary and precompute the sum of squares (l2norm^2) of each column

	//integer j: iterator
	int j;

	//double doubleNumberSamples: double version of number of samples
	double doubleNumberSamples = ((double) model->data.numberSamples);

	if(model->modelState.scaleFeatureMatrix==0){
		//if standardizing, then standardize
			for(j=0;j<model->data.numberPenalizedFeatures;j++){
				standardizeVector(extractPenalizedFeatureMatrixColumn(model,j),model->data.onesVector,model->data.numberSamples);
				model->data.penalizedDataMatrixColumnSumOfSquares[j] = doubleNumberSamples - 1;
			}
	}else{
			//if not, just precompute the vector of sum squares
			for(j=0;j<model->data.numberPenalizedFeatures;j++){
				model->data.penalizedDataMatrixColumnSumOfSquares[j]=vectorSumOfSquares(extractPenalizedFeatureMatrixColumn(model,columnIndex),model->data.numberSamples);
			}
	}
}


void initializeGaussianModel(double * epsilon,
					double * l0Vector,
					double * priorProbabilityVector,
					int * penalizeVariable,
					double * featureColumnScalingFactor,
					int * maximumIterations,
					int * l0VectorLength,
					int * numberOfRealizations,
					int * scale,
					int * est,
					int * approx,
					int * totalModelFits,
					double * penalizedDataMatrix,
					double * responseVariable,
					double * responseVariance,
					int * numberSamples,
					int * numberPenalizedFeatures,
					int * numberUnpenalizedFeatures,
					int * realizationMatrix,
					struct gaussianModelRealization * model){

	int k,l;

	//k, l: iterators
	//set the tolerance for assesing convergence
	model->modelState.epsilon = (*epsilon);

	//allocate memory for the logit penalty vector
	model->modelState.l0Vector = (double *) malloc(sizeof(double)*(*l0VectorLength));

	//allocate memory for the prior probability vector
	model->modelState.priorProbabilityVector = (double *) malloc(sizeof(double)*(*l0VectorLength));

	//copy the l0 and prior probability vectors to the internal model state
	for(k=0;k<*l0VectorLength;k++){
		model->modelState.l0Vector[k]=l0Vector[k];
		model->modelState.priorProbabilityVector[k]=priorProbabilityVector[k];
	}

	//this may be deprecated, as all variables will be penalized
	model->modelState.penalizeVariable = (int *) malloc(sizeof(int)*(*numberPenalizedFeatures));

	//copy the model scaling factor and model penalization indicator to the model struct
	model->modelState.featureColumnScalingFactor = (double *) malloc(sizeof(double)*(*numberPenalizedFeatures));
	for(k=0;k<*numberPenalizedFeatures;k++){
		model->modelState.penalizeVariable[k]=penalizeVariable[k];
		model->modelState.featureColumnScalingFactor[k]=featureColumnScalingFactor[k];
	}

	//define the maximimum number of iterations for the algorithm to run
	model->modelState.maximumIterations = (*maximumIterations);

	//define the path length
	model->modelState.l0VectorLength = (*l0VectorLength);

	//define the number of restarts of the algorithm
	model->modelState.numberOfRealizations = (*numberOfRealizations);

	//define the enums for whether to scale or not
	if((*scale)==1){
		model->modelState.scaleType = SCALE;
	} else{
		model->modelState.scaleType = NOSCALE;
	}


	//define the enums for what type of aggregation of results to return
	if((*est)==1){
		model->modelState.estType = BMA;
	} else{
		model->modelState.estType = MAXIMAL;
	}


	//indicate whether or not to apply the (computationally) intensive covariance correction
	if((*approx)==1){
		model->modelState.bType = APPR;
	} else{
		model->modelState.errType = EXACT;
	}

	//define the total number of model fits
	model->modelState.totalModelFits = (*totalModelFits);

	//allocate the memory for the penalized data column matrix
	model->data.penalizedDataMatrix = (struct doubleColumnVector *) malloc(sizeof(struct doubleColumnVector)*(*numberPenalizedFeatures));

	//allocate the memory for each column of the penalized data column matrix
	for(k=0;k<(*numberPenalizedFeatures);k++){
		(&(model->data.penalizedDataMatrix[k]))->column = (double *) malloc(sizeof(double)*(*n));
	}
	//copy the data into the column data matrix
	for(k=0;k<(*numberPenalizedFeatures);k++){
		for(l=0;l<(*n);l++){
			(&(model->data.penalizedDataMatrix[k]))->column[l] = penalizedDataMatrix[k*(*n)+l];
		}
	}

	//link the pointer for the response variable (as it is constant?)
	model->data.responseVariable = responseVariable;

	//set the responseVariance
	model->data.responseVariance = (*responseVariance);

	//set the number of samples
	model->data.numberSamples = (*numberSamples);

	//set the number of penalized features
	model->data.numberPenalizedFeatures = (*numberPenalizedFeatures);

	//set the number of unpenalized features
	model->data.numberUnpenalizedFeatures = (*numberUnpenalizedFeatures);

	//allocate the memory for the holder variable for the sum of squares of the penalized data matrix
	model->data.penalizedDataMatrixColumnSumOfSquares = (double *) malloc(sizeof(double)*(*numberPenalizedFeatures));

	//allocate the memory for the holder variable for the indicator matrix of ordering updates
	model->data.realizationMatrix = (struct integerColumnVector *) malloc(sizeof(struct integerColumnVector)*(*numberOfRealizations));
	//allocate the memory for each column (e.g. realization) of the orderings of penalized variables
	for(k=0;k<(*numberOfRealizations);k++){
		(&(model->data.realizationMatrix[k]))->column = (int *) malloc(sizeof(int)*(*numberPenalizedFeatures));
	}

	//copy the ordering matrix into the internal data structure
	for(k=0;k<(*numberOfRealizations);k++){
		for(l=0;l<(*numberPenalizedFeatures);l++){
			(&(model->data.realizationMatrix[k]))->column[l] = realizationMatrix[k*(*numberPenalizedFeatures)+l];
		}
	}

	//allocate the memory for the number of samples
	model->data.onesVector = (double *) malloc(sizeof(double)*(*numberSamples));

	//set the ones vector #winning
	for(k=0;k<(*numberSamples);k++){
		model->data.onesVector[k]= 1.0;
	}

	//standardize the data, what what
	dataStandardization(model);

	//allocate the memory for a model parameter realization
	model->modelParameterRealization = (struct modelParameters *) malloc(sizeof(struct modelParameters)*(*numberOfRealizations));

	//for each realization allocate the memory for the length of the l0 path
	for(k=0;k<(*numberOfRealizations);k++){
		(&(model->modelParameterRealization[k]))->modelParameters = (struct gaussianLinkModelParameters *) malloc(sizeof(struct gaussianLinkModelParameters)*(*l0VectorLength));
	}

	//for each of the realizations and penalty parameters, initialize the gaussian model parameters
	for(k=0;k<(*numberOfRealizations);k++){
		for(l=0;l<(*l0VectorLength);l++){
			initializeGaussianModelParameters((*numberSamples),(*numberPenalizedFeatures),k,l,model,responseVariable,*responseVariance);
		}
	}
}




void freeGaussianModel(struct gaussianModelRealization * model){

	//function to free all of the memory allocations for the gaussian model.  Basically a garbage collection step

	int i,j

	//free each column of the penalized data matrix
	for(i=0;i<(model->data.numberPenalizedFeatures);i++){
		free((&(model->data.penalizedDataMatrix[i]))->column);

	}

	//free the penalized data matrix
	free(model->data.penalizedDataMatrix);

	//free each column of the realization matrix
	for(i=0;i<(model->modelState.numberOfRealizations);i++){
		free((&(model->data.realizationMatrix[i]))->column);
	}

	//free the realization matrix
	free(model->data.realizationMatrix);

	//free each model struct realization
	for(i=0;i<model->modelState.numberOfRealizations;i++){
		for(j=0;j<model->modelState.l0VectorLength;j++){
			freeGaussianModelParameters(model, i, j);
		}
	}

	//free each model parameter realization struct
	for(k=0;k<(model->modelState.numberOfRealizations);k++){
		free((&(model->modelParameterRealization[k]))->modelParameters);
	}

	//free the model parameter realization
	free(model->modelParameterRealization);

	//free the data matrix sum of squares holder variable
	free(model->data.penalizedDataMatrixColumnSumOfSquares);

	//free the ones vector
	free(model->data.onesVector);

	//free the penalty vector
	free(model->modelState.l0Vector);

	//free the prior probability vector
	free(model->modelState.priorProbabilityVector);

	//free the column scaling factor vector
	free(model->modelState.featureColumnScalingFactor);

	//free the penalize variable vector
	free(model->modelState.penalizeVariable);
}

void copyGaussianModelState(struct gaussianModelRealization * model, int i, int j){
	//utility function that copies the result of the previous penalty path to the current penalty path state

	//k: iterator
	int k

	//previousPenaltyState- the indicator for the previous penalty index
	int previousPenaltyState = j-1;

	//copy all the beta parameters
	for(k=0;k<model->data.numberPenalizedFeatures;k++){
		getParameters(model,i,j)->betaMu[k] = getParameters(model,i,previousPenaltyState)->betaMu[k];
		getParameters(model,i,j)->betaSigmaSquared[k] = getParameters(model,i,previousPenaltyState)->betaSigmaSquared[k];
		getParameters(model,i,j)->betaChi[k] = getParameters(model,i,previousPenaltyState)->betaChi[k];
		getParameters(model,i,j)->betaPosteriorProbability[k] = getParameters(model,i,previousPenaltyState)->betaPosteriorProbability[k];
		getParameters(model,i,j)->expectationBeta[k] = getParameters(model,i,previousPenaltyState)->expectationBeta[k];
		getParameters(model,i,j)->expectationBetaSquared[k] = getParameters(model,i,previousPenaltyState)->expectationBetaSquared[k];
	}

	//copy the sigma squared error
	getParameters(model,i,j)->sigmaSquaredError = getParameters(model,i,previousPenaltyState)->sigmaSquaredError;

	//copy the lowerBound
	getParameters(model,i,j)->lowerBound = getParameters(model,i,previousPenaltyState)->lowerBound;

	//copy the psums
	getParameters(model,i,j)->sumOfBetaPosteriorProbabilities = getParameters(model,i,previousPenaltyState)->sumOfBetaPosteriorProbabilities;

	//copy the posterior probability entropy
	getParameters(model,i,j)->posteriorProbabilityEntropy = getParameters(model,i,previousPenaltyState)->posteriorProbabilityEntropy;

	//copy the betaSquaredExpectationCorrection
	getParameters(model,i,j)->betaSquaredExpectationCorrection = getParameters(model,i,previousPenaltyState)->betaSquaredExpectationCorrection;


	//copy the residual vector
	for(k=0;k<model->data.numberSamples;k++){
		getParameters(model,i,j)->residualVector[k] = getParameters(model,i,previousPenaltyState)->residualVector[k];
	}
}



void updateGaussianBetaDistribution(struct gaussianModelRealization * model, int i, int j){

	int k,l,exc,t;
	double mu, sigma,prec, chi, p, e_b,e_b2,l0;

	l0 = model->modelState.l0Vector[j];
	for(l=0;l< model->data.numberPenalizedFeatures ;l++){
		k = (&(model->data.realizationMatrix[i]))->column[l];

		innerProduct(model->data.numberSamples,extractPenalizedFeatureMatrixColumn(model,k),getParameters(model,realizationIndex,penaltyIndex)->residualVector,&mu);

		mu = mu + (model->data.penalizedDataMatrixColumnSumOfSquares[k])*(getParameters(realizationIndex,penaltyIndex)->expectationBeta[k]);

		mu = mu/model->data.penalizedDataMatrixColumnSumOfSquares[k];

		sigma = 1/((1/getParameters(realizationIndex,penaltyIndex)->sigmaSquaredError)*(model->data.penalizedDataMatrixColumnSumOfSquares[k]));

		chi = pow(mu,2)/sigma;
		p = 1/(1+exp(-0.5*(chi+l0+log(sigma))));
		e_b = p*mu;
		e_b2 = p*(pow(mu,2)+sigma);

		getParameters(realizationIndex,penaltyIndex)->betaSquaredExpectationCorrection = getParameters(realizationIndex,penaltyIndex)->betaSquaredExpectationCorrection + (pow(e_b,2)-e_b2)*(model->data.penalizedDataMatrixColumnSumOfSquares[k]);


		getParameters(realizationIndex,penaltyIndex)->sumOfBetaPosteriorProbabilities = getParameters(realizationIndex,penaltyIndex)->sumOfBetaPosteriorProbabilities + p;
		if(p>1-1e-10){
	  	getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy = getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy - p*log(p) + (1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
		}else if(p<1e-10){
	  	getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy = getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy + p - (1-p)*log(1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
		} else {
	  	getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy = getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy - p*log(p) - (1-p)*log(1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
		}

		scaledVectorAddition(model->data.numberSamples,extractPenalizedFeatureMatrixColumn(model,k),getParameters(realizationIndex,penaltyIndex)->residualVector,getParameters(realizationIndex,penaltyIndex)->expectationBeta[k]-e_b);

		getParameters(realizationIndex,penaltyIndex)->betaMu[k] = mu;
		getParameters(realizationIndex,penaltyIndex)->betaSigmaSquared[k] = sigma;
		getParameters(realizationIndex,penaltyIndex)->betaChi[k] = mu/sqrt(sigma);
		getParameters(realizationIndex,penaltyIndex)->expectationBeta[k] = e_b;
		getParameters(realizationIndex,penaltyIndex)->expectationBetaSquared[k] = e_b2;
		getParameters(realizationIndex,penaltyIndex)->betaPosteriorProbability[k] = p;
	}

}

void updateGaussianSigmaSquaredErrorEstimate(struct gaussianModelRealization * model, int i, int j){

	int t;
	double U;
	double nd = (double) model->data.numberSamples;
	innerProduct(model->data.numberSamples,getParameters(realizationIndex,penaltyIndex)->residualVector,getParameters(realizationIndex,penaltyIndex)->residualVector,&U);
	U = U - getParameters(realizationIndex,penaltyIndex)->betaSquaredExpectationCorrection;
	U = U/nd;
	getParameters(realizationIndex,penaltyIndex)->sigmaSquaredError = U;
	if(!R_FINITE(U)){
		freeGaussianModel(model);
		error("Penalized linear solution does not exist.\n");
	}

}

void updateGaussianLowerBound(struct gaussianModelRealization * model, int i, int j){

	double lba;
	double nd = (double) model->data.numberSamples;
	double md = (double) model->data.numberPenalizedFeatures;
	double pd = (double) model->data.numberUnpenalizedFeatures;
	md = md - pd;
	double p_beta;

	p_beta = model->modelState.priorProbabilityVector[j];
	int t;
  double U;

  innerProduct(model->data.numberSamples,getParameters(realizationIndex,penaltyIndex)->residualVector,getParameters(realizationIndex,penaltyIndex)->residualVector,&U);
	U = U - getParameters(realizationIndex,penaltyIndex)->betaSquaredExpectationCorrection;
	lba = -0.5*nd*(log(2*M_PI*getParameters(realizationIndex,penaltyIndex)->sigmaSquaredError) + 1);
	lba = lba + log(p_beta)*(getParameters(realizationIndex,penaltyIndex)->sumOfBetaPosteriorProbabilities);
	lba = lba + log(1-p_beta)*(md - getParameters(realizationIndex,penaltyIndex)->sumOfBetaPosteriorProbabilities);
	lba = lba + getParameters(realizationIndex,penaltyIndex)->posteriorProbabilityEntropy;

	getParameters(realizationIndex,penaltyIndex)->lowerBound = lba;

}

void runGaussianVariationalBayesSpikeRegression(struct gaussianModelRealization * model){
	//iterators i and j
	int i,j;

	//tolerance
	double tolerance=1;

	//define oldLowerBound
	double oldLowerBound;

	//algorithm iteration counter
	int count = 0;
	//#pragma omp parallel for private(i,j,count,tol,lb_old)
	for (i=0;i < model->modelState.numberOfRealizations;i++){
		for(j=0;j < model->modelState.l0VectorLength;j++){
			if(j>0){
				//copy the previous path to the new path
				copyGaussianModelState(model,i,j);
			}
			while(fabs(tolerance) > model->modelState.epsilon && count < model->modelState.maximumIterations){
				//set update to 0
				getParameters(model,i,j)->sumOfBetaPosteriorProbabilities = 0;
				getParameters(model,i,j)->betaSquaredExpectationCorrection = 0;
				getParameters(model,i,j)->posteriorProbabilityEntropy = 0;
				lowerBoundOld = getParameters(model,i,j)->lowerBound;

				//update penalized regression distributions
				updateGaussianBetaDistribution(model,i,j);

				//update unpenalized coefficient distributions
				updateGaussianAlphaDistribution(model,i,j);

				//update error term
				updateGaussianSigmaSquaredErrorEstimate(model,i,j);

				//update lower bound term
				updateGaussianLowerBound(model,i,j);
				tolerance = lowerBoundOld - getParameters(model,i,j)->lowerBound;
				count = count+1;
			}
			//if the data is going
			if(count>=model->modelState.maximumIterations){
				Rprintf("Maximum iterations exceeded!\n");
			}
			count =0;
			tol = 1;
		}
	}

}

void identifyUniqueModels(double * lb_t, double * post_p, int n,double tol){
	int i,j,count;
	double tv;
	count =0;
	//compute hash of the entire beta model parameters at a precision of 1e-6?
	for(i=0;i<n-1;i++){
		for(j=i+1;j<n;j++){
			if(i!=j){
				tv = fabs(lb_t[i]-lb_t[j]);
				if(tv < tol){
					post_p[j] = 0;
				}
			}
		}
	}


	tv =0;
	for(i=0;i<n;i++){
		if(post_p[i]>0){count=count+1;}
		tv = post_p[i]+tv;
	}
	for(i=0;i<n;i++){
		post_p[i]=post_p[i]/tv;

	}

}

void computeBayesianModelAveragingCovarianceCorrection(struct gaussianModelRealization * model,int k,double * post_prob,double * s_bma,int j){
	int t,l;
	double corv;
	s_bma[0] = 0;


	for (t=0;t<model->modelState.numberOfRealizations;t++){
		if(post_prob[t] > 0){
			s_bma[0] = s_bma[0] + pow(post_prob[t],2);
		}
	}


	for(t=0;t<model->modelState.numberOfRealizations-1;t++){
		for(l=t+1;l<model->modelState.numberOfRealizations;l++){
			if(post_prob[t]>0 && post_prob[l]>0){
				scaledVectorAddition(model->data.numberSamples,extractPenalizedFeatureMatrixColumn(model,k),getParameters(model,t,j)->residualVector,getParameters(model,t,j)->expectationBeta[k]);
				scaledVectorAddition(model->data.numberSamples,extractPenalizedFeatureMatrixColumn(model,k),getParameters(model,l,j)->residualVector,getParameters(model,l,j)->expectationBeta[k]);
				cor(getParameters(model,t,j)->residualVector, getParameters(model,l,j)->residualVector, model->data.onesVector,&corv,model->data.numberSamples);
				scaledVectorAddition(model->data.numberSamples,extractPenalizedFeatureMatrixColumn(model,k),getParameters(model,t,j)->residualVector,-getParameters(model,t,j)->expectationBeta[k]);
				scaledVectorAddition(model->data.numberSamples,extractPenalizedFeatureMatrixColumn(model,k),getParameters(model,l,j)->residualVector,-getParameters(model,l,j)->expectationBeta[k]);
				s_bma[0] = s_bma[0] + 2*post_prob[t]*post_prob[l]*(corv);

			}
		}
	}




}


void extractGaussianResults(struct gaussianModelRealization * model,
											double * betaMuResult,
											double * betaSigmaSquaredResult,
											double * expectationBetaResult,
											double * posteriorProbabilityBetaResult,
											double * lowerBoundResult,
											double * sigmaSquaredErrorResult,
											double * alphaResult{

	int i,j,k;
	double max_v,bc,bm,bs,eb,bp,Z,s_bma;
	double * post_prob = (double *) malloc(sizeof(double)*model->modelState.numberOfRealizations);
	double * lb_t = (double *) malloc(sizeof(double)*model->modelState.numberOfRealizations);
	int w_max;

	switch(model->modelState.estType){
		case BMA:
			for(j=0;j<model->modelState.l0VectorLength;j++){
				max_v = getParameters(model,0,j)->lowerBound;
				w_max = 0;
				Z =0;
				for(i=0;i<model->modelState.numberOfRealizations;i++){
					if(getParameters(realizationIndex,penaltyIndex)->lowerBound > max_v){
						max_v = getParameters(realizationIndex,penaltyIndex)->lowerBound;
						w_max = i;
					}
					lb_mat[(model->modelState.numberOfRealizations)*(j)+i] = getParameters(realizationIndex,penaltyIndex)->lowerBound;
					lb_t[i] = getParameters(realizationIndex,penaltyIndex)->lowerBound;
				}
				for(i=0;i<model->modelState.numberOfRealizations;i++){
					Z = Z + exp(getParameters(realizationIndex,penaltyIndex)->lowerBound-max_v);
				}
				for(i=0;i<model->modelState.numberOfRealizations;i++){
					post_prob[i] = exp(getParameters(realizationIndex,penaltyIndex)->lowerBound-max_v)/Z;
					//Rprintf("post_prob[%d]: %g\t",i,post_prob[i]);
					//lb_mat[(model->modelState.numberOfRealizations)*(j)+i] = post_prob[i];
					//bm = bm + post_prob*getParameters(realizationIndex,penaltyIndex)->betaChi
				}
				//Rprintf("\n");

				identifyUniqueModels(lb_t,post_prob,model->modelState.numberOfRealizations,model->modelState.epsilon*10);

				for(k=0;k<model->data.numberPenalizedFeatures;k++){
					bc =0;
					bm =0;
					bs=0;
					eb=0;
					bp=0;
					switch(model->modelState.errType){
					case APPR:
						s_bma = 1;
						break;
					case EXACT:
						computeBayesianModelAveragingCovarianceCorrection(model,k,post_prob,&s_bma,j);
						break;
					default:
						Rprintf("BMA computation not specified!\n");
						break;
					}

					for(i=0;i<model->modelState.numberOfRealizations;i++){
						bc = bc+ post_prob[i]*getParameters(realizationIndex,penaltyIndex)->betaChi[k];
						bm = bm+ post_prob[i]*getParameters(realizationIndex,penaltyIndex)->betaMu[k];
						bs = bs+ post_prob[i]*getParameters(realizationIndex,penaltyIndex)->betaSigmaSquared[k];
						eb = eb+ post_prob[i]*getParameters(realizationIndex,penaltyIndex)->expectationBeta[k];
						bp = bp+ post_prob[i]*getParameters(realizationIndex,penaltyIndex)->betaPosteriorProbability[k];
					}
					beta_chi_mat[(model->data.numberPenalizedFeatures)*(j)+k] = bc/sqrt(s_bma);
					beta_mu_mat[(model->data.numberPenalizedFeatures)*(j)+k] = bm;
					beta_sigma_mat[(model->data.numberPenalizedFeatures)*(j)+k] = bs;
					e_beta_mat[(model->data.numberPenalizedFeatures)*(j)+k] = eb;
					beta_p_mat[(model->data.numberPenalizedFeatures)*(j)+k] = bp;
				}
			}

			break;

		case MAXIMAL:
			////
			for(j=0;j<model->modelState.l0VectorLength;j++){
				max_v = getParameters(model,0,j)->lowerBound;
				w_max = 0;
				for(i=0;i<model->modelState.numberOfRealizations;i++){
					if(getParameters(realizationIndex,penaltyIndex)->lowerBound > max_v){
						max_v = getParameters(realizationIndex,penaltyIndex)->lowerBound;
						w_max = i;
					}
					lb_mat[(model->modelState.numberOfRealizations)*(j)+i] = getParameters(realizationIndex,penaltyIndex)->lowerBound;
				}
				for(k=0;k<model->data.numberPenalizedFeatures;k++){
					beta_chi_mat[(model->data.numberPenalizedFeatures)*(j)+k] = getParameters(model,w_max,j)->betaChi[k];
					beta_mu_mat[(model->data.numberPenalizedFeatures)*(j)+k] = getParameters(model,w_max,j)->betaMu[k];
					beta_sigma_mat[(model->data.numberPenalizedFeatures)*(j)+k] = getParameters(model,w_max,j)->betaSigmaSquared[k];
					e_beta_mat[(model->data.numberPenalizedFeatures)*(j)+k] = getParameters(model,w_max,j)->expectationBeta[k];
					beta_p_mat[(model->data.numberPenalizedFeatures)*(j)+k] = getParameters(model,w_max,j)->betaPosteriorProbability[k];
				}
			}
			break;

		default:
			////
			break;


	}
	free(post_prob);
	free(lb_t);

}





void runGaussianVariationalBayesSpikeRegression(double * epsilon,
			double * l0Vector,
			double * priorProbabilityVector,
			double * featureColumnScalingFactor,
			int * maximumIterations,
			int * l0VectorLength,
			int * numberOfRealizations,
			int * scale,
			int * est,
			int * approx,
			int * totalModelFits,
			double * penalizedDataMatrix,
			double * responseVariable,
			double * responseVariance,
			int * numberSamples,
			int * numberPenalizedFeatures,
			int * numberUnpenalizedFeatures,
			int * realizationMatrix,
			double * betaMuResult,
			double * betaSigmaSquaredResult,
			double * expectationBetaResult,
			double * posteriorProbabilityBetaResult,
			double * lowerBoundResult,
			double * sigmaSquaredErrorResult,
			double * alphaResult,
			int * nthreads){


	struct model_struct model;
	//omp_set_num_threads(*nthreads);
	//Rprintf("nthreads: %d, nthreads_o: %d\n",*nthreads,omp_get_max_threads());
	//Rprintf("Initializing model...\n");
	initialize_model(epsilon,l0Vector,priorProbabilityVector,penalizeVariable,featureColumnScalingFactor,maximumIterations,l0VectorLength,numberOfRealizations,scale,est,approx,totalModelFits,penalizedDataMatrix, responseVariable, responseVariance, n, m,realizationMatrix,&model);
	//Rprintf("Initialized model...\n");
	runGaussianVariationalBayesSpikeRegression(&model);
	//Rprintf("Model run...\n");
	extractGaussianResults(&model,betaMuResult,betaSigmaSquaredResult,expectationBetaResult,posteriorProbabilityBetaResult,lowerBoundResult,sigmaSquaredErrorResult,alphaResult);
	//Rprintf("Results computed..\n");
	freeGaussianModel(&model);

}
