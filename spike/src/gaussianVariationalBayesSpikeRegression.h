//gaussian variational Bayes spike regression (vbsr) R package C library declarations
//Copyright 2016, Benjamin A Logsdon


#include <R.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>

//modelSpaceSelection: whether to do Bayesian model averaging or
//		  take the mode with maxmimum lower bound
typedef enum {BMA, MAXIMAL, FULL} modelSpaceSelectionType;

//X_mat: the data structure used to contain
//	 a set of m vectors of length n
struct doubleColumnVector {

	//x_col: a given vector of length n
	double * column;

};

//matrix_i: int version of matrix_v.
struct integerColumnVector {

	//x_col: a given vector of length n
	int * column;

};


//control_param_struct: a data structure that contains all the relevant
//			control parameters to run the vbsr package
struct controlParameterSettings {

	//epsilon: tolerance used to asses convergence.
	//     default: 1e-6
	const double epsilon;

	//l0Vector: a vector of the 'l0' penalty parameters to run along
	const double * l0Vector;

	//priorProbabilityVector: logistic transformation of l0Vector
	//	   default: 1/(1+exp(-l0Vector))
	const double * priorProbabilityVector;

	//penalizeVariable: an indicator vector which is 1 if a variable is penalized,
	//         and 0 otherwise
	//	   default: everything is 1 except intercept
	const int * penalizeVariable;

	//featureColumnScalingFactor: a vector of optional rescaling of predictors
	//		  default: everything is 1
	const double * featureColumnScalingFactor;

	//maximumIterations: maximum iterations to run
	//	 default: 1e4
	const int maximumIterations;

	//l0VectorLength: length of path
	//	       default: 50
	const int l0VectorLength;

	//numberOfRealizations: number of restarts of algorithm
	//	       default: 100
	const int numberOfRealizations;

	//scaleFeatureMatrix: whether or not to scale the columns
	//	     default: SCALE
	const int scaleFeatureMatrix;

	//estType: whether to do Bayesian model averaging
	//		across identified modes or to take the
	//		maximal mode
	//		default: BMA
	modelSpaceSelectionType modelSpaceSelectionUsed;

	//bType: whether to do exact or approximate b.m.a
	//		 correction of z score
	//		 default: APPR
	const int applyBmaCovarianceCorrection;

	//totalModelFits: l0VectorLength*numberOfRealizations
	const int totalModelFits;


};

//model_param_struct: a data structure that contains all the relevant
//		      model parameters describing the state
//		      of a given run of the vbsr algorithm
//struct for gaussian link model
struct gaussianLinkModelParameters {
	//beta_mu: beta mean update from vbsr algorithm
	//	   default: everything initialized to 0
	//double * beta_mu;
	double * betaMu;

	//beta_sigma: beta variance update from vbsr algorithm
	//	      default: everything initialized to 0
	//double * beta_sigma;
	double * betaSigmaSquared;

	//beta_chi: betaMu/sqrt(betaSigmaSquared)
	//	    default: 0
	//double * beta_chi;
	double * betaChi;

	//beta_p: beta variational scaling update from vbsr algorithm
	//	  default: everything initialized to 0
	//double * beta_p;
	double * betaPosteriorProbability;

	//e_beta: expectation of beta from vbsr algorithm
	//	  default: everything initalized to 0
	//	  update: beta_p*beta_mu
	//double * e_beta;
	double * expectationBeta;

	//e_beta_sq: expectation of beta_sq from vbsr algorithm
	//	     default: everything is initialized to 0
	//	     update: beta_p*(beta_mu^2+beta_sigma)
	//double * e_beta_sq;
  double * expectationBetaSquared;

	//sigma_e: error variance update from vbsr algorithm
	//	   default: initialized to variance of phenotype
	//double sigma_e;
	double sigmaSquaredError;

	//lb: lower bound from vbsr algorithm
	//    default: 0
	//double lb;
	double lowerBound;

	//p_sums: sum of beta_p
	//	  default: 0
	//double p_sums;
	double sumOfBetaPosteriorProbabilities;

	//double entropy;
	double posteriorProbabilityEntropy;

	//v_sums_correct: a variable used to correct the lb
	//		  expectation for the beta^2 terms
	//		  sum_{j} (e_beta^2-e_beta_sq)*x_sum_sq
	//		  default: 0
	//double v_sums_correct;
	double betaSquaredExpectationCorrection;

	//resid_vec: the residual vector y-X%*%e_beta
	//	     default: y
	//double * resid_vec;
	double * residualVector;

	//ord_index: the ordering of the current model_param
	//	     default: 0
	//int ord_index;
	int realizationIndex;

	//path_index: the path index of the current model_param
	//	      default: 0
	//int path_index;
	int penaltyIndex;
};


//data_struct: the data structure that contains
//	       all of the fixed data vectors
struct dataRealization {

	//X: the data structure containing all of the
	//   relevant variables.
	const struct doubleColumnVector * penalizedDataMatrix;

	//add in unpenalized data matrix
	const struct double ** unpenalizedDataMatrix;

	//y: the vector containing the phenotype data
	const double * responseVariable;

	//responseVariance: the variance of the phenotype
	const double responseVariance;

	//n: the number of samples
	const int numberSamples;

	//m: the number of features
	const int numberPenalizedFeatures;

	//p: the number of unpenalized features
	const int numberUnpenalizedFeatures;

	//x_sum_sq: a vector of the l2^2 norm of the columns of X.
	//const double * x_sum_sq;
	const double * penalizedDataMatrixColumnSumOfSquares;

	//ordering: a vector of vectors of the multiple orderings
	//	    to be run by the algorithm
	const struct integerColumnVector * realizationMatrix;

	//one_vec: a vector of length n of ones
	const double * onesVector;

};


//order_struct: A data structure containing a given ordering
//		for all the runs of the algorithm

//struct order_struct {
struct gaussianLinkModelParameterRealization {

	struct gaussianLinkModelParameters * modelParameters;

};



//model_struct: A single data structure that contains
//		all of the control parameters,
//		model parameters, and data
//		for a given run of the vbsr algorithm

struct gaussianModelRealization {

	//control_param: the control parameter
	//		  struct for a given model
	struct controlParameterSettings modelState;


	//data: the data for a given model
	struct dataRealization data;

	//*model_param: a pointer to a pointer for the model
	//		 parameters for a given run of the
	// 		 algorithm.  The model_params are
	//		 indexed over the realizations first
	//		 then the l0 path.
	struct gaussianLinkModelParameterRealization * modelParameterRealization;


};

//need to do a more explicit model comparison check
void identifyUniqueModels(gaussianModelRealization * model);

double * extractPenalizedFeatureMatrixColumn(struct gaussianModelRealization * model, int columnIndex);

int * extractRealizationMatrixColumn(struct gaussianModelRealization * model, int columnIndex);

struct gaussianModelRealization * getParameters(struct gaussianModelRealization * model,int realizationIndex, int penaltyIndex);

void initializeGaussianModelParameters(int numberSamples,
				int numberPenalizedFeatures,
				int numberUnpenalizedFeatures,
				int realizationIndex,
				int penaltyIndex,
				struct gaussianModelRealization * model,
				double * responseVariable,
				double responseVariance);

void freeGaussianModelParameters(struct gaussianModelRealization * model, int realizationIndex, int penaltyIndex);

void initializeGaussianModel(double * eps,
			double * l0Vector,
			double * priorProbabilityVector,
			int * penalizeVariable,
			double * featureColumnScalingFactor,
			int * maximumIterations,
			int * l0VectorLength,
			int * numberOfRealizations,
			int * scale,
			int * est,
			int * applyBmaCovarianceCorrection,
			int * totalModelFits,
			double * penalizedDataMatrix,
			double * unpenalizedDataMatrix,
			double * responseVariable,
			double * responseVariance,
			int * numberSamples,
			int * numberPenalizedFeatures,
			int * numberUnpenalizedFeatures,
			int * realizationMatrix,
			struct gaussianModelRealization * model);

void freeGaussianModel(struct gaussianModelRealization * model);

void dataStandardization(struct gaussianModelRealization * model);

void copyGaussianModelState(struct gaussianModelRealization * model, int i, int j);

void updateGaussianBetaDistribution(struct gaussianModelRealization * model, int realizationIndex, int penaltyIndex);

void updateGaussianAlphaEstimate(struct gaussianModelRealization * model, int realizationIndex, int penaltyIndex);

void updateGaussianSigmaSquaredErrorEstimate(struct gaussianModelRealization * model, int i, int j);

void updateGaussianLowerBound(struct gaussianModelRealization * model, int i, int j);

void runGaussianVariationalBayesSpikeRegression(struct gaussianModelRealization * model);

void computeBayesianModelAveragingCovarianceCorrection(struct gaussianModelRealization * model,int k,double * post_prob,double * s_bma,int j);

void extractGaussianResults(struct gaussianModelRealization * model,
														double * betaMuResult,
														double * betaSigmaSquaredResult,
														double * expectationBetaResult,
														double * posteriorProbabilityBetaResult,
														double * lowerBoundResult,
														double * sigmaSquaredErrorResult,
														double * alphaResult);

void gaussianVariationalBayesSpikeRegression(double * eps,
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
			double * unpenalizedDataMatrix,
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
			int * nthreads);
