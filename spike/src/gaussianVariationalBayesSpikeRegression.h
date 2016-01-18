//generalized variational Bayes spike regression (vbsr) R package C library declarations
//Copyright 2016, Benjamin A Logsdon


#include <R.h>
#include <Rmath.h>
#include <stdio.h>
#include <R_ext/BLAS.h>

//generalizedLinearModelType: the type of genearlized linear model to fit
typedef enum {NORMAL, BINOMIAL, GAMMA, POISSON, MULTINOMIAL, INVERSEGAUSSIAN} generalizedLinearModelType;

//modelSpaceSelection: whether to do Bayesian model averaging or
//		  take the mode with maxmimum lower bound
typedef enum {BMA,MAXIMAL,FULL} modelSpaceSelectionType;

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

	//pb_path: logistic transformation of l0_path
	//	   default: 1/(1+exp(-l0_path))
	const double * priorProbabilityVector;

	//exclude: an indicator vector which is 1 if a variable is penalized,
	//         and 0 otherwise
	//	   default: everything is 1 except intercept
	const int * penalizeVariable;

	//penalty_factor: a vector of optional rescaling of predictors
	//		  default: everything is 1
	const double * featureColumnScalingFactor;

	//maxit: maximum iterations to run
	//	 default: 1e4
	const int maximumIterations;

	//path_length: length of path
	//	       default: 50
	const int l0VectorLength;

	//n_orderings: number of restarts of algorithm
	//	       default: 100
	const int numberOfRandomRestarts;

	//regressType: the type of regression used
	//	       default: LINEAR
	generalizedLinearModelType generalizedLinearModelUsed;

	//scaleType: whether or not to scale the columns
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

	//total_replicates: path_length*n_orderings
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

	//w_vec: the weights in the irls vbsr logistic regression
	//	 algorithms
	//	 default: 1
	//double * w_vec;
	double * irlsWeightVector;

	//mu_vec: the pred values in the irls vbsr logistic
	//	  regression algorithm
	//        default: 0
	//double * mu_vec;
	double * irlsPredictionVector;

	//resid_vec: the residual vector y-X%*%e_beta
	//	     default: y
	//double * resid_vec;
	double * residualVector;

	//pred_vec_old: the old prediction vector for irls vbsr
	//	 	default: 0
	//double * pred_vec_old;
	double * irlsOldPredictionVector;

	//pred_vec_new: the new prediction vector for irls vbsr
	//		default: 0
	//double * pred_vec_new;
	double * irlsNewPredictionVector;

	//x_w: the reweightings...
	//	default: 0;
	//double * x_w;
	double * irlsReweightings;


	//ord_index: the ordering of the current model_param
	//	     default: 0
	//int ord_index;
	int orderingIndex;

	//path_index: the path index of the current model_param
	//	      default: 0
	//int path_index;
	int penaltyPathIndex;


};

//struct for logistic regression parameters
struct binomialLinkModelParameters {
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

	//w_vec: the weights in the irls vbsr logistic regression
	//	 algorithms
	//	 default: 1
	//double * w_vec;
	double * irlsWeightVector;

	//mu_vec: the pred values in the irls vbsr logistic
	//	  regression algorithm
	//        default: 0
	//double * mu_vec;
	double * irlsPredictionVector;

	//resid_vec: the residual vector y-X%*%e_beta
	//	     default: y
	//double * resid_vec;
	double * residualVector;

	//pred_vec_old: the old prediction vector for irls vbsr
	//	 	default: 0
	//double * pred_vec_old;
	double * irlsOldPredictionVector;

	//pred_vec_new: the new prediction vector for irls vbsr
	//		default: 0
	//double * pred_vec_new;
	double * irlsNewPredictionVector;

	//x_w: the reweightings...
	//	default: 0;
	//double * x_w;
	double * irlsReweightings;


	//ord_index: the ordering of the current model_param
	//	     default: 0
	//int ord_index;
	int orderingIndex;

	//path_index: the path index of the current model_param
	//	      default: 0
	//int path_index;
	int penaltyPathIndex;


};

//data_struct: the data structure that contains
//	       all of the fixed data vectors
struct data {

	//X: the data structure containing all of the
	//   relevant variables.
	const struct doubleColumnVector * penalizedDataMatrix;

	//add in unpenalized data matrix
	const struct doubleColumnVector * unpenalizedDataMatrix;

	//y: the vector containing the phenotype data
	const double * responseVariable;

	//var_y: the variance of the phenotype
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
	const struct integerColumnVector * orderingMatrix;

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
	struct data dataRealization;

	//*model_param: a pointer to a pointer for the model
	//		 parameters for a given run of the
	// 		 algorithm.  The model_params are
	//		 indexed over the orderings first
	//		 then the l0 path.
	struct gaussianLinkModelParameterRealization * modelParameterRealization;


};


void identify_unique(double * lb_t, double * post_p, int numberSamples,double epsilon);

double * extractPenalizedFeatureMatrixColumn(struct gaussianModelRealization * model, int columnIndex);

int * extractOrderingMatrixColumn(struct gaussianModelRealization * model, int columnIndex);

struct gaussianModelRealization * getModelParameterRealization(struct gaussianModelRealization * model,int realizationIndex, int penaltyIndex);

void initialize_model_param(int numberSamples,
				int numberPenalizedFeatures,
				int realizationIndex,
				int penaltyIndex,
				struct gaussianModelRealization * model,
				double * responseVariable,
				double responseVariance);

void free_model_param(struct gaussianModelRealization * model, int realizationIndex, int penaltyIndex);

void initialize_model(double * eps,
			double * l0_path,
			double * pb_path,
			int * exclude,
			double * penalty_factor,
			int * maxit,
			int * path_length,
			int * n_orderings,
			int * regress,
			int * scale,
			int * est,
			int * error,
			double * kl,
			int * approx,
			int * total_replicates,
			double * X,
			double * y,
			double * var_y,
			int * n,
			int * m,
			int * ordering_mat,
			struct gaussianModelRealization * model);

void initialize_model_marg(double * eps,
			int * exclude,
			int * maxit,
			int * regress,
			int * scale,
			double * X,
			double * y,
			double * var_y,
			int * n,
			int * m,
			struct model_marg_struct * model);

void free_model(struct gaussianModelRealization * model);

void free_model_marg(struct model_marg_struct * model);

void process_data(struct gaussianModelRealization * model);

void process_data_marg(struct model_marg_struct * model);

void copy_model_state(struct gaussianModelRealization * model, int i, int j);

void update_beta(struct gaussianModelRealization * model, int i, int j);

void update_beta_marg(struct model_marg_struct * model, int * use_vec,int cv);

void update_error(struct gaussianModelRealization * model, int i, int j);

void update_error_marg(struct model_marg_struct * model);

void update_lb(struct gaussianModelRealization * model, int i, int j);

void update_lb_marg(struct model_marg_struct * model);

void run_vbsr(struct gaussianModelRealization * model);

void run_marg(struct model_marg_struct * model);

void compute_bma_correct(struct gaussianModelRealization * model,int k,double * post_prob,double * s_bma,int j);

void collapse_results(struct gaussianModelRealization * model,
			double * beta_chi_mat,
			double * beta_mu_mat,
			double * beta_sigma_mat,
			double * e_beta_mat,
			double * beta_p_mat,
			double * lb_mat,
			double * kl_mat);

void collapse_results_marg(struct model_marg_struct * model,
			double * beta_chi,
			double * beta_mu,
			double * beta_sigma,
			double * beta_p,
			double * lb);

void run_marg_analysis(double * eps,
			int * exclude,
			int * maxit,
			int * regress,
			int * scale,
			double * X,
			double * y,
			double * var_y,
			int * n,
			int * m,
			double * beta_chi,
			double * beta_mu,
			double * beta_sigma,
			double * beta_p_mat,
			double * lb);

void run_vbsr_wrapper(double * eps,
			double * l0_path,
			double * pb_path,
			int * exclude,
			double * penalty_factor,
			int * maxit,
			int * path_length,
			int * n_orderings,
			int * regress,
			int * scale,
			int * est,
			int * error,
			double * kl,
			int * approx,
			int * total_replicates,
			double * X,
			double * y,
			double * var_y,
			int * n,
			int * m,
			int * ordering_mat,
			double * beta_chi_mat,
			double * beta_mu_mat,
			double * beta_sigma_mat,
			double * e_beta_mat,
			double * beta_p_mat,
			double * lb_mat,
			double * kl_mat,
			int * nthreads);