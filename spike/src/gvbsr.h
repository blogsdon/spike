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
struct modelParameters {
	//beta_mu: beta mean update from vbsr algorithm
	//	   default: everything initialized to 0
	double * beta_mu;

	//beta_sigma: beta variance update from vbsr algorithm
	//	      default: everything initialized to 0
	double * beta_sigma;

	//beta_chi: beta_mu^2/beta_sigma
	//	    default: 0
	double * beta_chi;

	//beta_p: beta variational scaling update from vbsr algorithm
	//	  default: everything initialized to 0
	double * beta_p;

	//e_beta: expectation of beta from vbsr algorithm
	//	  default: everything initalized to 0
	//	  update: beta_p*beta_mu
	double * e_beta;

	//e_beta_sq: expectation of beta_sq from vbsr algorithm
	//	     default: everything is initialized to 0
	//	     update: beta_p*(beta_mu^2+beta_sigma)
	double * e_beta_sq;

	//sigma_e: error variance update from vbsr algorithm
	//	   default: initialized to variance of phenotype
	double sigma_e;

	//lb: lower bound from vbsr algorithm
	//    default: 0
	double lb;

	//p_sums: sum of beta_p
	//	  default: 0
	double p_sums;

	double entropy;

	//v_sums_correct: a variable used to correct the lb
	//		  expectation for the beta^2 terms
	//		  sum_{j} (e_beta^2-e_beta_sq)*x_sum_sq
	//		  default: 0
	double v_sums_correct;

	//w_vec: the weights in the irls vbsr logistic regression
	//	 algorithms
	//	 default: 1
	double * w_vec;

	//mu_vec: the pred values in the irls vbsr logistic
	//	  regression algorithm
	//        default: 0
	double * mu_vec;

	//resid_vec: the residual vector y-X%*%e_beta
	//	     default: y
	double * resid_vec;

	//pred_vec_old: the old prediction vector for irls vbsr
	//	 	default: 0
	double * pred_vec_old;

	//pred_vec_new: the new prediction vector for irls vbsr
	//		default: 0
	double * pred_vec_new;

	//x_w: the reweightings...
	//	default: 0;
	double * x_w;


	//ord_index: the ordering of the current model_param
	//	     default: 0
	int ord_index;

	//path_index: the path index of the current model_param
	//	      default: 0
	int path_index;


};




//data_struct: the data structure that contains
//	       all of the fixed data vectors
struct data {

	//X: the data structure containing all of the
	//   relevant variables.
	const struct doubleColumnVector * x;

	//y: the vector containing the phenotype data
	const double * y;

	//var_y: the variance of the phenotype
	const double responseVariance;

	//n: the number of samples
	const int n;

	//m: the number of features
	const int m;

	//p: the number of unpenalized features
	const int p;

	//x_sum_sq: a vector of the l2^2 norm of the columns of X.
	const double * x_sum_sq;

	//ordering: a vector of vectors of the multiple orderings
	//	    to be run by the algorithm
	const struct integerColumnVector * orderingMatrix;

	//one_vec: a vector of length n of ones
	const double * onesVector;

};


//order_struct: A data structure containing a given ordering
//		for all the runs of the algorithm

struct order_struct {

	struct model_param_struct * model_param;

};



//model_struct: A single data structure that contains
//		all of the control parameters,
//		model parameters, and data
//		for a given run of the vbsr algorithm

struct model_struct {

	//control_param: the control parameter
	//		  struct for a given model
	struct control_param_struct control_param;


	//data: the data for a given model
	struct data_struct data;


	//*model_param: a pointer to a pointer for the model
	//		 parameters for a given run of the
	// 		 algorithm.  The model_params are
	//		 indexed over the orderings first
	//		 then the l0 path.
	struct order_struct * order;


};

//control_param_marg: Control parameters for marginal analysis

struct control_param_marg {
	double eps;
	int * exclude;
	int maxit;
	regressionType regressType;
	scalingType scaleType;
};

struct model_param_marg {
	double * beta_mu;
	double * beta_sigma;
	double * beta_chi;
	//beta_p is a p-value, not posterior probability, in marginal analysis
	double * beta_p;
	double sigma_e;
	double lb;
	double * w_vec;
	double * mu_vec;
	double * resid_vec;
	double * pred_vec_old;
	double * pred_vec_new;
	double * x_w;
};



//marginal_model_struct: A single data structure that contains
//						all of the control parameters,
//						model parameters, and data
//						for a given marginal analysis

struct model_marg_struct {

	//control_param: the control parameter
	//				  struct for a given model
	struct control_param_marg control_param;

	//data: the data for a given model
	struct data_struct data;

	//model_param: the model params
	struct model_param_marg model_param;

};


void identify_unique(double * lb_t, double * post_p, int n,double tol);

inline double * xc(struct model_struct * model, int j);

inline double * xcm(struct model_marg_struct * model, int j);

inline int * oc(struct model_struct * model, int j);

inline struct model_param_struct * me(struct model_struct * model,int i, int j);

void initialize_model_param(int n,
				int m,
				int i,
				int j,
				struct model_struct * model,
				double * y,
				double var_y);

void free_model_param(struct model_struct * model, int i, int j);

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
			struct model_struct * model);

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

void free_model(struct model_struct * model);

void free_model_marg(struct model_marg_struct * model);

void process_data(struct model_struct * model);

void process_data_marg(struct model_marg_struct * model);

void copy_model_state(struct model_struct * model, int i, int j);

void update_beta(struct model_struct * model, int i, int j);

void update_beta_marg(struct model_marg_struct * model, int * use_vec,int cv);

void update_error(struct model_struct * model, int i, int j);

void update_error_marg(struct model_marg_struct * model);

void update_lb(struct model_struct * model, int i, int j);

void update_lb_marg(struct model_marg_struct * model);

void run_vbsr(struct model_struct * model);

void run_marg(struct model_marg_struct * model);

void compute_bma_correct(struct model_struct * model,int k,double * post_prob,double * s_bma,int j);

void collapse_results(struct model_struct * model,
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
