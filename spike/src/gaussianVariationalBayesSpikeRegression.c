//generalized variational Bayes spike regression (vbsr) R package C library
//Copyright 2016 Benjamin A Logsdon
#include "gaussianVariationalBayesSpikeRegression.h"

//xc: return jth column of data matrix x
double * extractPenalizedFeatureMatrixColumn(struct gaussianModelRealization * model,
	int columnIndex){
	return (&(model->data.penalizedDataMatrix[columnIndex]))->column;
}

//oc: return jth column of ordering matrix "ordering"
int * extractRealizationMatrixColumn(struct gaussianModelRealization * model,
	int columnIndex){
	return (&(model->data.realizationMatrix[columnIndex]))->column;
}

//me: return the ith, jth element of the ordering, path model parameters
struct gaussianModelRealization * getModelParameterRealization(struct gaussianModelRealization * model,
	 int realizationIndex,
	 int penaltyIndex){
	return (&((&(model->modelParameterRealization[realizationIndex]))->modelParameters[penaltyIndex]));
}

void initializeGaussianModelParameters(int numberSamples,
				int numberPenalizedFeatures,
				int realizationIndex,
				int penaltyIndex,
				struct gaussianModelRealization * model,
				double * responseVariable,
				double responseVariance){

	int k;

	getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaMu = (double *) malloc(sizeof(double)*numberPenalizedFeatures);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaSigmaSquared = (double *) malloc(sizeof(double)*numberPenalizedFeatures);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaChi = (double *) malloc(sizeof(double)*numberPenalizedFeatures);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaPosteriorProbability = (double *) malloc(sizeof(double)*numberPenalizedFeatures);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->expectationBeta = (double *) malloc(sizeof(double)*numberPenalizedFeatures);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->expectationBetaSquared = (double *) malloc(sizeof(double)*numberPenalizedFeatures);


	for(k=0;k<numberPenalizedFeatures;k++){
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaMu[k] = 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaSigmaSquared[k] = 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaChi[k]= 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaPosteriorProbability[k] = 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->expectationBeta[k] = 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->expectationBetaSquared[k] = 0;
	}


	getModelParameterRealization(model,realizationIndex,penaltyIndex)->sigma_e = var_y;
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->lb = 0;
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->p_sums = 0;
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->entropy = 0;
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->v_sums_correct = 0;

	getModelParameterRealization(model,realizationIndex,penaltyIndex)->resid_vec = (double *) malloc(sizeof(double)*n);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->pred_vec_old = (double *) malloc(sizeof(double)*n);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->pred_vec_new = (double *) malloc(sizeof(double)*n);
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->x_w = (double *) malloc(sizeof(double)*n);

	for(k=0;k<n;k++){
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->w_vec[k] = 0.25;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->mu_vec[k] = 0.5;
		switch(model->control_param.regressType){
			case LINEAR:
				getModelParameterRealization(model,realizationIndex,penaltyIndex)->resid_vec[k] = y[k];
				break;
			case LOGISTIC:
				getModelParameterRealization(model,realizationIndex,penaltyIndex)->resid_vec[k] = (y[k] - 0.5)/(0.25);
				break;
		}
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->pred_vec_old[k] = 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->pred_vec_new[k] = 0;
		getModelParameterRealization(model,realizationIndex,penaltyIndex)->x_w[k]=0;
	}

	getModelParameterRealization(model,realizationIndex,penaltyIndex)->ord_index = i;
	getModelParameterRealization(model,realizationIndex,penaltyIndex)->path_index = j;
}

void free_model_param(struct gaussianModelRealization * model, int i, int j){

	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaMu);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaSigmaSquared);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaChi);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->betaPosteriorProbability);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->expectationBeta);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->expectationBetaSquared);

	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->w_vec);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->mu_vec);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->resid_vec);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->pred_vec_old);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->pred_vec_new);
	free(getModelParameterRealization(model,realizationIndex,penaltyIndex)->x_w);

}

void process_data(struct gaussianModelRealization * model){
	int j;
  int exc;
	double nd = ((double) model->data.n);
	switch(model->control_param.scaleType){

		case SCALE:
			Rprintf("Scaling...\n");
		  Rprintf("m: %d\n",model->data.m);

			for(j=0;j<model->data.m;j++){
			  exc = model->control_param.exclude[j];
				if(exc==0){
					standardizeVector(extractPenalizedFeatureMatrixColumn(model,columnIndex),model->data.one_vec,model->data.n);
				}
				model->data.x_sum_sq[j] = nd - 1;
			  //model->data.x_sum_sq[j]=vectorSumOfSquares(extractPenalizedFeatureMatrixColumn(model,columnIndex),model->data.n);
			}
			break;

		case NOSCALE:
			//Rprintf("Sum of squares pre-compute...\n");
			for(j=0;j<model->data.m;j++){
				model->data.x_sum_sq[j]=vectorSumOfSquares(extractPenalizedFeatureMatrixColumn(model,columnIndex),model->data.n);
			}
			break;
	}


}


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
					struct gaussianModelRealization * model){

	//initialize: (*model).control_param;

	//Rprintf("eps: %g\n",eps[0]);
	//Rprintf("l0_path[0]: %g\n",l0_path[0]);
	//Rprintf("pb_path[0]: %g\n",pb_path[0]);
	//Rprintf("exclude[1]: %d\n",exclude[1]);
	//Rprintf("pf[0]: %g\n",penalty_factor[0]);
	//Rprintf("maxit: %d\n",maxit[0]);
	//Rprintf("path_length: %d\n",path_length[0]);
	//Rprintf("n_orderings: %d\n",n_orderings[0]);
	//Rprintf("regress: %d\n",regress[0]);
	//Rprintf("scale: %d\n",scale[0]);
	//Rprintf("est: %d\n",est[0]);
	//Rprintf("error: %d\n",error[0]);
	//Rprintf("kl: %g\n",kl[0]);
	//Rprintf("tr: %d\n",total_replicates[0]);
	//Rprintf("X[0,0]: %g\n",X[0]);
	//Rprintf("y[0]: %g\n",y[0]);
	//Rprintf("var_y: %g\n",var_y[0]);
	//Rprintf("n: %d\n",n[0]);
	//Rprintf("m: %d\n",m[0]);
	//Rprintf("ordering_mat[0]: %d\n",ordering_mat[0]);

	int k,l;
	model->control_param.eps = (*eps);
	//model->control_param.max_pb = (*max_pb);
	model->control_param.l0_path = (double *) malloc(sizeof(double)*(*path_length));
	model->control_param.pb_path = (double *) malloc(sizeof(double)*(*path_length));
	for(k=0;k<*path_length;k++){
		model->control_param.l0_path[k]=l0_path[k];
		model->control_param.pb_path[k]=pb_path[k];
	}

	model->control_param.exclude = (int *) malloc(sizeof(int)*(*numberPenalizedFeatures));
	model->control_param.penalty_factor = (double *) malloc(sizeof(double)*(*numberPenalizedFeatures));
	for(k=0;k<*m;k++){
		model->control_param.exclude[k]=exclude[k];
		model->control_param.penalty_factor[k]=penalty_factor[k];
	}

	model->control_param.maxit = (*maxit);
	model->control_param.path_length = (*path_length);
	model->control_param.n_orderings = (*n_orderings);
	if((*regress)==1){
		model->control_param.regressType = LINEAR;
	} else{
		model->control_param.regressType = LOGISTIC;
	}

	if((*scale)==1){
		model->control_param.scaleType = SCALE;
	} else{
		model->control_param.scaleType = NOSCALE;
	}

	if((*est)==1){
		model->control_param.estType = BMA;
	} else{
		model->control_param.estType = MAXIMAL;
	}

	if((*error)==1){
		model->control_param.errType = KL;
	} else{
		model->control_param.errType = NOKL;
	}

	if((*approx)==1){
		model->control_param.bType = APPR;
	} else{
		model->control_param.errType = EXACT;
	}


	model->control_param.kl_percentile = (*kl);
	model->control_param.total_replicates = (*total_replicates);
	//initialize: (*model).(data);
	//struct single_mod *single_mods;
	//single_mods= (single_mod *) malloc(sizeof(single_mod)*(n_order+1));

	model->data.X = (struct matrix_v *) malloc(sizeof(struct matrix_v)*(*numberPenalizedFeatures));
	for(k=0;k<(*numberPenalizedFeatures);k++){
		(&(model->data.X[k]))->col = (double *) malloc(sizeof(double)*(*n));
	}

	for(k=0;k<(*numberPenalizedFeatures);k++){
		for(l=0;l<(*n);l++){
			(&(model->data.X[k]))->col[l] = X[k*(*n)+l];
		}
	}

	model->data.y = y;
	model->data.var_y = (*var_y);
	model->data.n = (*n);
	model->data.m = (*numberPenalizedFeatures);
	int (pii) =0;
	for(k=0;k<(*numberPenalizedFeatures);k++){
		//Rprintf("exclude[%d]:%d\n",k,exclude[k]);
		if(exclude[k]==1){
			//Rprintf("worked\n");
			++(pii);
		}
	}
	model->data.p = (pii);
	//Rprintf("model->data.p = %d\n",model->data.p);
	model->data.x_sum_sq = (double *) malloc(sizeof(double)*(*numberPenalizedFeatures));


	model->data.ordering = (struct matrix_i *) malloc(sizeof(struct matrix_i)*(*n_orderings));
	for(k=0;k<(*n_orderings);k++){
		(&(model->data.ordering[k]))->col = (int *) malloc(sizeof(int)*(*numberPenalizedFeatures));
	}

	for(k=0;k<(*n_orderings);k++){
		for(l=0;l<(*numberPenalizedFeatures);l++){
			(&(model->data.ordering[k]))->col[l] = ordering_mat[k*(*numberPenalizedFeatures)+l];
		}
	}

	model->data.one_vec = (double *) malloc(sizeof(double)*(*n));
	for(k=0;k<(*n);k++){
		model->data.one_vec[k]= 1.0;
	}


	process_data(model);

	//initialize: (*model).getModelParameterRealization(model,realizationIndex,penaltyIndex);

	model->order = (struct order_struct *) malloc(sizeof(struct order_struct)*(*n_orderings));
	for(k=0;k<(*n_orderings);k++){
		(&(model->order[k]))->model_param = (struct model_param_struct *) malloc(sizeof(struct model_param_struct)*(*path_length));
	}

	for(k=0;k<(*n_orderings);k++){
		for(l=0;l<(*path_length);l++){
			initialize_model_param((*n),(*numberPenalizedFeatures),k,l,model,y,*var_y);
		}
	}


}




void free_model(struct gaussianModelRealization * model){
	//free X
	int i,j,k;
	for(k=0;k<(model->data.m);k++){
		free((&(model->data.X[k]))->col);

	}
	free(model->data.X);
	//free orderings

	for(k=0;k<(model->control_param.n_orderings);k++){
		free((&(model->data.ordering[k]))->col);
	}
	free(model->data.ordering);

	//free order:model_param
	for(i=0;i<model->control_param.n_orderings;i++){
		for(j=0;j<model->control_param.path_length;j++){
			free_model_param(model, i, j);
		}
	}


	for(k=0;k<(model->control_param.n_orderings);k++){
		free((&(model->order[k]))->model_param);
	}
	free(model->order);

	//free x_sum_sq

	free(model->data.x_sum_sq);

	//free one_vec

	free(model->data.one_vec);

	free(model->control_param.l0_path);
	free(model->control_param.pb_path);
	free(model->control_param.penalty_factor);
	free(model->control_param.exclude);


}

void copy_model_state(struct gaussianModelRealization * model, int i, int j){
	int k,l;
	l = j-1;

	for(k=0;k<model->data.m;k++){
		getModelParameterRealization(model,i,j)->betaMu[k] = getModelParameterRealization(model,i,l)->betaMu[k];
		getModelParameterRealization(model,i,j)->betaSigmaSquared[k] = getModelParameterRealization(model,i,l)->betaSigmaSquared[k];
		getModelParameterRealization(model,i,j)->betaChi[k] = getModelParameterRealization(model,i,l)->betaChi[k];
		getModelParameterRealization(model,i,j)->betaPosteriorProbability[k] = getModelParameterRealization(model,i,l)->betaPosteriorProbability[k];
		getModelParameterRealization(model,i,j)->expectationBeta[k] = getModelParameterRealization(model,i,l)->expectationBeta[k];
		getModelParameterRealization(model,i,j)->expectationBetaSquared[k] = getModelParameterRealization(model,i,l)->expectationBetaSquared[k];
	}


	getModelParameterRealization(model,i,j)->sigma_e = getModelParameterRealization(model,i,l)->sigma_e;
	getModelParameterRealization(model,i,j)->lb = getModelParameterRealization(model,i,l)->lb;
	getModelParameterRealization(model,i,j)->p_sums = getModelParameterRealization(model,i,l)->p_sums;
	getModelParameterRealization(model,i,j)->entropy = getModelParameterRealization(model,i,l)->entropy;
	getModelParameterRealization(model,i,j)->v_sums_correct = getModelParameterRealization(model,i,l)->v_sums_correct;

	for(k=0;k<model->data.n;k++){
		getModelParameterRealization(model,i,j)->w_vec[k] = getModelParameterRealization(model,i,l)->w_vec[k];
		getModelParameterRealization(model,i,j)->mu_vec[k] = getModelParameterRealization(model,i,l)->mu_vec[k];
		getModelParameterRealization(model,i,j)->resid_vec[k] = getModelParameterRealization(model,i,l)->resid_vec[k];
		getModelParameterRealization(model,i,j)->pred_vec_old[k] = getModelParameterRealization(model,i,l)->pred_vec_old[k];
		getModelParameterRealization(model,i,j)->pred_vec_new[k] = getModelParameterRealization(model,i,l)->pred_vec_new[k];
	}



}



void update_beta(struct gaussianModelRealization * model, int i, int j){

	int k,l,exc,t;
	double mu, sigma,prec, chi, p, e_b,e_b2,l0;


	//if(model->control_param.max_pb==1){
	//	l0 = getModelParameterRealization(model,i,j)->l0_max;
	//}else{
		l0 = model->control_param.l0_path[j];
	//}
	//Rprintf("l0: %g\n",l0);
	//Rprintf("m: %d\n",model->data.m);
	//error("!!!\n");
	switch (model->control_param.regressType){


		case LINEAR:
			//run linear updates
			for(l=0;l< model->data.m ;l++){
				k = (&(model->data.ordering[i]))->col[l];
				//k = l;
				//Rprintf("k: %d\n",k);
				exc = model->control_param.exclude[k];
				//Rprintf("exc: %d\n",exc);
				innerProduct(model->data.n,xc(model,k),getModelParameterRealization(model,realizationIndex,penaltyIndex)->resid_vec,&mu);
				//Rprintf("mu: %g\n",mu);
				//Rprintf("xumsq: %g, k: %d\n",model->data.x_sum_sq[k],k);
				mu = mu + (model->data.x_sum_sq[k])*(getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]);
				//Rprintf("mu: %g\n",mu);
				mu = mu/model->data.x_sum_sq[k];
				//Rprintf("mu: %g\n",mu);
				sigma = 1/((1/getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e)*(model->data.x_sum_sq[k]));
				//Rprintf("sigma: %g\n",sigma);
				chi = pow(mu,2)/sigma;
				//Rprintf("chi: %g, expectationBeta[%d]: %g\n",chi,k,getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]);
				if(exc==0){
					p = 1/(1+exp(-0.5*(chi+l0+log(sigma))));
					e_b = p*mu;
					e_b2 = p*(pow(mu,2)+sigma);
					//Rprintf("p: %g, e_b: %g, e_b2: %g\n",p,e_b,e_b2);
				}else{
					p = 0;
					e_b = mu;
					e_b2 = pow(mu,2);
					//Rprintf("p: %g, e_b: %g, e_b2: %g\n",p,e_b,e_b2);
				}


				if(exc==0){
				  getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct = getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct + (pow(e_b,2)-e_b2)*(model->data.x_sum_sq[k]);
				  getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums = getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums + p;
				  if(p>1-1e-10){
				    getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = getModelParameterRealization(realizationIndex,penaltyIndex)->entropy - p*log(p) + (1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
				  }else if(p<1e-10){
				    getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = getModelParameterRealization(realizationIndex,penaltyIndex)->entropy + p - (1-p)*log(1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
				  } else {
				    getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = getModelParameterRealization(realizationIndex,penaltyIndex)->entropy - p*log(p) - (1-p)*log(1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
				  }
				}

				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]-e_b);

				getModelParameterRealization(realizationIndex,penaltyIndex)->betaMu[k] = mu;
				getModelParameterRealization(realizationIndex,penaltyIndex)->betaSigmaSquared[k] = sigma;
				getModelParameterRealization(realizationIndex,penaltyIndex)->betaChi[k] = mu/sqrt(sigma);
				getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k] = e_b;
				getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBetaSquared[k] = e_b2;
				getModelParameterRealization(realizationIndex,penaltyIndex)->betaPosteriorProbability[k] = p;
			}
			break;
		case LOGISTIC:
			//run logistic updates
			for(l=0;l< model->data.m ;l++){
				k = (&(model->data.ordering[i]))->col[l];
				for(t=0; t< model->data.n;t++){
					getModelParameterRealization(realizationIndex,penaltyIndex)->x_w[t] = ((xc(model,k))[t])*(getModelParameterRealization(realizationIndex,penaltyIndex)->w_vec[t]);
				}

				//k = l;
				//Rprintf("k: %d\n",k);
				exc = model->control_param.exclude[k];
				//exc = 1;
				//Rprintf("exc: %d\n",exc);
				//sigma = 1/((1/getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e)*(model->data.x_sum_sq[k]));
				innerProduct(model->data.n,getModelParameterRealization(realizationIndex,penaltyIndex)->x_w,xc(model,k),&prec);
				sigma = 1/prec;
				//Rprintf("sigma: %g\n",sigma);
				innerProduct(model->data.n,getModelParameterRealization(realizationIndex,penaltyIndex)->x_w,getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,&mu);
				//Rprintf("mu: %g\n",mu);
				mu = mu + prec*(getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]);
				//Rprintf("mu: %g\n",mu);
				mu = mu/prec;
				//Rprintf("mu: %g\n",mu);

				chi = pow(mu,2)/sigma;
				//Rprintf("chi: %g, expectationBeta[%d]: %g\n",chi,k,getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]);
				if(exc==0){
					p = 1/(1+exp(-0.5*(chi+l0+log(sigma))));
					e_b = p*mu;
					e_b2 = p*(pow(mu,2)+sigma);
					//Rprintf("p: %g, e_b: %g, e_b2: %g\n",p,e_b,e_b2);
				}else{
					p = 0;
					e_b = mu;
					e_b2 = pow(mu,2);
					//Rprintf("p: %g, e_b: %g, e_b2: %g\n",p,e_b,e_b2);
				}

				getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums = getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums + p;
				if(p>1-1e-10){
					getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = getModelParameterRealization(realizationIndex,penaltyIndex)->entropy - p*log(p) + (1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
				}else if(p<1e-10){
					getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = getModelParameterRealization(realizationIndex,penaltyIndex)->entropy + p - (1-p)*log(1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
				} else {
					getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = getModelParameterRealization(realizationIndex,penaltyIndex)->entropy - p*log(p) - (1-p)*log(1-p) + 0.5*p*log(2*exp(1)*M_PI*sigma);
				}
				//getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct = getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct + (pow(e_b,2)-e_b2)*(model->data.x_sum_sq[k]);

				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]-e_b);
				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(realizationIndex,penaltyIndex)->pred_vec_new,e_b - getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k]);

				getModelParameterRealization(realizationIndex,penaltyIndex)->betaMu[k] = mu;
				getModelParameterRealization(realizationIndex,penaltyIndex)->betaSigmaSquared[k] = sigma;
				getModelParameterRealization(realizationIndex,penaltyIndex)->betaChi[k] = mu/sqrt(sigma);
				getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k] = e_b;
				getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta_sq[k] = e_b2;
				getModelParameterRealization(realizationIndex,penaltyIndex)->betaPosteriorProbability[k] = p;
			}

			break;

	}


}

void update_error(struct gaussianModelRealization * model, int i, int j){

	int t;
	double U;
	double nd = (double) model->data.n;

	switch(model->control_param.regressType){

		case LINEAR:

			innerProduct(model->data.n,getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,&U);
		  //Rprintf("U pre correction: %g\n",U);
			U = U - getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct;
		  //Rprintf("U post correction: %g, correction factor: %g\n",U, getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct);

			U = U/nd;
		  //Rprintf("U post division: %g\n",U);
			getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e = U;
		  //Rprintf("sigma_e: %g\n",getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e);
      //getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e = 1.0;
                        //Rprintf("no segfault\n");
			if(!R_FINITE(U)){
				free_model(model);
				//Rprintf("segfault\n");
				error("Penalized linear solution does not exist.\n");
				//error("uh oh\n");
			}

			break;

		case LOGISTIC:
			////
			for(t=0;t<model->data.n;t++){

				getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t] = 1/(1+exp(-getModelParameterRealization(realizationIndex,penaltyIndex)->pred_vec_new[t]));
				getModelParameterRealization(realizationIndex,penaltyIndex)->w_vec[t] = getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t]*(1-getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t]);
				getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec[t] = (model->data.y[t]-getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t])/getModelParameterRealization(realizationIndex,penaltyIndex)->w_vec[t];
				getModelParameterRealization(realizationIndex,penaltyIndex)->pred_vec_old[t] = getModelParameterRealization(realizationIndex,penaltyIndex)->pred_vec_new[t];
				if(getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t]==1 || getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t] ==0){
					//Rprintf("OVERFIT\n");
					free_model(model);
					error("Penalized logistic solution does not exist.\n");
				}

			}

			break;

	}

}

//void update_p_beta(struct gaussianModelRealization * model, int i, int j){

//	double md = (double) model->data.m;
//	double pd = (double) model->data.p;
//	double p_beta, l0;
//
//	p_beta = (getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums)/(md-pd);
//	//Rprintf("pd:%g, p_sums: %g, p_beta: %g, md: %g\n",pd,getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums,p_beta,md);
//	l0 = 2*(log(p_beta)-log(1-p_beta));
//	getModelParameterRealization(realizationIndex,penaltyIndex)->p_max = p_beta;
//	getModelParameterRealization(realizationIndex,penaltyIndex)->l0_max = l0;
//
//
//}

void update_lb(struct gaussianModelRealization * model, int i, int j){

	double lba;
	double nd = (double) model->data.n;
	double md = (double) model->data.m;
	double pd = (double) model->data.p;
	md = md - pd;
	double p_beta;
	//if(model->control_param.max_pb==1){
	//	p_beta = getModelParameterRealization(realizationIndex,penaltyIndex)->p_max;
	//}else{
		p_beta = model->control_param.pb_path[j];
	//}
	int t;
  double U;

	switch(model->control_param.regressType){

		case LINEAR:
      innerProduct(model->data.n,getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,getModelParameterRealization(realizationIndex,penaltyIndex)->resid_vec,&U);
			U = U - getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct;
      //Rprintf("here\n");
			lba = -0.5*nd*(log(2*M_PI*getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e) + 1);
			//Rprintf("lba: %g\n",lba);
			lba = lba + log(p_beta)*(getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums);
			//Rprintf("lba: %g\n",md);
			lba = lba + log(1-p_beta)*(md - getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums);
			//Rprintf("lba: %g\n",lba);
			lba = lba + getModelParameterRealization(realizationIndex,penaltyIndex)->entropy;
			//Rprintf("lba: %g\n",lba);
			//Rprintf("Entropy: %g\n",getModelParameterRealization(realizationIndex,penaltyIndex)->entropy);
			getModelParameterRealization(realizationIndex,penaltyIndex)->lb = lba;

			break;

		case LOGISTIC:
			////

			//lba = -0.5*(log(getModelParameterRealization(realizationIndex,penaltyIndex)->sigma_e)+1);
			innerProduct(model->data.n,model->data.y,getModelParameterRealization(realizationIndex,penaltyIndex)->pred_vec_new,&lba);
			for(t=0;t<model->data.n;t++){
				lba = lba + log(1-getModelParameterRealization(realizationIndex,penaltyIndex)->mu_vec[t]);
			}
			lba = lba + log(p_beta)*(getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums);
			lba = lba + log(1-p_beta)*(md - getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums);
			lba = lba + getModelParameterRealization(realizationIndex,penaltyIndex)->entropy;
			getModelParameterRealization(realizationIndex,penaltyIndex)->lb = lba;

			break;

	}



}

void run_vbsr(struct gaussianModelRealization * model){
	int i,j;
	double tol=1;
	double lb_old;
	int count = 0;
	//#pragma omp parallel for private(i,j,count,tol,lb_old)
	for (i=0;i < model->control_param.n_orderings;i++){
		for(j=0;j < model->control_param.path_length;j++){
			if(j>0){
				//copy the previous path to the new path
				copy_model_state(model,i,j);
				//Rprintf("Copied model state...\n");
			}
			while(fabs(tol) > model->control_param.eps && count < model->control_param.maxit){

				getModelParameterRealization(realizationIndex,penaltyIndex)->p_sums = 0;
				getModelParameterRealization(realizationIndex,penaltyIndex)->v_sums_correct = 0;
				getModelParameterRealization(realizationIndex,penaltyIndex)->entropy = 0;
				lb_old = getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
				//Rprintf("Updating beta...\n");
				update_beta(model,i,j);

				//if(model->control_param.max_pb==1){
				//	update_p_beta(model,i,j);
				//}
				//Rprintf("Updating error...\n");
				//Rprintf("entropy: %g\n",getModelParameterRealization(realizationIndex,penaltyIndex)->entropy);
				update_error(model,i,j);
				//Rprintf("Updating lower bound...\n");
				update_lb(model,i,j);
				tol = lb_old - getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
				count = count+1;

			}
			//Rprintf("lb: %g,i: %d, j: %d\n",lb_old,i,j);
			if(count>=model->control_param.maxit){
				Rprintf("Maximum iterations exceeded!\n");
			}
			count =0;
			tol = 1;
		}
	}

}

void identify_unique(double * lb_t, double * post_p, int n,double tol){
	int i,j,count;
	double tv;
	count =0;

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
		//if(post_p[i]>0){Rprintf("post_prob[%d]: %g\t",i,post_p[i]);}
	}
	//Rprintf("Identified: %d unique models\n",count);
}

void compute_bma_correct(struct gaussianModelRealization * model,int k,double * post_prob,double * s_bma,int j){
	int t,l;
	double corv;
	s_bma[0] = 0;

	//t ord ind
	//l ord ind
	//k marker ind
	//j path ind
	for (t=0;t<model->control_param.n_orderings;t++){
		if(post_prob[t] > 0){
			s_bma[0] = s_bma[0] + pow(post_prob[t],2);
		}
	}


	for(t=0;t<model->control_param.n_orderings-1;t++){
		for(l=t+1;l<model->control_param.n_orderings;l++){
			if(post_prob[t]>0 && post_prob[l]>0){
				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(model,t,j)->resid_vec,getModelParameterRealization(model,t,j)->expectationBeta[k]);
				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(model,l,j)->resid_vec,getModelParameterRealization(model,l,j)->expectationBeta[k]);
				cor(getModelParameterRealization(model,t,j)->resid_vec, getModelParameterRealization(model,l,j)->resid_vec, model->data.one_vec,&corv,model->data.n);
				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(model,t,j)->resid_vec,-getModelParameterRealization(model,t,j)->expectationBeta[k]);
				scaledVectorAddition(model->data.n,xc(model,k),getModelParameterRealization(model,l,j)->resid_vec,-getModelParameterRealization(model,l,j)->expectationBeta[k]);
				s_bma[0] = s_bma[0] + 2*post_prob[t]*post_prob[l]*(corv);
				//if(j==2 && k==0){Rprintf("correction: %g %g %g %g\n",s_bma[0],corv,post_prob[t],post_prob[l]);}
			}
		}
	}
	//Rprintf("correction: %g\n",s_bma[0]);




}


void collapse_results(struct gaussianModelRealization * model,
						double * beta_chi_mat,
						double * beta_mu_mat,
						double * beta_sigma_mat,
						double * e_beta_mat,
						double * beta_p_mat,
						double * lb_mat,
						double * kl_mat){

	int i,j,k;
	double max_v,bc,bm,bs,eb,bp,Z,s_bma;
	double * post_prob = (double *) malloc(sizeof(double)*model->control_param.n_orderings);
	double * lb_t = (double *) malloc(sizeof(double)*model->control_param.n_orderings);
	//max_v = -1e100;
	int w_max;
	//if(model->control_param.max_pb==1){
	//	for(i=0;i<model->control_param.n_orderings;i++){
	//		p_est[i]=getModelParameterRealization(model,i,0)->p_max;
	//	}
	//}

	switch(model->control_param.estType){


		case BMA:
			for(j=0;j<model->control_param.path_length;j++){
				max_v = getModelParameterRealization(model,0,j)->lb;
				w_max = 0;
				Z =0;
				for(i=0;i<model->control_param.n_orderings;i++){
					if(getModelParameterRealization(realizationIndex,penaltyIndex)->lb > max_v){
						max_v = getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
						w_max = i;
					}
					lb_mat[(model->control_param.n_orderings)*(j)+i] = getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
					lb_t[i] = getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
				}
				for(i=0;i<model->control_param.n_orderings;i++){
					Z = Z + exp(getModelParameterRealization(realizationIndex,penaltyIndex)->lb-max_v);
				}
				for(i=0;i<model->control_param.n_orderings;i++){
					post_prob[i] = exp(getModelParameterRealization(realizationIndex,penaltyIndex)->lb-max_v)/Z;
					//Rprintf("post_prob[%d]: %g\t",i,post_prob[i]);
					//lb_mat[(model->control_param.n_orderings)*(j)+i] = post_prob[i];
					//bm = bm + post_prob*getModelParameterRealization(realizationIndex,penaltyIndex)->betaChi
				}
				//Rprintf("\n");

				identify_unique(lb_t,post_prob,model->control_param.n_orderings,model->control_param.eps*10);

				for(k=0;k<model->data.m;k++){
					bc =0;
					bm =0;
					bs=0;
					eb=0;
					bp=0;
					switch(model->control_param.errType){
					case APPR:
						s_bma = 1;
						break;
					case EXACT:
						compute_bma_correct(model,k,post_prob,&s_bma,j);
						break;
					default:
						Rprintf("BMA computation not specified!\n");
						break;
					}

					for(i=0;i<model->control_param.n_orderings;i++){
						bc = bc+ post_prob[i]*getModelParameterRealization(realizationIndex,penaltyIndex)->betaChi[k];
						bm = bm+ post_prob[i]*getModelParameterRealization(realizationIndex,penaltyIndex)->betaMu[k];
						bs = bs+ post_prob[i]*getModelParameterRealization(realizationIndex,penaltyIndex)->betaSigmaSquared[k];
						eb = eb+ post_prob[i]*getModelParameterRealization(realizationIndex,penaltyIndex)->expectationBeta[k];
						bp = bp+ post_prob[i]*getModelParameterRealization(realizationIndex,penaltyIndex)->betaPosteriorProbability[k];
					}
					beta_chi_mat[(model->data.m)*(j)+k] = bc/sqrt(s_bma);
					beta_mu_mat[(model->data.m)*(j)+k] = bm;
					beta_sigma_mat[(model->data.m)*(j)+k] = bs;
					e_beta_mat[(model->data.m)*(j)+k] = eb;
					beta_p_mat[(model->data.m)*(j)+k] = bp;
				}
			}

			break;

		case MAXIMAL:
			////
			for(j=0;j<model->control_param.path_length;j++){
				max_v = getModelParameterRealization(model,0,j)->lb;
				w_max = 0;
				for(i=0;i<model->control_param.n_orderings;i++){
					if(getModelParameterRealization(realizationIndex,penaltyIndex)->lb > max_v){
						max_v = getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
						w_max = i;
					}
					lb_mat[(model->control_param.n_orderings)*(j)+i] = getModelParameterRealization(realizationIndex,penaltyIndex)->lb;
				}
				for(k=0;k<model->data.m;k++){
					beta_chi_mat[(model->data.m)*(j)+k] = getModelParameterRealization(model,w_max,j)->betaChi[k];
					beta_mu_mat[(model->data.m)*(j)+k] = getModelParameterRealization(model,w_max,j)->betaMu[k];
					beta_sigma_mat[(model->data.m)*(j)+k] = getModelParameterRealization(model,w_max,j)->betaSigmaSquared[k];
					e_beta_mat[(model->data.m)*(j)+k] = getModelParameterRealization(model,w_max,j)->expectationBeta[k];
					beta_p_mat[(model->data.m)*(j)+k] = getModelParameterRealization(model,w_max,j)->betaPosteriorProbability[k];
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
			int * nthreads){


	struct model_struct model;
	//omp_set_num_threads(*nthreads);
	//Rprintf("nthreads: %d, nthreads_o: %d\n",*nthreads,omp_get_max_threads());
	//Rprintf("Initializing model...\n");
	initialize_model(eps,l0_path,pb_path,exclude,penalty_factor,maxit,path_length,n_orderings,regress,scale,est,error,kl,approx,total_replicates,X, y, var_y, n, m,ordering_mat,&model);
	//Rprintf("Initialized model...\n");
	run_vbsr(&model);
	//Rprintf("Model run...\n");
	collapse_results(&model,beta_chi_mat, beta_mu_mat, beta_sigma_mat, e_beta_mat, beta_p_mat, lb_mat, kl_mat);
	//Rprintf("Results computed..\n");
	free_model(&model);

}
