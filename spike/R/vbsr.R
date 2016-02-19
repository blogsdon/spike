#getting rid of marginal analysis
#getting rid of kl statistics
#getting rid of screening
#add s3 methods to function
#clean solution ->
#fix scaling -> maybe in c

vbsr = function(responseVariable,
		penalizedDataMatrix,
		unpenalizedDataMatrix=NULL,
		ordering_mat=NULL,
		epsilon=1e-6,
		maximumIterations = 1e4,
		numberOfRealizations = 10,
    family = "gaussian",
		scaling = TRUE,
		modelSpaceSelectionType = "BMA",
		applyBmaCovarianceCorrection = TRUE,
		posteriorProbabilityBonferroniMapping=0.95,
		l0Vector=NULL,
    cleanSolution=FALSE){

	numberSamples <- nrow(penalizedDataMatrix)
	numberPenalizedFeatures <- ncol(penalizedDataMatrix)
	numberUnpenalizedFeatures <- ncol(unpenalizedDataMatrix)
	#add intercept parameter
	if(is.null(unpenalizedDataMatrix)){
		unpenalizedDataMatrix <- as.matrix(rep(1,numberSamples))
	}

	if(!is.null(posteriorProbabilityBonferroniMapping)){
		l0VectorLength=1
		l0Vector=-(qchisq(0.05/numberUnpenalizedFeatures,1,lower.tail=FALSE)-log(numberSamples)+2*log((1-posteriorProbabilityBonferroniMapping)/(posteriorProbabilityBonferroniMapping)))
	}else{
    l0VectorLength=length(l0Vector)
    if(l0VectorLength==0){
      stop("invalid penalty parameter path specification")
    }
	}

	if(scaling==TRUE){
		scale <- 1
	} else if (scaling==FALSE){
		scale <- 0
	} else {
		stop("Improper design matrix scaling parameter provided.")
	}


	if(modelSpaceSelectionType=="BMA"){
		est <- 1
	} else if (modelSpaceSelectionType =="MAXIMAL"){
		est <- 0
	} else {
		stop("Improper global estimation type.  Must be either 'BMA' or 'MAXIMAL'.")
	}


	if(applyBmaCovarianceCorrection==TRUE){
		approx <- 1
	} else if (applyBmaCovarianceCorrection==FALSE){
		approx <- 0
	} else {
		stop("Improper Bayesian model averaging z-score approximate estimation indicator.")
	}

	#define results
	totalModelFits = l0VectorLength*numberOfRealizations
	responseVariance = var(y)
	betaMuResult = double(numberPenalizedFeatures*l0VectorLength)
	betaSigmaSquaredResult = double(numberPenalizedFeatures*l0VectorLength)
	expectationBetaResult = double(numberPenalizedFeatures*l0VectorLength)
	posteriorProbabilityBetaResult = double(numberPenalizedFeatures*l0VectorLength)
	lowerBoundResult = double(numberOfRealizations*l0VectorLength)
	priorProbabilityVector = 1/(1+exp(-0.5*l0Vector))

	#compute sma p-values if pre-screening:

	#build the realizationMatrix

	result <- c()
	while(length(result)==0){
		try(result<-.C("gaussianVariationalBayesSpikeRegression",
			as.double(epsilon),
			as.double(l0Vector),
			as.double(priorProbabilityVector),
			as.integer(maximumIterations),
			as.integer(l0VectorLength),
			as.integer(numberOfRealizations),
			as.integer(totalModelFits),
			as.double(penalizedDataMatrix),
			as.double(unpenalizedDataMatrix),
			as.double(responseVariable),
			as.double(responseVariance),
			as.integer(numberSamples),
			as.integer(numberUnpenalizedFeatures),
			as.integer(numberPenalizedFeatures),
			as.integer(realizationMatrix),
			as.double(betaMuResult),
			as.double(betaSigmaSquaredResult),
			as.double(expectationBetaResult),
			as.double(posteriorProbabilityBetaResult),
			as.double(lowerBoundResult),
			as.double(sigmaSquaredErrorResult),
			as.double(alphaResult),
			PACKAGE="spike"),silent=TRUE)
		if(length(result)==0&&l0VectorLength>1){
			#rm(result)
			#gc()
			#result <- c()
			l0VectorLength <- l0VectorLength-1
			l0Vector <- l0Vector[-1]
			priorProbabilityVector <- priorProbabilityVector[-1]
			betaMuResult = double(numberPenalizedFeatures*l0VectorLength)
			betaSigmaSquaredResult = double(numberPenalizedFeatures*l0VectorLength)
			expectationBetaResult = double(numberPenalizedFeatures*l0VectorLength)
			posteriorProbabilityBetaResult = double(numberPenalizedFeatures*l0VectorLength)
			lowerBoundResult = double(numberOfRealizations*l0VectorLength)
		} else if (length(result)==0&&l0VectorLength<=1){
			stop("solution does not exist for any of path specified")
		}
	}


#get appropriate multiplier
# 	if(scale==1&&add.intercept==TRUE){
# 		mult <- c(1,apply(penalizedDataMatrix[,-1],2,sd)*sqrt((n-1)/numberSamples))
# 	} else if (scale==1){
# 		mult <- apply(penalizedDataMatrix[,-1],2,sd)*sqrt((n-1)/numberSamples)
# 	}else{
# 		mult <- rep(1,m)
# 	}
	  mult <- rep(1,m)

    return(collectedResults)
	}
}
