vbsrR <- function(y,
                  x,
                  z=NULL,
                  l0 = -1.545509,
                  eps = 1e-4,
                  seed=1){
  ###y = outcomes
  ###x = design matrix
  ###l0 = logit transfomred prior probability of being non-zero
  ###model <- list
  
  #seed default random seed to control behavior
  set.seed(seed)
  
  #n: number of samples
  n <- nrow(x)
  
  #m: number of features (to be selected on)
  m <- ncol(x)
  
  #p: number of fixed covariates
  
  #function to intialize the covariate matrix if it has not bee initialized
  initializeFixedCovariates <- function(z,n){
    if(is.null(z)){
      z <- as.matrix(rep(1,n))
    }
    return(z)
  }
  
  z <- initializeFixedCovariates(z,n)
  
  p <- ncol(z)
  
  #function to initialize the starting condition
  initializeModelState <- function(n,m,p,l0,eps,y,x,z){
    modelState <- list()
    
    #variable to track the state of expected beta
    modelState$ebeta <- rep(0,m)
    
    #variable to track the state of expectation of beta squared
    modelState$ebetasq <- rep(0,m)
    
    #variable to track the state of betamu
    modelState$betamu <- rep(0,m)
    
    #variable to track the state of betasigma
    modelState$betasigma <- rep(0,m)
    
    #variable to track the state of pbeta
    modelState$pbeta <- rep(0,m)
    
    #variable to track the sum of probabilities
    modelState$psums <- 0
    
    #variable to track vsums
    modelState$vsums <- 0
    
    modelState$vsums_correct <- 0
    
    #variable to track entropy
    modelState$entropy <- 0
    
    #variable to track the state of the fixed effect estimates
    modelState$alpha <- rep(0,p)
    
    #variable to strack the state of the error variance parameter
    modelState$sigma <- 1
    
    #variable to track the iteration of the gibbs sampler
    modelState$iteration <- 1
    
    #prior probability of being non-zero
    modelState$l0 <- l0
    
    #lower bound on marginal log likelihood
    modelState$lowerBound <- 0
    
    #residual vector
    modelState$residuals <- y
    
    #sum of squares
    modelState$xSumSquares <- apply(x^2,2,sum)
    
    #y - outcome vector
    modelState$y <- y
    
    #x - design matrix
    modelState$x <- x
    
    #z - covariate matrix
    modelState$z <- z
    
    #number of observations
    modelState$n <- n
    
    #nubmer of penalized variables
    modelState$m <- m
    
    #number of unpenalized variables
    modelState$p <- p

    #zhat matrix
    modelState$Zhat <- solve(t(z)%*%z)%*%t(z)
    
    #tolerance to assess convergence to a local maximum
    modelState$eps <- eps
    
    #need to transform l0 into pbeta
    modelState$pbetaParam <- 1/(1+exp(-0.5*modelState$l0))

    #return model state    
    return(modelState)
  }
  
  #function to update the beta parameters
  updateBeta <- function(modelState){
    for (j in 1:modelState$m){
      #muj - mean estimate
      
      muj <- t(modelState$x[,j])%*%modelState$residuals
      
      muj <- muj + modelState$ebeta[j]*modelState$xSumSquares[j]
      
      muj <- muj/modelState$xSumSquares[j]
      
      #sigmaj
      sigmaj <- modelState$sigma/modelState$xSumSquares[j]
      
      chi <- muj^2/sigmaj
      
      #pj
      
      pj <- 1/(1+exp(-0.5*(chi+modelState$l0+log(sigmaj))))
      
      #betaj
      #betaj <- rSpikeSlab(muj,sigmaj,pj)
      ebetaj <- pj*muj
      ebetajsq <- pj*(muj^2+sigmaj)

      #update sums of probabilties
      modelState$psums <- modelState$psums + pj
      
      #entropy
      if(pj > 1e-10){
        modelState$entropy <- modelState$entropy - pj*log(pj) + (1-pj) + 0.5*pj*log(2*exp(1)*pi*sigmaj)
      }else if (pj < 1e-10){
        modelState$entropy <- modelState$entropy + pj - (1-pj)*log(1-pj) + 0.5*pj*log(2*exp(1)*pi*sigmaj)
      }else {
        modelState$entropy <- modelState$entropy - pj*log(pj) - (1-pj)*log(1-pj) + 0.5*pj*log(2*exp(1)*pi*sigmaj)
      }
      
      #correction factor for vsums
      modelState$vsums_correct <- modelState$vsums_correct + (ebetaj^2-ebetajsq)*modelState$xSumSquares[j]
  
      #update residuals
      modelState$residuals <- modelState$residuals + modelState$x[,j]*(modelState$ebeta[j]-ebetaj)
      
      #set beta paramters
      modelState$ebeta[j] <- ebetaj
      modelState$ebetasq[j] <- ebetajsq
      modelState$betamu[j] <- muj
      modelState$betasigma[j] <- sigmaj
      modelState$pbeta[j] <- pj
      
    }
    
    #return the current model state
    return(modelState)
  }
  
  #function to update the alpha parameters
  updateAlpha <- function(modelState){
    #get new fixed parameter estimates
    alpha <- modelState$Zhat%*%(modelState$residuals+modelState$z%*%modelState$alpha)
    
    #update residuals
    modelState$residuals <- modelState$residuals + modelState$z%*%(modelState$alpha-alpha)
    
    #set model alpha to new alpha
    modelState$alpha <- alpha
    
    return(modelState)
  }
  
  #function to update the error parameters
  updateError <- function(modelState){
    #generate estimate of error variance
    #sigma <- sum(modelState$residuals^2)/modelState$n
    sigma <- t(modelState$residuals)%*%modelState$residuals
    sigma <- sigma - modelState$vsums_correct
    sigma <- sigma/modelState$n
    
    
    #set current state to new estimate
    modelState$sigma <- sigma
    
    return(modelState)
  }
  
  #function to update the likelihood parameters
  updateLowerBound <- function(modelState){
    
    lowerBound <- -0.5*modelState$n*(log(2*pi*modelState$sigma)+1)
    lowerBound <- lowerBound + log(modelState$pbetaParam)*modelState$psums
    lowerBound <- lowerBound + log(1-modelState$pbetaParam)*(modelState$m - modelState$psums)
    lowerBound <- lowerBound + modelState$entropy
    
    modelState$lowerBound <- lowerBound
    
    return(modelState)
  }
  
  
  modelState <- initializeModelState(n,m,p,l0,eps,y,x,z)
  lbold <- -Inf
  while(abs(modelState$lowerBound-lbold) > modelState$eps ){
    
    modelState$psums <- 0
    modelState$vsums_correct <- 0
    modelState$entropy <- 0
    lbold <- modelState$lowerBound
    #update coefficient distribution
    modelState <- updateBeta(modelState)
    
    #update fixed covariate effect distribution
    modelState <- updateAlpha(modelState)
    
    #update error variance estimate distribution
    modelState <- updateError(modelState)
    
    #update compelte log likliehod distribution
    modelState <- updateLowerBound(modelState)
    
    #update the iteration
    modelState$iteration <- modelState$iteration + 1
    
    if(modelState$iteration%%10==0){
      cat('iteration:',modelState$iteration,'\n')
    }
  }
  
  return(modelState)
}