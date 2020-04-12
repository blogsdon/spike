ell <- function(par,y,x,z,pb){
  #parameter order:
  #sigmae - par[1]
  #alphak - par[2:(p+1)]
  #muj - par[(p+2):(p+1+m)]
  #pj - par[(p+m+2):(p+1+2*m)]
  #sigmaj - par[(p+2*m+2):(p+1+3*m)]
  n <- length(y)
  m <- ncol(x)
  p <- ncol(z)
  sigma2e <- par[1]
  alphak <- par[2:(p+1)]
  muj <- par[(p+2):(p+1+m)]
  pj <- par[(p+m+2):(p+1+2*m)]
  pj[pj>(1-1e-16)] <- 1-1e-16
  pj[pj<(1e-16)] <- 1e-16
  
  sigma2j <- par[(p+2*m+2):(p+1+3*m)]
  fixed_hat <- z%*%alphak
  spike_hat <- x%*%(muj*pj)
  spike_neg_correction_hat <- (x^2)%*%((muj^2)*(pj^2))
  spike_pos_correction_hat <- (x^2)%*%(pj*((muj^2) + sigma2j))
  ones <- rep(1,n)
  U <- crossprod(y) - 2*crossprod(y,fixed_hat) - 2*crossprod(y,spike_hat) + 2*crossprod(fixed_hat,spike_hat) + crossprod(spike_hat) - sum(spike_neg_correction_hat) + sum(spike_pos_correction_hat)
  
  res <- -(n/2)*log(2*pi*sigma2e) - (U/(2*sigma2e)) + sum(pj*log(pb) + (pj/2)*log(2*exp(1)*pi*sigma2j) - pj*log(pj) + (1-pj)*log(1-pb) - (1-pj)*log(1-pj))
  return(res)
}