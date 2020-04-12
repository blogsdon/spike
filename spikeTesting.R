set.seed(2)
n <- 391
m <- 500
ntrue <- 30
e <- rnorm(n)
X <- matrix(rnorm(n*m),n,m)
colnames(X) <- paste0('var',1:m)
tbeta <- sample(1:m,ntrue)
beta <- rep(0,m)
beta[tbeta]<- rnorm(ntrue,0,2)
y <- X%*%beta+e
set.seed(1)
X<- scale(X)
resC<- spike::vbsr(y,X,family='normal',n_orderings=1,eps = 1e-8,scaling = F)
resR <- spike::vbsrR(y=y,x=X,eps=1e-8,l0 = resC$l0)

pb1 <- sqrt(exp(resR$l0))/(sqrt(exp(resR$l0)) + sqrt(2*pi))
par <- c(resR$sigma,resR$alpha,resR$betamu,resR$pbeta,resR$betasigma)
z <- matrix(rep(1,n),n,1)

spike::ell(par,y,X,z,pb1)


# profvis::profvis({resbootstrap <- spike::vbsrBootstrap(y=y,x=X,nsamp = 50,cores = 8)})
# system.time(resbootstrap <- spike::vbsrBootstrap(y=y,x=X,nsamp = 50,cores = 8))
# system.time(resbootstrap <- spike::vbsrBootstrap(y=y,x=X,nsamp = 50,cores = 2))
# profvis::profvis({resR <- spike::vbsrR(y=y,x=scale(X),eps=1e-8,l0 = resC$l0)})
# system.time(resOld <- vbsr::vbsr(y,scale(X),family='normal',n_orderings=1,eps=1e-8,scaling=F))

#resgpu <- profvis::profvis({spike::vbsrRgpu(y,scale(X),eps=1e-8,l0=resC$l0,gpuContex=1)})

resC$e_beta[-1][1:4]

resR$ebeta[1:4]
resOld$beta[1:4]
