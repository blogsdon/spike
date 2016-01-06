set.seed(2)
n <- 100
m <- 95
ntrue <- 10
e <- rnorm(n)
X <- matrix(rnorm(n*m),n,m)
tbeta <- sample(1:m,ntrue)
beta <- rep(0,m)
beta[tbeta]<- rnorm(ntrue,0,2)
y <- X%*%beta+e
set.seed(1)
resC<- spike::vbsr(y,scale(X),family='normal',n_orderings=1,eps = 1e-8,scaling = F)
resR <- spike::vbsrR(y=y,x=scale(X),eps=1e-8,l0 = resC$l0)
resOld <- vbsr::vbsr(y,scale(X),family='normal',n_orderings=1,eps=1e-8,scaling=F)
resC$e_beta[-1][1:4]

resR$ebeta[1:4]
resOld$beta[1:4]
