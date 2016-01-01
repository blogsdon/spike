set.seed(2)
n <- 100
m <- 10
ntrue <- 5
e <- rnorm(n)
X <- matrix(rnorm(n*m),n,m)
tbeta <- sample(1:m,ntrue)
beta <- rep(0,m)
beta[tbeta]<- rnorm(ntrue,0,2)
y <- X%*%beta+e
set.seed(1)
resC<- spike::vbsr(y,X,family='normal',n_orderings=1,eps = 1e-8)
resR <- spike::vbsrR(y=y,x=X,eps=1e-8)
resC$e_beta[-1][1:5]
resR$ebeta[1:5]
