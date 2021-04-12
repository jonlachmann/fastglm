# Title     : TODO
# Objective : TODO
# Created by: jonlachmann
# Created on: 2021-03-31

Rcpp::compileAttributes()


{
  library(devtools)
  build(vignettes = F)
  install.packages("../fastglm_0.0.2.tar.gz")
}
# detach("package:fastglm", unload=TRUE)
{
  library(fastglm)

library(microbenchmark)
library(RCurl)
### Download simulated logistic data as per example 2
logistic_x <- read.csv(header=F, text=getURL("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/Mode%20Jumping%20MCMC/supplementary/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-X.txt"))
logistic_y <- read.csv(header=F, text=getURL("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/Mode%20Jumping%20MCMC/supplementary/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-Y.txt"))
colnames(logistic_y) <- "Y"
# Modify data
logistic_x$V2<-(logistic_x$V10+logistic_x$V14)*logistic_x$V9
logistic_x$V5<-(logistic_x$V11+logistic_x$V15)*logistic_x$V12
logistic_data <- cbind(logistic_y, logistic_x)

logistic_xx <- rbind(logistic_x, logistic_x, logistic_x, logistic_x, logistic_x, logistic_x)
logistic_xx <- rbind(logistic_xx, logistic_xx)
logistic_yy <- rbind(logistic_y, logistic_y, logistic_y, logistic_y, logistic_y, logistic_y)
logistic_yy <- rbind(logistic_yy, logistic_yy)
}
# bench1 <- microbenchmark(glmfast <- fastglm(as.matrix(logistic_xx), as.matrix(logistic_yy), family=binomial(), method=0, tol=1e-1, quant=0.2))

#bench2 <- microbenchmark(glmfast <- fastglm(as.matrix(logistic_xx), as.matrix(logistic_yy), family=binomial(), method=1))

#fastglm(as.matrix(logistic_xx), as.matrix(logistic_yy), family=binomial(), method=0, tol=1e-1, quant=.4)$deviance

#bench1
#bench2

#system.time(glmfast <- fastglm(as.matrix((logistic_xx)), as.matrix(logistic_yy), family=binomial(), method=0, tol=0.01, quant=0.2))

#system.time(glmfast <- fastglm(as.matrix((logistic_xx)), as.matrix(logistic_yy), family=binomial(), method=0, tol=0.01))
#system.time(glmfast <- fastglm(as.matrix((logistic_xx)), as.matrix(logistic_yy), family=binomial(), method=1))

mod_count <- 500

mliks <- matrix(NA, mod_count, 2)
for (i in 1:mod_count) {
  model <- as.logical(c(T,intToBits(i*100)[1:20]))
  def_mod <- fastglm(as.matrix(logistic_x[,model]), as.matrix(logistic_y), family=binomial(), method=0, debug=F)
  sub_mod <- fastglm(as.matrix(logistic_x[,model]), as.matrix(logistic_y), family=binomial(), method=0, quant=0.25, maxit=12, maxit_s=15, debug=T)
  mliks[i,1] <- -def_mod$deviance/2
  mliks[i,2] <- -sub_mod$deviance/2
  print(i)
}

plot(mliks[,2], type="l", col="red", ylim=c(-2000, -700))
lines(mliks[,1])

model <- as.logical(c(T,intToBits(338*100)[1:20]))
  def_mod <- fastglm(as.matrix(logistic_x[,model]), as.matrix(logistic_y), family=binomial(), method=0, debug=F)
  sub_mod <- fastglm(as.matrix(logistic_x[,model]), as.matrix(logistic_y), family=binomial(), method=0, quant=0.5, maxit=10, maxit_s=5, debug=F)

which.max(mliks[,1] - mliks[,2])

#library(bigmemory)

million <- matrix(rnorm(10^6*15), 10^6)

milliondf <- as.data.frame(million)
milliondf$y <- milliondf$V1*5+milliondf$V7*3+milliondf$V14*6
milliondf$y2 <- as.integer(milliondf$V1*5+milliondf$V7*3+milliondf$V14*6>mean(milliondf$V1*5+milliondf$V7*3+milliondf$V14*6))

system.time(mill_sub <- fastglm(as.matrix(milliondf[,1:15]), as.matrix(milliondf[,17]), family=binomial(), quant=0.001, maxit=15, maxit_s=5, debug=T))
system.time(mill_full <- fastglm(as.matrix(milliondf[,1:15]), as.matrix(milliondf[,17]), family=binomial(), debug=T))

millx <- as.matrix(milliondf[,1:15])
milly <- as.matrix(milliondf[,17])

mill_sub <- fastglm(millx, milly, family=binomial(), quant=0.001, maxit=15, maxit_s=5)

mill_sub$deviance
mill_full$deviance

system.time(mill_sub_g <- fastglm(as.matrix(milliondf[,c(1,2,14)]), as.matrix(milliondf[,16]), family=gaussian(), quant=0.001, maxit=5, tol=0))
system.time(mill_full_g <- fastglm(as.matrix(milliondf[,c(1,2,14)]), as.matrix(milliondf[,16]), family=gaussian()))

hundredK <- big.matrix(rnorm(10^5*10^5), 10^5)