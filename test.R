# Title     : TODO
# Objective : TODO
# Created by: jonlachmann
# Created on: 2021-03-31

{
  library(devtools)
build(vignettes = F)
install.packages("../fastglm_0.0.2.tar.gz")
}
detach("package:fastglm", unload=TRUE)
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
bench1 <- microbenchmark(glmfast <- fastglm(as.matrix(logistic_xx), as.matrix(logistic_yy), family=binomial(), method=0, tol=1e-1))

bench2 <- microbenchmark(glmfast <- fastglm(as.matrix(logistic_xx), as.matrix(logistic_yy), family=binomial(), method=1))

bench1
bench2

glmfast <- fastglm(as.matrix((logistic_xx)), as.matrix(logistic_yy), family=binomial(), method=0, tol=0.01, quant=0.15)

system.time(glmfast <- fastglm(as.matrix((logistic_xx)), as.matrix(logistic_yy), family=binomial(), method=0, tol=0.01))
system.time(glmfast <- fastglm(as.matrix((logistic_xx)), as.matrix(logistic_yy), family=binomial(), method=1))

mliks <- matrix(NA, 1023, 1)

for (i in 1:1023) {
  model <- as.logical(c(T,intToBits(i)[1:20]))
  glmod_full <- glm(Y ~ . - 1, logistic_data[,model], family = binomial())
  mliks_2[i,1] <- logLik(glmod_full)
  betas_2[i,model[1:11]] <- glmod_full$coefficients
}