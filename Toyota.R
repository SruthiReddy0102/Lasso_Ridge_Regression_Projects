#Analyzing the input and output variables:
#Input Variables (x) = Other Variables
#Output Variable(y) = Price

# Importing Dataset

Toyota <- read.csv(file.choose())
View(Toyota)
attach(Toyota)

# Removing Uncessary columns
Toyota <- Toyota[ , 2:38]
View(Toyota)

# Reorder the variables
Toyota <- Toyota[,c(2,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37)]
View(Toyota)


install.packages("glmnet")
library(glmnet)

x <- model.matrix(Price ~ ., data = Toyota)[,-1]
View(x)
y <- Toyota$Price
View(y)

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid , standardize = TRUE)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
optimumlambda

y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min
optimumlambda_1

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)
