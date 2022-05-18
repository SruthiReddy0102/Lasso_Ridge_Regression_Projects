#Analyzing the input and output variables:
#Input Variables (x) = speed, hd, ram, screen, cd, multi, premium, ads, trend
#Output Variable(y) = Sales Price

# Importing Dataset

Computer_Data <- read.csv(file.choose())
View(Computer_Data)
attach(Computer_Data)

# Removing Unwanted columns
Computer_Data <- Computer_Data[ , 2:11]
View(Computer_Data)


install.packages("glmnet")
library(glmnet)

x <- model.matrix(price ~ ., data = Computer_Data)[,-1]
View(x)
y <- Computer_Data$price
View(y)

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid)
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
