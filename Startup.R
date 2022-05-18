# Analyzing the input and output variables
#Input Variables (x) = R.D Spend, Administration, Marketing Spend, State
#Output Variable(y) = Profit

# Importing the Dataset
Startup <- read.csv(file.choose())
colnames(Startup) <- c("RD","Adm","MS","State","Profit")
View(Startup)
attach(Startup)

# Reorder the variables
Startup <- Startup[,c(5,1,2,3,4)]
View(Startup)

install.packages("glmnet")
library(glmnet)

# Seperation of Input and Output variabes along with Dummy variable creation

x <- model.matrix(Profit ~ ., data = Startup)[,-1] # Dummy Variables creation is taken care by model.matrix command
y <- Startup$Profit


# Creating the grid values for lamda(Tuning Hyper-parameter) to minimize the error value

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

# Asthere is no much information from the model summary,we need to do cross validation or K-fold cross validation
# where k values can be any,as the K value increases the best will be the R^2 increased

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid ,standardize = TRUE)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
optimumlambda
cv_fit$lambda.1se


y_a <- predict(model_ridge, s = optimumlambda, newx = x)
sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared #  0.9507525

predict(model_ridge, s = optimumlambda, type="coefficients", newx = x)


# Lasso Regression
model_lasso <- glmnet(x, y, alpha = 1, lambda = grid)
summary(model_lasso)

cv_fit_1 <- cv.glmnet(x, y, alpha = 1, lambda = grid , standardize = TRUE)
plot(cv_fit_1)
optimumlambda_1 <- cv_fit_1$lambda.min
optimumlambda_1

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared #0.9498183

predict(model_lasso, s = optimumlambda, type="coefficients", newx = x)


#Elastic-net Regression

model_elastic <- glmnet(x, y, alpha = 0.5, lambda = grid)
summary(model_elastic)

cv_fit_2 <- cv.glmnet(x, y, alpha = 0.5, lambda = grid , standardize = TRUE)
plot(cv_fit_2)
optimumlambda_2 <- cv_fit_2$lambda.min
optimumlambda_2

y_a <- predict(model_elastic, s = optimumlambda_2, newx = x)

sse <- sum((y_a-y)^2)
sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared # 0.9495768

predict(model_elastic, s = optimumlambda, type="coefficients", newx = x)
