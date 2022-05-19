# Input Variables (x) = R.D Spend , Administration , Marketing Spend , State
# Output Variable(y) = Profit

# Importing the dataset
Startup <- read.csv(file.choose())
colnames(Startup) <- c("RD","Admin","MS","State","Profit") # RD = R.D.Spend , Admin = Administrartion & MS = Marketing Spend
View(Startup)
attach(Startup)


# Reorder the variables
Startup <- Startup[,c(5,1,2,3,4)]
View(Startup)

install.packages("glmnet")
library(glmnet)

# Seperation of Input & OutPut Variable using 

x <- model.matrix(Profit ~ ., data = Startup)[,-1]
View(x)
y <- Startup$Profit
View(y)



# Creating grid values for Lambda to minimise the error by using Hyper parameter tuning

grid <- 10^seq(10, -2, length = 100)
grid

# Ridge Regression
model_ridge <- glmnet(x, y, alpha = 0, lambda = grid)
summary(model_ridge)

# Performing K Fold Cross Validation when we have imbalance data set

cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = grid, standardize = TRUE)
plot(cv_fit)
optimumlambda <- cv_fit$lambda.min
cv_fit$lambda.1se

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
cv_fit_1$lambda.1se

y_a <- predict(model_lasso, s = optimumlambda_1, newx = x)

sse <- sum((y_a-y)^2)

sst <- sum((y - mean(y))^2)
rsquared <- 1-sse/sst
rsquared

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

