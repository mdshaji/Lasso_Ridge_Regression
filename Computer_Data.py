# Output Variable(y) = price
# Input Variables (x) = speed,hd,ram,screen,cd,multi,premium,ads,trend

# Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# Importing Dataset

Computer_data = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Computer_Data.csv")

# Removing Unnecessary Columns

Computer_data.columns = "S.No","price","speed","hd","ram","screen","cd","multi","premium","ads","trend"

Computer_data1 = Computer_data.drop(columns = "S.No")

Computer_data1

# Reodering of columns is not required as the output variables is at index "0"

# Checking the N/A values, Count of the variables and creating dummy variables

cat_Compudata1 = Computer_data1.select_dtypes(include = ['object']).copy()
cat_Compudata1
print(cat_Compudata1.isnull().values.sum()) 

print(cat_Compudata1['cd'].value_counts())
print(cat_Compudata1['multi'].value_counts())
print(cat_Compudata1['premium'].value_counts())

cat_Compudata1_onehot = cat_Compudata1
cat_Compudata1_onehot = pd.get_dummies(cat_Compudata1_onehot, columns=['cd','multi','premium'], prefix = ['cd','multi','premium'])
print(cat_Compudata1_onehot.head())


# Concatenation of the Dummy variables to data sheet and drop of original columns

Compudata_df = pd.concat([Computer_data1, cat_Compudata1_onehot], axis=1)
Compudata_df
Compudata_df = Compudata_df.drop(['cd','multi','premium'], axis=1)
Compudata_df


# Correlation matrix 
a = Compudata_df.corr()
a

# Strong correlation between Profit - Screen,Ram,hd,speed

# EDA
a1 = Compudata_df.describe()
a1


# sctterplot and histogram between variables
sns.pairplot(Compudata_df) 

# Creation of object for input and output variables

y = Compudata_df.iloc[:,0] # allocating the output variable(price)
y
x = Compudata_df.iloc[: , 1 :] # allocating the input variables
x

# preparing the model on train data 
model_train = smf.ols("y ~ x", data = Compudata_df).fit()
model_train.summary()

# R square value = 0.776 and Adjusted R square value = 0.775

# prediction
pred = model_train.predict(Compudata_df)
# Error
resid  = pred - Compudata_df.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
# The RMSE value = 275.13

# To overcome the issues, LASSO and RIDGE regression are used (L1 and L2 norm)
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.4, normalize = True)

lasso.fit(Compudata_df.iloc[:, 1:], Compudata_df.price)


# coefficient values for all independent variables#
lasso.coef_  
lasso.intercept_
# Intercept Value = 397.3587

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Compudata_df.columns[1:]))

lasso.alpha
# Taken alpha Value = 0.4 which acts as lambda value to find the R^2 value

pred_lasso = lasso.predict(Compudata_df.iloc[:, 1:])

# Adjusted r-square#
lasso.score(Compudata_df.iloc[:, 1:], Compudata_df.price)
# Adj R^2 = 0.7386

# RMSE
np.sqrt(np.mean((pred_lasso - Compudata_df.price)**2))
# RMSE = 296.90

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(Compudata_df.iloc[:, 1:], Compudata_df.price)


#coefficients values for all the independent vairbales
rm.coef_
rm.intercept_
# 494.54436022603954

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Compudata_df.columns[1:]))

rm.alpha
# Taken alpha value = 0.4 which acts as lambda value to find the R^2 value

pred_rm = rm.predict(Compudata_df.iloc[:, 1:])

# adjusted r-square
rm.score(Compudata_df.iloc[:, 1:], Compudata_df.price)
# Adj R^2 = 0.68

# RMSE
np.sqrt(np.mean((pred_rm - Compudata_df.price)**2))
# 327.16

# Rmse value is more for both Ridge and lasso regression 
# Rmse value is less in Multilinear model
# As a Conclusion after comparing 3 model, Multilinear seems to be the best fit model