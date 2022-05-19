#Input Variables (x) = R.D Spend, Administration, Marketing Spend, State
#Output Variable(y) = Profit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data and renaming the column names
Startup = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/50_Startups.csv")
Startup.columns = "RD","Admin","MS","State","Profit"
 
# Rearrange the order of the variables
Startup = Startup.iloc[:, [4, 0, 1, 2, 3]]
Startup.columns

# Correlation matrix 
a = Startup.corr()
a
# There is a strong correlation between Profit & RD as well as MS

#EDA
a1 = Startup.describe()
a1


# Checking of null values and creation of dummy variables

cat_Startup = Startup.select_dtypes(include = ['object']).copy()
cat_Startup.head()
print(cat_Startup.isnull().values.sum()) 

print(cat_Startup['State'].value_counts())

cat_Startup_onehot_sklearn = cat_Startup.copy()
cat_Startup_onehot_sklearn

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_Startup_onehot_sklearn['State'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

print(lb_results_df.head())

# concate the dummy variable to the data sheet

Startup_df = pd.concat([Startup, lb_results_df], axis=1)
Startup_df
Startup_df = Startup_df.drop(['State'], axis=1)
Startup_df

#sctterplot and histogram between variables
sns.pairplot(Startup) 

# preparing the model on train data 
model_train = smf.ols("Profit ~ RD+Admin+MS+State", data = Startup).fit()
model_train.summary()

# Over all P- value < 0.05 and R - squared value = 0.951 and Adjusted R Squared Value = 0.945
# Admin and state varibales show insignificant P- values

# prediction
pred = model_train.predict(Startup)
# Error
resid  = pred - Startup.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# The RMSE value = 8854.761029414496

# To overcome the issues, LASSO and RIDGE regression are used (L1 and L2 norm)
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.4, normalize = True)

lasso.fit(Startup.iloc[:, 1:4], Startup.Profit)
# Taken 3 varibales removing state variable as its P-value is insignificant

# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
# Intercept Value = 7.940125045817695

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Startup.columns[1:4]))

lasso.alpha
# Taken alpha value = 0.4 which acts as lambda value to find the R^2 value

pred_lasso = lasso.predict(Startup.iloc[:, 1:4])

# Adjusted r-square#
lasso.score(Startup.iloc[:, 1:4], Startup.Profit)
# Score = 0.9999999949751787

#RMSE
np.sqrt(np.mean((pred_lasso - Startup.Profit)**2))
# RMSE = 2.8284271247493598

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(Startup.iloc[:, 1:4], Startup.Profit)
# Taken 3 varibales removing state variable as its P-value is insignificant

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_
# Intercept Value = 34908.98110382576

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Startup.columns[1:4]))

rm.alpha
# Taken alpha value = 0.4 which acts as lambda value to find the R^2 value

pred_rm = rm.predict(Startup.iloc[:, 1:4])

# adjusted r-square#
rm.score(Startup.iloc[:, 1:4], Startup.Profit)
# Adj R^2 = 0.959604467464894

#RMSE
np.sqrt(np.mean((pred_rm - Startup.Profit)**2))
# 8019.574946771003

# AFter the regularization with Ridge the Rmse value has been reduce from 8854.76 to 8019.57 
# which indicate that the Ridge model is the best fit model