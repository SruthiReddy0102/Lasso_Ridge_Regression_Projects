# Multilinear Regression with Regularization using L1 and L2 norm

# Analyzing the input and output variables
#Input Variables (x) = R.D Spend, Administration, Marketing Spend, State
#Output Variable(y) = Profit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
Startup = pd.read_csv("C:/Users/personal/Desktop/50_Startups.csv")
Startup.columns = "RD","Adm","MS","State","Profit"

# Rearrange the order of the variables
Startup = Startup.iloc[:, [4, 0, 1, 2, 3]]
Startup.columns

# Correlation matrix 
a = Startup.corr()
a

# There is a strong correlation between Profit & RD and Profit & MS

#EDA
a1 = Startup.describe()

# Checking of null values

cat_Startup = Startup.select_dtypes(include = ['object']).copy()
cat_Startup.head()
print(cat_Startup.isnull().values.sum()) 

print(cat_Startup['State'].value_counts())

# Creation of Dummy Variable

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
model_train = smf.ols("Profit ~ RD+Adm+MS+State", data = Startup).fit()
model_train.summary()

# R^2 = 0.951 and Adj R^2 = 0.945

# prediction
pred = model_train.predict(Startup)
# Error
resid  = pred - Startup.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# The RMSE value = 8854.761029414496

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(Startup.iloc[:, 1:4], Startup.Profit)
# Exculded state varaible as it is insignifcant

# coefficient values for all independent variables#
lasso.coef_
#  0.80575097, -0.0268011 ,  0.02721082
lasso.intercept_
# 50121.38085553272

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Startup.columns[1:4]))

lasso.alpha

pred_lasso = lasso.predict(Startup.iloc[:, 1:4])

# Adjusted r-square#
lasso.score(Startup.iloc[:, 1:4], Startup.Profit)
# Score = 0.9507459924109123

#RMSE
np.sqrt(np.mean((pred_lasso - Startup.Profit)**2))
# RMSE = 8855.344638007662

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.13, normalize = True)

rm.fit(Startup.iloc[:, 1:4], Startup.Profit)
#Excluded state variable as it is insignificant

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_
# 46550.71446119002

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Startup.columns[1:4]))

rm.alpha

pred_rm = rm.predict(Startup.iloc[:, 1:4])

# adjusted r-square#
rm.score(Startup.iloc[:, 1:4], Startup.Profit)
# 0.9359761063647808

#RMSE
np.sqrt(np.mean((pred_rm - Startup.Profit)**2))
# 10096.14833524171
