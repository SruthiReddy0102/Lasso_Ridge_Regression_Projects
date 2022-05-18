# Multilinear Regression with Regularization using L1 and L2 norm

# Analyzing the input and output variables
#Input Variables (x) = speed,hd,ram,screen,cd,multi,premium,ads,trend
#Output Variable(y) = price

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# Import Dataset

Computer_data = pd.read_csv("C:/Users/personal/Desktop/Computer_Data.csv")

# Removing Unnecessary Columns

Computer_data.columns = "S.No","price","speed","hd","ram","screen","cd","multi","premium","ads","trend"

Computer_data1 = Computer_data.drop(columns = "S.No")

Computer_data1

# Checking the NA values and Count of the variables

cat_Compudata1 = Computer_data1.select_dtypes(include = ['object']).copy()
cat_Compudata1
print(cat_Compudata1.isnull().values.sum()) 


print(cat_Compudata1['cd'].value_counts())
print(cat_Compudata1['multi'].value_counts())
print(cat_Compudata1['premium'].value_counts())

# # Creation of Dummy Variabels

cat_Compudata1_onehot = cat_Compudata1
cat_Compudata1_onehot = pd.get_dummies(cat_Compudata1_onehot, columns=['cd','multi','premium'], prefix = ['cd','multi','premium'])
print(cat_Compudata1_onehot.head())


#Concatenation of the Dummy variables to data sheet and drop of original columns

Compudata_df = pd.concat([Computer_data1, cat_Compudata1_onehot], axis=1)
Compudata_df
Compudata_df = Compudata_df.drop(['cd','multi','premium'], axis=1)
Compudata_df


# Correlation matrix 
a = Compudata_df.corr()
a

# There is a strong correlation between Profit - Ram,hd,speed

#EDA
a1 = Compudata_df.describe()
a1


#sctterplot and histogram between variables
sns.pairplot(Compudata_df) 

# As the are multiple input variable , creating object for input and output variables
y = Compudata_df.iloc[:,0]
x = Compudata_df.iloc[: , 1 :]

# preparing the model on train data 
model_train = smf.ols("y ~ x", data = Compudata_df).fit()
model_train.summary()

# R^2 = 0.776 and Adj R^2 = 0.775

# prediction
pred = model_train.predict(Compudata_df)
# Error
resid  = pred - Compudata_df.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
# The RMSE value = 275.1298188638723

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(Compudata_df.iloc[:, 1:], Compudata_df.price)
# Exculded state varaible as it is insignifcant

# coefficient values for all independent variables#
lasso.coef_
#[ 8.64148222, 0.65933879,49.771742,113.0984382,0.52827241,-47.24277634,-44.97699744,0.,-66.30232167,   0.        , 451.94935849,  -0.    
lasso.intercept_
103.23428688912645

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Compudata_df.columns[1:]))

lasso.alpha
# alpha = 0.13

pred_lasso = lasso.predict(Compudata_df.iloc[:, 1:])

# Adjusted r-square#
lasso.score(Compudata_df.iloc[:, 1:], Compudata_df.price)
# Score = 0.7715870508830523

#RMSE
np.sqrt(np.mean((pred_lasso - Compudata_df.price)**2))
# RMSE = 277.5589460664742

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(Compudata_df.iloc[:, 1:], Compudata_df.price)
#Excluded state variable as it is insignificant

#coefficients values for all the independent vairbales#
rm.coef_
# 5.56804526,0.4455751 ,37.9024994,95.24294358,0.56368791,-27.58877205,-33.77921713,33.77921713,-6.09027152,6.09027152,166.85514139,-166.85514139]
rm.intercept_
# 494.54436022603954

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Compudata_df.columns[1:]))

rm.alpha
# alpha = 0.4

pred_rm = rm.predict(Compudata_df.iloc[:, 1:])

# adjusted r-square#
rm.score(Compudata_df.iloc[:, 1:], Compudata_df.price)
# 0.6826367248455276

#RMSE
np.sqrt(np.mean((pred_rm - Compudata_df.price)**2))
# 327.16979647969447