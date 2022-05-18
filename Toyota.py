#Multilinear Regression with Regularization using L1 and L2 norm

#Analyzing the input and output variables
#Input Variables (x) = Other Variables
#Output Variable(y)  = Price

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# Importing Dataset
Toyota = pd.read_csv("C:/Users/personal/Desktop/ToyotaCorolla.csv", encoding= 'unicode_escape')
Toyota

# Removing of unnecessary columns
Toyota1 = Toyota.drop(columns = "Id")
Toyota1

# Rearrange the order of the variables
Toyota = Toyota1.iloc[:, [1, 0, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]]
Toyota.columns


# Checking the NA values and Count of the variables
cat_Toyota1 = Toyota.select_dtypes(include = ['object']).copy()
cat_Toyota1
print(cat_Toyota1.isnull().values.sum()) 

print(cat_Toyota1['Model'].value_counts())
print(cat_Toyota1['Fuel_Type'].value_counts())
print(cat_Toyota1['Color'].value_counts())

 # Creation of Dummy Variabels
cat_Toyota1_onehot_sklearn = cat_Toyota1.copy()
cat_Toyota1_onehot_sklearn

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb_results1 = lb.fit_transform(cat_Toyota1_onehot_sklearn['Fuel_Type'])
lb_results1_df = pd.DataFrame(lb_results1, columns=lb.classes_)

print(lb_results1_df.head())

lb_results2 = lb.fit_transform(cat_Toyota1_onehot_sklearn['Color'])
lb_results2_df = pd.DataFrame(lb_results2, columns=lb.classes_)

print(lb_results2_df.head())

# concate the dummy variable to the data sheet

Toyota1_df = pd.concat([Toyota1,lb_results1_df,lb_results2_df], axis=1)
Toyota1_df
Toyota1_df = Toyota1_df.drop(['Model','Fuel_Type','Color'], axis=1)
Toyota1_df

# Correlation matrix 
a = Toyota1_df.corr()
a
# There is a strong correlation between Price - Mfg year,Cylinders,Automatic_airco ,Weight,

#EDA
a1 = Toyota1_df.describe()
a1

#sctterplot and histogram between variables
sns.pairplot(Toyota1_df) 

y = Toyota1_df.iloc[:,0]
y
x = Toyota1_df.iloc[: , 1 :]
x

# preparing the model on train data 
model_train= smf.ols('Toyota1_df.iloc[:,0] ~ Toyota1_df.iloc[: , 1 :]', data = Toyota1_df).fit()
model_train.summary()

# R^2 = 0.911 and Adj R^2 = 0.908

# prediction
pred = model_train.predict(Toyota1_df)
# Error
resid  = pred - Toyota1_df.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# The RMSE value =  1083.0089613788814

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.15, normalize = True)

lasso.fit(Toyota1_df.iloc[:, 1:], Toyota1_df.Price)


# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_
# -1403593.8172871785

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Toyota1_df.columns[1:]))

lasso.alpha
# alpha = 0.15

pred_lasso = lasso.predict(Toyota1_df.iloc[:, 1:])

# Adjusted r-square#
lasso.score(Toyota1_df.iloc[:, 1:], Toyota1_df.Price)
# Score = 0.9105484833231006

#RMSE
np.sqrt(np.mean((pred_lasso - Toyota1_df.Price)**2))
# RMSE = 1084.3909884746556

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 0.13, normalize = True)

rm.fit(Toyota1_df.iloc[:, 1:], Toyota1_df.Price)


#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_
# -1232760.68666374

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Toyota1_df.columns[1:]))

rm.alpha
# Alpha = 0.13
pred_rm = rm.predict(Toyota1_df.iloc[:, 1:])

# adjusted r-square#
rm.score(Toyota1_df.iloc[:, 1:], Toyota1_df.Price)
# 0.9069402051407645

#RMSE
np.sqrt(np.mean((pred_rm - Toyota1_df.Price)**2))
# 1106.0457491681952