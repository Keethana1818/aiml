#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.formula.api as smf 
from statsmodels.graphics.regressionplots import influence_plot 
import numpy as np


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP", "WT","MPG"])
cars.head()


# Description of columns
# 
# • MPG: Milege of the car (Mile per Gallon) (This is Y-column to be predicted)
# 
# • HP : Horse Power of the car (X1 column)
# 
# • VOL: Volume of the car (size) (X2 column)
# 
# • SP: Top speed of the car (Miles per Hour) (X3 column)
# 
# • WT: Weight of the car (Pounds) (X4 Column)

# In[6]:


cars.info()


# In[7]:


cars.isna().sum()


# Multiple Linear Regression
# 
# Multilinear regression, commonly referred to as multiple linear regression, is a statistical technique that models the relationship between two or more explanatory varial and a response variable by fitting a linear equation to observed data. Essentially, it extends the simple linear regression model to incorporate multiple predictors, therel providing a way to evaluate how multiple factors impact the outcome.
# Assumptions in Multilinear Regression
# 1. Linearity: The relationship between the predictors(X) and the response (Y) is linear.
# 2. Independence: Observations are independent of each other.
# 3. Homoscedasticity: The residuals (Y - Y_hat) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other.
# Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.
# The general formula for multiole linear rearession is

# In[14]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns. boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot

# Creating a histogram in the same x-axis
sns. histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt. tight_layout()
plt.show()


# In[16]:


cars[cars.duplicated()]


# In[18]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[19]:


cars.corr()


# In[20]:


cars["HP"].corr(cars["VOL"])


# In[24]:


model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[26]:


model.summary()


# In[28]:


#Build model
import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()

model1.summary()


# In[30]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[32]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[34]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print("MSE :", mean_squared_error(df1["actual_y1"],df1["pred_y1"]))


# In[36]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[38]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE : ", mse)
print("RMSE :",np.sqrt(mse))


# In[46]:


rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt = 1/(1-rsq_wt)

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared
vif_vol = 1/(1-rsq_vol)

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared
vif_sp = 1/(1-rsq_sp)



# In[48]:


d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)
Vif_frame


# In[50]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[54]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model2.summary()


# In[56]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[58]:


pred_y2 = model1.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[60]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# In[62]:


cars1.shape


# In[66]:


#Define variables and assign values
k = 3  # no of x-columns in cars1
n = 81 # no of observations (rows)
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[68]:


#Define variables and assign values
k = 3  # no of x-columns in cars1
n = 81 # no of observations (rows)
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[70]:


cars1[cars1.index.isin([65,70,76,78,80])]


# In[72]:


#Discard the data points which are influencers and reasign the row number (reset_
cars2=cars1.drop(cars1.index[[65,70, 76,78,79,80]], axis=0).reset_index(drop=True)


# In[74]:


cars2


# In[76]:


model3= smf.ols('MPG~VOL+SP+HP',data = cars).fit()


# In[78]:


model3.summary()


# In[80]:


df3=pd.DataFrame()
df3["actual_y3"] = cars2["MPG"]
df3.head()


# In[82]:


pred_y3 = model3.predict(cars.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[84]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[ ]:




