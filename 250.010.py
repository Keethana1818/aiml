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


# In[21]:


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


# In[11]:


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

# In[23]:


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


# In[25]:


cars[cars.duplicated()]


# In[27]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[29]:


cars.corr()


# In[31]:


cars["HP"].corr(cars["VOL"])


# In[37]:


model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[39]:


model.summary()


# In[ ]:





# In[ ]:




