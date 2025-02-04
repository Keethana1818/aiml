#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.formula.api as smf 
from statsmodels.graphics.regressionplots import influence_plot 
import numpy as np


# In[10]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[12]:


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

# In[19]:


cars.info()


# In[21]:


cars.isna().sum()


# In[ ]:




