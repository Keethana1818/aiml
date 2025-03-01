#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[4]:


iris = pd.read_csv("iris.csv")
iris


# In[6]:


iris.info()


# In[8]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[10]:


iris[iris.duplicated(keep=False)]


# #### Observations
# .There are 150 rows and 5 columns
# 
# .There are no null values
# 
# .There is one duplicated row
# 
# .The X-columns are sepal.length,sepal.width,petal.length and petal.width
# 
# .All the X-columns are continuous
# 
# .The Y - columns is "variety" which is categorical
# 
# .There are three flower categories

# In[16]:


iris = iris.reset_index(drop=True)
iris


# In[ ]:




