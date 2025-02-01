#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Load the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[51]:


data1 = pd.read_csv("NewspaperData.csv")
print(data1)


# In[53]:


data1.isnull().sum()


# In[55]:


data1


# In[57]:


data1.isnull().sum()


# In[59]:


data1.describe()


# In[71]:


data1.info()


# In[69]:


plt.figure (figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[74]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[76]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[78]:


data1["daily"].corr(data1["sunday"])


# In[84]:


data1[["daily","sunday"]].corr()


# In[86]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[88]:


model1.summary()


# In[ ]:




