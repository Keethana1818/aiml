#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[6]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[8]:


Univ1 = Univ.iloc[:,1:]


# In[10]:


Univ1


# In[12]:


cols = Univ1.columns


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols )
scaled_Univ_df


# In[ ]:




