#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ1 = Univ.iloc[:,1:]


# In[4]:


Univ1


# In[5]:


cols = Univ1.columns


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols )
scaled_Univ_df


# In[15]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[17]:


clusters_new.labels_


# In[19]:


set(clusters_new.labels_)


# In[21]:


#assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[23]:


Univ


# In[25]:


Univ.sort_values(by = "clusterid_new")


# In[29]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# Observations:
# 
# . Custer 2 appears to be the top rated universities cluster as the cut off score, Top10, SFRatio parameter mean values are highest
# 
# . Cluster 1 appears to occupy the middle level rated universities
# 
# . Cluster 0 comes as the lower level rated universities

# In[39]:


Univ[Univ['clusterid_new']==0]


# In[43]:


wcss = []
for i in range(1, 20):
    
    kmeans = KMeans(n_clusters=i, random_state=0 )
    kmeans.fit(scaled_Univ_df)
    #kmeans.fit(univ1)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method' )
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[45]:


Univ.info()


# In[ ]:




