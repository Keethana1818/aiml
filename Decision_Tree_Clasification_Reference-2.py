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


# In[3]:


iris = pd.read_csv("iris.csv")
iris


# In[4]:


iris.info()


# In[5]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[6]:


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

# In[13]:


iris = iris.reset_index(drop=True)
iris


# In[15]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[17]:


iris.info()


# In[19]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[21]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[23]:


x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# In[27]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[29]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[31]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[ ]:




