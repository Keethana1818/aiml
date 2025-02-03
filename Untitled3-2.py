#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Load the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[13]:


data1 = pd.read_csv("NewspaperData.csv")
print(data1)


# In[14]:


data1.isnull().sum()


# In[15]:


data1


# In[20]:


data1.isnull().sum()


# In[22]:


data1.describe()


# In[24]:


data1.info()


# In[26]:


plt.figure (figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[28]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[30]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[32]:


data1["daily"].corr(data1["sunday"])


# In[34]:


data1[["daily","sunday"]].corr()


# In[36]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[38]:


model1.summary()


# In[40]:


# Plot the scatter plot and overlay the fitted straight line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m",marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x

# plotting the regression Line
plt. plot(x, y_hat, color = "g")

# putting Labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[42]:


sns.regplot(x="daily", y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# In[44]:


newdata=pd.Series([200,300,1500])


# In[48]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[52]:


model1.predict(data_pred)


# In[54]:


pred = model1.predict(data1["daily"])
pred


# In[56]:


data1["Y_hat"] = pred
data1


# In[58]:


data1["residuals"]= data1["sunday"]-data1["Y_hat"]
data1


# In[60]:


mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[62]:


mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[64]:


plt.scatter(data1["Y_hat"], data1["residuals"])


# In[ ]:




