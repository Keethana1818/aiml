#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data = pd.read_csv("data_clean.csv")
data


# In[6]:


data.info()


# In[10]:


print(type(data))
print(data.shape)


# In[12]:


data.shape


# In[14]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[16]:


data['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[18]:


data1[data1.duplicated()]


# In[20]:


data1[data1.duplicated(keep = False)]


# In[22]:


data1.rename({'Solar.R':'Solar'},axis=1,inplace=True)
data1


# In[24]:


data1.info()


# In[26]:


#display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[28]:


#visualize data1 missing values
cols = data1.columns
colours = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[30]:


#find the mean and median values of each numeric column
median_ozone = data1['Ozone'].median()
median_solar = data1['Solar'].median()
median_wind = data1['Wind'].median()
mean_ozone = data1['Ozone'].mean()
mean_solar = data1['Solar'].mean()
mean_wind = data1['Wind'].mean()
print("Median of ozone: ",median_ozone)
print("Median of solar: ",median_solar)
print("Median of wind: ",median_wind)
print("Mean of ozone: ",mean_ozone)
print("Mean of solar: ",mean_solar)
print("Mean of wind: ",mean_wind)


# In[32]:


#replace the ozone missimg values with median
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[34]:


#replace the solar missing values with median
data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()
data1['Wind'] = data1['Wind'].fillna(median_wind)
data1.isnull().sum()


# In[36]:


#find the mode values of categorical column (weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[38]:


#find the mode values of categorical column (month)
print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[40]:


#Impute missing values (replace NaN with mode etc.) of "weather" using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[42]:


#Impute missing values (replace NaN with mode etc.) of "Month" using fillna()
data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[44]:


data1.tail()


# In[46]:


#Reset the index column
data1.reset_index(drop=True)


# In[48]:


# Create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 31]})

# Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes [0].set_title("Boxplot")
axes [0].set_xlabel("Ozone Levels")

# Plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes [1].set_xlabel("Ozone Levels")
axes [1].set_ylabel("Frequency")

# Adjust layout for better spacing
plt. tight_layout()

# Show the plot
plt.show()


# In[50]:


#create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2,1,figsize=(8,16), gridspec_kw={'height_ratios':[1,3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data1['Solar'],ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title('Boxplot')
axes[0].set_xlabel('Solar Levels')

# Plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='purple', bins=30)
axes [1] .set_title("Histogram with KDE")
axes [1] .set_xlabel("Solar Levels")
axes [1] .set_ylabel("Frequency")

#Adjust layout for better spacing
plt.tight_layout()

#show the plot
plt.show()


# In[58]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[60]:


data1["Ozone"].describe()


# In[62]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[67]:


import scipy.stats as stats

plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q_Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# /// Observations from Q-Q plot
# • The data does not follow normal distribution as the data points are deviating significcantly away from the red line
# • The data shows a right-skewed distribution and possible 

# In[ ]:




