#!/usr/bin/env python
# coding: utf-8

# In[29]:


import warnings
warnings.filterwarnings('ignore')


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[31]:


file_path = 'World Population by country 2024.csv'
df = pd.read_csv(file_path)


# In[32]:


#import libraries


# In[33]:


import pandas as pd


# In[34]:


import seaborn as sns


# In[35]:


df = pd.read_csv("World Population by country 2024.csv")


# In[36]:


#see the data that's been imported


# # Data overview 
# TOP 10

# In[38]:


df.head(11)


# In[39]:


# Convert 'Area (km2)' to numeric, removing commas and handling non-numeric values
df['Area (km2)'] = df['Area (km2)'].str.replace(',', '')
df['Area (km2)'] = pd.to_numeric(df['Area (km2)'], errors='coerce')

# Check for missing values
df.isnull().sum()


# In[51]:


top_10_population_countries = df.nlargest(10, 'Population 2024')
plt.figure(figsize=(12, 6))
sns.barplot(x='Country', y= 'Population 2024', data=top_10_population_countries)
plt.title('top_10_population_countries')
plt.xlabel('Country')
plt.ylabel('Population')
plt.xticks(rotation=45)
plt.show()

#visualize the 10 countries with the lowest population
bottom_10_population_countries = df.nsmallest(10, 'Population 2024')
plt.figure(figsize=(12, 6))
sns.barplot(x='Country', y='Population 2024', data=bottom_10_population_countries)
plt.title('in 10 Low Population Countries')
plt.xlabel('Country')
plt.ylabel('Country')
plt.xticks(rotation=45)
plt.show()


# In[55]:


print(df.head())


# In[57]:


# The 10 fastest growing countries
top_10_fastest_growing = df.nlargest(10, 'Growth Rate')
plt.figure(figsize=(14, 7))
sns.barplot(x='Country', y='Growth Rate', data=top_10_fastest_growing)
plt.title('Top 10 Fastest Growing Countries')
plt.xlabel('Country')
plt.ylabel('Growth Rate')
plt.xticks(rotation=45)
plt.show()


# In[40]:


# Plot the distribution of population in 2024
plt.figure(figsize=(10, 6))
sns.histplot(df['Population 2024'], bins=30, kde=True)
plt.title('Distribution of Population in 2024')
plt.xlabel('Population 2024')
plt.ylabel('Frequency')
plt.show()


# In[41]:


# Plot the correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[63]:


# The 10 most densely populated countries
top_10_dense_countries = df.nlargest(10, 'Density (/km2)')
plt.figure(figsize=(12, 6))
sns.barplot(x='Country', y='Density (/km2)', data=top_10_dense_countries)
plt.title('The 10 most densely populated countries')
plt.xlabel('Country')
plt.ylabel('Population density(/km²)')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




