
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model


# In[3]:


df = pd.read_csv(r"C:\Users\1348522\Desktop\Artificial Intelligence\Linear Regression MultiVariates\Home Price Prediction\homeprices.csv")
df


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']], df.price)


# In[9]:


reg.predict([[3000,2,15]])


# In[12]:


reg.coef_


# In[13]:


reg.intercept_

