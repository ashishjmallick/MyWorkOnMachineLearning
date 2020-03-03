
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[11]:


df = pd.read_csv(r"C:\Users\1348522\Desktop\Artificial Intelligence\Linear Regression MultiVariates\Hiring Salary Prediction\hiring.csv")
df


# In[13]:


df['test_score(out of 10)']


# In[17]:


import math
median_num = math.floor(df['test_score(out of 10)'].mean())
median_num


# In[18]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_num)
df['test_score(out of 10)']


# In[19]:


reg = linear_model.LinearRegression()


# In[25]:


reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df[['salary($)']])


# In[27]:


reg.predict([[2,7,9]])


# In[33]:


print(reg.coef_)
print(reg.intercept_)

