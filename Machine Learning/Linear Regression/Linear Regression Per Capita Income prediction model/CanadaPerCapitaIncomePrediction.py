
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[22]:


df =pd.read_csv(r"C:\Users\1348522\Desktop\Artificial Intelligence\Linear Regression Per Capita Income prediction model\Canada_Per_Capita_Income.csv")
df.head(10)


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Year", Size= 15)
plt.ylabel("Per Capita Income", Size= 15)
plt.scatter(df.Year, df.PerCapitaIncome,color='blue', marker ='+')

reg = linear_model.LinearRegression()
reg.fit(df[['Year']], df.PerCapitaIncome)
plt.plot(df.Year, reg.predict(df[['Year']]), color='red')


# In[34]:


reg.predict(2020)

