
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[52]:


df =pd.read_csv (r"C:\Users\1348522\Desktop\homeprices.csv")
df


# In[88]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sqr_ft)', size=25)
plt.ylabel('price(US$)', size =25)
plt.scatter(df.area,df.price, color='blue',marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color= 'red')


# In[110]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[111]:


reg.predict(3300)


# In[112]:


reg.coef_


# In[113]:


reg.intercept_


# In[114]:



135.78767123*3300+180616.43835616432


# In[115]:


d = pd.read_csv(r"C:\Users\1348522\Desktop\areas.csv")
d


# In[118]:


p = reg.predict(d)


# In[124]:


d['prices'] = p
d


# In[125]:


d.to_csv(r"C:\Users\1348522\Desktop\prediction.csv",index=False)
dl =pd.read_csv (r"C:\Users\1348522\Desktop\prediction.csv")
dl









