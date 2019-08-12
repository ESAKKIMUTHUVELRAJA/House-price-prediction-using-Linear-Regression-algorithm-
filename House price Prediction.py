
# coding: utf-8

# **Linear Regression

# ##House Price Prediction

# In[7]:


import numpy as np


# In[23]:


area=[2,2.5,6,3.8,7.2,5.5,8.9,8.3,5.5,2.4]
area=np.array(area)
print(area.shape)
print(area)
area=area.reshape(10,1)
print(area)
price=[800,500,600,200,600,900,700,600,400,550]
print(price)


# In[24]:


from sklearn.linear_model import LinearRegression


# In[27]:


model=LinearRegression()
model.fit(area,price)


# In[49]:


a=[5.5,7.8,9.9,10.89]
a=np.array(a).reshape(-1,1)
new_price=model.predict(a)
print(new_price)


# In[55]:


from matplotlib import pyplot as plt
plt.scatter(area,price,label='Old_datas')
plt.scatter(a,new_price,label='New_prediction')
plt.legend()
plt.title('House price prediction')
plt.xlabel('area in square feet')
plt.ylabel('price in 1000')
plt.show()

