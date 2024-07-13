#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as mp


# In[2]:


data=pd.read_csv("paymentFraudDet.csv")


# In[5]:


data.columns


# In[6]:


data.info()


# In[8]:


data.isnull().sum()


# In[10]:


data['type'].unique()


# In[11]:


type=data['type'].value_counts()


# In[12]:


trans=type.index


# In[13]:


quan=type.values


# In[14]:


import plotly.express as px


# In[15]:


px.pie(data,values=quan,names=trans,hole=0.3,title='DISTRIBUTION')


# In[16]:


print(data['isFraud'].isnull().sum())


# In[17]:


data


# In[18]:


data.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT'],value=[1,2,3],inplace=True)


# In[19]:


type


# In[20]:


data


# In[21]:


data['isFraud']=data['isFraud'].map({0:'No Fraud',1:'Fraud'})


# In[22]:


data


# In[23]:


x=data[['type','amount','oldbalanceOrg','newbalanceOrg']]


# In[24]:


y=data.iloc[:,-2]


# In[25]:


y


# In[26]:


from sklearn.tree import DecisionTreeClassifier


# In[27]:


model=DecisionTreeClassifier()


# In[28]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# model.fit(xtrain,ytrain)

# In[29]:


model.fit(xtrain,ytrain)


# In[30]:


model.score(xtest,ytest)


# In[31]:


model.predict([[1,9839.64,170136,160296.36]])


# In[32]:


model.predict([[3,182.00,182,0.00]])


# In[33]:


model.predict([[2,182.00,180.00,17088.24]])

