#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import mean
from scipy.stats import sem, t, f, levene, ttest_ind, kstest, chisquare, norm, kurtosis, kurtosistest, normaltest
from scipy.stats import ttest_1samp as ttest
from scipy import stats
from statsmodels.stats import gof
from numpy.random import seed, randn, rand
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import datasets
#%matplotlib notebook
#lt.style.use('seaborn-poster')
plt.rcParams['figure.figsize'] = [10,6]
import sympy as sp
import seaborn as sns
import datetime
from fbprophet import Prophet


# In[106]:


X=np.array([ 67, 64, 69, 88, 72, 80, 85,  77, 78, 81, 66, 91, 68, 73 ])


# In[107]:


Y=np.array([ 57, 53, 71, 61, 73, 50, 53,  63, 41, 78, 68, 86, 70, 74 ])


# In[108]:


print('Massimo di X:',X.max())
print('Minimo di X:',X.min())
print('Media di X:',X.mean())
print('Varianza di X:', np.var(X))


# In[110]:


print('Massimo di Y:',Y.max())
print('Minimo di Y:',Y.min())
print('Media di Y:',Y.mean())
print('Varianza di Y:', np.var(Y))


# In[111]:


plt.boxplot(X)


# In[122]:


bx=np.linspace(50,100,8)


# In[123]:


sns.distplot(X, bins= bx, color = 'blue', hist_kws={'edgecolor':'black'})


# In[125]:


plt.boxplot(Y)


# In[126]:


by=np.linspace(30,100,9)


# In[127]:


sns.distplot(Y, bins= by , color = 'red', hist_kws={'edgecolor':'black'})


# In[82]:


valori_X = randn(14)
scalati_X = np.mean(X) + np.std(X)*valori_X


# In[83]:


chisquare(X,f_exp=scalati_X)


# In[128]:


valori_Y = randn(14)
scalati_Y = np.mean(Y) + np.std(Y)*valori_Y
chisquare(Y,f_exp=scalati_Y)


# In[129]:


F = np.var(X) / np.var(Y)
df1 = len(X) - 1
df2 = len(Y) - 1
p_value = f.cdf(F, df1, df2)

if (p_value<0.05):
    print('Reject H0', p_value)
else:
    print('Cant Reject H0', p_value)


# In[130]:


F = np.var(Y) / np.var(X)
df1 = len(Y) - 1
df2 = len(X) - 1
p_value = f.cdf(F, df1, df2)

if (p_value<0.05):
    print('Reject H0', p_value)
else:
    print('Cant Reject H0', p_value)


# In[131]:


ttest_ind(X,Y,equal_var=True)


# In[ ]:





# In[80]:


print('Media di Z:' ,(0.5*X.mean()) + (0.5*Y.mean()) )


# In[44]:


Z = np.array (Y - X)


# In[45]:


print('Campione differenze di Z:' , Z)


# In[46]:


Z_segnato = np.mean(Z)
print('Media di Z', Z_segnato )


# In[48]:


T_0 = np.sqrt(14)* (Z_segnato / np.std(Z))
print('Valore statistica T_0', T_0)


# In[ ]:





# In[7]:


data=pd.read_csv('data/data.csv' , sep=';')


# In[8]:


data.describe()


# In[9]:


data.info()


# In[10]:


X= data.X
plt.boxplot(X)


# In[11]:


Y=data.Y
plt.boxplot(Y)


# In[12]:


data.hist()


# In[13]:


data.plot.scatter(x='X',y='Y')


# In[89]:


model=LinearRegression()


# In[92]:


X = data.X
y = data.Y+6

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[104]:


X = data.X
y = data.Y

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()

