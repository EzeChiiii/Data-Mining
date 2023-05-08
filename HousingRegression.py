#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


corr = df.corr() # allows us to get the correlations
sns.set(rc = {'figure.figsize':(30,20)})
sns.heatmap(corr, annot=True) # annot=True adds the values


# In[143]:


sns.set(rc = {'figure.figsize':(8,8)})
sns.scatterplot(data=df, x='1stFlrSF', y='SalePrice', hue='GrLivArea')


# In[ ]:





# In[142]:


sns.set(rc = {'figure.figsize':(8,8)})
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', hue='1stFlrSF')


# In[7]:


sns.set(rc = {'figure.figsize':(8,8)})
sns.scatterplot(data=df, x='GrLivArea', y='TotalBsmtSF', hue='SalePrice')


# In[148]:


df.isna().sum()


# In[149]:


df = df.dropna(axis=1)


# In[150]:


df.isna().sum()


# In[151]:


df.columns


# In[152]:


sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df)


# In[153]:


sns.scatterplot(x='GarageArea', y='SalePrice', data=df)


# In[49]:


sns.scatterplot(x='2ndFlrSF', y='SalePrice', data=df)


# In[154]:


y = df['SalePrice']


# In[155]:


X = df[['1stFlrSF','GarageArea', 'TotalBsmtSF']]


# In[156]:


from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(X)


# In[169]:


X_scaled = pd.DataFrame(scaled, columns = X.columns)

X_scaled


# In[170]:


from sklearn.feature_selection import SelectKBest, f_regression

fs = SelectKBest(score_func = f_regression, k=2)
best = fs.fit_transform(X_scaled, y)


# In[171]:


features = np.array(list(X_scaled.columns))

best_features = features[fs.get_support()]
print(best_features)


# In[172]:


from sklearn.model_selection import train_test_split


# In[173]:


from sklearn.linear_model import LinearRegression


# In[9]:


import statsmodels.api as sma


# In[8]:


sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df)


# In[176]:


X_train,X_test, y_train,y_test = train_test_split(X_scaled[best_features],y,test_size = 0.3, random_state = 7)


# In[177]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[178]:


from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[179]:


lr_model = lr.fit(X_train,y_train)


# In[183]:


#look at the intercept and slope of the model. 

print('Model Intercept:',lr_model.intercept_)
print('Model Slope:' ,lr_model.coef_[0])
#Get our coefficient of determination

print('Coefficient of Determination:',lr_model.score(X_test, y_test))
coeff_df = pd.DataFrame(lr_model.coef_, X_test.columns, columns=['Coefficients'])
coeff_df
y_pred = lr_model.predict(X_test)

from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred,squared = False)

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', rmse)


# In[184]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled[best_features], y, test_size = 0.4, random_state = 0)


# In[182]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[137]:


from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[138]:


lr_model = lr.fit(X_train, y_train)


# In[141]:


print('Model Intercept:',lr_model.intercept_)
print('Model Slope:' ,lr_model.coef_[0])
print('Coefficient of Determination:',lr_model.score(X_test, y_test))


coeff_df = pd.DataFrame(lr_model.coef_, X_test.columns, columns=['Coefficients'])
coeff_df


# In[194]:


plot_rnet = sns.regplot(data = df,x='TotalBsmtSF',y='SalePrice',line_kws={'color':'black',
                                                                           'label':'$y=%5.4sx+%5.5s$'%(lr_model.coef_[1],lr_model.intercept_)})


# In[ ]:





# In[196]:


plot_rnet = sns.regplot(data = df,x='GarageArea',y='SalePrice', line_kws={'color':'black',
                                                                           'label':'$y=%5.4sx+%5.5s$'%(lr_model.coef_[1],lr_model.intercept_)})


# In[185]:


y_pred = lr_model.predict(X_test)


# In[186]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error:', mae)


# In[187]:


mse = metrics.mean_squared_error(y_test,y_pred)
print('Mean Square Error:', mse)


# In[188]:


rmse = metrics.mean_squared_error(y_test,y_pred,squared = False)
print('Root Mean Square Error:', rmse)


# In[197]:


X = X = df[['OverallQual', 'GrLivArea', 'GarageArea','GarageCars']]
y = df['SalePrice']


# In[198]:


from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(X)


# In[199]:


X_scaled = pd.DataFrame(scaled, columns = X.columns)
X_scaled


# In[200]:


from sklearn.feature_selection import SelectKBest, f_regression

fs = SelectKBest(score_func = f_regression, k=2)
best = fs.fit_transform(X_scaled, y)


# In[201]:


features = np.array(list(X_scaled.columns))

best_features = features[fs.get_support()]
print(best_features)


# In[202]:


sns.regplot(data=df, y='SalePrice', x='OverallQual', fit_reg=True)


# In[203]:


sns.regplot(data=df, y='SalePrice', x='GrLivArea', fit_reg=True)


# In[204]:



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled[best_features],y,test_size=0.3, random_state = 7)


# In[205]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[206]:


lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)


# In[207]:


import statsmodels.api as sma
X2 = sma.add_constant(X_train)

est = sma.OLS(y_train,X2) 
est2 = est.fit()


# In[208]:


print(est2.summary())


# In[209]:


print('Intercept:',lr_model.intercept_)
print('Coefficient:',lr_model.coef_[1])


# In[210]:


y_pred = lr_model.predict(X_test)


# In[211]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_test,y_pred)
print(mae)


# In[212]:


mse = metrics.mean_squared_error(y_test,y_pred)
print(mse)


# In[213]:


rmse = metrics.mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# In[215]:


X


# In[216]:


X = df[['OverallQual', 'GrLivArea', 'GarageArea','GarageCars']]
y = df['SalePrice']
scaled = StandardScaler().fit_transform(X)


# In[217]:


scaled = StandardScaler().fit_transform(X)


# In[218]:


X_scaled = pd.DataFrame(scaled, columns = X.columns)


# In[219]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3, random_state = 7)


# In[220]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[221]:


lr_model = lr.fit(X_train, y_train)


# In[222]:


X2 = sma.add_constant(X_train)

est = sma.OLS(y_train,X2) 
est2 = est.fit()


# In[223]:


print(est2.summary())


# In[224]:


print('Intercept:',lr_model.intercept_)
print('Coefficient:',lr_model.coef_[1])


# In[225]:


y_pred = lr_model.predict(X_test)


# In[226]:


mae = metrics.mean_absolute_error(y_test,y_pred)
print(mae)


# In[227]:


mse = metrics.mean_squared_error(y_test,y_pred)
print(mse)


# In[228]:


rmse = metrics.mean_squared_error(y_test, y_pred, squared = False)
print(rmse)


# In[ ]:




