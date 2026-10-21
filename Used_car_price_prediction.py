#!/usr/bin/env python
# coding: utf-8

# # Part 1: Data preprocessing

# Dataset link: https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv

# ## Importing the libraries and the dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[ ]:


dataset = pd.read_csv('/content/car data.csv')

# In[ ]:


dataset.head()

# ## Data exploration

# In[ ]:


dataset.shape

# In[ ]:


dataset.columns

# In[ ]:


dataset.info()

# In[ ]:


# categorical columns
dataset.select_dtypes(include='object').columns

# In[ ]:


len(dataset.select_dtypes(include='object').columns)

# In[ ]:


# numerical columns
dataset.select_dtypes(include=['float64', 'int64']).columns

# In[ ]:


len(dataset.select_dtypes(include=['float64', 'int64']).columns)

# In[ ]:


dataset.describe()

# ## Dealing with missing values

# In[ ]:


dataset.isnull().values.any()

# In[ ]:


dataset.isnull().values.sum()

# ## Restructure the dataset

# In[ ]:


dataset.head()

# In[ ]:


dataset = dataset.drop(columns='Car_Name')

# In[ ]:


dataset.head()

# In[ ]:


# add a column
dataset['Current Year'] = 2020

# In[ ]:


dataset.head()

# In[ ]:


dataset['Years Old'] = dataset['Current Year'] - dataset['Year']

# In[ ]:


dataset.head()

# In[ ]:


dataset = dataset.drop(columns=['Current Year', 'Year'])

# In[ ]:


dataset.head()

# ## Encoding the categorical data

# In[ ]:


dataset.select_dtypes(include='object').columns

# In[ ]:


len(dataset.select_dtypes(include='object').columns)

# In[ ]:


dataset['Fuel_Type'].nunique()

# In[ ]:


dataset['Seller_Type'].nunique()

# In[ ]:


dataset['Transmission'].nunique()

# In[ ]:


dataset.shape

# In[ ]:


# one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)

# In[ ]:


dataset.head()

# In[ ]:


dataset.shape

# ## Correlation matrix

# In[ ]:


dataset_2 = dataset.drop(columns='Selling_Price')

# In[ ]:


dataset_2.corrwith(dataset['Selling_Price']).plot.bar(
    figsize=(16,9), title='Correlated with Selling Price', grid=True
)

# In[ ]:


corr = dataset.corr()

# In[ ]:


# heatmap
plt.figure(figsize=(16, 9))
sns.heatmap(corr, annot=True)

# ## Splitting the dataset

# In[ ]:


dataset.head()

# In[ ]:


# matrix of features
x = dataset.drop(columns='Selling_Price')

# In[ ]:


# target variable
y = dataset['Selling_Price']

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# In[ ]:


x_train.shape

# In[ ]:


y_train.shape

# In[ ]:


x_test.shape

# In[ ]:


y_test.shape

# ## Feature scaling

# In[ ]:


# we are not applying the feature scaling for this specific business problem

# # Part 2: Building the model

# ## 1) Multiple linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor_mlr = LinearRegression()
regressor_mlr.fit(x_train, y_train)

# In[ ]:


y_pred = regressor_mlr.predict(x_test)

# In[ ]:


from sklearn.metrics import r2_score

# In[ ]:


r2_score(y_test, y_pred)

# ## 2) Random forest regression

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(x_train, y_train)

# In[ ]:


y_pred = regressor_rf.predict(x_test)

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# # Part 3: Fing the optimal parameters using RandomizedSearchCV

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# In[ ]:


parameters = {
    'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'criterion':['mse', 'mae'],
    'max_depth':[10, 20, 30, 40, 50],
    'min_samples_split':[2, 5, 10, 20, 50],
    'min_samples_leaf':[1, 2, 5, 10],
    'max_features':['auto', 'sqrt', 'log2']
}

# In[ ]:


parameters

# In[ ]:


random_cv = RandomizedSearchCV(estimator=regressor_rf, param_distributions=parameters, n_iter=10,
                               scoring='neg_mean_absolute_error', cv=5, verbose=2, n_jobs=-1)

# In[ ]:


random_cv.fit(x_train, y_train)

# In[ ]:


random_cv.best_estimator_

# In[ ]:


random_cv.best_params_

# # Part 4: Final model (Random forest)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
regressor.fit(x_train, y_train)

# In[ ]:


y_pred = regressor.predict(x_test)

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# # Part 5: Predicting a single observation

# In[ ]:


dataset.head()

# In[ ]:


single_obs = [[8.50, 3500, 0, 5, 1, 0, 0, 1]]

# In[ ]:


regressor.predict(single_obs)
