#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Most Important 

import pandas as pd
import os

## Model Selection
from sklearn.model_selection import train_test_split 

## Preprocessing 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


# In[18]:


file_path = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(file_path)
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')
# add new features 
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedroms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_household'] = df_housing['population'] / df_housing['households']
## Split the Whole dataset to features and target
X = df_housing.drop(columns=['median_house_value'], axis=1)  ## features
y = df_housing['median_house_value']  ## target
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
#divide the data to numericla nad categorical 
num_cols = [col for col in  X_train.columns if X_train[col].dtype in ['float64', 'int64']]
categ_cols = [col for col in  X_train.columns if X_train[col].dtype not in ['float64', 'int64']]


# In[19]:


## We can get much much easier like the following
## numerical pipeline
num_pipeline = Pipeline([
                    ('selector', DataFrameSelector(num_cols)),    ## select only these columns
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())])

## categorical pipeline
categ_pipeline = Pipeline(steps=[
            ('selector', DataFrameSelector(categ_cols)),    ## select only these columns
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OHE', OneHotEncoder(sparse_output=False))])

## concatenate both two pipelines
total_pipeline = FeatureUnion(transformer_list=[
                                ('num_pip', num_pipeline),
                                ('categ_pipeline', categ_pipeline)])

## deal with (total_pipeline) as an instance -- fit and transform to train dataset and transform only to other datasets
X_train_final = total_pipeline.fit_transform(X_train)

            


# In[24]:


def processing_new_data(X_new):
    return total_pipeline.transform(X_new)

