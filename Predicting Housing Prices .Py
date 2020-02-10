# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:05:27 2020

@author: MinuLJ01
"""

"""
Initial thoughts on author's analysis. She doesn't categorize and look at some of the categorical values such as sale condition sale type 
and so on. Also look at factorization as part of a solution possibly?
"""

#taken from https://www.kaggle.com/goldens/house-prices-on-the-top-with-a-simple-model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV

#if having trouble finding filepath to load. use CD to get current directory to help find where the console is initially looking
#I had trouble when not putting the full file path. Kinda annoying but putting everything seemed to work.
train=pd.read_csv("C://Users/MinuLJ01/Desktop/Python/Housing Prices/train.csv")
test=pd.read_csv("C://Users/MinuLJ01/Desktop/Python/Housing Prices/test.csv")
test2=pd.read_csv("C://Users/MinuLJ01/Desktop/Python/Housing Prices/test.csv")
houses=pd.concat([train,test], sort=False)

#the next 10 lines get information about the data set to understand the variables/columns so that we can decide how to alter them
len_train=train.shape[0]
#This shows all the variables and will give you an idea of what types of variables you have
houses.dtypes
#This shows how many object type variables there are
houses.select_dtypes(include='object').head()
#While this one shows how many number/float/integer type variables. basically differentiating between fact and dim columns
#Author uses just int while I use int64. int64 holds more information and when I search using int64, I actually get all the columns i'm looking for
houses.select_dtypes(include=['float','int64']).head()
#so the author identifies that the column MSSubClass is categorical but stored as numerical. we'll need to change this to a 
#nomial variable.
houses['MSSubClass']=houses['MSSubClass'].astype(str)

#Now the author does some data cleanup
#What she does here is sum up the count of empty values. She looks at the categorical data and sees how many rows are empty for each
houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]
#Now she replaces all the na's with "None"
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train[col]=train[col].fillna('None')
    test[col]=test[col].fillna('None')
#Here she replaces the na's with the mode/most repeated answer for the varaible
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(train[col].mode()[0])
#Now she does the same thing with the numerical variables
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train[col]=train[col].fillna(0)
    test[col]=test[col].fillna(0)
#Same thing but she fills in with mean. This seems fairly dangerous considering 17% of the data set has missing value
#for the variable lotfrontage. In rework, i'll probably handle this variable differently    
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())
#Checks if there are any leftover na's in dataset
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())

#Correlation map
plt.figure(figsize=[30,15])
sns.heatmap(train.corr(), annot=True)
#Removing outliers recomended by author (taking out 4 rows that are above 4,000 ground living area)
train = train[train['GrLivArea']<4000]
len_train=train.shape[0]

#This simply makes the MSSubClass categorical since the labels are categories and not quantities
houses['MSSubClass']=houses['MSSubClass'].astype(str)

skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df=pd.DataFrame({'Skew':skew})
skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]

skewed_df.index

"""train[train['GrLivArea']<4000].count()
houses['GrLivArea'].dtypes
train['GrLivArea'].value_counts(bins=4)
houses['SalePrice'].mean()"""

train=houses[:len_train]
test=houses[len_train:]

#The author is using a Box Cox transformation ti turn the variables into a normal shape. The lamda is used in it as the exponent.
#This seems similar to using the log of a variable
lam=0.1
for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',
       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',
       'GarageYrBlt'):
    train[col]=boxcox1p(train[col],lam)
    test[col]=boxcox1p(test[col],lam)


