# Importing important libraries that will be used to analyze and predict the final sales price.

import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Read in datasets with pandas

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
label = train['SalePrice']

# EDA

# Looking at begin and end of the train and test datasets
train.head()
train.tail()

test.head()
test.tail()

# Statistical summary of the columns as well as info on number of rows and columns
train.info()
train.describe()

test.info()
test.describe()

# Looking at the unique value counts of each column. There are definitely imbalances within each categories.
for col in train.columns[1:]:
    print(col, train[col].value_counts(dropna=False))
    
for col in test.columns[1:]:
    print(col, test[col].value_counts(dropna=False))

# Drop duplicates in case there are any
train.drop_duplicates()
print(train.shape)

test.drop_duplicates()
print(test.shape)

# Plotting graphs
for x in num_cols:
    sns.lmplot(x, 'SalePrice', train)
    plt.show()
    
for x in cat_cols:
    sns.countplot(x=x, data=train)
    plt.show()

for col in cat_cols:
    sns.boxplot(col, 'SalePrice', data=train)
    plt.show()    

# Correlation graph
train.corr().style.background_gradient(cmap='coolwarm')

test.corr().style.background_gradient(cmap='coolwarm')

# Columns with correlation higher than 0.4
train_corr = train.corr()
high_corr = train_corr.index[abs(train_corr['SalePrice']) > 0.4]
plt.figure(figsize=(10,10))
sns.heatmap(train[high_corr].corr(), annot=True)

test_corr = test.corr()
high_corr = test_corr.index[abs(test_corr['SalePrice']) > 0.4]
plt.figure(figsize=(10,10))
sns.heatmap(test[high_corr].corr(), annot=True)

# Check for missing values
print(train.shape)
sns.heatmap(train.isnull(), cmap='viridis')
print(test.shape)
sns.heatmap(test.isnull(), cmap='viridis')

train_null = train.isnull().sum()
train_null[train_null > 0].sort_values(ascending=False)
test_null = test.isnull().sum()
test_null[test_null > 0].sort_values(ascending=False)

perc_null = (train_null.sort_values(ascending=False) / 1460) * 100
perc_null[perc_null > 50.0]
test_perc_null = (test_null.sort_values(ascending=False) / 1460) * 100
test_perc_null[test_perc_null > 50.0]

col_w_null = train_null[train_null > 0].keys()
test_col_w_null = test_null[test_null > 0].keys()

# Filling in missing data with appropriate value
train[train['MSSubClass'] == 80]['Electrical']
train['Electrical'].fillna(value='SBrKr', inplace=True)

for col in col_w_null:
    if train[col].dtypes == float:
        train[col].fillna(value=0, inplace=True)
for col in col_w_null:
    if train[col].dtypes == object:
        train[col].fillna(value='None', inplace=True)
train[col_w_null].count()

for col in col_w_null:
    if test[col].dtypes == float:
        test[col].fillna(value=0, inplace=True)
for col in col_w_null:
    if test[col].dtypes == object:
        test[col].fillna(value='None', inplace=True)
test[col_w_null].count()


# From the plots, we can see there are many ordinal columns that can be converted to numbers
comb_to_groups = ['LotFrontage', 'LotArea', 'Neighborhood', 'MasVnrArea']
nomial = ['MSSubClass']
discrete = ['BsmtFullBath', 'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars']
cat_to_ord = ['LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']
#ordinal = ['OverallQual','OverallCond']
binary = ['Street', 'Utilities','CentralAir','PavedDrive','PoolArea']
years = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
getdummies = ['Neighborhood', 'LotShape', 'Street']

from sklearn import preprocessing
enc = preprocessing.LabelEncoder()

# Binary columns: Utilities, CentralAir, PavedDrive, PoolArea
train['CentralAir'] = enc.fit_transform(train['CentralAir'])

# PavedDrive and PoolArea has values that can be combined
train['PavedDrive'].replace(['N', 'Y', 'P'], [0,1,1], inplace=True)
train['PavedDrive'].value_counts()


# Years
train['HomeAge'] = train['YrSold'] - train['YearBuilt']
train.head()
# train.drop('YearBuilt', 'YrSold', 'YearRemodAdd')
train['GarageAge'] = train['YrSold'] - train['GarageYrBlt']
train.head()
# train.drop('GarageYrBlt')
