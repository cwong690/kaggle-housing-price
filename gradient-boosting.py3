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

# Columns with correlation higher than 0.4
train_corr = train.corr()
high_corr = train_corr.index[abs(train_corr['SalePrice']) > 0.4]
plt.figure(figsize=(10,10))
sns.heatmap(train[high_corr].corr(), annot=True)

# From the plots, we can see there are many ordinal columns that can be converted to numbers



