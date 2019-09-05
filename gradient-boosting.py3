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

# Looking at the unique value counts of each column. There are definitely imbalances and possible outliers.
for col in train.columns[1:]:
    print(col, train[col].value_counts(dropna=False))
    
for col in test.columns[1:]:
    print(col, test[col].value_counts(dropna=False))
