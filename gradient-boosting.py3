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

#######################################################
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

#######################################################
# MISSING VALUES

#The shape of the training data is printed to see how many rows and columns we are dealing with.  
#A heatmap is created using seaborn to give a visualization of the null values throughout the dataset.

print(train.shape)
sns.heatmap(train.isnull(), cmap='viridis')

print(test.shape)
sns.heatmap(test.isnull(), cmap='viridis')

# Next, let's see how many null values are in each column.
train_null = train.isnull().sum()
train_null[train_null > 0].sort_values(ascending=False)
test_null = test.isnull().sum()
test_null[test_null > 0].sort_values(ascending=False)
test_null[test_null == 2].index

# Upon examination of the descriptions of these columns missing values, 
# it is determined that the house does not have those features. 
# Therefore, we will fill the null values with either 0 for numeric columns or 'None' for categorical columns.  
# However, there are some exceptions.
one_null = ['Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', 'KitchenQual', 'GarageCars', 'GarageArea', 'SaleType']

for col in one_null:
    print(col)
    print(test[test[col].isnull()])
    print('\n')
    
# Row 691 is missing Exterior1st, Exterior2nd.
# Row 660 is missing BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF.
# Row 95 is missing KitchenQual. Row 1116 is missing GarageCars, GarageArea.
# Row 1029 is missing SaleType.
# Checking out the foundation type of Row 691 and then check to see what the most common Exterior1st is amongst rows with the same foundation type.
# We will fill in the Exterior1st missing value with the mode.
test[test['Exterior1st'].isnull()]['Foundation']
print(test[test['Foundation']=='PConc']['Exterior1st'].mode())
test['Exterior1st'].fillna(value='VinylSd', inplace=True)
print(test[test['Exterior1st']=='VinylSd']['Exterior2nd'].mode())
test['Exterior2nd'].fillna(value='VinylSd', inplace=True)

# Row 660 is missing a lot of basement related values.
# We will see if the BsmtQual is also null because that means it does not have a basement.
test[test['Id']==2121][['BsmtQual','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
test[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].fillna(value=0, inplace=True)
print(test[test['KitchenQual'].isnull()]['KitchenAbvGr'])
print(test[test['KitchenAbvGr']==1]['KitchenQual'].mode())
test['KitchenQual'].fillna(value='TA', inplace=True)
test[test['Id']==2577][['GarageType', 'GarageFinish', 'GarageCond', 'GarageQual', 'GarageCars', 'GarageArea']]

# It is missing a lot of garage values which usually means there is no garage. However, with row 1116, it has a garagetype but no other values.
# Therefore, we will fill the rest with the mode/median of garages with the same type.
print(test[test['GarageType']=='Detchd']['GarageCars'].mode())
print(test[test['GarageType']=='Detchd']['GarageArea'].median())
test['GarageCars'].fillna(value=1, inplace=True)
test['GarageArea'].fillna(value=384, inplace=True)

# SaleCondition
print(test[test['Id']==2490]['SaleCondition'])
print(test[test['SaleCondition']=='Normal']['SaleType'].mode())
test['SaleType'].fillna(value='WD', inplace=True)

# Two null
two_null = ['Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional']

for col in two_null:
    print(col)
    print(test[test[col].isnull()])
    print('\n')
    
print(test['Utilities'].mode())
test['Utilities'].fillna(value='AllPub', inplace=True)

# For BsmtFullBath and BsmtHalfBath, it would be sensible that if there is no basement, there will not be any basement bathrooms.
# We can check for rows where BsmtQual is null (meaning no basement) and see what the bathroom values are.
test[test['BsmtQual'].isnull()]['BsmtFullBath']

# It seems we are correct! We will go ahead and fill the nulls with 0.0, representing none.
test[['BsmtFullBath', 'BsmtHalfBath']].fillna(value=0, inplace=True)

# Since functionality rating of a home can be so different between similar types, we will use the mode to fill in the null values.
print(test['Functional'].value_counts())
test['Functional'].fillna(value='Typ', inplace=True)

# MSZoning also shouldn't have any nulls because all of them should have a label.
test[test['MSZoning'].isnull()][['LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood']]
test[test['Neighborhood']=='IDOTRR']['MSZoning'].value_counts()
test['MSZoning'].fillna(value='RM', inplace=True)

# The exception is the 'Electrical' column in the train data.
# In the description, it does not have an option for no electrical system. Let's see what is going on in that row.
train[train['Electrical'].isnull()]

'''
To fill in the one missing value, we can try using the mode. 
To make it as accurate as possible, we will look at all rows with the same MsSubClass, the type of dwelling. 
We can then see what the mode is amongst rows of the same MsSubClass.
Basically all housing with MsSubClass 80, which is a split or multi-level dwelling, has Standard Circuit Breakers & Romex (SBrkr) for the electrical system.
So we will fill in the null value with 'SBrKr'. Double check to make sure the value was changed.
'''
train[train['MSSubClass'] == 80]['Electrical']
train['Electrical'].fillna(value='SBrKr', inplace=True)

####################################################
# Other Missing Values 
# The percentage of null values per column can be calculated.
perc_null = (train_null.sort_values(ascending=False) / 1460) * 100
perc_null[perc_null > 50.0]
test_perc_null = (test_null.sort_values(ascending=False) / 1460) * 100
test_perc_null[test_perc_null > 50.0]

col_w_null = train_null[train_null > 0].keys()
test_col_w_null = test_null[test_null > 0].keys()

for col in col_w_null:
    if train[col].dtypes == float:
        train[col].fillna(value=0, inplace=True)

for col in col_w_null:
    if train[col].dtypes == object:
        train[col].fillna(value='None', inplace=True)
train[col_w_null].count()

for col in test_col_w_null:
    if test[col].dtypes == float:
        test[col].fillna(value=0, inplace=True)

for col in test_col_w_null:
    if test[col].dtypes == object:
        test[col].fillna(value='None', inplace=True)
test[test_col_w_null].count()

######################################################
# Target Variable

label.isnull().any()

sns.distplot(label)
plt.show()
print('Skew: ', label.skew())
print('Kurtosis: ', label.kurt())

# plot log1p('SalePrice')
sns.distplot(np.log1p(label))
plt.show()
print('Skew: ', np.log1p(label).skew())
print('Kurtosis: ', np.log1p(label).kurt())

##################################################
# Encoding Features

from sklearn import preprocessing
enc = preprocessing.LabelEncoder()

# MSSubClass seem to have numerical values but they represent classes that is not higher than one another.
train.corr()['MSSubClass']['SalePrice']
# As expected, there is not much correlation between the MSSubClass and SalePrice.

map_dict = {}
map_dict.update(map_dict.fromkeys({20,30,40,120}, '1-Story'))
map_dict.update(map_dict.fromkeys({45,50,150}, '1.5-Story'))
map_dict.update(map_dict.fromkeys({60,70,75,160}, '2-Story'))
map_dict.update(map_dict.fromkeys({80,85,180}, 'Split/Multilevel'))
map_dict.update(map_dict.fromkeys({90,190}, 'Other'))
map_dict

train['MSSubClass'] = train['MSSubClass'].map(map_dict)
train['MSSubClass'].value_counts(dropna=False)
test['MSSubClass'] = test['MSSubClass'].map(map_dict)
test['MSSubClass'].value_counts(dropna=False)

print(train.corr()['LotArea']['SalePrice'])
print(train.corr()['LotFrontage']['SalePrice'])

plt.figure(figsize=(20,6))
sns.boxplot(y='SalePrice', x='Neighborhood', data=train)
train['Neighborhood'].value_counts()

print(train['MasVnrType'].value_counts())
print(train['MasVnrType'].describe())
sns.boxplot(x='MasVnrType', y='SalePrice', data=train)
sns.scatterplot('MasVnrArea', 'SalePrice', data=train, alpha=0.5)

train['HomeAge'] = train['YrSold'] - train['YearBuilt']
train.head()
test['HomeAge'] = test['YrSold'] - test['YearBuilt']
test.head()

train[train['HomeAge'] < 0][['YearBuilt', 'YrSold', 'HomeAge']]
test[test['HomeAge'] < 0][['YearBuilt', 'YrSold', 'HomeAge', 'SaleType', 'SaleCondition']]
for value in test['HomeAge']:
    if value < 0:
        test['HomeAge'].replace(value, 0, inplace=True)

# There is a typo in the test dataset where the year is 2207 instead of 2007.
test['GarageYrBlt'].replace(2207, 2007, inplace=True)
test[test['Id']==2593]['GarageYrBlt']

train['GarageAge'] = train['YrSold'] - train['GarageYrBlt']
train.head()
test['GarageAge'] = test['YrSold'] - test['GarageYrBlt']
test.head()

print(train[train['GarageAge'] < 0][['GarageYrBlt', 'YrSold', 'GarageAge']])
train[train['GarageAge'] > 150][['GarageYrBlt', 'YrSold', 'GarageAge']]
print(test[test['GarageAge'] < 0][['Id','GarageYrBlt', 'YrSold', 'GarageAge']])
test[test['GarageAge'] > 150][['Id','GarageYrBlt', 'YrSold', 'GarageAge']]

for value in test['GarageAge']:
    if value > 1000:
        test['GarageAge'].replace(value, 0, inplace=True)
    if value < 0:
        test['GarageAge'].replace(value, 0, inplace=True)
for value in train['GarageAge']:
    if value > 1000:
        train['GarageAge'].replace(value, 0, inplace=True)
        
###############################################################
# Binary features

train['drivepaved'] = train['PavedDrive'].apply(lambda x: 0 if x == 'N' else 1)
print('train', train['drivepaved'].value_counts(dropna=False))
test['drivepaved'] = test['PavedDrive'].apply(lambda x: 0 if x == 'N' else 1)
print('test', test['drivepaved'].value_counts(dropna=False))

train['hasutil'] = train['Utilities'].apply(lambda x: 1 if x == 'AllPub' else 0)
print(train['hasutil'].value_counts(dropna=False))
test['hasutil'] = test['Utilities'].apply(lambda x: 1 if x == 'AllPub' else 0)
print(test['hasutil'].value_counts(dropna=False))

train['CentralAir'] = enc.fit_transform(train['CentralAir'])
print(train['CentralAir'].value_counts(dropna=False))
test['CentralAir'] = enc.fit_transform(test['CentralAir'])
print(test['CentralAir'].value_counts(dropna=False))

train['haspool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
print(train['haspool'].value_counts(dropna=False))
test['haspool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
print(test['haspool'].value_counts(dropna=False))

###################################################################
# Discrete features

print(train['BsmtFullBath'].value_counts())
train['BsmtHalfBath'].value_counts(dropna=False)

# Feature engineering some new columns and then checking if the correlation increases.
train['Bathrooms'] = train['BsmtFullBath'] + train['BsmtHalfBath']*0.5 + train['FullBath'] + train['HalfBath']*0.5
test['Bathrooms'] = test['BsmtFullBath'] + test['BsmtHalfBath']*0.5 + test['FullBath'] + test['HalfBath']*0.5
train.corr()['Bathrooms']['SalePrice']

train['TotalPorchSF'] = train['WoodDeckSF'] + train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
test['TotalPorchSF'] = test['WoodDeckSF'] + test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']
print(train.corr()['TotalPorchSF']['SalePrice'])

print(train.corr()['TotRmsAbvGrd']['SalePrice'])
train['TotRmsAbvGrd'].value_counts(dropna=False)
sns.scatterplot(x='TotRmsAbvGrd', y='SalePrice', data=train)

print(train['GarageCars'].value_counts(dropna=False))
train.corr()['GarageCars']['SalePrice']

###################################################################
# Ordinal features

print(train.corr()['KitchenAbvGr']['SalePrice'])
sns.boxplot(x='KitchenQual', y='SalePrice', data=train)

train['Fireplaces'].value_counts(dropna=False)
print(train.corr()['Fireplaces']['SalePrice'])
sns.boxplot(x='FireplaceQu', y='SalePrice',data=train)

cat_to = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond','HeatingQC', 'KitchenQual',
          'FireplaceQu', 'GarageQual', 'GarageCond','PoolQC']
for x in cat_to:
    print(train[x].value_counts())
print('\n')
for x in cat_to:
    print(test[x].value_counts())
    
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for col in cat_to:
    train[col] = train[col].map(qual_map)
    print(train[col].value_counts(dropna=False))
    print(train.corr()[col]['SalePrice'])
    print('\n')
for col in cat_to:
    test[col] = test[col].map(qual_map)
    print(test[col].value_counts(dropna=False))
    print('\n')

train.corr()['SalePrice'][cat_to]

morecols = ['LotShape', 'LandSlope', 'BsmtExposure']
for x in morecols:
    print(train[x].value_counts(dropna=False))
for x in morecols:
    print(test[x].value_counts(dropna=False))

lotshape_map = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
train['LotShape'] = train['LotShape'].map(lotshape_map)
print(train['LotShape'].value_counts(dropna=False))
print(train.corr()['LotShape']['SalePrice'])
test['LotShape'] = test['LotShape'].map(lotshape_map)
print(test['LotShape'].value_counts(dropna=False))

landslope_map = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
train['LandSlope'] = train['LandSlope'].map(landslope_map)
print(train['LandSlope'].value_counts(dropna=False))
print(train.corr()['LandSlope']['SalePrice'])
test['LandSlope'] = test['LandSlope'].map(landslope_map)
print(test['LandSlope'].value_counts(dropna=False))

sns.boxplot(x='BsmtExposure', y='SalePrice', data=train)

bsmtexp_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
train['BsmtExposure'] = train['BsmtExposure'].map(bsmtexp_map)
print(train['BsmtExposure'].value_counts(dropna=False))
print(train.corr()['BsmtExposure']['SalePrice'])
test['BsmtExposure'] = test['BsmtExposure'].map(bsmtexp_map)
print(test['BsmtExposure'].value_counts(dropna=False))

train.corr().style.background_gradient(cmap='coolwarm')
train_corr = train.corr()
high_corr = train_corr.index[abs(train_corr['SalePrice']) > 0.4]
plt.figure(figsize=(10,10))
sns.heatmap(train[high_corr].corr(), annot=True)

num_cols = [c for c in train.drop('Id', axis=1) if train.dtypes[c] != object]
print(num_cols)
cat_cols = [c for c in train if train.dtypes[c] == object]
print(cat_cols)

for col in num_cols:
    sns.lmplot(col, 'SalePrice', train)
    plt.show()
    
train = train.drop(train[train['LotFrontage'] > 270].index)
sns.scatterplot(x='LotFrontage', y='SalePrice', data=train)

train = train.drop(train[train['LotArea'] > 100000].index)
print(train.corr()['LotArea']['SalePrice'])
sns.scatterplot(x='LotArea', y='SalePrice', data=train)

train[train['1stFlrSF'] > 4000]
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=train)

train = train.drop(train[(train['SalePrice'] < 200000) & (train['GrLivArea'] > 4000)].index)
print(train.corr()['GrLivArea']['SalePrice'])
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)

for col in cat_cols:
    sns.countplot(x=col, data=train)
    plt.show()
    
skewed = train[num_cols].apply(lambda x: skew(x))
skewed = skewed[skewed >0.75].index
train[skewed] = np.log1p(train[skewed])
test[skewed] = np.log1p(test[skewed])

####################################################
# Feature Preparation

from sklearn import preprocessing

# dropping columns
created = ['Id','Heating','Electrical','Exterior1st','Exterior2nd','Neighborhood','MiscFeature','Condition2','HouseStyle','RoofMatl','YearBuilt',
           'YearRemodAdd','GarageYrBlt','YrSold','PavedDrive','Utilities','PoolArea',
           'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','1stFlrSF','2ndFlrSF','LowQualFinSF','MiscVal',
           'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']

df2 = train.drop(created, axis=1)
test_df = test.drop(created, axis=1)
print(df2.shape)
print(test_df.shape)

num_cols = [c for c in df2.drop('SalePrice', axis=1) if df2.dtypes[c] != object]
print(num_cols)
cat_cols = [c for c in df2 if df2.dtypes[c] == object]
print(cat_cols)

Features = pd.get_dummies(data=df2[cat_cols])
pred_Features = pd.get_dummies(data=test_df[cat_cols])
print(Features.shape)
print(pred_Features.shape)

Features = np.concatenate([Features, np.array(df2[num_cols])], axis=1)
pred_Features = np.concatenate([pred_Features, np.array(test_df[num_cols])], axis=1)
print(Features.shape)
print(pred_Features.shape)

######################################################
# Training Model

import sklearn.metrics as sklm
from sklearn.model_selection import train_test_split

label = np.array(np.log1p(df2['SalePrice']))
X = Features
y = label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = preprocessing.StandardScaler().fit(X_train[:, 106:])
X_train[:, 106:] = scaler.transform(X_train[:, 106:])
X_test[:, 106:] = scaler.transform(X_test[:, 106:])
X_train[5]

print(X_train.shape)
print(X_test.shape)
np.argwhere(np.isnan(X_train))
np.argwhere(np.isnan(y_train))

#########################################################
# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

#########################################################
# Linear Regression

lin_mod = LinearRegression()
lin_mod.fit(X_train, y_train)
lin_pred = lin_mod.predict(X_test)
def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))
print_metrics(y_test, lin_pred, 0)

lin_final = lin_mod.predict(pred_Features)
for x in lin_final:
    print(round(np.exp(x),3))
    
##############################################################
# Gradient Boosting Regressor

GB_regressor = GradientBoostingRegressor(n_estimators = 600, random_state=0)
GB_regressor.fit(X_train, y_train)
GB_regressor.feature_importances_
gbr_pred = GB_regressor.predict(X_test)

def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))   
print_metrics(y_test, gbr_pred, 0)

gbr_final = GB_regressor.predict(pred_Features)
for x in gbr_final:
    print(round(np.exp(x),3))
