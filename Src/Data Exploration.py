
#import plotnine
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest



outliers = []
def detect_outlier(data_1):
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


#matplotlib inline
'''
print(ggplot(mpg)         # defining what data to use
 + aes(x='displ', y='hwy', color='class')    # defining what variable to use
 + geom_point() # defining the type of plot to use
#+ coord_flip()
 + labs(title='Engine Displacement vs. Highway Miles per Gallon', x='Engine Displacement, in Litres', y='Highway Miles per Gallon')
)
'''
data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')
dataset_head = dataset.head(100)
#print(ggplot(dataset_head)         # defining what data to use
# + aes(x='DISINTEGRATION_TIME', y='HARDNESS      (kg/cm2 , kp , kgF)   (mean)')    # defining what variable to use   , color='API'
# + geom_point() # defining the type of plot to use
#+ coord_flip()
# + labs(title='Disintegration time vs Hardness', x='Disintegration time', y='Hardness')
#)


#outlier_datapoints = detect_outlier(dataset.head().iloc[1][1])
#print(outlier_datapoints)

#import sklearn.ensemble.IsolationForest


dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)
print(dataset.shape)

iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(dataset)
mask = yhat != -1
dataset = dataset.iloc[mask][:]
print(dataset.shape)


dataset.drop(['num'], axis=1, inplace=True)

#### Factor Analysis

# Removing Constant features
# constant_filter = VarianceThreshold(threshold=0)

# Removing Quasi-Constant Features Using Variance Threshold
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(dataset)
len(dataset.columns[constant_filter.get_support()])

constant_columns = [column for column in dataset.columns
                    if column not in dataset.columns[constant_filter.get_support()]]

print(len(constant_columns))

for column in constant_columns:
    print(column)

dataset = constant_filter.transform(dataset)
# import sys
# np.set_printoptions(threshold=sys.maxsize)
# #pd.set_option('display.max_columns', None)
# j = 0
# for col_name in dataset.columns:
#     j+=1
#     plt.scatter(dataset.iloc[:,j], dataset['DISINTEGRATION_TIME'])
#     plt.xlabel(col_name)
# plt.show()

# Removing Duplicate Features
# Removing Duplicate Features using Transpose
train_features_T = dataset.T
train_features_T.shape

print(train_features_T.duplicated().sum())
unique_features = train_features_T.drop_duplicates(keep='first').T

duplicated_features = [dup_col for dup_col in dataset.columns if dup_col not in unique_features.columns]
duplicated_features
####----####

####### Correlation #####
num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(dataset.select_dtypes(include=num_colums).columns)
paribas_data = dataset[numerical_columns]

correlated_features = set()
correlation_matrix = paribas_data.corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

dataset.drop(labels=correlated_features, axis=1, inplace=True)


#####-------------   PCA   --------------######

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# standard scalar normalization to normalize our feature set.
sc = StandardScaler()
X_train = sc.fit_transform(dataset)


# Applying PCA
pca = PCA(n_components=20)  #n_components=1
X_train = pca.fit_transform(X_train)
X_train.shape
pca.explained_variance_ratio_


