from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# data_path = 'final data(reduce E ,C, F,SD)(normilized).csv'
data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)

# split data to test and train
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# target array
train_target = train_dataset.DISINTEGRATION_TIME
test_target = test_dataset.DISINTEGRATION_TIME

train_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
test_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)



# Feature Scaling
sc = preprocessing.StandardScaler()
train_dataset = sc.fit_transform(train_dataset)
test_dataset = sc.transform(test_dataset)



# rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=30, step=10, verbose=5)
# rfe_selector.fit(train_dataset, train_target)
# rfe_support = rfe_selector.get_support()
# rfe_feature = train_dataset.loc[:,rfe_support].columns.tolist()
# print(str(len(rfe_feature)), 'selected features')




# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# # generate dataset
# X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# define feature selection
fs = SelectKBest(score_func=f_regression, k=40)
# apply feature selection
X_selected = fs.fit_transform(train_dataset, train_target)
print(X_selected.shape)
train_dataset = X_selected


'''
# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# # generate dataset
# X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# define feature selection
fs = SelectKBest(score_func=f_regression, k=20)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)
X = X_selected

'''