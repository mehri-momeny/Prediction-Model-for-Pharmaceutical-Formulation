import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt



# dataframe = pd.read_csv('Data/final data(excipient as functional category)(value)reduce SD990709.csv', encoding='ISO-8859–1')
# dataframe = pd.read_csv('R code/real Data normal.csv', encoding='ISO-8859–1')
dataframe = pd.read_csv('../Data/Data as Ref type(990814).csv', encoding='ISO-8859–1')
dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])
# dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
#                                              bins = [0,60, 120, 180, 450],
#                                              labels = [1, 2, 3, 4])

dataframe.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset = dataframe.values
X = dataset[:,0:50].astype(float)
Y = dataset[:,87]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
'''
data_path_train = 'R code/trainingset.csv'
data_path_test = 'R code/testingset.csv'
dataframe_train = pd.read_csv(data_path_train, encoding='ISO-8859–1')
dataframe_test = pd.read_csv(data_path_test, encoding='ISO-8859–1')
# dataset = mf.normalize_data(dataset)

dataframe_train['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe_train['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe_train.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset = dataframe_train.values
X_train = dataset[:,0:79].astype(float)
Y_train = dataset[:,79]

dataframe_test['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe_test['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe_test.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset = dataframe_test.values
X_test = dataset[:,0:79].astype(float)
Y_test = dataset[:,79]
'''
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
# # Feature Scaling  *************
# sc= StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# Training the Algorithm
regressor = RandomForestClassifier(n_estimators=40,random_state=10) #,max_features =38
# The most important parameter of the RandomForest class is the n_estimators parameter.
# This parameter defines the number of trees in the random forest

regressor.fit(X_train,Y_train)


y_pred_RF = regressor.predict(X_train)
print ( metrics.accuracy_score(Y_train, y_pred_RF))
# Make predictions for the test set
y_pred_test = regressor.predict(X_test)
print(metrics.accuracy_score(Y_test, y_pred_test))
