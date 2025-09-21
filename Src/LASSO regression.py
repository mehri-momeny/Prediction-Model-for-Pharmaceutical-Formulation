from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import metrics

data_path = 'data.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

# target array
target = dataset.DISINTEGRATION_TIME
dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

# Feature Scaling
sc = preprocessing.StandardScaler()
X = sc.fit_transform(dataset)
y = target.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)


#  Ridge Regression
#from sklearn.linear_model import RidgeCV
#model = Ridge()

# ElasticNet Regression
# from sklearn.linear_model import ElasticNetCV
# model = ElasticNetCV()

# LASSO Regression
model = LassoCV()
hist = model.fit(X_train, y_train)
print ( "MSE for LassoCV : ",metrics.mean_squared_error(y_test, hist.predict(X_test)))


lasso = linear_model.Lasso()
print(cross_val_score(lasso, X_train, y_train,scoring="neg_mean_squared_error", cv=3))