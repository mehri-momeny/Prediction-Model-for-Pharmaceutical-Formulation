import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from keras.models import load_model
from matplotlib import pyplot as plt


data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

dataset.drop(['num'], axis=1, inplace=True)
dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)

target = dataset.DISINTEGRATION_TIME

dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2)
print(x_train,y_train)
print('.......')
print(x_test,y_test)

# create and train the Support vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3 , gamma=0.1)
svr_rbf.fit(x_train,y_train)

# Testing Model : Score returns the coefficient of determination R^2 of prediction.
# the best possible score is 1.0
svm_confidence = svr_rbf.score(x_test,y_test)
print('svm confidence:',svm_confidence)


#create and train the linear regression Model
lr = LinearRegression()
#train the model
lr.fit(x_train,y_train)

# Testing Model : Score returns the coefficient of determination R^2 of prediction.
# the best possible score is 1.0
lr_confidence = lr.score(x_test,y_test)
print('lr confidence:',lr_confidence)

'''
#####deep
model = load_model('model.h5')
score_test = model.evaluate(x_test, y_test, verbose=0)
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score_test[0],'Test accuracy:', score_test[1])
print('train loss:', score_train[0],'train accuracy:', score_train[1])

'''
