

##### این روش خوبی نبود و پاسخ خوبی نداشت

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


dataset = pd.read_csv('data.csv',encoding='ISO_8859_1')

# dataset['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataset['DISINTEGRATION_TIME'],
#                                              bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,250,450],
#                                              labels = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])

# dataset_Reduce_column = dataset.iloc[:][['Wetting time (s) (mean)','Neotame','Bulk Density\n(g/cc , gm/ml , gm/cm³) (Mean)','Angle of Repose (°) (Mean)'
#                                             ,'DIAMETER(mm) (Mean)','DISINTEGRATION_TIME_CAT']]
dataset['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataset['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataset.drop(['API'], axis=1, inplace=True)
dataset.drop(['num'], axis=1, inplace=True)
dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
# Data Preprocessing
y = dataset['DISINTEGRATION_TIME_CAT']
X = dataset.drop('DISINTEGRATION_TIME_CAT', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Training the Algorithm
svclassifier = SVC(kernel='sigmoid')   #### linear ,  Polynomial Kernel ,Gaussian Kernel, Sigmoid Kernel
svclassifier.fit(X_train, y_train)

# Making Predictions
y_pred = svclassifier.predict(X_test)

# Evaluating the Algorithm
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
