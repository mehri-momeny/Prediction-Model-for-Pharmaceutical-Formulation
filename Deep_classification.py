import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras



dataset = pd.read_csv('final_data_Reduce E,C,F ,SD.csv',encoding='ISO_8859_1')

dataset['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataset['DISINTEGRATION_TIME'],
                                             bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,250,450],
                                             labels = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
#
# dataset_Reduce_column = dataset.iloc[:][['Wetting time (s) (mean)','Neotame','Bulk Density\n(g/cc , gm/ml , gm/cm³) (Mean)','Angle of Repose (°) (Mean)'
#                                             ,'DIAMETER(mm) (Mean)','DISINTEGRATION_TIME_CAT']]

dataset.drop(['num'], axis=1, inplace=True)
dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)
dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
# Data Preprocessing
X = dataset.drop('DISINTEGRATION_TIME_CAT', axis=1)
y = dataset['DISINTEGRATION_TIME_CAT']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


#model design
model = keras.Sequential()
model.add(keras.layers.Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train,y_train, epochs=500, batch_size=10)



# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))