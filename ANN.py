from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
import math
import sklearn.utils as ut

data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

# dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)

###### Standardize dataset
# Get column names first
names = dataset.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(dataset)
dataset = pd.DataFrame(scaled_df, columns=names)
##########
# split data to test and train
train_dataset = dataset.sample(frac=0.5,random_state=0)
## bootstraping method for choosing data
# train_dataset = ut.resample(dataset,replace=True, n_samples=math.floor(0.8*len(dataset)), random_state=1)
test_dataset = dataset.drop(train_dataset.index)

# target array
train_target = train_dataset.HARDNESS# DISINTEGRATION_TIME
test_target = test_dataset.HARDNESS

train_dataset.drop(['HARDNESS'], axis=1, inplace=True)
test_dataset.drop(['HARDNESS'], axis=1, inplace=True)


'''
# Feature Scaling
sc = preprocessing.StandardScaler()
train_dataset = sc.fit_transform(train_dataset)
test_dataset = sc.transform(test_dataset)
'''
model = keras.Sequential()
model.add(keras.layers.Dense(100, input_dim=train_dataset.shape[1], activation='relu'))
# model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(50, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))) #sigmoid
# model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(25, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))#softmax
#IMPORTANT PART
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='mean_absolute_error')

hist = model.fit(train_dataset, train_target,
          batch_size=64, epochs=1000, validation_split=0.2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper right')
plt.show()

from sklearn import metrics
y_pred = model.predict(test_dataset)
#####   evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_target, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_target, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_target, y_pred)))   #### should be less 10% of the mean value(48.97) ### np.average(test_target)


