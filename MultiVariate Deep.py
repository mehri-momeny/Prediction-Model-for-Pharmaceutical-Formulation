
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

# data_path = 'final data(reduce E ,C, F,SD)(normilized).csv'
data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)


# split data to test and train
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# target array

# pd.set_option('display.max_columns', None)
target = 'HARDNESS','FRIABILITY','Water absorption ratio','DISINTEGRATION_TIME'

train_target = train_dataset[['HARDNESS','FRIABILITY','Water absorption ratio','DISINTEGRATION_TIME']]
test_target = test_dataset[['HARDNESS','FRIABILITY','Water absorption ratio','DISINTEGRATION_TIME']]

# drop target feature from dataset
for name in target:
    train_dataset.drop([name], axis=1, inplace=True)
    test_dataset.drop([name], axis=1, inplace=True)


# Feature Scaling
sc = preprocessing.StandardScaler()
train_dataset = sc.fit_transform(train_dataset,train_target)
test_dataset = sc.transform(test_dataset)


# create model
model = keras.Sequential(name='Deep')
# input layer
model.add(keras.layers.Dense(128,kernel_initializer='normal',input_dim=train_dataset.shape[1],activation='relu'))  #kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1)
# hidden layer
model.add(keras.layers.Dense(128,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))  #,kernel_regularizer=keras.regularizers.l2(0.99)
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))
#model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))
model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.Dense(512,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))
model.add(keras.layers.Dense(128,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))



# output layer
model.add(keras.layers.Dense(4,kernel_initializer='normal',activation='linear'))

# compile the network
model.compile(loss='mean_absolute_error',optimizer=keras.optimizers.Adam(0.001))  #Adamax(0.005)  val_loss: 11.04  #Adam(0.001) val_loss: 10.96
# model.summary()


history = model.fit(train_dataset,train_target,epochs=300,batch_size=32,validation_split=0.2)


#plot the loss value
plt.plot(history.history['loss'][100:])
plt.plot(history.history['val_loss'][100:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper right')
plt.show()


y_pred = model.predict(test_dataset)
#####   evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_target, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_target, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_target, y_pred)))   #### should be less 10% of the mean value(48.97) ### np.average(test_target)

#np.average(abs(test_target))*0.1
#pd.set_option('display.max_rows', None)
pd.DataFrame(y_pred)



