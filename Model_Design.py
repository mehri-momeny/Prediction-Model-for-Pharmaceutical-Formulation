from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#data_path = 'test_reduce_data.csv'
# data_path = 'final_data_withOut_SD.csv'
#data_path = 'final_data_Reduce E,C,F.csv'
data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

#dataset.drop(['num'], axis=1, inplace=True)
dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)
#dataset.drop(['DISINTEGRATION TIME(s) (SD)'], axis=1, inplace=True)
# train_dataset.drop(['DISINTEGRATION TIME(s) (SD)'], axis=1, inplace=True)



# print(dataset.to_numpy())
# # normalize the data attributes
# normalized = preprocessing.normalize(dataset)
# print("Normalized Data = ", normalized)
# print(dataset.shape)

'''
#remove outlier data
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(dataset)
mask = yhat != -1
dataset = dataset.iloc[mask][:]
'''


# dataset = dataset.iloc[:][['Wetting time (s) (mean)','Neotame','Bulk Density\n(g/cc , gm/ml , gm/cm³) (Mean)','Angle of Repose (°) (Mean)'
#                             ,'DIAMETER(mm) (Mean)','Gum','Sodium croscarmellose','Aspartame','Mannitol','Hydrogen Bond Donor Count'
#                             ,'Colloidal silicon dioxide (Aerosil)','Sodium saccharin','Topological surface area( A^2)','polyvinyl acetate'
#                             ,'Pharmaburst','Kaolin','GNSP','Silicon dioxide','FRIABILITY(%) (mean)','DISINTEGRATION_TIME']]

# split data to test and train
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# target array
#target = dataset.DISINTEGRATION_TIME
#dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

#train_dataset, test_dataset, train_target, test_target = train_test_split(dataset, target, test_size=0.2)

# target array
train_target = train_dataset.DISINTEGRATION_TIME
test_target = test_dataset.DISINTEGRATION_TIME

train_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
test_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
#test_dataset.drop(['DISINTEGRATION TIME(s) (SD)'], axis=1, inplace=True)

# create model
model = keras.Sequential()
# input layer
model.add(keras.layers.Dense(128,kernel_initializer='normal',input_dim=train_dataset.shape[1],activation='relu'))
# hidden layer
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu'))
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu'))
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128,kernel_initializer='normal',activation='relu'))
# output layer
model.add(keras.layers.Dense(1,kernel_initializer='normal',activation='linear'))

# compile the network
model.compile(loss='mean_absolute_error',optimizer=keras.optimizers.Adam(0.001),metrics=['mean_absolute_error'])  #Adamax(0.005)  val_loss: 11.04  #Adam(0.001) val_loss: 10.96
# model.summary()

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]
#
# history = model.fit(train_dataset,train_target,epochs=200,batch_size=32,validation_split=0.2,callbacks=callbacks_list)
# print('call back : ',callbacks_list)

history = model.fit(train_dataset,train_target,epochs=300,batch_size=32,validation_split=0.2)
# load weights file of the best model:
#weights_file = 'Weights-355--10.94746.hdf5'
#model.load_weights(weights_file) # load it
#model.compile(loss='mean_absolute_error',optimizer='adam', metrics=['mean_absolute_error'])


# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")



#plot the loss value
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()




score_test = model.evaluate(test_dataset, test_target, verbose=0)
score_train = model.evaluate(train_dataset, train_target, verbose=0)
print('Test loss:', score_test[0])
print('train loss:', score_train[0])


#### find the record that have wrong prediction 
#test_dataset=test_dataset.head()
# target = dataset.DISINTEGRATION_TIME
#
# dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

test_result = model.predict(test_dataset)
test_target = test_target.to_numpy()
diff = 0
List_Of_Change = []
unchange = []
for i in range(test_dataset.shape[0]):
   # print()
    print('result formulation number ',test_dataset.to_numpy()[i][0],'is : ',test_result[i],test_target[i])
    diff_pred =np.abs(test_result[i]-test_target[i])
    t = diff_pred/test_target[i]
   # print(t)
    if diff_pred > test_target[i]*0.1:
        List_Of_Change.append(int(test_dataset.to_numpy()[i][0]))
    else :
        unchange.append(int(test_dataset.to_numpy()[i][0]))
    diff = diff + t
print(diff)
print(diff/test_dataset.shape[0])

#print(test_dataset.head(10))
# print(len(dataset))
# print(len(unchange))
print(List_Of_Change)
print(unchange)



#### Scatter plot for prediction data
test_predictions_ = model.predict(test_dataset).flatten()
test_labels_ = test_target.to_numpy().flatten()
fig, ax = plt.subplots(figsize=(14,8))
plt.scatter(test_labels_, test_predictions_, alpha=0.6,
            color='#ff7043', lw=1, ec='black')
lims = [0, max(test_predictions_.max(), test_labels_.max())]
plt.plot(lims, lims, lw=1, color='#00acc1')
plt.tight_layout()


for label, x, y in zip(test_labels_, test_labels_, test_predictions_):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))


plt.show()





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


sc= StandardScaler()
x_train = sc.fit_transform(train_dataset)
x_test = sc.transform(test_dataset)

# Training the Algorithm
regressor = RandomForestRegressor(n_estimators=30,random_state=0)
# The most important parameter of the RandomForestRegressor class is the n_estimators parameter.
# This parameter defines the number of trees in the random forest

regressor.fit(x_train,train_target)
y_pred = regressor.predict(x_test)

# Evaluating the Algorithm
print('Mean Absolute Error:',metrics.mean_absolute_error(test_target, y_pred))
print('Mean Squared Erorr:',metrics.mean_squared_error(test_target, y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(test_target, y_pred)))

#np.average(y_test)


#test_dataset=test_dataset.head()
test_result_RF = regressor.predict(x_test)
diff = 0
List_Of_Change = []
for i in range(x_test.shape[0]):
   # print()
    print('result formulation number ',x_test[i][0],'is : ',test_result_RF[i],test_target[i])
    diff_pred =np.abs(test_result_RF[i]-test_target[i])
    t = diff_pred/test_target[i]
   # print(t)
    if diff_pred > test_target[i]*0.1:
        List_Of_Change.append(int(x_test[i][0]))
    diff = diff + t
print(diff)
print(diff/x_test.shape[0])

print(x_test.head(10))
print(len(x_test))
print(List_Of_Change)


####Compare Deep learning and Random Forest
plt.plot(test_target[1:40][:])
plt.plot(test_result[1:40][:])
plt.plot(test_result_RF[1:40][:])
plt.title('Model answer')
plt.ylabel('Time')
plt.xlabel('Num')
plt.legend(['real','prediction','RandomForest_pred'],loc='upper left')
plt.show()






y_pred = model.predict(test_dataset)
y_pred= np.concatenate( y_pred, axis=0 )  ####array of arrays python to array
df = pd.DataFrame({'Actual': test_target, 'Predicted': y_pred})
df1 = df.head(75)

#####   plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#####   evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_target, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_target, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_target, y_pred)))   #### should be less 10% of the mean value(48.97) ### np.average(test_target)



