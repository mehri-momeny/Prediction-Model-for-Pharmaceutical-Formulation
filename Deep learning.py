from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


def acc_compute(y_pred, y_real):
    ct = 0  # count number of record
    for i in range(len(y_real)):
        if abs(y_pred[i]-y_real[i]) <= 0.1 * y_real[i]:
            ct += 1
    return ct / len(y_pred)


def creat_model(input_dim):
    # create model
    model = keras.Sequential(name='Deep')
    # input layer
    model.add(keras.layers.Dense(128, kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1), input_dim=input_dim,
                                 activation='relu'))  # kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1)
    # hidden layer
    model.add(keras.layers.Dense(128, kernel_initializer='normal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.99)))
    model.add(keras.layers.Dense(256, kernel_initializer='normal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(
                                     0.99)))  # ,kernel_regularizer=keras.regularizers.l2(0.99)
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(256, kernel_initializer='normal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.99)))
    # model.add(keras.layers.Dense(256,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))
    model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.Dense(512,kernel_initializer='normal',activation='relu',kernel_regularizer=keras.regularizers.l2(0.99)))
    model.add(keras.layers.Dense(128, kernel_initializer='normal', activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.99)))

    # output layer
    model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='linear'))

    # compile the network
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))  # Adamax(0.005)  val_loss: 11.04  #Adam(0.001) val_loss: 10.96
    # model.summary()
    return model


def load_data(data_path):
   # data_path = 'final_data_Reduce E,C,F ,SD.csv'
    dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

    dataset.drop(['API num'], axis=1, inplace=True)
    dataset.drop(['API'], axis=1, inplace=True)
    dataset.drop(['num'], axis=1, inplace=True)
    return dataset

def data_split(data_set):
    X_train = data_set.sample(frac=0.8, random_state=0)
    X_test = data_set.drop(X_train.index)
    # target array
    y_train = X_train.DISINTEGRATION_TIME
    y_test = X_test.DISINTEGRATION_TIME

    X_train.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
    X_test.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

    return X_train,y_train,X_test,y_test

def plot_loss(data_set,target) :
    y_pred = model.predict(data_set)
    print('Mean Squared Error:', metrics.mean_squared_error(target, y_pred))
    # plot the loss value
    plt.plot(history.history['loss'][100:])
    plt.plot(history.history['val_loss'][100:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


dataset = load_data('final data(excipient as functional category)(value)reduce SD.csv')
# split data to test and train
train_dataset, train_target, test_dataset, test_target = data_split(dataset)

# Feature Scaling
sc = preprocessing.StandardScaler()
train_dataset = sc.fit_transform(train_dataset, train_target)
test_dataset = sc.transform(test_dataset)

model = creat_model(train_dataset.shape[1])
history = model.fit(train_dataset, train_target, epochs=300, batch_size=32, validation_split=0.2)



plot_loss(test_dataset, test_target)
y_pred_train = model.predict(train_dataset)
accuracy_train = acc_compute(y_pred_train, train_target.to_numpy())

y_pred_test = model.predict(test_dataset)
accuracy_test = acc_compute(y_pred_test,test_target.to_numpy())

print("accuracy of train data : ", accuracy_train)
print("accuracy of test data : ", accuracy_test)

# from sklearn.model_selection import KFold
# k_fold = KFold(10)
# for k, (train, test) in enumerate(k_fold.split(X, y)):
#     model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))  # Adamax(0.005)  val_loss: 11.04  #Adam(0.001) val_loss: 10.96
#     # model.summary()
#     history = model.fit(train_dataset[train], target[train], epochs=300, batch_size=64, validation_split=0.15)
#
#     print("MSE for Deep : ", metrics.mean_squared_error(train_dataset[test], model.predict(train_dataset[test])))


'''
target = dataset.DISINTEGRATION_TIME

dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
test_result = model.predict(dataset)
test_target = target.to_numpy()
diff = 0
List_Of_Change = []
unchange = []
for i in range(dataset.shape[0]):
   # print()
    print('result formulation number ',dataset.to_numpy()[i][0],'is : ',test_result[i],test_target[i])
    diff_pred =np.abs(test_result[i]-test_target[i])
    t = diff_pred/test_target[i]
   # print(t)
    if diff_pred > test_target[i]*0.1:
        List_Of_Change.append(int(dataset.to_numpy()[i][0]))
    else :
        unchange.append(int(dataset.to_numpy()[i][0]))
    diff = diff + t
print(diff)
print(diff/dataset.shape[0])

#print(test_dataset.head(10))
# print(len(dataset))
# print(len(unchange))
# print(len(List_Of_Change))
print(List_Of_Change)
print(unchange)
'''


'''
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
'''


'''
y_pred = model.predict(test_dataset)
#####   evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_target, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_target, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_target, y_pred)))   #### should be less 10% of the mean value(48.97) ### np.average(test_target)

#np.average(abs(test_target))*0.1

y_pred = model.predict(test_dataset)
y_pred= np.concatenate( y_pred, axis=0 )  ####array of arrays python to array
df = pd.DataFrame({'Actual': test_target, 'Predicted': y_pred})
df1 = df.head(75)

#####   plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

'''


'''


##### Desicion Tree 
from sklearn.tree import DecisionTreeRegressor
model_name = 'Decision tree'
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train_dataset,train_target)

# predicting a new value

# test the output by changing values, like 3750
y_pred_DT = regressor.predict(test_dataset)

#### Random Forest 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# Feature Scaling  *************
sc= StandardScaler()
x_train = sc.fit_transform(train_dataset)
x_test = sc.transform(test_dataset)

# Training the Algorithm
regressor = RandomForestRegressor(n_estimators=30,random_state=0)
# The most important parameter of the RandomForestRegressor class is the n_estimators parameter.
# This parameter defines the number of trees in the random forest

regressor.fit(train_dataset,train_target)
y_pred_RF = regressor.predict(test_dataset)




####Compare Deep learning and Random Forest
test_target = test_target.to_numpy()
plt.plot(test_target[0:50][:])
plt.plot(y_pred[0:50][:])
plt.plot(y_pred_RF[0:50][:])
plt.plot(y_pred_DT[0:50][:])
plt.title('Model answer')
plt.ylabel('Time')
plt.xlabel('Num')
plt.legend(['real','DeepLearning_prediction','RandomForest_pred','DecisionTree_pred'],loc='upper left')
plt.show()

'''