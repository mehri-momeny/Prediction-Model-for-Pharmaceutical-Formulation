import numpy as np
import pandas as pd
import keras
import main_func as mf

def creat_ANN(input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # sigmoid
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(25, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # softmax
    # IMPORTANT PART
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='mean_absolute_error')
    return model

def creat_Deep(input_dim):
    # create model
    model = keras.Sequential(name='Deep')
    # input layer
    model.add(keras.layers.Dense(128, kernel_initializer='normal', input_dim=input_dim,
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
    model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='relu'))

    # compile the network
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))  # Adamax(0.005)  val_loss: 11.04  #Adam(0.001) val_loss: 10.96
    # model.summary()
    return model


train_dataset = pd.read_csv('R code/trainingset.csv', encoding='ISO-8859–1')
test_dataset = pd.read_csv('R code/testingset.csv', encoding='ISO-8859–1')

train_dataset = pd.read_csv('other data/train_data.csv', encoding='ISO-8859–1')
test_dataset = pd.read_csv('other data/test_data.csv', encoding='ISO-8859–1')

train_target = train_dataset.y
test_target = test_dataset.y   #DISINTEGRATION_TIME

train_dataset.drop(['y'], axis=1, inplace=True)
test_dataset.drop(['y'], axis=1, inplace=True)

# DNN
Deep_model = creat_Deep(train_dataset.shape[1])
Deep_history = Deep_model.fit(train_dataset, train_target, epochs=500, batch_size=64, validation_split=0.2)
y_pred = Deep_model.predict(test_dataset)
mf.plot_loss(y_pred,test_target,Deep_history)
mf.acc_compute(y_pred,test_target)

# ANN
ANN_model = creat_ANN(train_dataset.shape[1])
ANN_history =  ANN_model.fit(train_dataset, train_target,batch_size=64, epochs=1000, validation_split=0.2)
y_pred = ANN_model.predict(test_dataset)
mf.plot_loss(y_pred,test_target,ANN_history)
