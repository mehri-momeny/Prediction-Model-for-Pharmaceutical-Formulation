from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
import math
import sklearn.utils as ut
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def make_model(input_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_dim=input_size, activation='relu'))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # sigmoid
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(25, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # softmax
    # IMPORTANT PART
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.categorical_crossentropy,metrics='accuracy')
    return model

# data_path = 'final data(reduce E ,C, F,SD)(normilized).csv'
data_path = 'Data(990712).csv' # 'final data(excipient as functional category)(value)reduce SD.csv'
dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')

dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
# dataset = dataframe
# dataset = dataframe.values
# X = dataset[:,0:72].astype(float) #38 #72
# Y = dataset[:,72]
# X = preprocessing.normalize(X)
# X_train = X
# Y_train = Y
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)

# split data to test and train
# train_dataset = dataset.sample(frac=0.8,random_state=0)

## bootstraping method for choosing data
n_iterations = 100
n_size = int(len(dataframe) * 0.80)
# train_dataset = dataframe.sample(frac=0.8,random_state=0)

# run bootstrap
stats = list()
for i in range(n_iterations):
    # prepare train and test sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)
    #train_dataset = ut.resample(dataset, replace=True, n_samples=1585, random_state=1)
    train = ut.resample(dataframe,replace=True, n_samples=n_size, random_state=1)
    test = dataframe.drop(train.index)
    # target array
    train = train.values
    train_dataset = train[:,0:72].astype(float) #38 #72
    train_target = train[:,72]

    test = test.values
    test_dataset = test[:,0:72].astype(float) #38 #72
    test_target = test[:,72]

    # one hot encoding
    encoder = LabelEncoder()
    encoder.fit(train_target)
    encoded_Y = encoder.transform(train_target)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = np_utils.to_categorical(encoded_Y)
    # type(dummy_y_train)
    encoder = LabelEncoder()
    encoder.fit(test_target)
    encoded_Y = encoder.transform(test_target)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_test = np_utils.to_categorical(encoded_Y)

    # fit model
    model = make_model(train_dataset.shape[1])
    hist = model.fit(train_dataset, dummy_y_train,batch_size=64, epochs=300, validation_split=0.2)
    # evaluate model
    score, acc = model.evaluate(test_dataset, dummy_y_test, batch_size=32)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    # predictions = model.predict(test_dataset)
    score =acc
    print(score)
    stats.append(score)
# plot scores
plt.hist(stats)
plt.show()

# confidence intervals
'''
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
'''
