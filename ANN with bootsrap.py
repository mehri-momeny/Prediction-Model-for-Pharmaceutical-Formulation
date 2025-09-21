from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
import math
import sklearn.utils as ut
from sklearn import metrics

def make_model(input_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_dim=input_size, activation='relu'))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # sigmoid
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dense(25, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))  # softmax
    # IMPORTANT PART
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='mean_absolute_error')
    return model


data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

dataset.drop(['API num'], axis=1, inplace=True)
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
# train_dataset = dataset.sample(frac=0.8,random_state=0)

## bootstraping method for choosing data
n_iterations = 100
n_size = int(len(dataset) * 0.50)

# run bootstrap
stats = list()
for i in range(n_iterations):
    # prepare train and test sets
    train_dataset = ut.resample(dataset,replace=True, n_samples=n_size, random_state=1)
    test_dataset = dataset.drop(train_dataset.index)
    # target array
    train_target = train_dataset.DISINTEGRATION_TIME
    test_target = test_dataset.DISINTEGRATION_TIME
    train_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
    test_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
    # fit model
    model = make_model(train_dataset.shape[1])
    hist = model.fit(train_dataset, train_target,batch_size=64, epochs=300, validation_split=0.2)
    # evaluate model
    predictions = model.predict(test_dataset)
    score =metrics.mean_absolute_error(test_target, predictions)
    print(score)
    stats.append(score)
# plot scores
plt.hist(stats)
plt.show()

# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


