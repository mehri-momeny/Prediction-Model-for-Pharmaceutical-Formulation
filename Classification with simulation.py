# multi-class classification with Keras
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import PCA as pca

# create a scatter plot of points colored by class value
def plot_samples(X, y, classes=10):
    # plot points for each class
    for i in range(classes):
        # select indices of points with each class label
        samples_ix = np.where(y == i)
        # plot points for this class with a given color
        plt.scatter(X[samples_ix, 0], X[samples_ix, 1])


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='tanh'))
    # model.add(keras.layers.ActivityRegularization(l1=0.9,l2=0.99))
    model.add(Dense(256, activation='tanh'))
    # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(256, activation='tanh',kernel_regularizer=keras.regularizers.l2(0.99)))
    # model.add(keras.layers.ActivityRegularization(l1=0.9,l2=0.99))
    # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(128, activation='tanh'))
    # model.add(keras.layers.ActivityRegularization(l1=0.9))
    # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(256, activation='tanh'))#,kernel_regularizer=keras.regularizers.l2(0.99)))
    # # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(128, activation='tanh'))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(Dense(128, activation='tanh'))
    model.add(Dense(9, activation='softmax')) #10 15
    # Compile model
    # .RMSprop(0.001)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(lr=0.001), metrics=['accuracy']) #,decay=0.01
    return model


# load dataset
#Data more 6 formulations)-990709.csv
# data_path = 'normalize_data(minMax)(5-180).csv'#normalize_data(minMax).csv  #'Data(990712).csv' #Data(990712)(without Powdr specification ).csv
# data_path = 'final data(excipient as functional category)(value)reduce SD990712.csv'
data_path = 'R code/new_data2.csv'
dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')
'''
dataset = dataframe.sample(frac=0.7, random_state=0)
dataset_test = dataframe.drop(dataset.index)
# write into file
filepath = 'Dataset70.xlsx'
filepath_test = 'Dataset_test.xlsx'
## convert your array into a dataframe
df = pd.DataFrame(dataset)
df.to_excel(filepath, index=False)

df_test = pd.DataFrame(dataset_test)
df_test.to_excel(filepath_test, index=False)
'''
# np.average(dataframe['DISINTEGRATION_TIME'])
dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])
# dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
#                                              bins = [0,5,10,15, 20,25, 30,35, 40,45, 50, 60, 90, 120, 180,450],
#                                              labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15])
# dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
#                                              bins = [0,15, 30, 45, 60, 75, 90, 120, 180, 450],
#                                              labels = [1, 2, 3, 4, 5, 6, 7, 8, 9])
# dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
#                                              bins = [0,20, 40, 60, 80, 100, 120, 180, 450],
#                                              labels = [1, 2, 3, 4, 5, 6, 7, 8])   # 64%
# dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
#                                              bins = [0,30, 60,90, 120,150, 180],
#                                              labels = [1, 2, 3 ,4 ,5, 6])

dataframe.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset = dataframe.values
X = dataset[:,0:72].astype(float) #38 #72
Y = dataset[:,72]
# max(dataframe['DISINTEGRATION_TIME'])

# X = preprocessing.normalize(X)
# X_train = X
# Y_train = Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)
'''
############################
dataframe_test = pd.read_csv('Dataset_test.csv', encoding='ISO-8859–1')

dataframe_test['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe_test['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe_test.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset_test = dataframe_test.values
X_test = dataset_test[:,0:72].astype(float) #38 #72
Y_test = dataset_test[:,72]


'''
'''
data_path_train = 'R code/trainingset(readyforcheck).csv'
data_path_test = 'R code/testingset(readyforcheck).csv'
dataframe_train = pd.read_csv(data_path_train, encoding='ISO-8859–1')
dataframe_test = pd.read_csv(data_path_test, encoding='ISO-8859–1')
# dataset = mf.normalize_data(dataset)

dataframe_train['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe_train['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe_train.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset = dataframe_train.values
X_train = dataset[:,0:79].astype(float) # 79
Y_train = dataset[:,79]

# X_train = preprocessing.normalize(X_train)

dataframe_test['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe_test['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe_test.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
dataset = dataframe_test.values
X_test = dataset[:,0:79].astype(float)
Y_test = dataset[:,79]

# X_test = preprocessing.normalize(X_test)
'''

# X_train , X_test = pca.PCA_Data(X_train,X_test)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y)
# type(dummy_y_train)
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y = encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test = np_utils.to_categorical(encoded_Y)
if dummy_y_test.shape[1] < dummy_y_train.shape[1]:
    diff = dummy_y_train.shape[1] - dummy_y_test.shape[1]
    dummy_y_test_ = np.zeros((Y_test.shape[0],diff))
    dummy_y_test = np.append(dummy_y_test,dummy_y_test_,axis=1)

'''
# write into file
filepath = 'my_excel_file.xlsx'
## convert your array into a dataframe
df = pd.DataFrame(dataset)
df.to_excel(filepath, index=False)

'''
model = baseline_model()
hist = model.fit(X_train, dummy_y_train, batch_size=128, epochs=1000, verbose=1)  #, validation_split=0.2
# estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=64, verbose=1)
## hist = estimator.fit(X_train,dummy_y_train)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X_train, dummy_y_train, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

'''
plt.plot(hist.history['accuracy'][100:])
plt.plot(hist.history['val_accuracy'][100:])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(hist.history['loss'][100:])
plt.plot(hist.history['accuracy'][100:])
plt.title('Model loss & Acc')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'acc'], loc='upper right')
plt.show()
'''

score, acc = model.evaluate(X_test, dummy_y_test,batch_size=64)

print('Test score:', score)
print('Test accuracy:', acc)



