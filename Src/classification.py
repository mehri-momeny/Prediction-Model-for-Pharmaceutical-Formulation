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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
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
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'
                    ,kernel_initializer=keras.initializers.RandomUniform))
    # model.add(keras.layers.ActivityRegularization(l1=0.9,l2=0.99))
    model.add(Dense(128, activation='relu',kernel_initializer=keras.initializers.RandomUniform))
    # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(256, activation='relu',kernel_initializer=keras.initializers.RandomUniform))
    # model.add(keras.layers.ActivityRegularization(l1=0.9,l2=0.99))
    # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(128, activation='relu',kernel_initializer=keras.initializers.RandomUniform))
    # model.add(keras.layers.Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(13, activation='softmax')) #10 15
    # Compile model
    # .RMSprop(0.001)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(lr=0.001), metrics=['accuracy']) #,decay=0.01
    #categorical_crossentropy  #kullback_leibler_divergence
    return model


# load dataset
#Data more 6 formulations)-990709.csv
data_path = 'normalize_data(minMax)(5-180).csv'  #normalize_data(minMax).csv  #'Data(990712).csv' #Data(990712)(without Powdr specification ).csv
# data_path = 'final data(excipient as functional category)(value)reduce SD990712.csv'
dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')

# np.average(dataframe['DISINTEGRATION_TIME'])
# dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
#                                              bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
#                                              labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])
dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
                                             bins = [5,15,25,35,45,55,65,75,85,95,105,120,150, 180],
                                             labels = [0, 1, 2 ,3 ,4, 5,6,7,8,9,10,11,12])

dataframe.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
# The frac keyword argument specifies the fraction of rows to return in the random sample,
# so frac=1 means return all rows (in random order).
dataframe = dataframe.sample(frac=1)
dataset = dataframe.values
X = dataset[:,0:72].astype(float) #38 #72
Y = dataset[:,72]
# max(dataframe['DISINTEGRATION_TIME'])
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)
Y = to_categorical(Y)  # one hot encoded
# X = preprocessing.normalize(X)
# X_train = X
# Y_train = Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=16)

# X_train , X_test = pca.PCA_Data(X_train,X_test)
#
# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y_train)
# encoded_Y = encoder.transform(Y_train)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y_train = np_utils.to_categorical(encoded_Y)
# # type(dummy_y_train)
# encoder = LabelEncoder()
# encoder.fit(Y_test)
# encoded_Y = encoder.transform(Y_test)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y_test = np_utils.to_categorical(encoded_Y)
# if dummy_y_test.shape[1] < dummy_y_train.shape[1]:
#     diff = dummy_y_train.shape[1] - dummy_y_test.shape[1]
#     dummy_y_test_ = np.zeros((Y_test.shape[0],diff))
#     dummy_y_test = np.append(dummy_y_test,dummy_y_test_,axis=1)


model = baseline_model()
history = model.fit(X_train, Y_train, batch_size=64, epochs=500, verbose=1)  #, validation_split=0.2
# estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=64, verbose=1)
## hist = estimator.fit(X_train,dummy_y_train)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X_train, dummy_y_train, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


score, acc = model.evaluate(X_test, Y_test,batch_size=64)

print('Test score:', score)
print('Test accuracy:', acc)

'''
how calculate the accuracy
Y_pred = model.predict(X_test)

# Table(dummy_y_test, Y_pred)
for i in range(len(Y_pred)):
    index = np.argmax(Y_pred[i])
    Y_pred[i][:] = 0
    Y_pred[i][index] =1
ct=0
for i in range(len(Y_test)):
    if (Y_pred[i] == Y_test[i]).all(): # Change
        ct =ct+1
        print(i)
acc = ct / len(Y_pred)       
'''

# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


'''
# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
'''


'''
# write into file
filepath = 'my_excel_file.xlsx'
## convert your array into a dataframe
df = pd.DataFrame(dataset)
df.to_excel(filepath, index=False)

'''
