import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def load_data(data_path):
   # data_path = 'final_data_Reduce E,C,F ,SD.csv'
    dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

    dataset.drop(['API num'], axis=1, inplace=True)
    dataset.drop(['API'], axis=1, inplace=True)
    dataset.drop(['num'], axis=1, inplace=True)
    return dataset


def load_categorical_data(data_path,att_num,data_path_sim=''):
    if data_path_sim == '':
        dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')
    else:
        dataframe_sim = pd.read_csv(data_path_sim, encoding='ISO-8859–1')
        dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')
        dataframe = dataframe.append(dataframe_sim,sort = False)
    # dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
    #                                              bins = [0, 16,21,25,30,35,41,50,62,93, 450],
    #                                              labels = [0,1, 2, 3, 4, 5,6,7,8,9])
    dataframe['DISINTEGRATION_TIME_CAT'] = pd.cut(x=dataframe['DISINTEGRATION_TIME'],
                                                  bins=[0, 24, 35, 55, 432],
                                                  labels=[0, 1, 2, 3])
    dataframe.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
    # The frac keyword argument specifies the fraction of rows to return in the random sample,
    # so frac=1 means return all rows (in random order).

    # dataframe.describe()
    dataset = dataframe.sample(frac=1, random_state=2)  # 2,replace=True
    dataset = dataset.values
    X = dataset[:, 0:att_num].astype(float)  # 38 #71
    Y = dataset[:, att_num].astype(int)
    return X, Y


def data_split(data_set):
    X_train = data_set.sample(frac=0.8, random_state=0)
    X_test = data_set.drop(X_train.index)
    # target array
    y_train = X_train.DISINTEGRATION_TIME
    y_test = X_test.DISINTEGRATION_TIME

    X_train.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
    X_test.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

    return X_train,y_train,X_test,y_test


'''
# Feature Scaling
sc = preprocessing.StandardScaler()
train_dataset = sc.fit_transform(train_dataset, train_target)
test_dataset = sc.transform(test_dataset)
'''

def normalize_data(dataset):
    # normalize the data attributes
    # Get column names first
    names = dataset.columns
    normalized = preprocessing.normalize(dataset)
    # print("Normalized Data = ", normalized)
    normalized = pd.DataFrame(normalized, columns=names)
    return normalized

def standardize_data(train,test):
    # Get column names first
    names = train.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    standard_train = pd.DataFrame(train, columns=names)
    standard_test = pd.DataFrame(test,columns=names)
    return standard_train,standard_test


def acc_compute(y_pred, y_real):
    ct = 0  # count number of record
    for i in range(len(y_real)):
        if abs(y_pred[i]-y_real[i]) <= 0.2 * y_real[i]: #10 :  # 0.1 * y_real[i]  ##10
            ct += 1
    return ct / len(y_pred)


def plot_loss(history) :
    # plot the loss value
    plt.plot(history.history['loss'][100:])
    plt.plot(history.history['val_loss'][100:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def output_excel(size,test_result,test_target):
    out = []
    for i in range(size):
        out.append([test_result[i, 0], test_target[i]])
    out = np.array(out)

    # write into file
    filepath = 'outout.xlsx'
    ## convert your array into a dataframe
    df = pd.DataFrame(out)
    df.to_excel(filepath, index=False)


