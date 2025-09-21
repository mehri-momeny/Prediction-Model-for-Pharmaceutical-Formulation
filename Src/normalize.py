import pandas as pd
import numpy as np
from sklearn import preprocessing

data_path = 'final data(reduce E ,C, F,SD)(normilized).csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')
print(dataset)
print(dataset.to_numpy())
# normalize the data attributes
normalized = preprocessing.normalize(dataset)
print("Normalized Data = ", normalized)
print(dataset.shape)
np.savetxt("t.csv",normalized, delimiter=",")



###  display full Dataframe i.e. print all rows & columns without truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
print(dataset.corr()["DISINTEGRATION_TIME"].abs().sort_values(ascending=False))