
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# data_path = 'normalize_data(minMax)target_normalize.csv'#normalize_data(minMax).csv  #'Data(990712).csv' #Data(990712)(without Powdr specification ).csv
data_path = 'final data(excipient as functional category)(value)reduce SD990712.csv'
dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')


C_mat = dataframe.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()
C_mat['DISINTEGRATION_TIME']
'''
# write into file
filepath = 'Correlations_Excipient based on functionality.xlsx'
## convert your array into a dataframe
df = pd.DataFrame(C_mat)
df.to_excel(filepath, index=False)

'''

dataframe['DISINTEGRATION_TIME_CAT']=pd.cut(x = dataframe['DISINTEGRATION_TIME'],
                                             bins = [0,10, 20, 30, 40, 50, 60, 90, 120, 180,450],
                                             labels = [1, 2, 3, 4, 5, 6, 7, 8, 9,10])

dataframe.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)


# dataset = dataframe.values
# X = dataset[:,0:57].astype(float) #38 #72
# Y = dataset[:,57]


C_mat = dataframe.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

# dataframe.corr()
# plt.matshow(dataframe.corr())
# plt.show()
