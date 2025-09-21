import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)


# split data to test and train
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# target array
train_target = train_dataset.DISINTEGRATION_TIME
test_target = test_dataset.DISINTEGRATION_TIME

train_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
test_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)


model_name = 'Decision tree'
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train_dataset,train_target)

# predicting a new value

# test the output by changing values, like 3750
y_pred_DT = regressor.predict(test_dataset)


#####   evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_target, y_pred_DT))
print('Mean Squared Error:', metrics.mean_squared_error(test_target, y_pred_DT))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_target, y_pred_DT)))   #### should be less 10% of the mean value(48.97) ### np.average(test_target)

#np.average(abs(test_target))*0.1


#y_pred = regressor.predict(test_dataset)
# y_pred= np.concatenate( y_pred, axis=0 )  ####array of arrays python to array
df = pd.DataFrame({'Actual': test_target, 'Predicted': y_pred_DT})
df1 = df.head(75)

#####   plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()