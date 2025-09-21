####  Multiple Linear Regression
#####  https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data_path = 'final_data_Reduce E,C,F ,SD.csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

# dataset.isnull().any()
dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['DISINTEGRATION_TIME'])

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# target array
train_target = train_dataset.DISINTEGRATION_TIME
test_target = test_dataset.DISINTEGRATION_TIME

train_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
test_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)



#### Train model
regressor = LinearRegression()
regressor.fit(train_dataset, train_target)

coeff_df = pd.DataFrame(regressor.coef_, train_dataset.columns, columns=['Coefficient'])
coeff_df

y_pred = regressor.predict(test_dataset)

df = pd.DataFrame({'Actual': test_target, 'Predicted': y_pred})
df1 = df.head(75)

#####   plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#####   evaluate the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(test_target, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_target, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_target, y_pred)))   #### should be less 10% of the mean value(48.97) ### np.average(test_target)



