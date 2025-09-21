import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pyplot as plt


dataset = pd.read_csv('Data/final_data_Reduce E,C,F ,SD.csv',encoding='ISO-8859–1')
# dataset = pd.read_csv('Data/Data as Ref type(990814).csv', encoding='ISO-8859–1')
# dataset = pd.read_csv('R code/real Data normal.csv', encoding='ISO-8859–1')
dataset.head()

#dataset.drop(['num'], axis=1, inplace=True)
dataset.drop(['API num'], axis=1, inplace=True)
dataset.drop(['API'], axis=1, inplace=True)


# Preparing Data For Training
target = dataset.DISINTEGRATION_TIME
dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

x_train , x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=0)

# Feature Scaling  *************
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the Algorithm
regressor = RandomForestRegressor(n_estimators=30,random_state=0)
# The most important parameter of the RandomForestRegressor class is the n_estimators parameter.
# This parameter defines the number of trees in the random forest

regressor.fit(x_train,y_train)
y_pred_RF = regressor.predict(x_test)

# Evaluating the Algorithm
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred_RF))
print('Mean Squared Erorr:',metrics.mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)))

#np.average(y_test)

#y_pred_RF = regressor.predict(test_dataset)
# y_pred_RF= np.concatenate( y_pred_RF, axis=0 )  ####array of arrays python to array
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_RF})
df1 = df.head(75)

#####   plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



