#1 Importing the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
#from pasta.augment import inline
from sklearn import svm
#from mlxtend.plotting import plot_decision_regions
import main_func as mf
#%matplotlib inline
#2 Importing the dataset

# dataset = pd.read_csv('final_data_Reduce E,C,F ,SD.csv',encoding='ISO_8859_1')
dataset = pd.read_csv('Data(990712).csv',encoding='ISO_8859_1')
# dataset.drop(['API num'], axis=1, inplace=True)
# dataset.drop(['API'], axis=1, inplace=True)
# dataset.drop(['num'],axis=1, inplace=True)
dataset.describe()

dataset.dropna(how='all',axis=1,inplace=True)


#pd.plotting.scatter_matrix(dataset.loc[0:,dataset.columns],c=['red','blue'],alpha=0.5,figsize=[25,25],diagonal='hist',s=200,marker='.',edgecolor='black')
#plt.show()


# Split data
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# target array
train_target = train_dataset.DISINTEGRATION_TIME
test_target = test_dataset.DISINTEGRATION_TIME

train_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)
test_dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

#3 Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(train_dataset)
# y = sc_y.transform(train_target)
train_dataset,test_dataset = mf.standardize_data(train_dataset, test_dataset)


#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here
from sklearn.svm import SVR
# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
regressor = SVR(kernel='rbf')
regressor.fit(train_dataset,train_target)
#5 Predicting a new result
y_pred = regressor.predict(test_dataset)

#6 Visualising the Support Vector Regression results
#plt.scatter(train_dataset,train_target, color = 'magenta')
#plt.plot(train_target, regressor.predict(train_dataset), color = 'green')
#plt.title('Truth or Bluff (Support Vector Regression Model)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()


y_pred_train = regressor.predict(train_dataset)
accuracy_train = mf.acc_compute(y_pred_train, train_target.to_numpy())

y_pred_test = regressor.predict(test_dataset)
accuracy_test = mf.acc_compute(y_pred_test, test_target.to_numpy())

print("accuracy of train data : ", accuracy_train)
print("accuracy of test data : ", accuracy_test)