import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data_path = 'normalize_data(minMax).csv'
dataset = pd.read_csv(data_path, encoding='ISO-8859–1')

# target array
target = dataset.DISINTEGRATION_TIME
dataset.drop(['DISINTEGRATION_TIME'], axis=1, inplace=True)

# Feature Scaling
# sc = preprocessing.StandardScaler()
# X = sc.fit_transform(dataset)
# y = target.to_numpy()
X = dataset
y = target.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)

# X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)




# X_train, X_test = dataset[:200], dataset[200:]
# y_train, y_test = target[:200], target[200:]
est = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))



#####  Voting Regressor #####
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn import metrics
import Deep_learning as dp

# Training classifiers
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
reg1_model = reg1.fit(X_train, y_train)
print ( "MSE for GradientBoostingRegressor : ",mean_squared_error(y_test, reg1_model.predict(X_test)))

reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
reg2_model = reg2.fit(X_train, y_train)
# reg2_model = reg2.fit(X_train, y_train)
print ( "MSE for RandomForestRegressor : ",mean_squared_error(y_test, reg2_model.predict(X_test)))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, reg2_model.predict(X_test)))


reg3 = LinearRegression()
reg3_model = reg3.fit(X_train, y_train)
print ( "MSE for LinearRegression : ",mean_squared_error(y_test, reg3_model.predict(X_test)))


## deep
deep = dp.creat_model(X_train.shape[1])
deep_model = deep.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2,verbose=0)
print ( "MSE for LinearRegression : ",mean_squared_error(y_test, deep.predict(X_test)))

ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3),('dp', deep)])
ereg = ereg.fit(X_train, y_train)

print ( "MSE for VotingRegressor : ",mean_squared_error(y_test, ereg.predict(X_test)))


xt = X[:20]
yt = y[:20]

pred1 = reg1_model.predict(xt)
pred2 = reg2_model.predict(xt)
pred3 = reg3_model.predict(xt)
pred4 = ereg.predict(xt)
# Plot the results
### Finally, we will visualize the 20 predictions.
# The red stars show the average prediction made by VotingRegressor.
plt.figure()
plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'ys', label='LinearRegression')
plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')
plt.plot(yt, 'r^', ms=10, label='RealValues')

plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Regressor predictions and their average')

plt.show()

#### K Fold Cross validation
from sklearn.model_selection import KFold
k_fold = KFold(10)
for k, (train, test) in enumerate(k_fold.split(X, y)):
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg2_model = reg2.fit(X[train], y[train])
    print ( "MSE for RandomForestRegressor : ",mean_squared_error(y[test], reg2_model.predict(X[test])))
