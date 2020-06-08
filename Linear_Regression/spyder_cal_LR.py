import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing.csv')

dataset['total_bedrooms'].fillna(537.870553, inplace= True)

dataset=dataset.drop('ocean_proximity',axis=1)

X = dataset.drop(['median_house_value'],axis=1)
y = dataset['median_house_value']
print(X.shape,y.shape)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


reg = linear_model.LinearRegression()

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

reg.fit(X_train,Y_train)


pred = reg.predict(X_test)



res = pd.DataFrame({'Predicted':pred,'Actual':Y_test})

res = res.reset_index()
res = res.drop(['index'],axis=1)

plt.plot(res[:30])
plt.legend(['Actual','Predicted'])


import statsmodels.formula.api as sm
X = np.append(arr= np.ones((20640,1)).astype(int),values=X,axis=1)
X_opt = X[: , [0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 

X_opt = X[: , [0,1,2,3,4,5,7,8]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 