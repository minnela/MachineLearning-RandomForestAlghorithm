# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the dataset
dataset= pd.read_csv('50_Startups.csv')
X= dataset.iloc[:,1:2].values
y= dataset.iloc[:,4].values



#Fitting random forest regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators=30, random_state=0)
regressor.fit(X,y)

#predicting a new result
y_pred=regressor.predict(X)


#Visualizing the regression results
X_grid=np.arange(min(X), max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='purple')
plt.plot(X_grid,regressor.predict(X_grid), color='green')
plt.title('Random Forest Regression')
plt.xlabel('Spendings')
plt.ylabel('Profit')
plt.show()






