#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:17:16 2019

@author: caffeines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set values
y_pred = regressor.predict(X_test)

# Visualising the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience");
plt.ylabel("Salary")
plt.show()

# Visualising the test set result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience");
plt.ylabel("Salary")
plt.show()