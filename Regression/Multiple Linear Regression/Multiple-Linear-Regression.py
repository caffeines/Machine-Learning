#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:58:34 2019

@author: caffeines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()

# Avoiding the dummy variables trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)

# Building the optimal model using Backword elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)), values = X, axis = 1)


