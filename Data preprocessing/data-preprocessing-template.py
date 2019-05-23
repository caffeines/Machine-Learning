#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 03:48:32 2019

@author: caffeines
"""

# data-preprocessing

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# missing data handling
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy="mean");
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.transform(X_test);
 























