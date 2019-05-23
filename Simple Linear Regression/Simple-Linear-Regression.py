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
