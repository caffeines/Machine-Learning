#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 02:56:01 2019

@author: caffeines
"""

# Polinomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
