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
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]



