# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:43:48 2020

@author: Wenmo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report#这个包是评价报告
weight = loadmat(r"D:\git\Machine-learning\Code\Andrew Ng\3_Neural network\ex3weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']
theta1.shape, theta2.shape
data = loadmat(r"D:\git\Machine-learning\Code\Andrew Ng\3_Neural network\ex3data1.mat")
X = data['X']
X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(X.shape[0]), axis=1))
y2 = np.matrix(data['y'])
X2.shape, y2.shape
a1 = X2
z2 = a1 * theta1.T
z2.shape
a2 = sigmoid(z2)
a2.shape
a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
z3 = a2 * theta2.T
z3.shape
a3 = sigmoid(z3)
a3
y_pred2 = np.argmax(a3, axis=1) + 1
y_pred2.shape
print(classification_report(y2, y_pred))