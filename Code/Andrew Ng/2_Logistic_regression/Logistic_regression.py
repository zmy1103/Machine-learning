# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:07:13 2020

@author: Wenmo
"""

#逻辑回归、正则化
# 设想你是大学相关部分的管理者
# 想通过申请学生两次测试的评分，来决定他们是否被录取。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv(r"D:\git\Machine-learning\Code\Andrew Ng\2_Logistic_regression\data1.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#实现sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#实现代价函数
def cost(X, y, theta):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
#数据处理
data.insert(0, 'Ones', 1)
cols = data.shape[1]#列数
X = data.iloc[:,:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)
X = np.array(X.values)
y = np.array(y.values)
cost(X, y, theta)

#实现梯度计算
def gradientdescent(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

gradientdescent(theta, X, y)

# result = opt.fmin_tnc(func=cost, x0=theta,fprime=gradientdescent, args=(X, y))
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradientdescent)











