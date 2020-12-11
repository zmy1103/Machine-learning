# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:31:16 2020

@author: Wenmo
"""

#线性回归简单练习
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.eye(5)

#单变量的线性回归
#需要根据城市人口数量，预测开小吃店的利润

#读入数据，并可视化
data = pd.read_csv(r'D:\git\Machine-learning\Code\Andrew Ng\Linear_regression\1data1.txt', header = None, names=['Populations', 'Profit'])
data.plot(kind='scatter', x='Populations', y='Profit', figsize=(12,8))

#梯度下降
    #1.计算J(Ѳ)，X是矩阵
    def computeCost(X, y, theta):
        inner = np.power(((X*theta.T)-y),2)
        return np.sum(inner)/(2*len(X))
    #2.初始化X和y
    data.insert(0, 'Ones', 1)#加入一列，更新theta_0
    cols = data.shape[1]
    X = data.iloc[:,:-1]
    y = data.iloc[:,cols-1:cols]
    #3.代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0]))
    # 4.梯度下降函数
    def gradientDescent(X, y, theta, alpha, iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.ravel().shape[1])#有几个参数
        cost = np.zeros(iters)
        for i in range(iters):
            error = (X * theta.T) - y
            for j in range(parameters):
                term = np.multiply(error,X[:,j])
                temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term))
            theta = temp
            cost[i] = computeCost(X, y, theta)
        return theta, cost
    #5.将学习率初始化为0.01，迭代次数为1500次
    alpha = 0.01
    iters = 1500
    #6.运行梯度下降算法来将我们的参数θ适合于训练集。
    g, cost = gradientDescent(X, y, theta, alpha, iters)
g
#预测
predict1 = [1, 3.5]*g.T
print("predict1",predict1)
predict2 = [1,7]*g.T
print("predict2",predict2)    
#进行数据可视化
x = np.linspace(data.Populations.min(), data.Populations.max(),100) 
f = g[0,0]+(g[0,1] * x)
#这是函数，可视化留个坑

#多变量线性回归
data2 = pd.read_csv(r'D:\git\Machine-learning\Code\Andrew Ng\Linear_regression\1data2.txt', header = None, names=['Size', 'Bedrooms', 'Price'])
#1.特征归一化 每类特征减去他的平均值后除以标准差
data2 = (data2-data2.mean())/data2.std()
#2.进行数据处理 
#2.1加一列常数项
data2.insert(0, 'Ones', 1)
#2.2初始化X和y
cols = data2.shape[1]
X = data2.iloc[:,:-1]
y = data2.iloc[:,cols-1:cols]
#2.3转换成matrix格式，初始化theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))
#2.4运行梯度下降算法
g, cost = gradientDescent(X, y, theta, alpha, iters)
g


#3.正规方程解法
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta
result_theta = normalEqn(X, y)
result_theta

