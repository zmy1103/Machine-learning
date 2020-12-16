# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:52:54 2020

@author: Wenmo
"""

import numpy as np
import pandas as pd
import random as rnd
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#导入数据
train_df = pd.read_csv(r'D:\git\Machine-learning\Competition\Titanic\train.csv',dtype=({'Ticket':str}))
train_df.info()
test_df = pd.read_csv(r'D:\git\Machine-learning\Competition\Titanic\test.csv',dtype=({'Ticket':str}))
gender_data = pd.read_csv(r'D:\git\Machine-learning\Competition\Titanic\gender_submission.csv')

combine = [train_df, test_df]
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#数据特征：离散型的变量有：Survived，Sex 和 Embarked。基于序列的有：Pclass
#连续型的数值特征有：Age，Fare。离散型数值有：SibSp，Parch
# Ticket是混合了数值型以及字母数值型的数据类型，Cabin是字母数值型数据

# 一共有891个样本
# Survived的标签是通过0或1来区分
# 大概38%的样本是survived
# 大多数乘客（>75%）没有与父母或是孩子一起旅行
# 大约30%的乘客有亲属和/或配偶一起登船
# 票价的差别非常大，少量的乘客（<1%）付了高达$512的费用
# 很少的乘客（<1%）年纪在64-80之间
# a = train_df.describe() 
# train_df.describe(percentiles=[.61, .62])#来查看数据集可以了解到生存率为 38%