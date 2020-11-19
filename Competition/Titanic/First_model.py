# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:52:54 2020

@author: Wenmo
"""

import numpy as np
import pandas as pd
import matplotlib as plt
#导入数据
train_data = pd.read_csv(r'D:\git\Machine-learning\Competition\Competition\train.csv',dtype=({'Ticket':str}))
train_data.info()


# 生存	0 =否，1 =是
# p类	机票舱位	1 = 1、2 = 2、3 = 3
# 性别	性别	
# 年龄	年岁	
# 同胞	泰坦尼克号上的兄弟姐妹/配偶数	
# 胹	泰坦尼克号上的父母/子女数量	
# 票	票号	
# 票价	旅客票价	
# 舱	机舱号	
# 出发	登船港口