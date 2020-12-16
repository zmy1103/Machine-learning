## Titannic

[TOC]

### 总体思路：

1. 运用pandas库对数据进行处理，用matplotlib进行数据可视化
2. 建立数据分析模型：XXX
3. 对结果进行验证评估

### 数据处理：

- 导入数据with Pandas
- 清洗数据
- 对数据进行可视化

1. 数据特征

   ![image-20201115201909816](https://tvax1.sinaimg.cn/large/005IQUPRgy1gkq4hvs0xhj30dk0do74t.jpg)

   训练数据集一共891条

2. 泰坦尼克号基本特征工程

   - 职称：[‘Name’]->['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                         'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                         'Don', 'Jonkheer']
   - 机舱 [‘Cabin’] = [‘A’,’B’.......’Unknown’]
   - 家庭规模 df[‘Family_Size’] = df[‘SibSp’] + df[‘Parch’]
   - 年龄*类别 交互术语地方
   - 每人票价

### 数据分析模型：

### 遇到的问题及积累经验：

- 文件路径问题

  ```python
  import os
  f_path = os.path.join(*['D:', 'git', 'Machine-learning', 'Competition', 'train.csv'])
  df = pd.read_csv(f_path)
  
  import os
  import pandas as pd
  script_dir = os.getcwd()
  file = 'example_file.csv'
  data = pd.read_csv(os.path.normcase(os.path.join(script_dir, file)))
  ```

   导入os库可以自动生成格式，但仍然会出现文件读入失败

  ```python
  解决：train_data = pd.read_csv(r'D:\git\Titanic\train.csv',dtype=({'Ticket':str}))
  ```

  Python原始字符串是通过在字符串文字前加上'r'或'R'来创建的。Python原始字符串将反斜杠视为文字字符。当我们想要一个包含反斜杠的字符串并且不希望将其视为转义字符时，这很有用。

  

  

