# 运用wine变量进行线性回归并预测葡萄酒的评分

import pandas as pd
import statsmodels.api as sma
import matplotlib.pyplot as plt

inputExcel = 'winequality-both.csv'
wine = pd.read_csv(inputExcel)
print(wine.columns)
print(wine)
print(wine.iloc[:, 1:])
# 、将列标题的空格用下划线替代
wine.columns = wine.columns.str.replace(' ', '_')

# 将wine的最后10行数据并剔除'tpye'字段和'quality'字段作为测试集
wineLast = wine.tail(10)
# 生成自变量并添加常数项
wineTest = sma.add_constant(wineLast[wineLast.columns.difference(['type', 'quality'])])

# 将wine第一行到倒数第11行的数据作为训练集,并为自变量加上常数项
wineHead = wine.iloc[0:-10, :]
# 生成自变量并添加常数项
wineTrainInd = sma.add_constant(wineHead[wineHead.columns.difference(['type', 'quality'])])
# 生成因变量
wineTrainDep = wineHead['quality']

# 调用statsmodels模块的api.ols进行最小二乘线性回归
lm = sma.OLS(wineTrainDep, wineTrainInd)
# 生成回归结果
res = lm.fit()

# 显示模型结果
print(res.summary())
print(res.params[0])
# 输出测试结果
predict = res.predict(wineTest)
print(predict)
