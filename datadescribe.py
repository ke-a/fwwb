#看看数据有没有什么特征，方便以后优化算法。
import pandas as pd
# 创建一个DataFrame
mini_weather = pd.read_csv('micro1.csv')  # 读取微型气象站数据
std_weather = pd.read_csv('standard.csv')  # 读取标准站数据
x = mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
y = std_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# 使用 describe() 函数输出数据的描述性统计信息
# print(x.describe())
# print(y.describe())
import matplotlib.pyplot as plt
plt.scatter(x['PM2.5'], x['temperature'])
plt.show()