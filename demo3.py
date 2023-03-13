from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
# 读取微型气象站和标准站的数据
mini_weather = pd.read_csv('micro1.csv')  # 读取微型气象站数据
std_weather = pd.read_csv('standard.csv')  # 读取标准站数据

# 校正PM2.5、PM10和NO2数据


def correct_data(row):
    pm25_diff = abs(row['PM2.5'] - std_weather.loc[row.name]
                    ['PM2.5'])  # 计算微型气象站PM2.5数据与标准站PM2.5数据之差
    pm10_diff = abs(row['PM10'] - std_weather.loc[row.name]
                    ['PM10'])  # 计算微型气象站PM10数据与标准站PM10数据之差
    no2_diff = abs(row['NO2'] - std_weather.loc[row.name]
                   ['NO2'])  # 计算微型气象站NO2数据与标准站NO2数据之差
    
    humidity_diff = abs(row['humidity'] - std_weather.loc[row.name]
                   ['humidity'])  # 计算微型气象站NO2数据与标准站NO2数据之差

    temperature_diff = abs(row['temperature'] - std_weather.loc[row.name]
                   ['temperature'])  # 计算微型气象站NO2数据与标准站NO2数据之差
    # 如果PM2.5数据误差大于20或为NaN，则将其修正为标准站数据
    if pm25_diff > 2 or np.isnan(pm25_diff):
        row['PM2.5'] = std_weather.loc[row.name]['PM2.5']
    # 如果PM10数据误差大于20或为NaN，则将其修正为标准站数据
    if pm10_diff > 2 or np.isnan(pm10_diff):
        row['PM10'] = std_weather.loc[row.name]['PM10']
    if no2_diff > 2 or np.isnan(no2_diff):  # 如果NO2数据误差大于20或为NaN，则将其修正为标准站数据
        row['NO2'] = std_weather.loc[row.name]['NO2']
    if humidity_diff > 2 or np.isnan(humidity_diff):
        row['humidity'] = std_weather.loc[row.name]['humidity']
    if temperature_diff > 2 or np.isnan(temperature_diff):  # 如果NO2数据误差大于20或为NaN，则将其修正为标准站数据
        row['temperature'] = std_weather.loc[row.name]['temperature']
    return row


mini_weather = mini_weather.apply(
    correct_data, axis=1)  # 应用correct_data函数对每一行进行校正

# 发现并标记异常数据
scaler = StandardScaler()  # 创建StandardScaler对象，用于将数据缩放至均值为0，方差为1
# 对PM2.5、PM10和NO2数据进行缩放
X = scaler.fit_transform(mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']])

dbscan = DBSCAN(eps=0.5, min_samples=5)  # 创建DBSCAN对象，用于检测聚类
dbscan.fit(X)  # 对X进行聚类

mini_weather['cluster'] = dbscan.labels_  # 将聚类结果存储到mini_weather中

# 实现拟合精度逐步提升

# 选择PM2.5、PM10、NO2、温度和湿度作为自变量
X = mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# 选择标准站数据的PM2.5、PM10、NO2、温度和湿度作为因变量
y = std_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)  # 将数据划分为训练集和测试集



# while mse > 0.01:  # 当均方误差大于0.01时，重新训练模型
regressor = MLPRegressor(hidden_layer_sizes=(100,100), activation='relu', solver='adam', max_iter=500)    
regressor.fit(X_train, y_train)  # 对训练数据进行模型拟合

y_pred = regressor.predict(X_test)  # 对测试数据进行预测
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
print(mse)

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.scatterplot(x='temperature', y='humidity', hue='cluster', data=mini_weather) # 绘制温度和湿度散点图，并根据聚类结果进行着色
# plt.show() # 显示图形

import joblib

# 将训练好的MLPRegressor模型保存到名为“model.pkl”的文件中
joblib.dump(regressor, 'model.pkl')