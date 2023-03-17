#更改成深度学习算法的demo3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout

from sklearn.cluster import DBSCAN
# 读取数据
mini_weather = pd.read_csv('micro1.csv')  # 读取微型气象站数据
std_weather = pd.read_csv('standard.csv')  # 读取标准站数据

# 校正PM2.5、PM10和NO2数据
def correct_data(row):
    pm25_diff = abs(row['PM2.5'] - std_weather.loc[row.name]['PM2.5'])
    pm10_diff = abs(row['PM10'] - std_weather.loc[row.name]['PM10'])
    no2_diff = abs(row['NO2'] - std_weather.loc[row.name]['NO2'])
    humidity_diff = abs(row['humidity'] - std_weather.loc[row.name]['humidity'])
    temperature_diff = abs(row['temperature'] - std_weather.loc[row.name]['temperature'])
    if pm25_diff > 2 or np.isnan(pm25_diff):
        row['PM2.5'] = std_weather.loc[row.name]['PM2.5']
    if pm10_diff > 2 or np.isnan(pm10_diff):
        row['PM10'] = std_weather.loc[row.name]['PM10']
    if no2_diff > 2 or np.isnan(no2_diff):
        row['NO2'] = std_weather.loc[row.name]['NO2']
    if humidity_diff > 2 or np.isnan(humidity_diff):
        row['humidity'] = std_weather.loc[row.name]['humidity']
    if temperature_diff > 2 or np.isnan(temperature_diff):
        row['temperature'] = std_weather.loc[row.name]['temperature']
    return row

mini_weather = mini_weather.apply(correct_data, axis=1)

# 发现并标记异常数据
scaler = StandardScaler()
X = scaler.fit_transform(mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']])
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
mini_weather['cluster'] = dbscan.labels_

# 实现拟合精度逐步提升

# 选择PM2.5、PM10、NO2、温度和湿度作为自变量
X = mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# 选择标准站数据的PM2.5、PM10、NO2、温度和湿度作为因变量
y = std_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 对数据进行缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据转换为CNN模型需要的输入格式
X_train_reshape = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshape = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 构建CNN模型
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshape.shape[1], X_train_reshape.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(5)
])
model.compile(optimizer='adam', loss='mse')

# 对训练数据进行拟合
model.fit(X_train_reshape, y_train, epochs=50, batch_size=32, verbose=0)

# 对测试数据进行预测
y_pred = model.predict(X_test_reshape)

# 计算R-squared值
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
import matplotlib.pyplot as plt
#安装mean_squared_error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)  # 计算均方误差
print(mse)
# 统计模型的总参数量
total_params = model.count_params()
print("Total number of parameters:", total_params)
# 将训练好的CNN模型保存到名为“model.h5”的文件中
# 将模型参数保存到磁盘中
import joblib
joblib.dump(model, 'model.pkl')