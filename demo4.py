#针对实时模型精度拟合的实验品
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import time

# 实时监控微型气象站数据，并自动运行校正和聚类算法来发现异常数据
while True:
    mini_weather = pd.read_csv('micro1.csv')  # 读取微型气象站数据
    std_weather = pd.read_csv('standard.csv')  # 读取标准站数据

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

    # 实时缩放和聚类数据，并发现异常点
    scaler = StandardScaler()  # 创建StandardScaler对象，用于将数据缩放至均值为0，方差为1
    # 对PM2.5、PM10和NO2数据进行缩放
    X = scaler.fit_transform(mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']])

    dbscan = DBSCAN(eps=0.5, min_samples=5)  # 创建DBSCAN对象，用于检测聚类
    dbscan.fit(X)  # 对X进行聚类

    mini_weather['cluster'] = dbscan.labels_  # 将聚类结果存储到mini_weather中

    # 按时间顺序拆分训练集和测试集
    mini_weather = mini_weather.sort_values(by=['time'])
    X = mini_weather[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
    # 选择标准站数据的PM2.5、PM10、NO
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()
    n_samples = len(mini_weather)
    if n_samples > 100:
        train_size = int(n_samples * 0.8) # 80%训练数据
        test_size = n_samples - train_size # 20%测试数据
        mini_train = mini_weather.head(train_size)
        mini_test = mini_weather.tail(test_size)
        X_train = mini_train[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
        y_train = std_weather.loc[mini_train.index][['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
        X_test = mini_test[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
        y_test = std_weather.loc[mini_test.index][['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]

        # 训练模型
        regressor = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500)
        regressor.fit(X_train, y_train)

        # 对测试集进行预测
        y_pred = regressor.predict(X_test)

        # 计算均方误差和R2得分
        mse = mean_squared_error(y_test, y_pred)
        r2score = regressor.score(X_test, y_test)

        # 打印结果
        print(f"mse: {mse}, R2 score: {r2score}")

        # 保存模型
        filename = 'model.joblib'
        import joblib
        joblib.dump(regressor, filename)

        # 睡眠1小时，等待下一轮数据输入
        time.sleep(3)