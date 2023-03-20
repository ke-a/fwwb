# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 读取数据
data1 = pd.read_csv('micro1.csv')
data2 = pd.read_csv('standard.csv')
data1 = data1[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
data2 = data2[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# 将时间转换为时间戳格式，并设置为索引列
data1['time'] = pd.to_datetime(data1['time'])
data1.set_index('time', inplace=True)
data2['time'] = pd.to_datetime(data2['time'])
data2.set_index('time', inplace=True)
def correct_data(row):
    pm25_diff = abs(row['PM2.5'] - data2.loc[row.name]
                    ['PM2.5'])  # 计算微型气象站PM2.5数据与标准站PM2.5数据之差
    pm10_diff = abs(row['PM10'] - data2.loc[row.name]
                    ['PM10'])  # 计算微型气象站PM10数据与标准站PM10数据之差
    no2_diff = abs(row['NO2'] - data2.loc[row.name]
                   ['NO2'])  # 计算微型气象站NO2数据与标准站NO2数据之差
    
    humidity_diff = abs(row['humidity'] - data2.loc[row.name]
                   ['humidity'])  # 计算微型气象站NO2数据与标准站NO2数据之差

    temperature_diff = abs(row['temperature'] - data2.loc[row.name]
                   ['temperature'])  # 计算微型气象站NO2数据与标准站NO2数据之差
    # 如果PM2.5数据误差大于20或为NaN，则将其修正为标准站数据
    if pm25_diff > 2 or np.isnan(pm25_diff):
        row['PM2.5'] = data2.loc[row.name]['PM2.5']
    # 如果PM10数据误差大于20或为NaN，则将其修正为标准站数据
    if pm10_diff > 2 or np.isnan(pm10_diff):
        row['PM10'] = data2.loc[row.name]['PM10']
    if no2_diff > 2 or np.isnan(no2_diff):  # 如果NO2数据误差大于20或为NaN，则将其修正为标准站数据
        row['NO2'] = data2.loc[row.name]['NO2']
    if humidity_diff > 2 or np.isnan(humidity_diff):
        row['humidity'] = data2.loc[row.name]['humidity']
    if temperature_diff > 2 or np.isnan(temperature_diff):  # 如果NO2数据误差大于20或为NaN，则将其修正为标准站数据
        row['temperature'] = data2.loc[row.name]['temperature']
    return row


data1 = data1.apply(
    correct_data, axis=1)  # 应用correct_data函数对每一行进行校正


# 合并两个数据集，按照时间排序
data = pd.concat([data1, data2], axis=1)
data = data.loc[:,~data.columns.duplicated()]
data = data.reindex(sorted(data.columns), axis=1)

# 标准化数据
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 分离训练集和测试集
train_size = int(len(data) * 0.7)
train_data = data_scaled[:train_size, :]
test_data = data_scaled[train_size:, :]

# 定义函数用于生成批次数据
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])
    return np.array(X), np.array(Y)

# 准备训练和测试数据
look_back =3
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# 创建LSTM模型
model = Sequential()
model.add(LSTM(units=96, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))#原本是2
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=32))
model.add(Dropout(0.15))
# model.add(LSTM(units=16))
# model.add(Dropout(0.1))
model.add(Dense(Y_train.shape[1]))

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
history = model.fit(X_train, Y_train, epochs=400, batch_size=40, validation_data=(X_test, Y_test), verbose=1)

# 预测结果
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train)
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test)

# 计算r2分数
train_score = r2_score(Y_train[:,0], train_predict[:,0])
print('Train R2 Score: %.2f' % (train_score))
test_score = r2_score(Y_test[:,0], test_predict[:,0])
print('Test R2 Score: %.2f' % (test_score))
import numpy as np
from sklearn.ensemble import IsolationForest
# 定义孤立森林模型的树数量和子采样大小
n_trees = 100
subsample_size = 256

# 创建字典以存储每个特征的孤立森林模型
clfs = {}
train_data = pd.DataFrame(train_data)
# 循环遍历数据集中的每个特征并训练一个孤立森林模型
for column_name in train_data.columns:
    # 提取特征值
    column_data = train_data[column_name].values.reshape(-1, 1)
    
    # 训练孤立森林模型
    model = IsolationForest(n_estimators=n_trees, max_samples=subsample_size, random_state=42)
    model.fit(column_data)
    
    # 存储已训练的模型
    clfs[column_name] = model


def diff(predict_data,original_data):
    # 定义函数来计算预测数据和原始数据之间的差异
    diff = 0
    for i,value in enumerate(predict_data):
        diff+=abs(value-original_data[i]);
    return diff/len(predict_data);
# def check_data(predict_data, original_data, threshold, threshold2, row, queue):
#     diff = diff(predict_data, original_data)
#     if diff <= threshold:
#        #小于的话就是可校准数据
#         data1[row] = predict_data
#         queue.append(row)
#         total = sum(queue)
#         if total / len(queue) > threshold2:
#             # 重新训练模型
#             train_size = int(row)
#             train_data = data_scaled[:train_size, :]
#             test_data = data_scaled[train_size:, :]
#             X_train, Y_train = create_dataset(train_data, look_back)
#             X_test, Y_test = create_dataset(test_data, look_back)
#             model.fit(X_train, Y_train, epochs=400, batch_size=40, validation_data=(X_test, Y_test), verbose=1)
#             queue.clear()
#         return queue
#     else:
#         #否则属于异常数据
#         #还需要判断具体是微站损坏还是微站异常
#         print("异常数据：", row)

# train_len = int(len(data) * 0.3)
# import queue
# q = queue.Queue(maxsize=3)
# for i in range(train_len,len(data)):
#     predict_data = model.predict(data[i][0])
#     #反归一化
#     predict_data = scaler.inverse_transform(predict_data)
#     original_data = data[i]
#     q = check_data(predict_data,original_data[0:6],0.05,0.3,i,q)


