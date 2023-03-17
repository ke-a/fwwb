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

# 合并两个数据集，按照时间排序
data = pd.concat([data1, data2], axis=1)
data = data.loc[:,~data.columns.duplicated()]
data = data.reindex(sorted(data.columns), axis=1)

# 标准化数据
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 分离训练集和测试集
train_size = int(len(data) * 0.8)
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
look_back = 3
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# 创建LSTM模型
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
history = model.fit(X_train, Y_train, epochs=200, batch_size=64, validation_data=(X_test, Y_test), verbose=1)

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

# # 定义函数检查新数据
# print("start check data")
# def check_data(new_data):
#     print("start check data")
#     # 转换为numpy数组
#     new_data = np.array(new_data).reshape(1, -1)
#     # 归一化
#     new_data_scaled = scaler.transform(new_data)
#     # 生成序列数据
#     X_new = []
#     for i in range(len(new_data_scaled)-look_back):
#         a = new_data_scaled[i:(i+look_back), :]
#         X_new.append(a)
#     # 预测输出值
#     y_new = model.predict(np.array(X_new))
#     # 反归一化
#     y_new = scaler.inverse_transform(y_new)[0]
#     # 计算偏差
#     deviation = np.abs(new_data - y_new)
#     # 检查是否异常
#     if np.any(deviation > 0.1):
#         return False, '数据异常'
#     else:
#         return True, y_new


#从data1中拿出一行数据
# final_data = [5,6,5,5,5]
# is_valid, prediction = check_data(final_data)
# if is_valid:
#     print('新数据预测结果为：', prediction)
# else:
#     print('新数据异常')
