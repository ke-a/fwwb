# #使用demo3计算的模型
# import joblib
# import pandas as pd
# loaded_model = joblib.load('model.pkl')
# test = pd.read_csv('micro2.csv')
# standard = pd.read_csv('standard.csv')
# X = test[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# Y = standard[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# x_pred = loaded_model.predict(X)
# #计算r2误差
# from sklearn.metrics import r2_score
# print(r2_score(Y, x_pred))
# import matplotlib.pyplot as plt
# import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv('micro1.csv')

# # 绘制箱线图
# plt.boxplot([df['PM2.5'], df['PM10'], df['NO2']])
# plt.xticks([1, 2, 3], ['PM2.5', 'PM10', 'NO2'])
# plt.ylabel('Concentration (μg/m³)')
# plt.title('Air quality data')
# plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 读取数据集
df = pd.read_csv('micro1.csv')

# 预处理数据
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.iloc[:, 1:].values)

# 构建特征和标签序列
time_steps = 10
X = []
y = []
for i in range(len(scaled_data) - time_steps):
    X.append(scaled_data[i:i+time_steps])
    y.append(scaled_data[i+time_steps, 0])

X_train = np.array(X)
y_train = np.array(y)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(
    time_steps, scaled_data.shape[1]-1), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1))

# 编译模型
model.compile(loss='mse', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=64)

# 预测结果
X_test = scaled_data[-time_steps:]
X_test = np.expand_dims(X_test, axis=0)
pred = model.predict(X_test)
pred = scaler.inverse_transform(np.concatenate(
    (pred, np.zeros((1, scaled_data.shape[1]-1))), axis=1))

print(f"预测值为: {pred[0][0]}")
