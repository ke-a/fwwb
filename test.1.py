
# 导入必要的库和模块
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.semi_supervised import LabelPropagation
from keras.models import Sequential
from keras.layers import Dense

# 加载微型站和标准站的数据
micro_data = pd.read_csv('micro1.csv')
standard_data = pd.read_csv('standard.csv')

# 提取微型站中与PM2.5相关的特征数据
X_micro = micro_data[['temperature', 'humidity', 'NO2']].values
y_micro = micro_data['PM2.5'].values

# 构建半监督学习模型
model = Sequential()  # 创建序列模型
model.add(Dense(64, activation='relu', input_dim=3))  # 添加一个具有64个神经元的全连接层，并使用ReLU激活函数
model.add(Dense(64, activation='relu'))  # 添加第二个具有64个神经元的全连接层，并使用ReLU激活函数
model.add(Dense(1))  # 添加一个输出神经元的全连接层，用于预测PM2.5的值
model.compile(optimizer='adam', loss='mean_squared_error')  # 编译模型并指定优化器和损失函数

# 利用标准站数据训练半监督学习模型
X_standard_labeled = standard_data[['temperature', 'humidity', 'NO2']].dropna().values  # 提取标准站中与PM2.5相关的特征数据，并删除缺失值
y_standard_labeled = standard_data['PM2.5'].dropna().values  # 提取标准站中的PM2.5浓度值，并删除缺失值
model.fit(X_standard_labeled, y_standard_labeled, epochs=50, batch_size=32)  # 使用标准站数据对模型进行有标记的训练

# 对微型站数据进行预测和校准
X_micro_unlabeled = micro_data[['temperature', 'humidity', 'NO3']].fillna(0).values  # 提取微型站中与PM2.5相关的特征数据，并将缺失值填充为0
y_pred = model.predict(X_micro_unlabeled)  # 利用训练好的模型对微型站数据进行预测
lp_model = LabelPropagation()  # 创建标签传播模型
# 利用未标记的数据对模型进行半监督训练和校准
lp_model.fit(X_micro_unlabeled, y_pred.ravel().astype('int'))

# 创建两个零数组来表示额外的特征
zeros1 = np.zeros_like(y_pred)
zeros2 = np.zeros_like(y_pred)

# 将y_pred与两个零数组连接起来以创建一个包括3个特征的输入
X = np.concatenate((y_pred, zeros1, zeros2), axis=1)

# 使用X作为lp_model.predict()方法的输入
y_calibrated = lp_model.predict(X)
# 计算模型的均方误差
mse = mean_squared_error(y_micro, y_calibrated)
print('Model Mean Squared Error: {:.4f}'.format(mse))
