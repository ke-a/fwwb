# 导入必要的库和模块
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.semi_supervised import LabelPropagation
from keras.models import Sequential
from keras.layers import Dense

# 加载微型站和标准站的数据
micro_data = pd.read_csv('micro1.csv')
standard_data = pd.read_csv('standard.csv')

# 划分 PM2.5 值为 5 个区间，并将每个区间视为一个离散的类别
bins = [-np.inf, 35, 75, 115, 150, np.inf]
labels = [0, 1, 2, 3, 4]
standard_data['PM2.5_category'] = pd.cut(
    standard_data['PM2.5'], bins=bins, labels=labels)
standard_data.dropna(subset=['PM2.5_category'], inplace=True)

# 提取微型站中与PM2.5相关的特征数据
X_micro = micro_data[['temperature', 'humidity', 'NO2']].values
y_micro = micro_data['PM2.5'].values

# 构建半监督学习模型
model = Sequential()  # 创建序列模型
# 添加一个具有64个神经元的全连接层，并使用ReLU激活函数
model.add(Dense(64, activation='relu', input_dim=3))
model.add(Dense(64, activation='relu'))  # 添加第二个具有64个神经元的全连接层，并使用ReLU激活函数
model.add(Dense(1))  # 添加一个输出神经元的全连接层，用于预测PM2.5的值
model.compile(optimizer='adam', loss='mean_squared_error')  # 编译模型并指定优化器和损失函数

# 利用标准站数据训练半监督学习模型
X_standard_labeled = standard_data[[
    'temperature', 'humidity', 'NO2']].values  # 提取标准站中与PM2.5相关的特征数据
y_standard_labeled = standard_data['PM2.5_category'].astype(
    int).values  # 提取标准站中的 PM2.5 分类标签，并转换为整数类型
model.fit(X_standard_labeled, y_standard_labeled,
          epochs=50, batch_size=32)  # 使用标准站数据对模型进行有标记的训练

# 对微型站数据进行预测和校准
X_micro_unlabeled = micro_data[['temperature', 'humidity', 'NO2']].fillna(
    0).values  # 提取微型站中与PM2.5相关的特征数据，并将缺失值填充为0
y_pred = model.predict(X_micro_unlabeled).ravel()
lp_model = LabelPropagation()  # 创建标签传播模型
lp_model.fit(X_micro_unlabeled, y_pred.round().astype(int))
y_calibrated = lp_model.label_propagation(
    y_pred.round().astype(int))  # 对预测结果进行校准

# 计算模型的均方误差和 R2 得分
mse = mean_squared_error(y_micro, y_calibrated)
r2 = r2_score(y_micro, y_calibrated)


# 打印输出模型的性能指标
print('Model Mean Squared Error: {:.4f}'.format(mse))
print('Model R2 Score: {:.4f}'.format(r2))
