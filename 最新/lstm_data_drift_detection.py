import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
class DriftDetection:
    def __init__(self, threshold=0.95):
        self.threshold = threshold           # 设定漂移检测的阈值
        self.scaler = MinMaxScaler()          # 对数据进行MinMax缩放
        self.model = None                     # 定义模型

    def fit(self, X_train, y_train):
        self.scaler.fit(X_train)             # 对训练集进行拟合
        X_train_scaled = self.scaler.transform(X_train)
        # 将输入重塑为3D [样本数，时间步长，特征数]
        X_train_reshaped = X_train_scaled.reshape(
            (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        # 定义模型
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu',
                       input_shape=(1, X_train_scaled.shape[1])))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        # 拟合模型
        self.model.fit(X_train_reshaped, y_train, epochs=50, verbose=0)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        # 将输入重塑为3D [样本数，时间步长，特征数]
        X_test_reshaped = X_test_scaled.reshape(
            (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        yhat = self.model.predict(X_test_reshaped, verbose=0)
        return yhat.flatten()

    def detect_drift(self, X_new):
        y_pred = self.predict(X_new)
        r2 = r2_score(y_true=X_new[:, -1], y_pred=y_pred)
        if r2 < self.threshold:
            return True
        else:
            return False

    def detect_outliers(self, X_all, eps=0.5, min_samples=10):
        X_all_scaled = self.scaler.transform(X_all)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_all_scaled)
        labels = db.labels_
        return np.where(labels == -1)[0]

    def update_model(self, X_train, y_train, outliers_idx):
        X_train_scaled = self.scaler.transform(X_train)
        X_train_cleaned = np.delete(X_train_scaled, outliers_idx, axis=0)
        y_train_cleaned = np.delete(y_train, outliers_idx, axis=0)
        # 将输入重塑为3D [样本数，时间步长，特征数]
        X_train_reshaped = X_train_cleaned.reshape(
            (X_train_cleaned.shape[0], 1, X_train_cleaned.shape[1]))
        # 重新拟合模型
        self.model.fit(X_train_reshaped, y_train_cleaned, epochs=50, verbose=0)

# 读入数据并进行预处理
df_micro = pd.read_csv('micro1.csv')
df_micro = df_micro[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
df_standard = pd.read_csv('standard.csv')
df_standard = df_standard[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]

df_merged = pd.concat([df_micro, df_standard])     # 合并两个DataFrame
df_merged = df_merged.sort_values(by=['time'])     # 根据时间排序
df_merged = df_merged.fillna(0)                    # 填充缺失值为0
scaler = MinMaxScaler()                            # 实例化MinMaxScaler
data_norm = scaler.fit_transform(df_merged.iloc[:, 2:].values)   # 对特征进行缩放

# 划分训练集和测试集
train_size = int(len(data_norm) * 0.8)
train, test = data_norm[:train_size, :], data_norm[train_size:, :]
X_train, y_train = train[:,2:], train[:, :2]
X_test, y_test = test[:, 2:], test[:, :2]

drift_detection = DriftDetection()                # 实例化DriftDetection类
drift_detection.fit(X_train, y_train)             # 训练模型

new_data = np.random.rand(10, 6)                   # 生成新数据
if drift_detection.detect_drift(new_data):         # 如果检测到漂移，则进行模型更新
    outliers_idx = drift_detection.detect_outliers(
            np.concatenate((train, new_data), axis=0))   # 检测异常点的索引
    drift_detection.update_model(np.concatenate((X_train, new_data[:, 2:]), axis=0),
                                    np.concatenate((y_train, new_data[:, :2]), axis=0),
                                    outliers_idx)         # 更新模型

# 不断生成新数据，直到检测到漂移为止
while True:
    new_data = np.random.rand(10, 6)                  # 生成新数据
    if drift_detection.detect_drift(new_data):         # 如果检测到漂移，则进行模型更新
        outliers_idx = drift_detection.detect_outliers(
            np.concatenate((train, test, new_data), axis=0))   # 检测异常点的索引
        drift_detection.update_model(np.concatenate((X_train, X_test, new_data[:, 2:]), axis=0),
                                      np.concatenate((y_train, y_test, new_data[:, :2]), axis=0),
                                      outliers_idx)        # 更新模型
        break

# 最后输出拟合精度和异常点的位置
y_pred_all = drift_detection.predict(data_norm[:, 2:])
r2_all = r2_score(y_true=data_norm[:, -1], y_pred=y_pred_all)
print('R2 score:', r2_all)
outliers_idx_all = drift_detection.detect_outliers(data_norm)
print('Outliers index:', outliers_idx_all)