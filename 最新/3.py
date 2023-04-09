import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest
import queue
# Read data
data1 = pd.read_csv('micro1.csv')
data2 = pd.read_csv('standard.csv')
data1 = data1[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
data2 = data2[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
data1['time'] = pd.to_datetime(data1['time'])
data1.set_index('time', inplace=True)
data2['time'] = pd.to_datetime(data2['time'])
data2.set_index('time', inplace=True)

def correct_data(row):
    pm25_diff = abs(row['PM2.5'] - data2.loc[row.name]['PM2.5'])  
    pm10_diff = abs(row['PM10'] - data2.loc[row.name]['PM10'])  
    no2_diff = abs(row['NO2'] - data2.loc[row.name]['NO2'])  
    humidity_diff = abs(row['humidity'] - data2.loc[row.name]['humidity'])  
    temperature_diff = abs(row['temperature'] - data2.loc[row.name]['temperature'])  
    
    if pm25_diff > 2 or np.isnan(pm25_diff):
        row['PM2.5'] = data2.loc[row.name]['PM2.5']
    if pm10_diff > 2 or np.isnan(pm10_diff):
        row['PM10'] = data2.loc[row.name]['PM10']
    if no2_diff > 2 or np.isnan(no2_diff):
        row['NO2'] = data2.loc[row.name]['NO2']
    if humidity_diff > 2 or np.isnan(humidity_diff):
        row['humidity'] = data2.loc[row.name]['humidity']
    if temperature_diff > 2 or np.isnan(temperature_diff):
        row['temperature'] = data2.loc[row.name]['temperature']
    return row

data1.iloc[:int(len(data1)*0.3)] = data1.iloc[:int(len(data1)*0.3)].apply(correct_data, axis=1)
data = pd.concat([data1, data2], axis=1)
data = data.loc[:,~data.columns.duplicated()]
data = data.reindex(sorted(data.columns), axis=1)

    # Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split train and test data
train_size = int(len(data) * 0.7)
train_data = data_scaled[:train_size, :]
test_data = data_scaled[train_size:, :]
train_clf = data1.iloc[:train_size, :]
train_clf = train_clf[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
# Define function to generate batch data
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])
    return np.array(X), np.array(Y)
look_back =3
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Define LSTM model
def create_model(units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=int(units/2), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=int(units/4)))
    model.add(Dropout(dropout))
    model.add(Dense(Y_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Use KerasRegressor wrapper to make model compatible with GridSearchCV
model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=40, verbose=1)

# Define hyperparameters to optimize
param_grid = {'units': [64, 96, 128],
              'dropout': [0.1, 0.2, 0.3]}

# Use TimeSeriesSplit for cross-validation
# tscv = TimeSeriesSplit(n_splits=5)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, verbose=1, n_jobs=-1)

# Train model and output best score and parameters
# grid_result = grid.fit(X_train, Y_train)
# print("Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# # Predict results
# train_predict = grid.predict(X_train)
# test_predict = grid.predict(X_test)

# Reverse scaling
# train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train)
# test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test)

# Calculate r2 score
# train_score = r2_score(Y_train[:,0], train_predict[:,0])
# print('Train R2 Score: %.2f' % (train_score))
# test_score = r2_score(Y_test[:,0], test_predict[:,0])
# print('Test R2 Score: %.2f' % (test_score))

import pickle

# # Save model
# with open('model.pkl', 'wb') as file:
#     pickle.dump(grid_result.best_estimator_, file)

# Load model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)


#训练孤独森林模型用于异常检测
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1),max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(train_clf)
#预测
outlier = queue.Queue(5)
def detect_outliers(predict_data,original_data,outlier):
    
    original_data = original_data[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
    if clf.predict() == -1:
        outlier.put((predict_data,original_data))
    if outlier.full():
        temp = []
        for i in outlier:
            predict_data = i[0]
            original_data = i[1]
            #计算z分数
            std = np.std(original_data)
            mean = np.mean(original_data)
            z_score = (predict_data - mean)/std
            temp.append(z_score)
        if temp.sum()/temp.size >3:
            return 'z_score_max'
        temp = []
        for i in outlier:
            predict_data = i[0]
            original_data = i[1]
            #计算绝对值
            abs_score = abs(predict_data - original_data)
            temp.append(abs_score)
        if temp.sum()/temp.size >20:
            return 'abs_max'
        a = 0
        for i in outlier:
            predict_data = i[0]
            for j in predict_data:
                if j ==0:
                    a+=1
        if a/len(predict_data) >0.5:
            return 'zero_max'
        return  'normal'
        
def check_data(predict_data,original_data,row,queue,threshold):
    sign = detect_outliers(predict_data,original_data,outlier)
    if sign == 'normal':
        data1[row] = predict_data
        queue.append(predict_data)
        if len(queue) ==queue.maxsize and sum(queue)/queue.maxsize > threshold:
            # 如果队列已满，并且平均误差小于阈值，则重新训练模型
            train_size = int (row)
            train_data = data_scaled[:train_size, :]
            test_data = data_scaled[train_size:, :]
            X_train, Y_train = create_dataset(train_data, look_back)
            X_test, Y_test = create_dataset(test_data, look_back)
            model.fit(X_train, Y_train,epochs=200, batch_size=40, verbose=1)
            queue.clear()
        return queue,'calibration'
    else:
        if sign == 'z_score_max' or sign == 'abs_max':
            print('微站损坏',row)
            return queue,'damage'
        else:
            print('微站异常',row)
            return queue,'abnormal'
        
train_len = int(len(data)*0.3)
import queue
q = queue.Queue(maxsize=5)
data = pd.read_csv('micro1.csv')
data = data[['time','PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
data = data.apply(correct_data,axis=1)
# data['time'] = pd.to_datetime(data['time'])

# data.set_index('time', inplace=True)
look_back = 3
num_features = 5
for index, row in data[train_len+1:].iterrows():
# for i in range(train_len,len(data)):
    timestamp = index 
    print(timestamp)
    print(data.loc[timestamp])
    data_point = data.loc[timestamp].values.reshape(1, look_back, num_features)
    scaled_data_point = scaler.transform(data_point)
    prediction = model.predict(scaled_data_point)
    prediction = scaler.inverse_transform(prediction)
    timestamp = pd.to_datetime(data['time'])
    timestamp = index
    print(row)
    # 获取前n个时间步长的数据
    predict_data = data.loc[timestamp].values.reshape(1, -1)
    predicted_data_scaled = scaler.transform(predict_data)
    predicted_data,_ = create_dataset(predicted_data_scaled, look_back)
    
    previous_data = data.loc[timestamp - pd.Timedelta(minutes=60*(look_back-1)):timestamp]
    # 缩放数据
    # 检查数据是否为空
    if previous_data.empty:
        continue

    # 重新塑造数据

    # 使用模型进行预测
    prediction = model.predict(predicted_data)
    prediction = scaler.inverse_transform(prediction)
    # 反向缩放
    predicted_data = scaler.inverse_transform(predicted_data_scaled)

    print(predict_data)
    # 反归一化
    predict_data = scaler.inverse_transform(predict_data)
    
    predict_data = row
    predict_data_scaled = scaler.transform(predict_data)
    

    original_data = index[1:7] 
    q,sign = check_data(predict_data[1:7],original_data,index,q,0.1)
    if sign == 'damage':
    # 如果是微站损坏，则记录日志并发送警报
        print("微站损坏：", index)
    # TODO: 发送警报到管理系统
    elif sign == 'abnormal':
    # 如果是微站异常，则记录日志和异常类型，并进行维护
        print("微站异常：", index)
        print("异常类型：", q[-1])
    # TODO: 记录日志和异常类型，进行维护操作


