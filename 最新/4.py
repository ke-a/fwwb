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

data1 = data1.apply(correct_data, axis=1) 

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
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, verbose=1, n_jobs=-1)

# Train model and output best score and parameters
# grid_result = grid.fit(X_train, Y_train)
# print("Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Predict results
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
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
import pydot
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 绘制模型图
# plot_model(model.model, to_file='rnn.png', show_shapes=True, show_layer_names=True)

plot_model(model.model, to_file='rnn.png', show_shapes=True, show_layer_names=True)

# 生成DOT代码
with open("rnn.png", 'rb') as f:
    graph = pydot.graph_from_dot_data(f.read())

dot_code = graph.to_string()

# 将DOT代码保存到文件中
with open("rnn.dot", "w") as f:
    f.write(dot_code)
    
# 使用Graphviz将DOT代码渲染为图片
(graph,) = pydot.graph_from_dot_file('rnn.dot')
graph.write_png('rnn_structure.png')