import joblib
import pandas as pd
loaded_model = joblib.load('model.pkl')
test = pd.read_csv('micro2.csv')
standard = pd.read_csv('standard.csv')
X = test[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
Y = standard[['PM2.5', 'PM10', 'NO2', 'temperature', 'humidity']]
x_pred = loaded_model.predict(X)
#计算r2误差
from sklearn.metrics import r2_score
print(r2_score(Y, x_pred))
