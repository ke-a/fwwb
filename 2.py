import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import queue
from sklearn.ensemble import IsolationForest
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
print("TimeSeriesSplit version:", TimeSeriesSplit.VERSION)
print("IsolationForest version:", IsolationForest().__class__.__name__, sklearn.__version__)