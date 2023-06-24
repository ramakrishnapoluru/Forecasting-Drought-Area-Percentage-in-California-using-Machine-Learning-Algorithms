import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import data_preprocessing as dp
import feature_understanding as fu
import DecisionTree_Model as DTM 
import Random_Forest_Model as RFM 
import Lstm_model as LSTMM 
import Artificial_NN as ANNM
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def drop_columns(self, columns):
        self.data = self.data.drop(columns=columns)

    def standard_scale(self, columns):
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def minmax_scale(self, columns):
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def replace_missing_values(self, method='mean'):
        if method == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif method == 'median':
            self.data = self.data.fillna(self.data.median())
        elif method == 'mode':
            self.data = self.data.fillna(self.data.mode().iloc[0])

    def datetime_to_features(self, column):
        self.data[column] = pd.to_datetime(self.data[column])
        self.data['Month'] = self.data[column].dt.month
        self.data['Year'] = self.data[column].dt.year
        self.data = self.data.drop(columns=[column])

    def set_index(self):
        self.data = self.data.set_index('Date')

    def rename_the_columns(self, column_name, rename_name):
        self.data = self.data.rename(columns={column_name:rename_name})
    def return_the_dataset(self):
        return self.data

