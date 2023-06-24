from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.callbacks import EarlyStopping
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class LstmModel:
    
    def __init__(self, n_lstm_units=50, n_dense_units=32, n_epochs=50, batch_size=32, dropout=0.1, lr=0.001):
        self.n_lstm_units = n_lstm_units
        self.n_dense_units = n_dense_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.model = None
        
    def build_model(self,X_train):
        self.model = Sequential()
        self.model.add(LSTM(units=self.n_lstm_units, input_shape=(1,X_train.shape[2])))
        self.model.add(Dense(units=self.n_dense_units, activation='relu'))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(units=1))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.lr))
    
    def summary_the_model(self):
        return self.model.summary()

    def fit(self, X_train, y_train,X_val,Y_val):
        self.model.fit(X_train, y_train,validation_data=(X_val,Y_val), epochs=self.n_epochs, batch_size=self.batch_size)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred.flatten()

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return [mae,mse,rmse]

    def return_the_model(self):
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        # Define the hyperparameter grid
        param_grid = {'n_lstm_units': [32, 64, 128], 'lr': [0.1, 0.01, 0.001], 'dropout': [0.1, 0.2, 0.5]}
        
        # Initialize GridSearchCV with the model, parameter grid, and scoring metric
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
        
        # Fit GridSearchCV on your training data
        grid_search.fit(X_train, y_train)
        
        # Retrieve the best hyperparameters and use them to build the final model
        best_params = grid_search.best_params_
        self.n_lstm_units = best_params['n_lstm_units']
        self.lr = best_params['lr']
        self.dropout = best_params['dropout']
        self.build_model(X_train)
