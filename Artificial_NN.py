from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.callbacks import EarlyStopping

class ANN:
    
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_units=64, dropout_rate=0.2, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_units, activation='relu', input_dim=self.input_dim))
        model.add(Dropout(self.dropout_rate))
        for i in range(self.hidden_layers - 1):
            model.add(Dense(self.hidden_units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.output_dim))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def summary_the_model(self):
        return self.model.summary()
    
    def fit(self, X_train, y_train):
        if not self.model:
            self.build_model()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, callbacks=[es])
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        #rmsle = mean_squared_error(np.log(y_true), np.log(y_pred), squared=False)
        #r2 = r2_score(y_true, y_pred)
        return [mae,mse,rmse]

    
    def return_the_model(self):
        return self.model
        
