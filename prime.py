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
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


data_set = pd.read_csv("/Users/dolly/Drought_Prediction/ccsnd.csv")
print(data_set.info())

preprocessor = dp.DataPreprocessor(data_set)

#make the date as index
preprocessor.set_index()

#Renaming the target column 
preprocessor.rename_the_columns('D0-D4','Drought_Target')

#drop columns 

dropping_columns = ["D1-D4","D2-D4","D3-D4","DSCI","county_encoded","Year","Month","None","D4","Lat","Lon"]

preprocessor.drop_columns(dropping_columns)



#Scaling the Values 
numerical_features = numerical_features = data_set.select_dtypes(include=['float64', 'int64']).columns.tolist()

preprocessor.minmax_scale(['T2MDEW', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WD10M', 'T2M', 'PRECTOTCORR', 'SPEI_1', 'ndvi_value'])

#Get the preprocessed data

data_set = preprocessor.return_the_dataset() 

print(data_set.head(),data_set.info())

data_set1 = data_set.copy()
feature_understanding_caller = fu.FeatureUnderstanding(data_set1)

print("Correlation Scores Matrix According to Feature vs Target")
feature_understanding_caller.correlation_matrix()

print("Scatter Plots target vs features")
for col in data_set1.columns:
  if col != 'Drought_Target':
    feature_understanding_caller.scatterplot(col,'Drought_Target')

print("Mi_Scores With respect to target")
from sklearn.feature_selection import mutual_info_regression
y = data_set1['Drought_Target']
X = data_set1.drop(['Drought_Target',"county"], axis=1)

mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
print(mi_scores)

from sklearn.decomposition import PCA
pca = PCA(n_components=8)
data_set1 = data_set.copy()
data_set1 = data_set1.drop(["Drought_Target","county"],axis=1)
pca_data = pca.fit_transform(data_set1)
df_pca = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2', 'PC3',"PC4","PC5","PC6",'PC7',"PC8"],index=data_set1.index)
print(pca.explained_variance_ratio_)
print("So the total of 8 component can explain {} of the dataset".format(sum(pca.explained_variance_ratio_)))
df_pca["county"] = data_set["county"]
df_pca["Drought_Target"] = data_set["Drought_Target"]
df_pca.head()
counties = df_pca["county"].value_counts()
print(counties)

model_LSTM_Summary =LSTMM.LstmModel()
model_LSTM_Summary.build_model(np.array([1]*8).reshape(1,1,8))
print("Summary of the LSTM_Model : ", model_LSTM_Summary.summary_the_model())


model_ANN_Summary =ANNM.ANN(8,1)
model_ANN_Summary.build_model()
print("Summary of the ANN_Model : ", model_ANN_Summary.summary_the_model())

predictions = []

counties = df_pca["county"].value_counts().index.tolist()


for county in counties:

    print("\n\n")
    print("*"*15, county , "*"*15 )
    print("\n\n")
    
    county_data = df_pca[df_pca['county']==county] 

    
    Target = county_data["Drought_Target"]

    county_data = county_data[['PC1', 'PC2', 'PC3',"PC4","PC5","PC6",'PC7',"PC8"]]

    for i in range(2,0,-1): 
        
        
        train_x = county_data.iloc[:len(county_data)-i]
        val_x = county_data.iloc[len(county_data)-24:len(county_data)-i]
        X_val = np.reshape(val_x.values, (val_x.shape[0],1, val_x.shape[1]))
        val_y = Target[len(county_data)-24:len(county_data)-i]
        train_y = Target[:len(county_data)-i]
        test_x_index = county_data.iloc[len(county_data)-i].name
        test_x = county_data.iloc[len(county_data)-i].values
        test_y = Target[len(county_data)-i]
        print("\n\n")
        print("*"*15, "{} Drought Prediction for month {}".format(county,test_x_index) , "*"*15 )
        print("\n\n")
        
        DT = DTM.DecisionTree()
        DT_param_grid = {
    'criterion': [ 'friedman_mse'],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2,],
    'max_features': ['auto']
      }


        DT.hyperparameter_tuning(train_x,train_y,DT_param_grid)
        DT_y_pred = DT.predict(test_x.reshape(1,-1))
        print(DT_y_pred,test_y)
        DT_eval_results = DT.evaluate(DT_y_pred,[test_y])
        predictions.append({"County": county, "Algorithm": "Decision Tree", "Month": test_x_index, 
                        "Actual": test_y, "Predicted": DT_y_pred, "MAE": DT_eval_results[0],
                        "MSE": DT_eval_results[1], "RMSE": DT_eval_results[2]})
        #Random Forest Model
        RF = RFM.RandomForest()
        RT_param_grid = {
          'n_estimators': [50, 100],
          'max_features': ['auto', 'sqrt', 'log2'],
          'max_depth': [5, 10],
          'min_samples_split': [2, 5],
          'min_samples_leaf': [1, 2],
        }
        RF.hyperparameter_tuning(train_x,train_y,RT_param_grid)
        RF_y_pred = RF.predict(test_x.reshape(1,-1))
        RF_eval_results = RF.evaluate(RF_y_pred, [test_y])

        predictions.append({"County": county, "Algorithm": "Random Forest","Month": test_x_index, "Actual": test_y, "Predicted":RF_y_pred ,"MAE" :RF_eval_results[0],"MSE":RF_eval_results[1],"RMSE":RF_eval_results[2]})
        
        #LSTM Model
        train_x = np.reshape(train_x.values, (train_x.shape[0],1, train_x.shape[1]))
        test_x = np.reshape(test_x,(1,1,8))
        val_x = county_data.iloc[len(county_data)-24:len(county_data)-i]
        X_val = np.reshape(val_x.values, (val_x.shape[0],1, val_x.shape[1]))
        
        LSTM_obj = LSTMM.LstmModel()
        LSTM_obj.build_model(train_x)
        LSTM_obj.fit(train_x,train_y,X_val,val_y)
        LSTM_y_pred = LSTM_obj.predict(test_x)
        LSTM_eval_results = LSTM_obj.evaluate([test_y],LSTM_y_pred)

        predictions.append({"County": county, "Algorithm":"LSTM","Month": test_x_index, "Actual": test_y, "Predicted": LSTM_y_pred,"MAE" :LSTM_eval_results[0],"MSE":LSTM_eval_results[1],"RMSE":LSTM_eval_results[2]})
        #ANN Model
        train_x = county_data.iloc[:len(county_data)-i]
        val_x = county_data.iloc[len(county_data)-24:len(county_data)-i]
        val_y = Target[len(county_data)-24:len(county_data)-i]
        train_y = Target[:len(county_data)-i]
        test_x_index = county_data.iloc[len(county_data)-i].name
        test_x = county_data.iloc[len(county_data)-i].values.reshape(1,8)
        test_y = Target[len(county_data)-i]
        
        ANN_Model = ANNM.ANN(input_dim=train_x.shape[1],output_dim = 1)
        ANN_Model.build_model() 
        ANN_Model.fit(train_x,train_y)
        ANN_y_pred = ANN_Model.predict(test_x)
        ANN_eval_results  = ANN_Model.evaluate([test_y],ANN_y_pred)

        predictions.append({"County": county, "Algorithm":"ANN","Month": test_x_index, "Actual": test_y, "Predicted": ANN_y_pred,"MAE" :ANN_eval_results[0],"MSE":ANN_eval_results[1],"RMSE":ANN_eval_results[2] })
df = pd.DataFrame(predictions)

# Save DataFrame as CSV file
df.to_csv('result_data.csv', index=False)