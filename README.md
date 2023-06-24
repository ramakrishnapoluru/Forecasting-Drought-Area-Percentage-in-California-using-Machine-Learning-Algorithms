# Forecasting-Drought-Area-Percentage-in-California-using-Machine-Learning-Algorithms

# Abstract

California is highly susceptible to frequent, severe droughts that significantly impact ecosystems, agriculture, and water supplies. To identify the adverse effects, it’s crucial to forecast future drought scenarios based on past occurrences in California. Prior research has shown that using SPEI to predict drought is viable, with the majority of studies limited to predicting SPEI or SPI values but not the actual drought percentage. This research aimed to expand on the previous research by creating an accurate model for predicting the area percentage of drought-affected regions beyond just SPEI values for each county from 2000 to 2020 by utilizing climatic, drought percentage, and NDVI values. The calculation of SPEI values involves using Thornthwaite equation along with climatic and NDVI data. After preparing the dataset with climatic and SPEI values, Decision Tree Regressor, Random Forest, LSTM, and ANN models are used to forecast drought area percentage for California. The Random Forest model outperforms other models with MAE value of 6.81 and R2 value of 0.74. ANN and decision tree models achieved good results, with MAE of 7.43 and 7.21, and an R2 score of 0.70 and 0.72, respectively. These results suggest that decision tree-based models and traditional neural networks are effective in predicting the target variable. The findings indicate that RF models can assist in identifying drought-prone areas and help to implement effective mitigation measures.

# Introduction

# Project Background and Executive Summary

California is prone to frequent, severe droughts that can seriously affect ecosystems, agriculture, and water supplies. Kiem et al. (2016) said that because of its many influencing factors that operate at different spatial and temporal scales, drought is a complicated phenomenon and one of the least well-known environmental disasters. Belayneh et al. (2016); Hernández-Espinosa et al. (2018) suggested that in comparison to traditional models, machine learning techniques are better equipped to assess the hierarchical and nonlinear interactions between independent factors and dependent variables. The research aims to solve this issue by creating a machine learning-based model that can precisely predict the proportion of California that will experience a drought. An early warning system for the government and better water management decisions are the driving forces behind the initiative. The research can assist the government in preparing for and lessening the effects of droughts in California by precisely estimating the area percentage of drought. The research also aims to advance understanding of the causes of the current drought in California and shed light on how climate change can affect the frequency and intensity of droughts in the future. The recurrent droughts in California have an adverse effect on the state's water resources, agricultural sector, and ecosystems. The project will enable the government to make better decisions about water management, including water distribution and conservation strategies, by delivering accurate and timely information.

The goal is to create machine learning models that use historical weather data to forecast the drought area percentage of California. The project involves the CRISP-DM methodology, a widely used process model for data mining and analytics projects that includes six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. After careful examination, given the nature of models and the iterative nature of this machine learning project, a CRISP-DM Agile methodology is more appropriate. Agile techniques are made to be adaptable, iterative, and responsive to shifting needs and new information which will be best-aligned characteristics for a machine learning project. The six phases of the CRISP-DM technique will be divided into smaller, more manageable iterations or sprints to be converted to an Agile approach. Each sprint focuses on one or two CRISP-DM phases, enabling the team to advance the project while maintaining flexibility and teamwork as its top priorities.

According to Wilhite et al. (2007), hydrometeorological variables including precipitation, evapotranspiration, sun radiation, and others can be used to calculate drought indices. A complicated interaction between precipitation, air temperature, water vapor pressure, solar radiation, and other factors causes it. Thus, obtained data from the NASA Power website that included daily data on the temperature of the air, the temperature of the earth, the humidity, the wind speed, and the amount of precipitation in California. This data can be used to calculate the Standardized Precipitation Evapotranspiration Index (SPEI) using the Thornthwaite equation, which displays the balance between precipitation and evapotranspiration. The state's NDVI is calculated from satellite imagery. The approach involves using machine learning algorithms to predict the percentage of drought-prone areas in the California region based on historical weather data. It is highly data-driven and involves collecting the required climatic and historical drought area percentage data from accessible and authorized public websites, as well as satellite data. Exploratory analysis is then implemented to identify patterns and trends that can aid in prediction. Later, four machine learning models are built to analyze the historical data and make accurate predictions. The implementation of the approach involves building, training, and testing
4
the following models: a Decision tree model, a Long Short-Term Memory(LSTM) model, an Artificial Neural Network (ANN) model, and a Random Forest model using the various Python libraries. The temperature of Earth, two-meter temperature, two-meter specific humidity, and two-meter wind speed Correcting for precipitation are the major features and metrics commonly used in remote sensing studies are NDVI and SPEI.

Based on past meteorological data, the suggested project seeks to create a dependable and accurate machine-learning model for forecasting the proportion of areas in California that will be impacted by drought. The goal is to improve drought forecast accuracy by analyzing historical weather data with sophisticated machine-learning methods. This initiative aims to improve water resource management and preparation planning in California. Because of the considerable effects that drought has on agriculture, water availability, and the state's economy, this project is crucial. The developed machine learning model can aid in better planning and management of events in the state and identify areas that are likely to experience drought conditions in the future, which can aid in early warning and preparedness. The metrics/ attributes that are considered are vast compared to other proposed models. The study also advances knowledge of the connection between climatic factors and the frequency and severity of California's droughts. The SPEI, and vegetation index are important indicators that will be used to forecast the percentage of drought-prone areas in the California region. The initiative also attempts to evaluate how well various machine learning algorithms forecast the percentage of California's land that will experience a drought. The model can also be adapted and used for other regions that experience similar climatic conditions and are prone to droughts.

# Project Breakdown
The following are the six phases of the project breakdown
Business Understanding: Going through the available resources online and considering the factors which majorly impact the drought to identify and rank the project's requirements and objectives.
Data Understanding: Understanding satellite data along with easily accessible and permitted public websites to gather the necessary climatic and historical drought region percentage information. The factors considered for each model will be studied to gain domain knowledge of the project.
Data Preparation: Before training machine learning models to accurately forecast drought-prone area percentages, the acquired data needs to be cleaned and preprocessed. Missing values are dealt with through data imputation and data interpolation techniques, handling outliers, transforming the data into the necessary format, feature engineering, data regularization, and dividing the data into training and testing datasets.
Modeling: Building four machine learning models: a Decision tree model using the scikit-learn library, an LSTM model using various Python libraries, an ANN model using a Python library, and an RF model using the scikit-learn library.
Evaluation: Use data to evaluate the model's accuracy. choosing the model that is most useful for determining the severity of the drought in California in various situations by comparing their accuracy.
Deployment: Making the developed application accessible to end users, making sure it satisfies all technical specifications, and providing essential support are all parts of the deployment process. This procedure makes it possible for the application to be used and accessed by the target market.

# Data Engineering

# Data Process

This project's goal is to study climatic data over the last 20 years to discover drought patterns and anticipate the percentage of each county in California that may experience drought conditions. Climate data and area percentage statistics were gathered from a variety of sources, including the NASA Power website, which provided daily climate data for each county in California, including temperature, humidity, wind speed, and precipitation, among other things. The Google Earth Engine for computing NDVI values for each county using LANDSAT satellite images, as well as the US Drought Monitor website, which contains weekly and monthly statistics on the area percentages of each county in California that is affected by drought conditions. For further analysis, the data were combined into one dataset based on the county and time dimension. But following data collection, the data was verified for consistency and accuracy, and several issues such as missing values, inconsistent formatting, and outliers were discovered.

To make the data suitable for analysis, a method was used that involved creating a date range covering the entire dataset and filling in any missing values by using weekly values. For each week, the drought area percentage value was assigned to all dates in that week to ensure consistency in handling missing data across all counties and years. Additionally, irrelevant or duplicate data was removed, and outliers beyond the interquartile range were eliminated. When a machine learning model fits the training data too closely, it can lead to overfitting, which is characterized by poor generalization to new data. To overcome this issue, regularization techniques such as ridge and lasso regression are utilized to alter the model's complexity.

The dataset was subsequently reduced in dimensionality by using principal component analysis (PCA) to determine the most critical features contributing to drought severity. PCA is a useful approach for finding the essential variables and lowering the dataset's dimensionality. And determined the appropriate number of principal components to use for further analysis by generating two plots, the cumulative explained variance plot and the scree plot.

The dataset was then divided into three sections: training, validation, and testing. The training set included prior-year data, whereas the test set included data from the most recent year. The training set was used to train the machine learning models, while the validation set was used to tweak the hyperparameters and evaluate the models performance. Experimented with several machine learning algorithms, including Random Forest, Decision Tree ensemble with XGBoost, LSTM, and ANN to predict the area percentage of each county in California that may face drought conditions.

# Data Collection
# Sources
Data has been collected from three sources, including the climatic data of California, the Earth Engine Data Catalog, and the Drought area percentage dataset of California, all of them retrieved from official websites covering the period from 2000 to 2020. The climatic variables and Drought area percentages have been sourced from the following websites respectively:
1. The data for climatic variables from January 2000 to December 2020 has been obtained from the website https://power.larc.nasa.gov/data/.This data comprises daily values covering the entire period from 2000 to 2020.
  
2. The data on drought area percentage in California from January 2000 to December 2020 has been sourced from the website https://droughtmonitor.unl.edu/DmData/DataTables.aspx?state,ca. This data is available on a weekly basis and has been retrieved for each county in California.
  
3. The data of NDVI values for California from January 2000 to December 2020 has been sourced from the Earth Engine Data Catalog using Python API by importing the ‘ee’ package in Python.
 
The data collection plan for the climatic dataset is shown in Figure 14, including all the necessary details based on the dataset features. The collection plan specifies key variables as many variables available in the dataset don’t have much significance to calculate SPEI or drought area percentage. The dataset has a total of 3 Million records of numerical time series climatic variables such as Temperature, Specific humidity, Precipitation, Wind Speed at 10 meters, etc. This dataset has historical data from 2000 to 2020 with 20 climatic variables used to calculate the SPEI value. The dataset includes latitude and longitude values, along with date
66
values in separate columns, which are not practical for Pandas operations. To calculate the SPEI values and drought area percentages at the county level instead of latitude and longitude, the next step is to combine the date columns into a single column and obtain county names based on the latitude and longitude values. After obtaining the county names, the dataset will be preprocessed to address any missing, inconsistent, or noisy data.

#  Data Cleaning: Handling of Incomplete & Missing Data
The Drought Area percentage dataset provides the data of drought percentages for 51 counties in California on a weekly basis. As the climate dataset values are all on a daily basis, the missing values for each county in the remaining week of the 21 years need to be filled.

To address the issue of missing values, all the dates ranging from Jan 1, 2000, to Dec 31, 2020, were generated and used to fill in the drought areas percentage based on the weekly values and sample code is shown in Appendix A. This approach entailed assigning the same drought area percentage value for the entire week's dates, based on the weekly value. This process was applied to all the counties throughout the years 

After populating daily basis values in the Drought dataset, missing values were identified. These missing values occurred due to the dataset's starting week being 4 Jan 2000, rendering all county values before that date as NULL. The data consisted of 153 null values per column.

To address the issue of missing values, the dataset underwent a process of row elimination where the rows containing null values were removed. These rows were deemed insignificant for the modeling task. Additionally, the climatic data available in the dataset initiated from Jan 4, 2000, coincides with the onset of drought data that is free of any missing values.

The NDVI dataset has missing values for many dates between 2000 and 2020. These values are extracted from satellite images for 51 counties during the same time period. The missing values need to be handled before the data can be integrated. Figure 32 shows the missing dates in the dataset for Alameda County.

The missing values in the dataset were filled using an interpolation technique. This technique involves estimating the missing values by interpolating between two available data points. After performing the interpolation, the dataset size increased from 220K records to 391K records, as all the data was now available for all the days in the dataset

# Data Cleaning: Handling of Noisy Data

The climatic dataset manifests anomalous values in certain columns, commonly known as outliers, which deviate significantly from the rest of the dataset. These outliers can lead to inaccurate analysis and modeling. Visual analysis methods such as scatter plots and box plots are frequently employed to detect outliers. In this study, outliers have been identified using box plots, and eliminated from the dataset that falls beyond the Max and Min range of the IQR region.

Anomalies lying beyond the interquartile range for the feature 'T2MDEW' in the climatic dataset were detected. 

The identification and removal of outliers are carried out by detecting the values that lie outside the Max and Min range of the IQR region leading to a more consistent dataset that can enhance the performance of modeling 

# Data Cleaning: Handling of Inconsistent Data

Inconsistencies in the data were addressed by employing several strategies, which involved transforming the dataset into the requisite format, generating novel features through the process of feature engineering, and converting the data types into a uniform format to facilitate more effective analysis

The snippet highlights how inconsistent data are managed in a climate dataset. The county names were derived from the latitude and longitude coordinates using the geopy.geocoders library, a Python program for geocoding and reverse geocoding. Since the county names could not be established since the lat and lon coordinates were in marine regions rather than land regions, the County values with None were eliminated from the dataset. The counties that did not belong to California were eliminated, including several Nevada counties, to make sure that the dataset solely contained California county names. The final dataset was reduced to just 51 different counties in California.

This process emphasizes how crucial it is to spot and deal with conflicting data in datasets. While tools like geopy.geocoders can make it easier to extract location-based data from datasets, it is important to take into account the constraints and any problems that can occur.

The geopy package was used to extract county names from the latitude and longitude values in the dataset. However, some of these values are located in the ocean and not in California, resulting in the retrieval of null values to address this issue, the null values were removed from the dataset in the stage of handling inconsistent data.

# Data Transformation

Data transformation is an important step in this project. It involves converting raw data into a format that is suitable for analysis and modeling. Techniques such as normalization, feature engineering, and dimensionality reduction can be applied during data transformation to improve model performance.

# Feature Engineering

Feature engineering is a type of data transformation that involves generating new features or altering existing features in a dataset to enhance the efficiency of machine learning models. In this project, A feature is being generated from the available dataset by utilizing the SPEI to demonstrate the drought index, which is calculated based on the precipitation and temperature over different time intervals. SPEI can be calculated using climatic water balance which will again be calculated by considering the difference between precipitation and Potential Evapotranspiration on a monthly time scale. To calculate the SPEI value, it is necessary to aggregate the values into monthly data for all the counties which will make the data size limited to 12K records. Potential Evapotranspiration can be calculated based on average temperature values and humidity values.

The range of SPEI values can vary from negative to positive values, where negative values indicate drought conditions and positive values indicate wet conditions. The exact range of SPEI value depends on the specific calculation method used and the time scale being analyzed. However, in general, SPEI values can range from approximately -4 to +4, with extreme values outside of this range being rare 

# Data Integration


Data Integration is one of the techniques used in Data Transformation in which data from multiple sources is combined into a single, consistent, and meaningful format. The process involves identifying the sources of data, resolving any inconsistencies and conflicts, and combining them into a single dataset that can be used for modeling.

The current project involves the utilization of three distinct datasets for the development of a machine-learning model. The initial dataset encompasses climatic data for California counties over a 21-year period, inclusive of latitude and longitude coordinates, and daily dates. The second dataset consists of drought area percentages for California counties over the same 21-year period, including daily dates. Lastly, the third dataset comprises NDVI data derived from satellite images, containing NDVI values and dates.

The integration of three datasets was accomplished by leveraging the shared attributes of Date and County names. This allowed for the consolidation of climatic variable data for each county on a specific month across all datasets, serving as a valuable opportunity to create a cohesive dataset with an emphasis on data integrity, consistency, and semantic relevance. Subsequently, PCA will be implemented to facilitate advanced analysis and modeling of the combined dataset. The utilization of PCA is among the intended next steps.

# Data Regularization

Overfitting is a significant issue that needs to be considered in any Machine Learning model. In overfitting, the ML model tries to fit each data point on the curve because of its learning from the training dataset, which is noisy. Since the model has very little flexibility, it fails to predict the new data points during prediction. Regularization techniques are employed to modify the linear regression model to reduce the adjusted loss function and avoid the problems of overfitting.

In this data regularization process, the first step followed is to split the data into training and testing sets. A linear regression model was then trained using the training set to predict the target variable. The model was evaluated by making predictions on the test set and calculating the mean absolute error and mean squared error.

To improve the performance of the model, ridge regression and lasso regression models were implemented. The Ridge() and Lasso() functions were used to train the models, and the mean squared error was calculated for each. The coefficients of the ridge and lasso models were also obtained using the coef_ attribute. Later L1 and L2 regression is applied from the results obtained from Linear regression.

The lasso model's effectiveness is limited as a vast majority of the coefficients have become precisely zero.

The lasso regression model disregards some of the features in the training set, with only 9 of the 24 being utilized.

# Data Transformation: Data Reduction Using PCA

The PCA, or the principal component analysis technique, is often employed in data analysis to identify the most significant elements that contribute to changes in the data. PCA is used in this climate data project to minimize the number of features while making the data simpler to understand and display. Date, latitude, longitude, county, and various climate variables such as temperature, dew point, and precipitation all had been included in the dataset, which included a total of 28 columns. Plotted a heatmap to show the correlation between the numeric features.

To ensure that all features after PCA receive the same importance, the non-numerical data was removed, then the numerical columns were standardized using scikit-learn's StandardScaler function. Post-data preprocessing, PCA with eight principle components was applied to scaled data. The reduced dimensionality space was created by converting the data with the retrieved primary components. The generated dataset after PCA contained over two million rows and eight columns.

To determine the appropriate number of principal components to use for further analysis, a cumulative explained variance plot was generated.

This is a line graph that displays the cumulative explained variance for each number of principal components, indicating that the first few principal components account for the majority of the variance in the data. More than 95% of the variance in the data is explained by 8 primary components.

After performing PCA on the dataset, the dimensionality of the data was reduced to 8 principal components. These 8 principal components captured the majority of the variance in the data while still maintaining the important information needed for predicting the target variable. Using these 8 principal components, identified the 8 original columns from the dataset that had the highest impact on the target variabl

Columns were chosen because the mentioned eight variables had the highest impact on the target variable, and thus were the most important features for training the machine learning model.

Selecting specific columns reduced the dataset's complexity while maintaining high prediction accuracy. This method produced a more efficient and simplified mechanism for accurately anticipating new data.

#  Data Preparation

Data preparation is a key component in any data modeling task, following data cleaning, regularization, and PCA. One important aspect of data preparation is dividing the data into training, validation, and testing sets, which is an essential step in modeling.

The training set is the largest subset of the data and is used to train the model and all the models are built using this training data. The validation set is a smaller subset of the data that is used to evaluate the performance of the model during the training process. The model is not trained on this data, but it is used to assess how well the model is generalizing to new data. The validation set is used to tune the hyperparameters of the model to improve its performance. The test set is a separate subset of the data that is used to evaluate the final performance of the model. The model is not trained or tuned on this data, and it is only used to assess how well the model is able to generalize to new or unseen data.

Given that the dataset for this project is a time series data type spanning from 2000 to 2020, it was partitioned into training, validation, and test sets, with an approximate ratio of 70, 20, and 10 percent, respectively. The splitting was carried out based on the years, where the data from 2000 to 2018 was utilized for training, the data from 2019 was employed as validation data, and the data from 2020 was used as test data.

Splitting the data into the above mentioned ratio for training, validation, and testing datasets respectively provides a reliable and efficient method for maintaining the temporal dependencies between the time series data by evaluating the performance of machine learning models and improving their accuracy and effectiveness. The dataset's train, validation, and test sizes, which will be used to implement models for predicting the percentage of drought area,

# Model Development

# Model Proposal

Predicting drought area percentage from time series data involves dealing with complex relationships between the climatic input variables and the target variable. To accurately model these relationships, it is important to choose a machine learning or deep learning model that can effectively capture non-linear dependencies and scale well to large datasets. In this research, Decision Tree Regressor, Random Forest, ANN, and LSTM were chosen as the models.

# Decision Tree

Addressing complex relationships between climatic variables being used and the resultant target variable is necessary for accurately estimating the proportion of drought-affected areas using time series data. The choice of an appropriate machine learning or deep learning model becomes essential to accurately capture the non-linear relationships and manage massive datasets. For estimating the proportion of a region that is experiencing drought, many models like neural networks, decision trees, and support vector machines have been used. The Decision Tree model was especially chosen for this research because it can handle non-linear connections and scales effectively to fit large datasets. By analyzing feature importance and comprehending the decision-making process, the Decision Tree model offers interpretability. In order to identify drought patterns and their causes, it recursively separates data using a set of input variables. The model is appropriate for data with errors because of its resilience in managing missing values and outliers. Its potential for drought prediction is increased by its capability to handle categorical and numerical information. The goal of this research is to utilize the Decision Tree Regressor model to obtain an understanding of the variables affecting the likelihood of drought area percentage in California.

# Random Forest

Random Forest creates an ensemble of decision trees, each using a random subset of input variables and observations, and aggregates their outputs to produce a final prediction. This technique reduces overfitting and improves generalization performance, which is important when dealing with complex relationships in climatic and SPEI data.

In the study by Dash et al. (2022), Random Forest was identified as a highly effective model for handling heterogeneous data. Unlike other models, Random Forest does not require faster GPU servers and is known for its computational efficiency. In time series modeling of climatic data, overfitting can be a major challenge due to the high dimensionality and nonlinearity of the data. According to Liu et al. (2016), they introduced a self-adjusting mechanism to the Random Forest model to prevent overfitting. This mechanism modifies the number of trees in the forest to achieve a balance between model complexity and prediction accuracy. They applied this approach to time series modeling of climatic data and compared it with conventional Random Forest models.

# Artificial Neural Network

ANN has the ability to handle complex, non-linear correlations between input features and target variables. Furthermore, ANNs can generalize effectively to previously unseen data, which is critical in this research project because the model must properly predict dry areas for future periods. ANNs are also quite adaptable, as they can be utilized for regression as well as classification problems, making them a good fit for this research. Finally, ANNs have been implemented successfully to forecast drought conditions in previous research studies, indicating their potential utility in the present study.

In a study conducted by Bodri and Čermàk (2000), the authors used a collection of historical precipitation data from two meteorological stations in Moravia to create a neural network model to predict extreme precipitation events. Because radial basis function neural networks and backpropagation neural networks have proved effective in previous studies on precipitation forecasting, the authors trained those neural network models. The first model achieved an accuracy of 85% and was able to predict the precipitation events for up to ten days ahead. The neural network, according to the researchers, is an appropriate model for this research because of its non-linearity, because of which complex relationships between precipitation and variables like temperature and humidity can be captured. These conclusions are essential to this research since a nonlinear model, like a neural network, could be efficient at capturing the complex relationships among drought area percentages and other variables such as precipitation, temperature, and vegetation index.

# Long Short Term Memory

The climatic data, drought data, and Normalized Difference Vegetation Index data are data sequences that extend over long timeframes used to predict future drought conditions. LSTM is effective in modeling sequences with long-term dependencies because it has memory cells that can selectively include or exclude information over time, which enables it to maintain critical information from earlier time steps and discard irrelevant information. This is particularly important when dealing with time-series data, where the previous states can have a significant impact on the current state.

In their study, Poornima and Pushpalatha (2019) compared the performance of LSTM and ARIMA for time series prediction. The authors found that LSTM outperformed ARIMA for longer time scales and when additional parameters with positive correlation were added to Standard Precipitation Index and SPEI. However, they noted that LSTM was more resource-intensive than ARIMA. Despite this, the authors highlighted the strengths of LSTM, including its ability to overcome vanishing gradients, learn from past experiences, classify processes, and make predictions for time series.

The author's focus was to develop an LSTM model that achieved high accuracy and learning rate while minimizing the number of epochs required. Their use of univariate and multivariate approaches found that scaling up the number of layers improved model accuracy. In the multivariate approach with two layers of LSTM, they were able to reduce the number of epochs required by half compared to using only one layer of LSTM. The authors also addressed the constraint of increasing layers, as this increases time complexity since hidden layers undergo recursion. Despite the computational expense of LSTM, the authors observed very high accuracy and less Root Mean Square Error (RMSE) scores for different timescales and parameter combinations.

# Model Architecture and Data flow

# Decision Tree

The model was trained, tested, and evaluated using a dataset of historical weather, NDVI, and drought data for various counties to predict the Area percentage of drought. The training set covered a period of 20 years, from January 2000 to December 2019, and the test set consists of the last year of available data, from January 2020 to December 2020. For training, the model utilized a feature set consisting of Relative_Humidity, Min_Temperature, Max_Temperature, Wind_Speed, Temperature, Precipitation, SPEI_1, and NDVI. The training set consists of 12240 records. A TimeSeriesSplit cross-validation technique with 4 splits was employed to value the model's effect on the training data. Each split involved a training set with a maximum of 2448 records and a validation set with 612 records and performed hyperparameter tuning using Gridsearch. After cross-validation, the model was fit into the entire training dataset, to evaluate its performance on a test set of 612 records. The model is created to train on the data from each county separately in order to forecast drought and analyzes evaluation metrics such as MAE, RMSE, and R2 for each county and record them in a list. The individual county metrics are aggregated to determine the overall drought area percentage and metrics for the entire state of California. 

The climatic, NDVI, and drought data were collected from official websites, followed by a rigorous data cleaning process to handle missing values, null entries, and inconsistencies. Following the integration of the three datasets, data transformation operations such scaling, regularization, and dimensionality reduction (PCA) were performed.

The data was divided into a training set covering 19 years. i.e. from 2000 to 2019 and a test set covering one year (2020) in order to prepare it for modeling. The model was tuned and optimized using a time-split cross-validation approach after being trained independently for each county. The model was then retrained using the whole training dataset, and its performance was assessed using the test dataset. In order to generate overall metrics, the evaluation metrics for each county were aggregated, giving information on the drought area percentage of California.

# Random Forest

The training dataset which is being split into k number of random samples with a replacement where the size of the random sample should be less than the training data. This approach is known as bagging. OOB is a validation technique that uses the samples that were not included in the training of a particular decision tree to evaluate its performance. This ensures that each sample is used in the training or evaluation of at least one tree. OOB error rate in the decision tree provides a way to estimate the performance of the model during training and helps to avoid overfitting. And then the predicted data will be aggregated from all the predictions shown by the Decision Trees. Figure 63 shows the architecture of Random Forest model

For each county, the data was trained and validated, and the evaluation metrics were computed at the county level. These metrics were stored in a list, and in the end, they were aggregated to obtain the overall metric value for the entire dataset, which can be considered for the entire state of California. This approach allowed for an evaluation of the model's performance at both the county and state levels, providing insight into its effectiveness in predicting drought in different regions.

After the validation of the baseline model, the training dataset from 2000 to 2018 was then again validated on the 2019 data using hyperparameter tuning. During this process, hyperparameter tuning using Bayesian Optimization was applied to the model in which a probabilistic model for the regression model's validation loss is constructed. The search space for this optimization technique is specified by passing the values of n_estimators ranging from 50 to 500, max_depth ranging from 2 to 20, and min_samples_split ranging from 2 to 20, from which the best possible combination of hyperparameters was chosen for the model. This optimization technique performs 32 iterations to determine the optimal hyperparameter combination. The resulting model showed better performance than earlier runs without any hyperparameters.

Using the hyperparameters identified during tuning, the model was trained on the entire dataset from 2000 to 2019 and then tested on the 2020 dataset. In comparison, another version of the model was also trained on the dataset from 2000 to 2019 but without passing any hyperparameters, and tested on the same 2020 dataset. The model trained with hyperparameters consistently outperformed the baseline model, as shown by the evaluation metrics.

# ANN

A total of four layers were employed to build the model: a single input layer, two hidden layers, plus one output layer. The two hidden layers have 100 and 50 nodes, respectively. Each node in the model was given random weights and biases at the start.

The input variables were then propagated from the input layer to the output layer via hidden layers where the activation function mathematically transforms the input variables. The activation function that was used in this research was the Rectified Linear Unit (reLu) function, which is provided by (1). After obtaining the output data, the error is calculated using the loss function MSE. During the backpropagation phase, a gradient g for each neuron is calculated. The weights and biases are then changed depending on that value to achieve model convergence. The optimizer function, such as Adam, is responsible for this.

Following the development of the model, it was trained on the training dataset and tested
on the validation set. Based on the results, hyperparameter tuning is done using grid search to select the optimal set of hyperparameters that can improve the performance of the model. Finally, predictions were made using the model on the testing set. The model was then evaluated using metrics such as MAE, MSE, RMSE, and R2 for each county and recorded in a list. Individual county indicators are aggregated to get the overall drought percentage and metrics for California.

# LSTM

The transformed time series dataset is then divided based on the time period, the training data spans from the year 2000 to 2018, while the validation data is for the year 2019. The model is trained at the county level, where the data for each county train with the model separately. Models performance is then evaluated at the county level using various evaluation metrics. Finally, the county-level results are aggregated and rolled up to the state level for the state of California.

After the validation of the model using the 2019 dataset, it was found that the model's accuracy needed improvement. Therefore, to enhance the model's performance, hyperparameter tuning was carried out using Grid Search, and the model was retrained using the data from 2000 to 2018. The hyperparameters that were tuned were n_lstm_units, learning_rate, and dropout rate. By adjusting these hyperparameters, the model's performance was significantly improved, as the accuracy of the model increased. With this improvement, the model was further retrained using the data from 2000 to 2019 and tested on the dataset from 2020 to evaluate its generalization capability.

The data flow within the different components of the LSTM model, the input data is first loaded into the LSTM layer, which is designed to handle sequential data. The LSTM layer performs a series of computations on the input data to generate a sequence of hidden states. These hidden states contain information about the input sequence, and are updated and passed along to the next time step in the sequence. This allows the LSTM layer to record long-term dependencies in the input data, making it apt for tasks such as time series forecasting. Figure 65 shows the architecture of the LSTM model

The sequence of hidden states output by the LSTM layer is then passed on to the dense layer with the ReLu activation function applied. The dense layer is responsible for learning a non-linear relationship between the output of the LSTM layer and the target variable. This is achieved by applying a series of matrix operations to the input data, followed by the application of the ReLu activation function. The ReLu activation function is commonly used in neural networks as it is simple, computationally efficient, and has been shown to be effective in preventing the vanishing gradient problem.

To prevent overfitting, a dropout layer is incorporated after the dense layer. The dropout layer arbitrarily drops some of the neurons in the course of the training process, thereby preventing the model from overfitting to the training dataset. The dropout rate is set to a value of self.dropout, which determines the fraction of neurons that are randomly dropped during each training iteration. This helps to ensure that the model is generalizing well to new data and not simply memorizing the training set.

Finally, the output from the dropout layer is passed through the final dense layer, which contains a single neuron that produces the predicted value. This output represents the model's prediction for the target variable at the next time step in the sequence.

# Model Evaluation

In any machine learning or deep learning study, it is crucial to evaluate the performance of the models used to ensure that they provide reliable and accurate results. In this particular research, the target value is a continuous numerical value, for which the Random Forest regression model is one of the appropriate choices. To measure the accuracy of the predicted values compared to the actual values, various evaluation metrics are used. Some common evaluation metrics used in regression problems include MAE, MSE, RMSE, and R2 score. These metrics provide different ways of quantifying the performance of the models, with some prioritizing accuracy over precision and vice versa. Careful consideration must be given to the choice of evaluation metric, as it can impact the interpretation and usefulness of the model's results.

# Mean Absolute Error

The MAE value is commonly utilized in evaluating the effectiveness of a regression model. It determines the average absolute difference between the actual and predicted values of a dataset. The L1 norm, commonly known as the MAE, is calculated by averaging the absolute differences between predicted and actual outcomes. The advantage of using this metric is that it provides information on the extent of the error in the predictions, irrespective of the direction of the error. It is also less influenced by outlier values, unlike other metrics used for evaluation. The MAE formula is seen (2).

𝑀𝐴𝐸 = (1/𝑛) *∑ (𝑦𝑖 −𝑦𝑖)

where n is the number of samples, and 𝑦𝑖, 𝑦𝑖 are the actual and predicted values, respectively.

# Mean Squared Error

Another frequently used metric for evaluation in problems related to regression is a MSE. The average of the squared differences between the actual and predicted values is how it is calculated. Lower numbers signify greater model performance, and the MSE value varies between 0 to infinity. By averaging the squared differences between the expected and actual values, the MSE formula is calculated. 

𝑀𝑆𝐸 = (1/𝑛) *∑ (𝑦𝑖 −𝑦𝑖)^2

where n is the number of samples, and 𝑦𝑖, 𝑦𝑖 are the actual and predicted values, respectively

# Root Mean Squared Error

Another often-used evaluation metric for regression models is RMSE. The root mean square error, or RMSE, calculates the average of the squared deviations between predicted and actual values. It is a more sensitive metric to outliers as it squares the errors. RMSE can help identify how close the predictions are to the actual values on average. Better model performance is associated with a lower RMSE. 

R𝑀𝑆𝐸 = SQRT((1/𝑛) *∑ (𝑦𝑖 −𝑦𝑖))

where n is the number of samples, and 𝑦𝑖, 𝑦𝑖 are the actual and predicted values, respectively.

# R-Squared Score

The R2 value is a statistical metric used to measure how effectively the regression model fits the data, which is also known as the coefficient of determination. Its range is 0 to 1, with 1 signifying the most optimal model fit to the data. By comparing the variances of predicted and actual values, the R2 value is determined. A negative R2 number means the model performs worse than predicting the mean value of the target variable, whereas a value of 0 indicates poor model fit. This value is a useful tool for comparing the performance of different regression models. One drawback of the R2 value is that it does not indicate whether the model is overfitting or underfitting the data

where n is the number of samples in the collection, 𝑦𝑖, 𝑦𝑖 are the actual and predicted values respectively and 𝑦 is the mean value of a dependent variable across all observations.

The research used the four evaluation metrics mentioned above to assess the model's
performance. In conclusion, the evaluation metrics are crucial in determining the dependability and precision of regression models, making them essential for any machine learning or deep learning research that employs regression techniques.

# Model Validation and Evaluation

In this study, four different models were employed to forecast the percentage of drought-affected areas. Prior to prediction, preprocessing, transformation, and splitting procedures were conducted. The models underwent initial validation using 2019 data, wherein each model was trained on data spanning from 2000 to 2018. Baseline models were trained on the training data and subsequently evaluated using the validation data. Following this, hyperparameter optimization was performed on each model using various techniques. Upon evaluating the validation data results, noticeable improvements in the performance of all models were observed. Utilizing the identified hyperparameters, the training data was extended to include data from 2000 to 2019, while the test data consisted of 2020 data. In comparison, an alternative version of the model was trained on the dataset from 2000 to 2019 without applying any hyperparameters and then tested on the same 2020 dataset.

# Decision Tree Base Model

Using the Decision Tree Base model, the prediction results yielded an MAE value of 10.21, an MSE value of 87.79, an RMSE value of 9.37, and an R2 score of 0.62. It is important to note that no hyperparameter tuning was performed on the data during the predictions.

# Decision Tree Hypertuned Model

The Gridsearch Optimization technique was utilized to perform hyperparameter tuning and the model's performance was marginally improved by selecting the optimal hyperparameters of max_tree_depth as 13, min_samples_split as 4, and min_samples_leaf as 6, criterion as MSE, CCP as 0.07, as evidenced by the results. The predictions produced an MAE score of 8.86, an MSE score of 76.03, an RMSE score of 8.72, and an R2 score of 0.66. Gridsearch Optimization took a considerable amount of time to generate the results because the search space was extensive due to the iterations in Gridsearch, which resulted in a longer execution time.

# Random Forest Base Model

Using the Random Forest Base model, the prediction results yielded an MAE value of 7.61, an MSE value of 43.28, an RMSE value of 6.57, and an R2 score of 0.70. It is important to note that no hyperparameter tuning was performed on the data during the predictions.

# Random Forest Hypertuned Model

The Bayesian Optimization technique was utilized to perform hyperparameter tuning and the model's performance was marginally improved by selecting the optimal hyperparameters of n_estimators as 423, max_depth as 18, and min_samples_split as 5, as evidenced by the results.

The predictions produced an MAE score of 6.81, an MSE score of 39.73, an RMSE score of 6.3, and an R2 score of 0.74. Bayesian Optimization took a considerable amount of time to generate the results because the search space was extensive, and the number of iterations was set to 32, which resulted in a longer execution time.

# ANN Baseline Model

The original ANN base model generated prediction results with an R2 score of 0.67, an MAE of 8.91, an MSE of 84.20, an RMSE of 9.17, and an RMSE of 84.20. It's crucial to point out that the dataset was never tuned for the hyperparameters prior to obtaining these results.

# ANN Hypertuned Model

The grid search technique is utilized, in which a range of values for each hyperparameter is established, including the activation function, hidden layer count, and neurons count, and then the algorithm is trained on the model on each combination of hyperparameters in the grid. After doing the hyperparameter tuning, the algorithm provided the optimal combination of parameters as the two hidden layers, 100 neurons in the first hidden layer and 50 neurons in the second, the activation method being reLu, and the optimizer being the Adam function. Running the model with these hyperparameters yielded results of 7.43 for MAE, 52.67 for MSE, 7.25 for RMSE, and 0.70 for R2.

# LSTM Base Model

A common practice in time series analysis is to ensure that the model is accurately predicting future values based on the patterns it has learned from past data. So the data of the years prior to 2020 is used for training. The results of the evaluation metrics for the model are as follows: an MAE value of 12.86, an MSE value of 168.52, an RMSE value of 12.98, and an R2 score of 0.54.

# Post Parameter Tuned LSTM Model

The results of the baseline model were not satisfactory with very less accuracy rates. To boost the models performance Grid Search model is employed to perform hyperparameter tuning. Hyperparameter tuning involves searching for the best combination of hyperparameters that will result in the highest performance of a model. This procedure helps to optimize the model's effectiveness for a particular dataset.

The dictionary param_grid was used to specify the values for each hyperparameter. For the n_lstm_units hyperparameter 32, 64, and 128 values are used to determine the optimal number of LSTM units. The lr hyperparameter was tuned using values of 0.1, 0.01, and 0.001 to determine the optimal learning rate. And we tuned the dropout hyperparameter using values of 0.1, 0.2, and 0.5 to determine the optimal dropout rate. The grid search explored all possible combinations of these hyperparameter values to find the best combination that minimizes the MSE. Though the model took more time than the baseline model, it showed improved performance with the following hyperparameters n_lstm_units: 32, lr: 0.01, dropout: 0.2. The predictions produced an MAE score of 11.20, an MSE score of 126.31, an RMSE score of 11.23, and an R2 score of 0.56

# Comparison of Models

Comparing evaluation metrics of Decision Tree, Random Forest, ANN, and LSTM models, Table 19 demonstrates the impact of hyperparameter tuning on their performance. Notably, all models exhibited improved results after the optimization process, underscoring the significance of hyperparameter tuning for achieving better performance.

# Conclusions

From the results obtained, Random Forest demonstrated superior performance compared to other models that were used in the research after performing hyperparameter tuning on the model with an R2 value of 0.74. When compared to ANN and LSTM, Random Forest outperformed them due to its capability to identify non-linear relationships, resistance to overfitting, and generalization ability. On the other hand, LSTM was the least effective model among the ones employed in the research, yielding an R2 value of 0.56. Among the Neural Networks algorithms, ANN performed better than LSTM, ranking second in terms of performance, just after Random Forest. The primary factor for this sub-par performance of Neural Networks algorithms is their dependency on a significant amount of data to learn patterns and avoid overfitting which is one of the main reasons Random Forest performed better than all other models.

# Limitations

The primary limitation of the study is that the model was trained at the county level with a limited dataset and predicted target values in the same county. This resulted in a shortage of data points to detect temporal dependencies and a significant number of irregularities in the test dataset, leading to a suboptimal performance from all the models. Basically, Decision Tree struggles with handling continuous and complex relationships between variables, which limits its ability to capture the complex relations of drought. Similarly, the Random Forest model is limited to predicting the values within the range of the training data, and it may struggle to make accurate predictions when given new data points outside of the training dataset. Both Neural Networks models face a significant limitation as they need huge data to predict accurate results which is lacking in this research. As a result, these models were prone to overfitting leading to suboptimal performance. In addition to the limited number of data points in the training dataset, the model lacked more features that contribute to the effects of drought and have an optimal correlation with the target variable.

# Future Scope

In the future, to enhance the accuracy of the drought prediction model used in this research, the dataset could be expanded by incorporating additional features, such as underground water balances, soil moisture index, and average rainfall, among other things. By including these features, the dataset would be more complete, and the models would be able to identify more patterns in the data, resulting in better accuracy. Also, the models can be further refined by gathering more detailed data at the city level, enabling government officials to make more accurate predictions and take appropriate action to mitigate the effects of drought in vulnerable regions.

In the future, if the data is more complex and varied, a hybrid approach involving models like Random Forest and XG Boost can be used to achieve more precise results. Bagging and boosting techniques can be applied to the ANN model by mixing numerous neural networks trained with various initial weights or structures. For LSTM, various time series analysis techniques can be applied to capture the patterns and trends over time such as autoregression or moving averages.





