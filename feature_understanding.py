import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class FeatureUnderstanding:
    
    def __init__(self, data):
        self.data = data
    
    def correlation_matrix(self, figsize=(10, 10)):
        corr_matrix = self.data.corr()
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, cmap='YlGnBu', annot=True)
        plt.show()
    
    def scatterplot(self, x, y, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.data, x=x, y=y)
        plt.title("Scatter plot {} vs Target: {}".format(x,y))
        plt.show()
    
    def boxplot(self, x, y, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        sns.boxplot(data=self.data, x=x, y=y)
        plt.show()
    

