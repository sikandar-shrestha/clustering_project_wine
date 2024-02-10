#tabular data imports :
import pandas as pd
import numpy as np
from pydataset import data

# visualization imports:
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# success metrics from earlier in the week: mean squared error and r^2 explained variance
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.cluster import KMeans

#stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import shapiro

import warnings
warnings.filterwarnings("ignore")
import wrangle as w
import os
directory = os.getcwd()
pd.set_option('display.max_columns', None)



def summarize(df) -> None:
    '''
    Summarize will take in a dataframe and report out statistics
    regarding the dataframe to the console.
    
    this will include:
     - the shape of the dataframe
     - the info reporting on the dataframe
     - the descriptive stats on the dataframe
     - missing values by column
     - missing values by row
     
    '''
    print('--------------------------------')
    print('--------------------------------')
    print('Information on DataFrame: ')
    print(f'Shape of Dataframe: {df.shape}')
    print('--------------------------------')
    print(f'Basic DataFrame info:')
    print(df.info())
    print('--------------------------------')
    # print out continuous descriptive stats
    print(f'Continuous Column Stats:')
    print(df.describe().T)
    print('--------------------------------')
    # print out objects/categorical stats:
    print(f'Categorical Column Stats:')
    print(df.select_dtypes('O').describe().T)
    print('--------------------------------')
    print('--------------------------------')



def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train, validate_test = train_test_split(df, train_size=0.60, random_state=123, stratify = df.quality)
    
    # Create train and validate datsets
    validate, test = train_test_split(validate_test, test_size=0.50, random_state=123, stratify = validate_test.quality)

    # Take a look at your split datasets

    print(f"""
    train -> {train.shape}
    validate -> {validate.shape}
    test -> {test.shape}""")
    
    return train, validate, test



def preprocess_wine(df):
    '''
    preprocess_mall will take in values in form of a single pandas dataframe
    and make the data ready for spatial modeling,
    including:
     - splitting the data
     - encoding categorical features
     - scaling information (continuous columns)

    return: three pandas dataframes, ready for modeling structures.
    '''
    # Create dummy variables for the 'color' column
    color_dummies = pd.get_dummies(df['color'], drop_first=True).astype(int)
    # Add the dummy variables to the DataFrame
    df = pd.concat([df, color_dummies], axis=1)
    
    # Drop original 'color' and 'wine_quality' columns
    df = df.drop(columns=['color'])
    # split data:
    train, validate, test = split_data(df)
    # scale continuous features:
    scaler = MinMaxScaler()
    train = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns)
    validate = pd.DataFrame(
        scaler.transform(validate),
        index=validate.index,
        columns=validate.columns
    )
    test = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns)
    return train, validate, test




def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return dataframe (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    scaler = MinMaxScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 33)
    kmeans.fit(X)
    kmeans.predict(X)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
