import time, sys

import pandas as pd

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

import numpy as np
from scipy.stats import sem
from scipy import stats

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import BernoulliRBM
from multilayer_perceptron import MultilayerPerceptronClassifier
from sklearn.dummy import DummyClassifier

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.externals import joblib

from operator import itemgetter


def process_csv(train_loc, test_loc, binary):
    #ToDo: need to re-comment to match new code
    ########
    # Takes in a path for a csv file and returns a list containing test and training sets for a data set
    # Output: [x_test, x_train, y_test, y_train]
    ########

    print('\n#############')
    print('\nProcessing CSV file...')
    print('\n#############\n')

    def build_df(csv_location):

        ########
        # Builds a pandas dataframe from a csv file
        ########

        print('Dataframe is being built...')
        df = pd.read_csv(csv_location)
        df.rename(columns=lambda i: i.replace(' ', '_').lower(), inplace=True)
        if binary:
            df = df.replace(to_replace='Transferred', value='Dropped')
            df = df.replace(to_replace='Dropped', value=0)
            df = df.replace(to_replace='Graduated', value=1)

        return df

    def impute_missing_values(x_train, x_test):
        ########
        # Takes in a dataframe and imputes missing values
        ########
        print('Values are being imputed in dataframe...')
        # df = df.fillna(0)
        # Impute numerical variables with mean
        x_train = x_train.fillna(x_train.mean())
        x_test = x_test.fillna(x_train.mean())
        # Impute categorical variables with mode
        # x_train = x_train.apply(lambda i: i.fillna(i.mode()[0]))
        x_train = x_train.fillna(x_train.mode())
        x_test = x_test.fillna(x_train.mode())

        return x_train, x_test

    def process_features(df):
        ######
        # Takes in a dataframe and converts categorical variables into indicator variables
        #######
        # Split dataset in two, one with all numerical features and one with all categorical features
        print('Processing features...')
        num_df = df.select_dtypes(include=['float64'])
        cat_df = df.select_dtypes(include=['object'])
        # Convert categorical features into indicator variables
        if len(cat_df.columns) > 0:
            cat_df = convert_to_indicators(cat_df)
        # Recombine categorical and numerical feature into one dataframe
        df = num_df.join(cat_df)
        # Replace NaN's that were caused by division by 0 when rescaling with 0's
        # This occurs when all values are 0 (eg. indicator variables)
        # df = df.fillna(0)

        return df

    def convert_to_indicators(df):
        ########
        # Takes in a dataframe and makes a new dataframe with indicator variables
        # for each column of the provided dataframe
        ########
        # Create a new data frame with indicator variables for the first column
        # Needs to be done with this so that other indicator variables can be joined by iteration
        print('Converting some features to indicator variables...')
        df1 = pd.get_dummies(df.iloc[:, 0])
        df1 = df1.iloc[:, 0:1]
        # Iterate through columns creating indicator variables for each column and
        # join the indicator variables to the new dataframe created above
        if len(df.columns) > 1:
            for i in range(1, len(df.columns)):
                df2 = pd.get_dummies(df.iloc[:, i])
                df1 = df1.join(df2.iloc[:, 0:len(df2.columns)-1])
        return df1

    def transform(x_train, y_train, x_test, y_test):

        ########
        # Takes in two dataframes for the features and labels of a dataset and
        # outputs a dictionary with training and keys relating to training testing sets for each
        ########

        # scaler = StandardScaler()
        # scaler.fit(x_train)
        # x_train = scaler.transform(x_train)
        # x_test = scaler.transform(x_test)

        # scaler2 = MinMaxScaler()
        # scaler2.fit(x_train)
        # x_train = scaler2.transform(x_train)
        # x_test = scaler2.transform(x_test)

        x_train = (x_train - x_train.min())/(x_train.max()-x_train.min())
        x_test = (x_test - x_train.min())/(x_train.max()-x_train.min())

        x_test = x_test.fillna(0)
        x_train = x_train.fillna(0)

        data_dict = {'x_test': x_test, 'x_train': x_train,
                     'y_test': y_test, 'y_train': y_train}

        return data_dict

    # Create a dataframe from a provided path
    train_data = build_df(train_loc)
    test_data = build_df(test_loc)
    # Separate dataframe into labels and features
    train_y = train_data.pop(train_data.columns[len(train_data.columns)-1])
    train_x = train_data
    test_y = test_data.pop(test_data.columns[len(test_data.columns)-1])
    test_x = test_data

    train_x, test_x = impute_missing_values(train_x, test_x)

    print(train_x.shape, test_x.shape)
    train_x = process_features(train_x)
    test_x = process_features(test_x)

    train_col = train_x.columns
    test_col = test_x.columns
    features = list(set(train_col.union(test_col)))

    train_x = train_x.reindex(columns=features, fill_value=0)
    test_x = test_x.reindex(columns=features, fill_value=0)

    return transform(train_x, train_y, test_x, test_y)
