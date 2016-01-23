__author__ = 'Ricardo'

import time
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np
from scipy.stats import sem
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.grid_search import GridSearchCV


def process_csv(location):
    ########
    # Takes in a path for a csv file and returns a list containing test and training sets for a data set
    # Output: [x_test, x_train, y_test, y_train]
    ########
    def build_df(csv_location):
        ########
        # Builds a pandas dataframe from a csv file
        ########
        df = pd.read_csv(csv_location)
        df.rename(columns=lambda i: i.replace(' ', '_').lower(), inplace=True)
        return df

    def impute_missing_values(df):
        ########
        # Takes in a dataframe and imputes missing values
        ########
        # Impute numerical variables with mean
        df = df.fillna(df.mean())
        # Impute categorical variables with mode
        df = df.apply(lambda i: i.fillna(i.mode()[0]))
        return df

    def process_features(df):
        ########
        # Takes in a dataframe and converts categorical variables into indicator variables
        # and rescales numerical variables between -1 and 1
        ########
        # Split dataset in two, one with all numerical features and one with all categorical features
        num_df = df.select_dtypes(include=['float64'])
        cat_df = df.select_dtypes(include=['object'])
        # Convert categorical features into indicator variables
        if len(cat_df.columns) > 0:
            cat_df = convert_to_indicators(cat_df)
        # Rescale numerical features between -1 and 1
        num_df = ((1/(num_df.max()-num_df.min()))*(2*num_df-num_df.max()-num_df.min()))
        # Recombine categorical and numerical feature into one dataframe
        df = num_df.join(cat_df)
        # Replace NaN's that were caused by division by 0 when rescaling with 0's
        #### This may need to be verified for accuracy ####
        df = df.fillna(0)
        return df

    def convert_to_indicators(df):
        ########
        # Takes in a dataframe and makes a new dataframe with indicator variables
        # for each column of the provided dataframe
        ########
        # Create a new data frame with indicator variables for the first column
        # Needs to be done with this so that other indicator variables can be joined by iteration
        df1 = pd.get_dummies(df.iloc[:, 0])
        df1 = df1.iloc[:, 0:1]
        # Iterate through columns creating indicator variables for each column and
        # join the indicator variables to the new dataframe created above
        if len(df.columns) > 1:
            for i in range(1, len(df.columns)):
                df2 = pd.get_dummies(df.iloc[:, i])
                df1 = df1.join(df2.iloc[:, 0:len(df2.columns)-1])
        return df1

    def transform_and_split(features, labels):
        ########
        # Takes in two dataframes for the features and labels of a dataset and
        # outputs a list with training and testing sets for each
        ########
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)
        data_list = [x_test, x_train, y_test, y_train]
        return data_list

    # Create a dataframe from a provided path
    data = build_df(location)
    # Separate dataframe into labels and features
    y = data.pop(data.columns[len(data.columns)-1])
    x = process_features(impute_missing_values(data))
    return transform_and_split(x, y)


def model_select(data_list):

    model_list = [[SVC(), 'SVC()'],
                  [KNeighborsClassifier(), 'KNeighborsClassifier()']]

    def get_param_grid(cur_model, points):
        c_range = 10.0 ** np.arange(-2, 9)
        gamma_range = 10.0 ** np.arange(-5, 4)
        neighbor_range = np.arange(2, points + 1)
        leaf_range = np.arange(10, 101)
        # fix to optimize iterations
        paramaters = {'SVC()': [{'C': c_range,
                                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                 'degree': [1, 2, 3, 4, 5, 6],
                                 'gamma': gamma_range,
                                 'random_state': [33]}],
                      'KNeighborsClassifier()': [{'n_neighbors': neighbor_range,
                                                  'weights': ['uniform', 'distance'],
                                                  'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                                                  'leaf_size': leaf_range}]}
        return paramaters[cur_model]
    #find out why this needs to be scaled down (by 0.65) (maybe the dataset is split again
    num = int(data_list[3].size * 0.65)
    for model in model_list:
        param_grid = get_param_grid(model[1], num)
        clf = GridSearchCV(model[0], param_grid)
        clf.fit(data_list[1], data_list[3])
        train_acc = str(clf.score(data_list[1], data_list[3]))
        print('accuracy on training set for ' + model[1] + ': ' + train_acc)


# Begin timer for output
start = time.time()
# Generate df and split into labels/features
data = process_csv('Trial_Data(50).csv')
model_select(data)

end = time.time()
print("\n\nTime to run: " + str(end - start))