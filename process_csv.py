__author__ = 'Ricardo'

import pandas as pd
from sklearn.cross_validation import train_test_split


def build_df(location):
    """

    :param csv_location:
    :return:
    """

    print('Dataframe is being built...')
    df1 = pd.read_csv(location)
    df1.rename(columns=lambda i: i.replace(' ', '_').lower(), inplace=True)

    return df1


def _impute_missing_values(df):
    ########
    # Takes in a dataframe and imputes missing values
    ########

    print('Values are being imputed in dataframe...')

    # Impute numerical variables with mean
    df1 = df.fillna(df.mean())
    # Impute categorical variables with mode
    df1 = df1.apply(lambda i: i.fillna(i.mode()[0]))

    return df1


def _process_features(df):
    ########
    # Takes in a dataframe and converts categorical variables into indicator variables
    # and rescales numerical variables between -1 and 1
    ########
    # Split dataset in two, one with all numerical features and one with all categorical features
    print('Processing features...')
    # num_df = df.select_dtypes(include=['float64'])
    # cat_df = df.select_dtypes(include=['object'])
    # # Convert categorical features into indicator variables
    # if len(cat_df.columns) > 0:
    #     cat_df = _convert_to_indicators(cat_df)
    # # Rescale numerical features between -1 and 1
    # if len(num_df.columns) > 0:
    #     num_df = ((1/(num_df.max()-num_df.min()))*(2*num_df-num_df.max()-num_df.min()))
    ### Since data was preprocessed
    df1 = ((1/(df.max()-df.min()))*(2*df-df.max()-df.min()))
    # # Recombine categorical and numerical feature into one dataframe
    # df = num_df.join(cat_df)
    # Replace NaN's that were caused by division by 0 when rescaling with 0's
    # This occurs when all values are 0 (eg. indicator variables)
    return df1.fillna(0)


def _convert_to_indicators(df):
    ########
    # Takes in a dataframe and makes a new dataframe with indicator variables
    # for each column of the provided dataframe
    ########

    print('Converting some features to indicator variables...')

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


def split(df):
    ########
    # Takes in two dataframes for the features and labels of a dataset and
    # outputs a dictionary with training and keys relating to training testing sets for each
    ########

    print('Performing prelimianry datasplit')

    labels = df.pop(df.columns[len(df.columns)-1])
    features = df
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.25,
                                                        random_state=33)
    data_dict = {'x_test': x_test, 'x_train': x_train,
                 'y_test': y_test, 'y_train': y_train}
    return data_dict


def process_csv(location, impute_values=True, indicator_variable=False):

    df = build_df(location)

    # Separate dataframe into labels and features
    if impute_values and not indicator_variable:
        df = _impute_missing_values(df)

    df_dict = split(df)

    return df_dict