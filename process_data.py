import pandas as pd
from sklearn.cross_validation import train_test_split


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

print process_csv('Trial_Data(pp).csv')