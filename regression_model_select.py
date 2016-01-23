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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score


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
        np.round(df, 2)
        df = df.apply(lambda i: i.astype(np.float32))
        #num_df = df.select_dtypes(include=['float64'])
        #cat_df = df.select_dtypes(include=['object'])
        # Convert categorical features into indicator variables
        # if len(cat_df.columns) > 0:
        #     cat_df = convert_to_indicators(cat_df)
        # Rescale numerical features between -1 and 1
        df = ((1/(df.max()-df.min()))*(2*df-df.max()-df.min()))
        np.round(df, 2)
        # Recombine categorical and numerical feature into one dataframe
        # df = num_df.join(cat_df)
        # Replace NaN's that were caused by division by 0 when rescaling with 0's
        #### This may need to be verified for accuracy ####
        df = df.apply(lambda i: i.replace([np.inf, -np.inf], np.nan))
        df = df.fillna(0)
        return df

    def transform_and_split(features, labels):
        ########
        # Takes in two dataframes for the features and labels of a dataset and
        # outputs a list with training and testing sets for each
        ########
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)
        data_list = [x_test, x_train, y_test, y_train]
        return data_list

    # Create a dataframe from a provided path
    data = build_df(location[0])
    # Separate dataframe into labels and features
    y = data.pop(data.columns[len(data.columns)-1])
    y = y[0:location[1]]
    #y = y.apply(lambda i: i.astype(np.float32))
    x = process_features(impute_missing_values(data))
    if location[0] == 'Oil Forecast Data 1Q.csv' or location == 'Nat Gas Forecast 1Q.csv':
        x_pred = x[location[2] - 62:location[2] + 1]
    else:
        x_pred = x[location[2] - 251:location[2] + 1]
    x = x[0:location[1]]
    return [transform_and_split(x, y), x_pred]


def model_select(data_list, model_list, k, output):

    def get_param_grid(cur_model, points):
        try:
            #c_range = 10.0 ** np.arange(-2, 5)
            print 'Getting Parameter grid...'
            out_txt.write('Getting Parameter grid...')
            gamma_range = 10.0 ** np.arange(-2, 2)
            neighbor_range = 2.0 ** np.arange(2, 5)
            leaf_range = np.arange(10, 101)
            depth_range = np.arange(1,10)
            # fix to optimize iterations
            paramaters = {'GradientBoostingRegressor()': [{'loss': ['lad', 'huber'],
                                      'max_depth': depth_range}],
                          'RandomForestRegressor()': [{}],
                          'DecisionTreeRegressor()': [{}],
                          'KNeighborsRegressor()': [{'n_neighbors': neighbor_range}],
                          'SVR()': [{'kernel': ['rbf', 'sigmoid'], 'gamma': gamma_range}]}
            return paramaters[cur_model]
        except:
            print('could not get parameter grid')

    def train_and_eval(model, data_list, k, output):
        print '\nEavaluating...'
        out_txt.write('\nEavaluating...')
        x_train = data_list[0][0]
        x_test = data_list[0][1]
        y_train = data_list[0][2]
        y_test = data_list[0][3]
        x_pred = data_list[1]

        scores = ['r2']
        #find out why this needs to be scaled down (by 0.65) (maybe the dataset is split again
        num = int(data_list[0][3].size * 0.65)
        for cur_score in scores:
            print('*****************')
            out_txt.write('*****************')
            print("\nTuning hyper-parameters for %s." % cur_score)
            out_txt.write("\nTuning hyper-parameters for %s." % cur_score)
            param_grid = get_param_grid(model[1], num)
            # print('Found parameter grid.')
            # out_txt.write('Found parameter grid.')

            # print('Setting grid search object.')
            # out_txt.write('Setting grid search object.')
            reg = GridSearchCV(model[0], param_grid, cv=5, scoring=cur_score)
            # print('Grid search object is set.')
            # out_txt.write('Grid search object is set.')
            # print('Model will now be fit.')
            # out_txt.write('Model will now be fit.')
            reg.fit(x_train, y_train)
            print(model[1] + ' is fit.')
            out_txt.write(model[1] + ' is fit.')

            print('Best parameters set found on development set for ' + cur_score + ':\n')
            out_txt.write('Best parameters set found on development set for ' + cur_score + ':\n')
            print(reg.best_estimator_)
            out_txt.write(str(reg.best_estimator_))
            # print('\nGrid scores on development set ' + ' for ' + cur_score + ': ')
            # out_txt.write('\nGrid scores on development set ' + ' for ' + cur_score + ': ')
            # for params, mean_score, scores in reg.grid_scores_:
            #     print("\n%0.3f (+/-%0.03f) for %r"
            #           % (mean_score, scores.std() / 2, params))
            #     out_txt.write("\n%0.3f (+/-%0.03f) for %r"
            #           % (mean_score, scores.std() / 2, params))
            # print('\n')
            # out_txt.write('\n')

            # print('Detailed regression report ' + ' for ' + cur_score + ': ')
            # out_txt.write('Detailed regression report ' + ' for ' + cur_score + ': ')
            # print("The model is trained on the full development set.")
            # out_txt.write("The model is trained on the full development set.")
            # print("The scores are computed on the full evaluation set.")
            # out_txt.write("The scores are computed on the full evaluation set.")
            y_true, y_pred = y_test, reg.predict(x_test)
            output.write('\n5-Fold Cross Validation ' + ' for ' + cur_score + ': ')
            print('\n5-Fold Cross Validation ' + ' for ' + cur_score + ': ')
            cv = KFold(len(y_train), k, shuffle=True, random_state=0)
            scores = cross_val_score(reg, x_train, y_train, scoring=cur_score, cv=cv)
            output.write('\nCross Validation Score: ' + ' for ' + cur_score + ': ' + '\n' + str(scores))
            output.write('\n' + str('Mean Score ' + ' for ' + cur_score +
                                    ': {0:.3f} (+/-{1:.3f})'.format(np.mean(scores), sem(scores))))
            print('\nCross Validation Score: ' + ' for ' + cur_score + ': ' + '\n' + str(scores))
            print('\n' + str('Mean Score ' + ' for ' + cur_score +
                             ': {0:.3f} (+/-{1:.3f})'.format(np.mean(scores), sem(scores))))
            print('***********************')
            out_txt.write('***********************')
            forecast = reg.best_estimator_.predict(x_pred)
            out_txt.write('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('\nExpected value for the period: ')
            out_txt.write('\nExpected value for the period: ')
            print(forecast[-1])
            out_txt.write('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('\nWith the following scores:')
            out_txt.write('\nWith the following scores:')
            print('Explained Variance:')
            out_txt.write('Explained Variance:')
            print(explained_variance_score(y_true, y_pred))
            out_txt.write(explained_variance_score(y_true, y_pred))
            print('\nMean Absolute Error:')
            out_txt.write('\nMean Absolute Error:')
            print(mean_absolute_error(y_true, y_pred))
            out_txt.write(mean_absolute_error(y_true, y_pred))
            print('\nMean Squared Error:')
            out_txt.write('\nMean Squared Error:')
            print(mean_squared_error(y_true, y_pred))
            out_txt.write(mean_squared_error(y_true, y_pred))
            print('\nR-square:')
            out_txt.write('\nR-square:')
            r2 = r2_score(y_true, y_pred)
            print(r2)
            out_txt.write(r2)
            # if r2 > best_r2:
            #     reg_best = reg
        # return best_r2, reg_best

    best_score = 0
    best_model = ""
    # for model in model_list:
    # try:
    print("*************************************************************")
    out_txt.write("*************************************************************")
    print 'Testing: ' + model_list[1]
    out_txt.write('Testing: ' + model_list[1])
    # cur_score, cur_model = train_and_eval(model_list, data_list, k, output)
    train_and_eval(model_list, data_list, k, output)
    # print('The best model was:')
    # out_txt.write('The best model was:')
    # print '\n\n' + str(cur_model) + '\n' + 'Score: ' + str(cur_score)
    # out_txt.write('\n\n' + str(cur_model) + '\n' + 'Score: ' + str(cur_score))
    # if cur_score > best_score:
    #     best_model = str(cur_model)
    # except:
    #     print '\n\n' + str(model[1]) + '\n\n The above model had an error. It was not considered'
    #     out_txt.write('\n\n' + str(model[1]) + '\n\n The above model had an error. It was not considered')
    # return best_model


out_txt = open('output.txt', "wb")

files = [['Nat Gas Forecast 1Q.csv', 4534, 4597],
         ['Nat Gas Forecast 1Y.csv', 4344, 4597],
         ['Nat Gas Forecast 2Y.csv', 4093, 4597],
         ['Nat Gas Forecast 3Y.csv', 3836, 4597],
         ['Nat Gas Forecast 4Y.csv', 3582, 4597],
         ['Nat Gas Forecast 5Y.csv', 3326, 4597]]
#          ['Oil Forecast Data 1Q.csv', 7361, 7424],
#          ['Oil Forecast Data 1Y.csv', 7171, 7424],
#          ['Oil Forecast Data 2Y.csv', 6921, 7424],
#          ['Oil Forecast Data 3Y.csv', 6665, 7424],
#          ['Oil Forecast Data 4Y.csv', 6401, 7424],
#          ['Oil Forecast Data 5Y.csv', 6150, 7424]]



model_list = [SVR(), 'SVR()']

for cur_file in files:
    print(cur_file[0])
    data = process_csv(cur_file)
    best = model_select(data, model_list, 5, out_txt)
    out_txt.write('\n\n*************************************************************************')
    out_txt.write('\n\nThe test is finished for: ' + cur_file[0])
    out_txt.write('\n\n*************************************************************************')