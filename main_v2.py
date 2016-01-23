__author__ = 'Ricardo'

import time, sys
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
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
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from operator import itemgetter

# Begin timer for output
start_main = time.time()

# Set variable for a file in the current path for generating logs
out_txt = open('output.txt', "wb")


def process_csv(location):
    """
    :param location: A folder path for a csv file
    :return: A list containing the test and training set [x_test, x_train, y_test, y_train]
    """

    print('#############')
    print('\nProcessing CSV file...')
    print('#############')

    def build_df(csv_location):
        """ Builds a pandas dataframe from a csv file
        :param csv_location: A folder path for a csv file
        :return: A dataframe with the source csv's data
        """

        print('Dataframe is being built...')
        df = pd.read_csv(csv_location)
        df.rename(columns=lambda i: i.replace(' ', '_').lower(), inplace=True)
        return df

    def impute_missing_values(df):
        """
        :param df: A raw dataframe with missing values
        :return: A clean dataframe with the missing values filled in
        """

        print('Values are being imputed in dataframe...')
        # Impute numerical variables with mean
        df = df.fillna(df.mean())
        # Impute categorical variables with mode
        df = df.apply(lambda i: i.fillna(i.mode()[0]))

        return df

    def process_features(df):
        """ Converts categorical variables into indicator variables and rescales numerical variables between -1 and 1
        :param df: A clean dataframe
        :return: A dataframe with standardized values
        """

        # Split dataset in two, one with all numerical features and one with all categorical features
        print('Processing features...')
        num_df = df.select_dtypes(include=['float64'])
        cat_df = df.select_dtypes(include=['object'])

        # Convert categorical features into indicator variables
        if len(cat_df.columns) > 0:
            cat_df = convert_to_indicators(cat_df)

        # Rescale numerical features between -1 and 1
        if len(num_df.columns) > 0:
            num_df = ((1/(num_df.max()-num_df.min()))*(2*num_df-num_df.max()-num_df.min()))

        # Recombine categorical and numerical feature into one dataframe
        df = num_df.join(cat_df)

        # Replace NaN's that were caused by division by 0 when rescaling with 0's
        # ToDo: This may need to be verified for accuracy
        df = df.fillna(0)

        return df

    def convert_to_indicators(df):
        """
        :param df: A dataframe with only categorical features
        :return: A new dataframe with indicator variables for each column of the provided dataframe
        """

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

    def transform_and_split(features, labels):
        """
        :param features: A dataframe with the features
        :param labels: A dataframe with the labels of the data set
        :return: A dictionary with each key being either the training or test set, and the respective value being
        a dataframe of processed/standardized data for that particular set
        """

        print('Performing preliminary data split')
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)
        data_dict = {'x_test': x_test, 'x_train': x_train,
                     'y_test': y_test, 'y_train': y_train}

        return data_dict

    # Create a dataframe from the provided path
    data = build_df(location)
    # Separate dataframe into labels and features; the last column has the labels
    y = data.pop(data.columns[len(data.columns)-1])
    x = process_features(impute_missing_values(data))
    return transform_and_split(x, y)


def gen_model_results(data_dict, model_list, output):

    print('################')
    print('\nNow generating model results...')
    print('################')
    x_train, x_dev, y_train, y_dev = train_test_split(
        data_dict['x_train'], data_dict['y_train'], test_size=0.5, random_state=33)

    def param_optimization(cur_model, x_dev, y_dev, output):
        print('Conducting parameter optimization...')

        def get_param_grid(model, points, rand):
            print('Retrieving parameter grid...')

            try:
                c_range = 10.0 ** np.arange(-1, 1)
                # print 'Getting Parameter grid...'
                # out_txt.write('Getting Parameter grid...')
                gamma_range = 10.0 ** np.arange(-1, 1)
                neighbor_range = 2.0 ** np.arange(4, 5)
                leaf_range = np.arange(10, 15)

                # ToDo: fix to optimize iterations
                if not rand:
                    grid_params = {'SVC()':
                                       [{'C': c_range,
                                         'kernel': ['linear'],
                                         'gamma': gamma_range,
                                         'random_state': [33],
                                         'probability': [True]}],
                                   'KNeighborsClassifier()':
                                       [{'n_neighbors': neighbor_range,
                                         'weights': ['uniform', 'distance'],
                                         'algorithm': ['auto'],
                                         'leaf_size': leaf_range}],
                                   'NearestCentroid()':
                                       [{'metric': ['euclidian', 'manhattan']}]}

                    return grid_params[model]

                else:
                    rand_params = {'SVC()':
                                       [{'C': c_range,
                                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                         'degree': [1, 2, 3, 4, 5, 6],
                                         'gamma': gamma_range,
                                         'probability': ['True', 'False'],
                                         'shrinking': ['True', 'False'],
                                         'random_state': 33}],
                                   'KNeighborsClassifier()':
                                       [{'n_neighbors': neighbor_range,
                                         'weights': ['uniform', 'distance'],
                                         'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                                         'metric': ['euclidian', 'manhattan', 'chebyshev', 'minkowski'],
                                         'leaf_size': leaf_range}],
                                   'NearestCentroid()':
                                       [{'metric': ['euclidian', 'manhattan', 'chebyshev', 'minkowski']}]}

                    return rand_params[model]

            except:
                print('Could not get parameter grid')

        def optimize_params(model, cur_score, x_dev, y_dev, rand):

            def report(grid_scores, n_top=1):
                top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

                for i, score in enumerate(top_scores):
                    print("Model with rank: {0}".format(i + 1))
                    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        score.mean_validation_score, np.std(score.cv_validation_scores)))
                    print("Parameters: {0}".format(score.parameters))
                    print("")

            if not rand:
                start = time.time()
                param_grid = get_param_grid(model[1], int(y_dev.size * 0.65), rand)
                print('Optimizing parameters for: ' + model[1] + '. With score: ' + cur_score + '.')
                grid_model = GridSearchCV(model[0], param_grid, cv=5, scoring=cur_score, n_jobs=1)
                grid_model.fit(x_dev, y_dev)
                print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                      % (time.time() - start, len(grid_model.grid_scores_)))
                report(grid_model.grid_scores_)

                return grid_model

            else:
                start = time.time()
                param_grid = get_param_grid(cur_model[1], int(y_dev.size * 0.65), rand)
                rand_model = RandomizedSearchCV(model[0], param_grid, cv=5, scoring=cur_score)
                rand_model.fit(x_dev, y_dev)
                print("RandomSearchCV took %.2f seconds for %d candidate parameter settings."
                      % (time.time() - start, len(rand_model.grid_scores_)))
                report(rand_model.grid_scores_)

                return rand_model

        # scores = ['confusion matrix', 'accuracy_score', 'f1_score',
        #           'fbeta_score', 'hamming_loss','jacccard_similarity_score',
        #           'log_loss', 'precision_recall_fscore_support',
        #           'precision score', 'recall_score', 'zero_one_loss',
        #           'average_precision_score', 'roc_auc_score']
        # The following list contains the only possible values for multi-class classification
        scores = ['accuracy', 'log_loss', 'recall']

        out_list = []
        # for cur_score in scores:
        #     opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
        #     out_list.append([cur_model, cur_score, opt_model.best_estimator_])
        #     #optimize_params(cur_model, cur_score, x_dev,y_dev, False)

        def main(cur_score, out_list):
            opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
            out_list.append([cur_model, cur_score, opt_model.best_estimator_])
            #optimize_params(cur_model, cur_score, x_dev,y_dev, False)

        multithread(main, scores, out_list, threads=2)

        return out_list

    def train_and_eval(x_train, y_train, x_test, y_test, model, param_result):
        print('\nTraining and evaluating...')

        for result_list in param_result:
            print('Fitting: ' + str(result_list[2]))

            opt_model = result_list[2]
            opt_model.fit(x_train, y_train)
            y_pred = opt_model.predict(x_test)

            print('\nClassification Report:')
            print(metrics.classification_report(y_test, y_pred))
            print('\nAccuracy Score:')
            print(metrics.accuracy_score(y_test, y_pred))
            print('\nConfusion Matrix:')
            print(metrics.confusion_matrix(y_test, y_pred))
            print('\nF1-Score:')
            print(metrics.f1_score(y_test, y_pred))
            print('\nHamming Loss:')
            print(metrics.hamming_loss(y_test, y_pred))
            print('\nJaccard Similarity:')
            print(metrics.jaccard_similarity_score(y_test, y_pred))
            #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # print('\nLog Loss:')
            # print(metrics.log_loss(y_test, y_pred))
            #vvv multiclass not supported
            # print('\nMatthews Correlation Coefficient:')
            # print(metrics.matthews_corrcoef(y_test, y_pred))
            print('\nPrecision:')
            print(metrics.precision_score(y_test, y_pred))
            #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # print('\nRecall:')
            # print(metrics.recall(y_test, y_pred))
            print()

    for model in model_list:
        cur_result = param_optimization(model, x_dev, y_dev, output)
        train_and_eval(x_train, y_train, data_dict['x_test'],
                       data_dict['y_test'], model, cur_result)


def multithread(function, items, extra_variable='', threads=2):
    """ Takes the main function to run in parallel, inputs the variable(s) and returns the results.
    :param function: The main function to process in parallel.
    :param items: A list of strings that are passed into the function for each thread.
    :param extra_variable: One additional variable that can be passed into the function.
    :param threads: The number of threads to use. The default is 2, but the threads are not CPU core bound.
    :return: The results of the function passed into this function.
    """

    if __name__ == '__main__':

        # """ A CPU core dependent multiprocessing technique.
        # The synchronized variant, which locks the main program until a process is finished. Order is retained. """
        # pool = Pool(threads)
        # results = [pool.apply(function, args=(item, extra_variable)) for item in items]
        # pool.close()
        # pool.join()

        # """ A thread dependent multiprocessing technique. Theoretically, an unlimited number of threads can be used.
        # The synchronized variant, which locks the main program until a process is finished. Order is retained. """
        # pool = ThreadPool(threads)
        # results = [pool.apply(function, args=(item, extra_variable)) for item in items]
        # pool.close()
        # pool.join()

        """ A thread dependent multiprocessing technique. Theoretically, an unlimited number of threads can be used.
        The async variant, which submits all processes at once and retrieve the results as soon as finished. """
        pool = ThreadPool(threads)
        output = [pool.apply_async(function, args=(item, extra_variable)) for item in items]
        results = [p.get() for p in output]

        return results

#Copy from previous code
#NameError: global name 'clf' is not defined
    # out_txt.write('\n\n************************************')
    # out_txt.write('\n\n' + str(clf))
    # out_txt.write('\nAccuracy on training set: ')
    # out_txt.write('\n' + str(new_clf.score(x_train, y_train)))
    # out_txt.write('\nAccuracy on testing set: ')
    # out_txt.write('\n' + str(new_clf.score(x_test, y_test)))
    # y_pred = new_clf.predict(x_test)
    # out_txt.write('\nClassification Report: ')
    # out_txt.write('\n' + str(metrics.classification_report(y_test, y_pred)))
    # out_txt.write('\nConfusion Matrix: ')
    # out_txt.write('\n' + str(metrics.confusion_matrix(y_test, y_pred)))
    # return new_clf.score(x_test, y_test), str(new_clf)
    ######


# model_list = [[SVC(), 'SVC()'],
#               [NearestCentroid(), 'NearestCentroid()'],
#               [KNeighborsClassifier(), 'KNeighborsClassifier()']]
# Need to test compatibility with other Classifiers other than SVM()
model_list = [[SVC(), 'SVC()']]

# Generate df from file in path and split into labels/features
# Then generate model results for objects in model_list
gen_model_results(process_csv('Trial_Data.csv'), model_list, out_txt)
print('Total runtime: ' + str(time.time() - start_main) + ' s')

if sys.version_info < (3, 0):
    raw_input('Press Enter to close:')
else:
    input('Press Enter to close:')

# # To-Do
# # Need to find a multinomial logit implementation (sci-kit only has one-vs-all classification not true multinomial)
# # Need to reassess neural net implementation and bring into main
# # Need to improve implementation of GridSearch and add RandomSearch
# # Last Read:
#
# C:\Python27\python.exe "C:/Users/Ricardo/PycharmProjects/Student Attrition/main.py"
#
# #############
# Processing CSV file...
# #############
# Dataframe is being built...
# Values are being imputed in dataframe...
# Processing features...
# Performing prelimianry datasplit
# CSV was successfully processed!
#
# ################
# Now generating model results...
# ################
# Conducting parameter optimization...
# Retrieving parameter grid...
# Optimizing parameters for: SVC(). With score: accuracy.
# GridSearchCV took 2.44 seconds for 256 candidate parameter settings.
# Model with rank: 1
# Mean validation score: 0.579 (std: 0.080)
# Parameters: {'kernel': 'poly', 'C': 0.01, 'probability': True, 'degree': 3, 'random_state': 33, 'gamma': 10.0}
#
# Retrieving parameter grid...
# Optimizing parameters for: SVC(). With score: log_loss.
# GridSearchCV took 3.37 seconds for 256 candidate parameter settings.
# Model with rank: 1
# Mean validation score: -1.014 (std: 0.051)
# Parameters: {'kernel': 'poly', 'C': 10.0, 'probability': True, 'degree': 4, 'random_state': 33, 'gamma': 0.01}
#
# Retrieving parameter grid...
# Optimizing parameters for: SVC(). With score: recall.
# GridSearchCV took 3.24 seconds for 256 candidate parameter settings.
# Model with rank: 1
# Mean validation score: 0.579 (std: 0.080)
# Parameters: {'kernel': 'poly', 'C': 0.01, 'probability': True, 'degree': 3, 'random_state': 33, 'gamma': 10.0}
#
#
# Training and evaluating...
# Fitting: SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#   gamma=10.0, kernel='poly', max_iter=-1, probability=True,
#   random_state=33, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#     Dropped       0.64      0.60      0.62        15
#   Graduated       0.50      0.20      0.29         5
# Transferred       0.33      0.60      0.43         5
#
# avg / total       0.55      0.52      0.52        25
#
#
# Accuracy Score:
# 0.52
#
# Confusion Matrix:
# [[9 1 5]
# [3 1 1]
# [2 0 3]]
#
# F1-Score:
# 0.515270935961
#
# Hamming Loss:
# 0.48
#
# Jaccard Similarity:
# 0.52
#
# Precision:
# 0.552380952381
# ()
# Fitting: SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=4,
#   gamma=0.01, kernel='poly', max_iter=-1, probability=True,
#   random_state=33, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#     Dropped       0.60      1.00      0.75        15
#   Graduated       0.00      0.00      0.00         5
# Transferred       0.00      0.00      0.00         5
#
# avg / total       0.36      0.60      0.45        25
#
#
# Accuracy Score:
# 0.6
#
# Confusion Matrix:
# [[15  0  0]
# [ 5  0  0]
# [ 5  0  0]]
#
# F1-Score:
# 0.45
# C:\Python27\lib\site-packages\sklearn\metrics\metrics.py:1771: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#
#   'precision', 'predicted', average, warn_for)
# Hamming Loss:
# C:\Python27\lib\site-packages\sklearn\metrics\metrics.py:1771: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
# 0.4
#   'precision', 'predicted', average, warn_for)
#
# C:\Python27\lib\site-packages\sklearn\metrics\metrics.py:1771: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
# Jaccard Similarity:
#   'precision', 'predicted', average, warn_for)
# 0.6
#
# Precision:
# 0.36
# ()
# Fitting: SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#   gamma=10.0, kernel='poly', max_iter=-1, probability=True,
#   random_state=33, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#     Dropped       0.64      0.60      0.62        15
#   Graduated       0.50      0.20      0.29         5
# Transferred       0.33      0.60      0.43         5
#
# avg / total       0.55      0.52      0.52        25
#
#
# Accuracy Score:
# 0.52
#
# Confusion Matrix:
# [[9 1 5]
# [3 1 1]
# [2 0 3]]
#
# F1-Score:
# 0.515270935961
#
# Hamming Loss:
# 0.48
#
# Jaccard Similarity:
# 0.52
#
# Precision:
# 0.552380952381
# ()
#
# Process finished with exit code 0
