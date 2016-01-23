__author__ = 'Ricardo'

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

from multiprocessing.dummy import Pool as ThreadPool

from operator import itemgetter

# Begin timer for output
start_main = time.time()

# Set variable for a file in the current path for generating logs
out_txt = open('output.txt', "wb")

# Setup list for column order of output dataframe
column_list2 = ['model', 'kernel', 'degree', 'gamma', 'probability',
                'C', 'penalty', 'solver', 'leaf_size', 'metric', 'algorithm',
                'n_neighbors', 'n_hidden', 'alpha', 'eta0', 'learning_rate', 'search type',
                'optimization score function', 'optimization  score', 'Accuracy']

column_list = ['model', 'kernel', 'degree', 'gamma', 'probability',
               'C', 'penalty', 'solver', 'leaf_size', 'metric', 'algorithm',
               'n_neighbors', 'n_hidden', 'alpha', 'eta0', 'learning_rate', 'search type',
               'optimization score function', 'optimization score', 'Accuracy']


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
            df = df.replace(to_replace='Trans', value='Drop')
            df = df.replace(to_replace='Drop', value=0)
            df = df.replace(to_replace='Grad', value=1)

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


def gen_model_results(data_dict, model_list, output):

    print('\n################')
    print('\nNow generating model results...')
    print('\n################\n')

    x_train, x_dev, y_train, y_dev = train_test_split(
        data_dict['x_train'], data_dict['y_train'], test_size=0.5, random_state=33)

    def param_optimization(cur_model, x_dev, y_dev, output):

        print('\nConducting parameter optimization for: ' + model[1] + '...\n')

        def get_param_grid(model, points, rand):
            print('\nRetrieving parameter grid...')
            try:
                c_range = 10.0 ** np.arange(0, 3)
                # print 'Getting Parameter grid...'
                # out_txt.write('Getting Parameter grid...')
                gamma_range = 10.0 ** np.arange(-3, 1)
                # neighbor_range = np.arange(2, points, step=5)
                # leaf_range = np.arange(10, points, step=5)
                neighbor_range = np.arange(10, 100, step=4)
                leaf_range = np.arange(1, 65, step=5)
                alpha_range = 10.0**np.arange(-7, 0)
                eta_range = 10.0**np.arange(-3, 0)
                learning_rate_range = 10.0**np.arange(-3, 1)
                component_range = 2**np.arange(1, 7, step=1)
                if not rand:
                    grid_params = {'SVC()': [{'C': c_range,
                                              'kernel': ['poly'],
                                              'degree': [2, 3, 5, 8],
                                              'gamma': gamma_range,
                                              'probability': [True],
                                              'class_weight': ['auto'],
                                              'cache_size': [1000],
                                              'max_iter': [250000],
                                              'verbose': [False]},
                                             {'C': c_range,
                                              'kernel': ['rbf', 'sigmoid'],
                                              'gamma': gamma_range,
                                              'probability': [True],
                                              'cache_size': [1000],
                                              'class_weight': ['auto'],
                                              'max_iter': [250000],
                                              'verbose': [False]},
                                             {'C': c_range,
                                              'kernel': ['linear'],
                                              'probability': [True],
                                              'cache_size': [1000],
                                              'class_weight': ['auto'],
                                              'max_iter': [250000],
                                              'verbose': [False]}],
                                   'KNeighborsClassifier()': [{'n_neighbors': neighbor_range,
                                                               'weights': ['uniform'],
                                                               'algorithm': ['brute'],
                                                               'metric': ['euclidean', 'manhattan']}],
                                   'LogisticRegression()': [{'penalty': ['l1', 'l2'],
                                                             'C': c_range,
                                                             'class_weight': ['auto'],
                                                             'multi_class': ['multinomial', 'ovr'],
                                                             'solver': ['lbfgs']}],
                                   'ANNLogisticRegression()': [{'annlogisticregression__penalty': ['l1', 'l2'],
                                                                'annlogisticregression__C': c_range,
                                                                'annlogisticregression__class_weight': ['auto'],
                                                                'annlogisticregression__multi_class': ['multinomial'],
                                                                'annlogisticregression__solver': ['lbfgs']}],
                                   'Perceptron()': [{'perceptron__penalty': ['l1', 'l2', 'elasticnet', None],
                                                     'perceptron__alpha': alpha_range,
                                                     'perceptron__eta0': eta_range,
                                                     'perceptron__class_weight': ['auto'],
                                                     'perceptron__warm_start': [True],
                                                     'rbm__learning_rate': learning_rate_range,
                                                     'rbm__n_components': component_range}],
                                   'MultilayerPerceptronClassifier()': [{'alpha': alpha_range,
                                                                         'n_hidden': component_range,
                                                                         'learning_rate': ['constant'],
                                                                         'warm_start': [True],
                                                                         'eta0': eta_range}]}
                    return grid_params[model]
                else:
                    rand_params = {'SVC()': {'C': stats.beta(loc=1, scale=1000, a=.85, b=1),
                                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                             'degree': [2, 3, 4, 5, 6],
                                             'gamma': stats.beta(loc=.0001, scale=30, a=1.1, b=50),
                                             'random_state': [10],
                                             'probability': [True],
                                             'cache_size': [1000],
                                             'class_weight': ['auto'],
                                             'max_iter': [250000],
                                             'verbose': [False]},
                                   'KNeighborsClassifier()': {'n_neighbors': stats.randint(low=2, high=100),
                                                              'weights': ['uniform', 'distance'],
                                                              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                                              'metric': ['euclidean', 'manhattan'],
                                                              'leaf_size': stats.randint(low=4, high=60)},
                                   'LogisticRegression()': {'penalty': ['l1', 'l2'],
                                                            'C': stats.beta(loc=1, scale=1000, a=.85, b=1),
                                                            'class_weight': ['auto'],
                                                            'multi_class': ['multinomial'],
                                                            'solver': ['lbfgs']},
                                   'ANNLogisticRegression()': {'annlogisticregression__penalty': ['l1', 'l2'],
                                                               'annlogisticregression__C': c_range,
                                                               'annlogisticregression__class_weight': ['auto'],
                                                               'annlogisticregression__multi_class': ['multinomial'],
                                                               'annlogisticregression__solver': ['lbfgs']},
                                   'Perceptron()': {'perceptron__penalty': ['l1', 'l2', 'elasticnet', None],
                                                    'perceptron__alpha': stats.expon(scale=.005),
                                                    'perceptron__eta0': stats.expon(scale=.8),
                                                    'perceptron__class_weight': ['auto'],
                                                    'perceptron__warm_start': [True],
                                                    'rbm__learning_rate': learning_rate_range,
                                                    'rbm__n_components': stats.randint(1, 140)},
                                   'MultilayerPerceptronClassifier()': {'alpha': stats.beta(loc=.0000001,
                                                                                            scale=1, a=.7, b=1),
                                                                        'learning_rate': ['constant', 'invscaling'],
                                                                        'warm_start': [True],
                                                                        'n_hidden': stats.randint(1, 256),
                                                                        'eta0': stats.uniform(10**-3, .999)}}
                    return rand_params[model]
            except:
                print('could not get parameter grid')

        def optimize_params(model, cur_score, x_dev, y_dev, rand):

            def report(grid_scores, n_top=1):
                top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
                for i, score in enumerate(top_scores):
                    print("Model with rank: {0}".format(i + 1))
                    print("Mean validation " + cur_score + " score: {0:.3f} (std: {1:.3f})".format(
                        score.mean_validation_score, np.std(score.cv_validation_scores)))
                    print("Parameters: {0}".format(score.parameters))
                    print("")

            if not rand:
                start = time.time()
                param_grid = get_param_grid(model[1], int(y_dev.size), rand)
                print('\nOptimizing parameters for: ' + model[1] +
                      ' using Grid Search with score: ' + cur_score + '.')
                if model[1] != 'Perceptron()' and model[1] != 'ANNLogisticRegression()':
                    grid_model = GridSearchCV(model[0], param_grid, cv=5, scoring=cur_score, n_jobs=3)
                elif model[1] != 'ANNLogisticRegression()':
                    perceptron = Perceptron()
                    rbm = BernoulliRBM()
                    clf = Pipeline(steps=[('rbm', rbm), ('perceptron', perceptron)])
                    grid_model = GridSearchCV(clf, param_grid, cv=5, scoring=cur_score, n_jobs=3)
                else:
                    annlogisticregression = LogisticRegression()
                    rbm = BernoulliRBM()
                    clf = Pipeline(steps=[('rbm', rbm), ('annlogisticregression', annlogisticregression)])
                    grid_model = GridSearchCV(clf, param_grid, cv=5, scoring=cur_score, n_jobs=3)
                grid_model.fit(x_dev, y_dev)
                print("\nGridSearchCV took %.2f seconds for %d candidate parameter settings."
                      % (time.time() - start, len(grid_model.grid_scores_)))
                report(grid_model.grid_scores_)
                return grid_model
            else:
                start = time.time()
                param_grid = get_param_grid(cur_model[1], int(y_dev.size), rand)
                print('\nOptimizing parameters for: ' + model[1] +
                      ' using Random Search with score: ' + cur_score + '.')
                if model[1] != 'Perceptron()' and model[1] != 'ANNLogisticRegression()':
                    rand_model = RandomizedSearchCV(model[0], param_distributions=param_grid,
                                                    cv=5, scoring=cur_score, n_jobs=3, n_iter=200)
                    rand_model.fit(x_dev, y_dev)
                    print("\nRandomSearchCV took %.2f seconds for %d candidate parameter settings."
                          % (time.time() - start, len(rand_model.grid_scores_)))
                    report(rand_model.grid_scores_)
                    return rand_model
                elif model[1] != 'ANNLogisticRegression()':
                    perceptron = Perceptron()
                    rbm = BernoulliRBM()
                    clf = Pipeline(steps=[('rbm', rbm), ('perceptron', perceptron)])
                    rand_model = RandomizedSearchCV(clf, param_distributions=param_grid,
                                                    cv=5, scoring=cur_score, n_jobs=3, n_iter=200)
                    rand_model.fit(x_dev, y_dev)
                    print("\nRandomSearchCV took %.2f seconds for %d candidate parameter settings."
                          % (time.time() - start, len(rand_model.grid_scores_)))
                    report(rand_model.grid_scores_)
                    return rand_model
        # scores = ['confusion matrix', 'accuracy_score', 'f1_score',
        #           'fbeta_score', 'hamming_loss','jacccard_similarity_score',
        #           'log_loss', 'precision_recall_fscore_support',
        #           'precision score', 'recall_score', 'zero_one_loss',
        #           'average_precision_score', 'roc_auc_score']
        # The following list contains the only possible values for multi-class classification

        scores = ['accuracy']
        out_list = []

        # def main(cur_score, out_list):
        for cur_score in scores:
            if cur_model[1] == "Perceptron()":
                if cur_score != 'log_loss':
                    opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                    out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                     opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
                    opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, True)
                    out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                     opt_model.best_score_, opt_model.best_params_, 'RandomSearch'])
            elif cur_model[1] == "ANNLogisticRegression()":
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
            else:
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, True)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'RandomSearch'])
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'GridSearch'])

        # multithread(main, scores, out_list, threads=1)

        return out_list

    def train_and_eval(x_train, y_train, x_test, y_test, model, param_result):
        print('\nTraining and evaluating...')

        # def get_kfold_scores(cur_model, score, score_name, result):
        #     # try:
        #     print('\n10-Fold CV ' + score_name + ' Score:')
        #     scores = cross_val_score(estimator=cur_model, X=x_test, y=y_test, cv=5,
        #                              scoring=score, n_jobs=3)
        #     print(score_name + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #     result.update({score_name: scores.mean()})
        #     # except:

        all_results = []
        for result_list in param_result:

            print('\n########\nEvaluating: ' + str(result_list[2]))
            opt_model = result_list[2]
            result = {'model': result_list[0][0:-2], 'optimization score function': result_list[1],
                      'optimization score': result_list[3], 'search type': result_list[5]}
            result.update(result_list[4])

            # result = {'model': str(result_list[2])[0:str(result_list[2]).find('(')]}
            avail_scores = [['accuracy', 'Accuracy']]
            #### vv Only metrics available for multiclass classification
            # avail_scores = [['accuracy', 'Accuracy'],
            #                 ['f1_weighted', 'F1 Score'],
            #                 ['precision_weighted', 'Precision'],
            #                 ['recall_weighted', 'Recall']]
            # for this_score in avail_scores:
            #     get_kfold_scores(cur_model=opt_model, score=this_score[0], score_name=this_score[1], result=result)

            ##### Not using k-fold cross validation
            opt_model.fit(x_train, y_train)
            joblib.dump(opt_model, 'multi_class-' +
                        str(result_list[0][0:-2]) +
                        '-' + str(result_list[1]) +
                        '-' + str(result_list[5]) + '.pkl')
            y_pred = opt_model.predict(x_test)
            print('\nClassification Report:')
            print(metrics.classification_report(y_test, y_pred))
            print('\nAccuracy Score:')
            print(metrics.accuracy_score(y_test, y_pred))
            print('\nConfusion Matrix:')
            print(metrics.confusion_matrix(y_test, y_pred))
            result.update({'Accuracy': metrics.accuracy_score(y_test, y_pred)})
            all_results.append(result)
            # print('\nF1-Score:')
            # print(metrics.f1_score(y_test, y_pred))
            # print('\nHamming Loss:')
            # print(metrics.hamming_loss(y_test, y_pred))
            # print('\nJaccard Similarity:')
            # print(metrics.jaccard_similarity_score(y_test, y_pred))
            # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # print('\nLog Loss:')
            # print(metrics.log_loss(y_test, y_pred))
            # #vvv multiclass not supported
            # print('\nMatthews Correlation Coefficient:')
            # print(metrics.matthews_corrcoef(y_test, y_pred))
            # print('\nPrecision:')
            # print(metrics.precision_score(y_test, y_pred))
            # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # print('\nRecall:')
            # print(metrics.recall(y_test, y_pred))
        return all_results

    results_list = []

    for model in model_list:
        cur_result = param_optimization(model, x_dev, y_dev, output)
        print('\nEvaluating ' + model[1] + '.')
        results_list.extend(train_and_eval(x_train, y_train, data_dict['x_test'],
                                           data_dict['y_test'], model, cur_result))
        print('\n##############################################')

    return results_list


def gen_model_results2(data_dict, model_list, output):

    print('\n################')
    print('\nNow generating model results...')
    print('\n################\n')

    x_train, x_dev, y_train, y_dev = train_test_split(
        data_dict['x_train'], data_dict['y_train'], test_size=0.5, random_state=33)

    def param_optimization(cur_model, x_dev, y_dev, output):

        print('\nConducting parameter optimization for: ' + model[1] + '...\n')

        def get_param_grid(model, points, rand):
            print('\nRetrieving parameter grid...')
            try:
                c_range = 10.0 ** np.arange(0, 3)
                # print 'Getting Parameter grid...'
                # out_txt.write('Getting Parameter grid...')
                gamma_range = 10.0 ** np.arange(-3, 1)
                # neighbor_range = np.arange(2, points, step=5)
                # leaf_range = np.arange(10, points, step=5)
                neighbor_range = np.arange(10, 100, step=4)
                leaf_range = np.arange(1, 65, step=5)
                alpha_range = 10.0**np.arange(-7, 0)
                eta_range = 10.0**np.arange(-3, 0)
                learning_rate_range = 10.0**np.arange(-3, 1)
                component_range = 2**np.arange(1, 7, step=1)
                if not rand:
                    grid_params = {'SVC()': [{'C': c_range,
                                              'kernel': ['poly'],
                                              'degree': [2, 3, 5, 8],
                                              'gamma': gamma_range,
                                              'probability': [True],
                                              'class_weight': ['auto'],
                                              'cache_size': [1000],
                                              'max_iter': [250000],
                                              'verbose': [False]},
                                             {'C': c_range,
                                              'kernel': ['rbf', 'sigmoid'],
                                              'gamma': gamma_range,
                                              'probability': [True],
                                              'cache_size': [1000],
                                              'class_weight': ['auto'],
                                              'max_iter': [250000],
                                              'verbose': [False]},
                                             {'C': c_range,
                                              'kernel': ['linear'],
                                              'probability': [True],
                                              'cache_size': [1000],
                                              'class_weight': ['auto'],
                                              'max_iter': [250000],
                                              'verbose': [False]}],
                                   'KNeighborsClassifier()': [{'n_neighbors': neighbor_range,
                                                               'weights': ['uniform'],
                                                               'algorithm': ['brute'],
                                                               'metric': ['euclidean', 'manhattan']}],
                                   'LogisticRegression()': [{'penalty': ['l1', 'l2'],
                                                             'C': c_range,
                                                             'class_weight': ['auto'],
                                                             'multi_class': ['multinomial', 'ovr'],
                                                             'solver': ['lbfgs']}],
                                   'ANNLogisticRegression()': [{'annlogisticregression__penalty': ['l1', 'l2'],
                                                                'annlogisticregression__C': c_range,
                                                                'annlogisticregression__class_weight': ['auto'],
                                                                'annlogisticregression__multi_class': ['multinomial'],
                                                                'annlogisticregression__solver': ['lbfgs']}],
                                   'Perceptron()': [{'perceptron__penalty': ['l1', 'l2', 'elasticnet', None],
                                                     'perceptron__alpha': alpha_range,
                                                     'perceptron__eta0': eta_range,
                                                     'perceptron__class_weight': ['auto'],
                                                     'perceptron__warm_start': [True],
                                                     'rbm__learning_rate': learning_rate_range,
                                                     'rbm__n_components': component_range}],
                                   'MultilayerPerceptronClassifier()': [{'alpha': alpha_range,
                                                                         'n_hidden': component_range,
                                                                         'learning_rate': ['constant'],
                                                                         'warm_start': [True],
                                                                         'eta0': eta_range}]}
                    return grid_params[model]
                else:
                    rand_params = {'SVC()': {'C': stats.beta(loc=1, scale=1000, a=.85, b=1),
                                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                             'degree': [2, 3, 4, 5, 6],
                                             'gamma': stats.beta(loc=.0001, scale=30, a=1.1, b=50),
                                             'random_state': [10],
                                             'probability': [True],
                                             'cache_size': [1000],
                                             'class_weight': ['auto'],
                                             'max_iter': [250000],
                                             'verbose': [False]},
                                   'KNeighborsClassifier()': {'n_neighbors': stats.randint(low=2, high=100),
                                                              'weights': ['uniform', 'distance'],
                                                              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                                              'metric': ['euclidean', 'manhattan'],
                                                              'leaf_size': stats.randint(low=4, high=60)},
                                   'LogisticRegression()': {'penalty': ['l1', 'l2'],
                                                            'C': stats.beta(loc=1, scale=1000, a=.85, b=1),
                                                            'class_weight': ['auto'],
                                                            'multi_class': ['multinomial'],
                                                            'solver': ['lbfgs']},
                                   'ANNLogisticRegression()': {'annlogisticregression__penalty': ['l1', 'l2'],
                                                               'annlogisticregression__C': c_range,
                                                               'annlogisticregression__class_weight': ['auto'],
                                                               'annlogisticregression__multi_class': ['multinomial'],
                                                               'annlogisticregression__solver': ['lbfgs']},
                                   'Perceptron()': {'perceptron__penalty': ['l1', 'l2', 'elasticnet', None],
                                                    'perceptron__alpha': stats.expon(scale=.005),
                                                    'perceptron__eta0': stats.expon(scale=.8),
                                                    'perceptron__class_weight': ['auto'],
                                                    'perceptron__warm_start': [True],
                                                    'rbm__learning_rate': learning_rate_range,
                                                    'rbm__n_components': stats.randint(1, 140)},
                                   'MultilayerPerceptronClassifier()': {'alpha': stats.beta(loc=.0000001,
                                                                                            scale=1, a=.7, b=1),
                                                                        'learning_rate': ['constant', 'invscaling'],
                                                                        'warm_start': [True],
                                                                        'n_hidden': stats.randint(1, 256),
                                                                        'eta0': stats.uniform(10**-3, .999)}}
                    return rand_params[model]
            except:
                print('could not get parameter grid')

        def optimize_params(model, cur_score, x_dev, y_dev, rand):

            def report(grid_scores, n_top=1):
                top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
                for i, score in enumerate(top_scores):
                    print("Model with rank: {0}".format(i + 1))
                    print("Mean validation " + cur_score + " score: {0:.3f} (std: {1:.3f})".format(
                        score.mean_validation_score, np.std(score.cv_validation_scores)))
                    print("Parameters: {0}".format(score.parameters))
                    print("")

            if not rand:
                start = time.time()
                param_grid = get_param_grid(model[1], int(y_dev.size), rand)
                print('\nOptimizing parameters for: ' + model[1] +
                      ' using Grid Search with score: ' + cur_score + '.')
                if model[1] != 'Perceptron()' and model[1] != 'ANNLogisticRegression()':
                    grid_model = GridSearchCV(model[0], param_grid, cv=5, scoring=cur_score, n_jobs=3)
                elif model[1] != 'ANNLogisticRegression()':
                    perceptron = Perceptron()
                    rbm = BernoulliRBM()
                    clf = Pipeline(steps=[('rbm', rbm), ('perceptron', perceptron)])
                    grid_model = GridSearchCV(clf, param_grid, cv=5, scoring=cur_score, n_jobs=3)
                else:
                    annlogisticregression = LogisticRegression()
                    rbm = BernoulliRBM()
                    clf = Pipeline(steps=[('rbm', rbm), ('annlogisticregression', annlogisticregression)])
                    grid_model = GridSearchCV(clf, param_grid, cv=5, scoring=cur_score, n_jobs=3)
                grid_model.fit(x_dev, y_dev)
                print("\nGridSearchCV took %.2f seconds for %d candidate parameter settings."
                      % (time.time() - start, len(grid_model.grid_scores_)))
                report(grid_model.grid_scores_)
                return grid_model
            else:
                start = time.time()
                param_grid = get_param_grid(cur_model[1], int(y_dev.size), rand)
                print('\nOptimizing parameters for: ' + model[1] +
                      ' using Random Search with score: ' + cur_score + '.')
                if model[1] != 'Perceptron()' and model[1] != 'ANNLogisticRegression()':
                    rand_model = RandomizedSearchCV(model[0], param_distributions=param_grid,
                                                    cv=5, scoring=cur_score, n_jobs=3, n_iter=200)
                    rand_model.fit(x_dev, y_dev)
                    print("\nRandomSearchCV took %.2f seconds for %d candidate parameter settings."
                          % (time.time() - start, len(rand_model.grid_scores_)))
                    report(rand_model.grid_scores_)
                    return rand_model
                elif model[1] != 'ANNLogisticRegression()':
                    perceptron = Perceptron()
                    rbm = BernoulliRBM()
                    clf = Pipeline(steps=[('rbm', rbm), ('perceptron', perceptron)])
                    rand_model = RandomizedSearchCV(clf, param_distributions=param_grid,
                                                    cv=5, scoring=cur_score, n_jobs=3, n_iter=200)
                    rand_model.fit(x_dev, y_dev)
                    print("\nRandomSearchCV took %.2f seconds for %d candidate parameter settings."
                          % (time.time() - start, len(rand_model.grid_scores_)))
                    report(rand_model.grid_scores_)
                    return rand_model
        # scores = ['confusion matrix', 'accuracy_score', 'f1_score',
        #           'fbeta_score', 'hamming_loss','jacccard_similarity_score',
        #           'log_loss', 'precision_recall_fscore_support',
        #           'precision score', 'recall_score', 'zero_one_loss',
        #           'average_precision_score', 'roc_auc_score']
        # The following list contains the only possible values for multi-class classification

        scores = ['accuracy']
        out_list = []

        # def main(cur_score, out_list):
        for cur_score in scores:
            if cur_model[1] == "Perceptron()":
                if cur_score != 'log_loss':
                    opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                    out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                     opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
                    opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, True)
                    out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                     opt_model.best_score_, opt_model.best_params_, 'RandomSearch'])
            elif cur_model[1] == "ANNLogisticRegression()":
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
            else:
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, True)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'RandomSearch'])
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'GridSearch'])

        # multithread(main, scores, out_list, threads=1)

        return out_list

    def train_and_eval(x_train, y_train, x_test, y_test, model, param_result):
        print('\nTraining and evaluating...')

        # def get_kfold_scores(cur_model, score, score_name, result):
        #     # try:
        #     print('\n10-Fold CV ' + score_name + ' Score:')
        #     scores = cross_val_score(estimator=cur_model, X=x_test, y=y_test, cv=5,
        #                              scoring=score, n_jobs=3)
        #     print(score_name + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #     result.update({score_name: scores.mean()})
        #     # except:

        all_results = []
        for result_list in param_result:

            print('\n########\nEvaluating: ' + str(result_list[2]))
            opt_model = result_list[2]
            result = {'model': result_list[0][0:-2], 'optimization score function': result_list[1],
                      'optimization score': result_list[3], 'search type': result_list[5]}
            result.update(result_list[4])

            # result = {'model': str(result_list[2])[0:str(result_list[2]).find('(')]}
            avail_scores = [['accuracy', 'Accuracy']]
            #### vv Only metrics available for multiclass classification
            # avail_scores = [['accuracy', 'Accuracy'],
            #                 ['f1_weighted', 'F1 Score'],
            #                 ['precision_weighted', 'Precision'],
            #                 ['recall_weighted', 'Recall']]

            opt_model.fit(x_train, y_train)
            joblib.dump(opt_model, 'one_class-' +
                        str(result_list[0][0:-2]) +
                        '-' + str(result_list[1]) +
                        '-' + str(result_list[5]) + '.pkl')

            # for this_score in avail_scores:
            #     get_kfold_scores(cur_model=opt_model, score=this_score[0], score_name=this_score[1], result=result)
            #

            ##### Not using k-fold cross validation

            y_pred = opt_model.predict(x_test)
            print('\nClassification Report:')
            print(metrics.classification_report(y_test, y_pred))
            print('\nAccuracy Score:')
            print(metrics.accuracy_score(y_test, y_pred))
            print('\nConfusion Matrix:')
            print(metrics.confusion_matrix(y_test, y_pred))
            result.update({'Accuracy': metrics.accuracy_score(y_test, y_pred)})
            all_results.append(result)
            # print('\nF1-Score:')
            # print(metrics.f1_score(y_test, y_pred))
            # print('\nHamming Loss:')
            # print(metrics.hamming_loss(y_test, y_pred))
            # print('\nJaccard Similarity:')
            # print(metrics.jaccard_similarity_score(y_test, y_pred))
            # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # print('\nLog Loss:')
            # print(metrics.log_loss(y_test, y_pred))
            # #vvv multiclass not supported
            # print('\nMatthews Correlation Coefficient:')
            # print(metrics.matthews_corrcoef(y_test, y_pred))
            # print('\nPrecision:')
            # print(metrics.precision_score(y_test, y_pred))
            # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # print('\nRecall:')
            # print(metrics.recall(y_test, y_pred))
        return all_results

    results_list = []

    for model in model_list:
        cur_result = param_optimization(model, x_dev, y_dev, output)
        print('\nEvaluating ' + model[1] + '.')
        results_list.extend(train_and_eval(x_train, y_train, data_dict['x_test'],
                                           data_dict['y_test'], model, cur_result))
        print('\n##############################################')

    return results_list


def multithread(function, items, extra_variable, threads=2):
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

models = [[LogisticRegression(), 'LogisticRegression()'],
          [KNeighborsClassifier(), 'KNeighborsClassifier()'],
          [MultilayerPerceptronClassifier(), 'MultilayerPerceptronClassifier()'],
          [SVC(), 'SVC()']]

# Generate df from file in path and split into labels/features
# Then generate model results for objects in model_list
if __name__ == '__main__':

    data = process_csv('new_student_data(08).csv',
                       'new_student_data(09).csv',
                       True)
    results = gen_model_results2(data, models, out_txt)
    df = pd.DataFrame(results)
    df = df[column_list]
    pd.DataFrame.to_csv(df, 'results_dataframe(one_class)final.csv')

    print('Binary-class runtime: ' + str(time.time() - start_main) + ' s')

    data2 = process_csv('new_student_data(08).csv',
                        'new_student_data(09).csv',
                        False)
    results2 = gen_model_results(data2, models, out_txt)
    df2 = pd.DataFrame(results2)
    df2 = df2[column_list2]
    pd.DataFrame.to_csv(df2, 'results_dataframe(multi_class)final.csv')

    clf1 = DummyClassifier(strategy='most_frequent')
    clf2 = DummyClassifier(strategy='stratified')

    clf1.fit(data['x_train'], data['y_train'])
    clf2.fit(data['x_train'], data['y_train'])

    y_pred1 = clf1.predict(data['x_test'])
    y_pred2 = clf2.predict(data['x_test'])

    print('---BinaryClass---')
    print('most frequent:')
    print(metrics.accuracy_score(data['y_test'], y_pred1))
    print(metrics.classification_report(data['y_test'], y_pred1))
    print('stratified:')
    print(metrics.accuracy_score(data['y_test'], y_pred2))
    print(metrics.classification_report(data['y_test'], y_pred2))
    print(metrics.confusion_matrix(data['y_test'], y_pred2))

    clf1.fit(data2['x_train'], data2['y_train'])
    clf2.fit(data2['x_train'], data2['y_train'])

    y_pred1 = clf1.predict(data2['x_test'])
    y_pred2 = clf2.predict(data2['x_test'])

    print('---MultiClass---')
    print('most frequent:')
    print(clf1.score(data2['x_test'], data2['y_test']))
    print(metrics.accuracy_score(data2['y_test'], y_pred1))
    print('stratified:')
    print(clf2.score(data2['x_test'], data2['y_test']))
    print(metrics.accuracy_score(data2['y_test'], y_pred2))
    print(metrics.confusion_matrix(data2['y_test'], y_pred2))
    print('Total runtime: ' + str(time.time() - start_main) + ' s')

    # if sys.version_info < (3, 0):
    #     raw_input('Press Enter to close:')
    # else:
    #     input('Press Enter to close:')

# # To-Do
## Consider verbose to track progress of algorithms
# # Why is SVM so slow?

# C:\Python27\python.exe "C:/Users/Ricardo/PycharmProjects/Student Attrition/main.py"
#
# #############
#
# Processing CSV file...
#
# #############
#
# Dataframe is being built...
# Dataframe is being built...
# Values are being imputed in dataframe...
# ((3564, 43), (3355, 43))
# Processing features...
# Converting some features to indicator variables...
# Processing features...
# Converting some features to indicator variables...
#
# ################
#
# Now generating model results...
#
# ################
#
#
# Conducting parameter optimization for: LogisticRegression()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: LogisticRegression() using Random Search with score: log_loss.
#
# RandomSearchCV took 59.57 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -0.662 (std: 0.005)
# Parameters: {'penalty': 'l1', 'multi_class': 'multinomial', 'C': 4.960064790253416, 'solver': 'lbfgs', 'class_weight': 'auto'}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: LogisticRegression() using Grid Search with score: log_loss.
#
# GridSearchCV took 7.13 seconds for 6 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -0.663 (std: 0.004)
# Parameters: {'penalty': 'l1', 'multi_class': 'multinomial', 'C': 1.0, 'solver': 'lbfgs', 'class_weight': 'auto'}
#
#
# Evaluating LogisticRegression().
#
# Training and evaluating...
#
# ########
# Evaluating: LogisticRegression(C=4.96006479025, class_weight='auto', dual=False,
#           fit_intercept=True, intercept_scaling=1, max_iter=100,
#           multi_class='multinomial', penalty='l1', random_state=None,
#           solver='lbfgs', tol=0.0001, verbose=0)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.81      0.89      0.85      2704
#           1       0.25      0.16      0.19       651
#
# avg / total       0.70      0.75      0.72      3355
#
#
# Accuracy Score:
# 0.746050670641
#
# Confusion Matrix:
# [[2401  303]
#  [ 549  102]]
#
# ########
# Evaluating: LogisticRegression(C=1.0, class_weight='auto', dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='multinomial',
#           penalty='l1', random_state=None, solver='lbfgs', tol=0.0001,
#           verbose=0)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.82      0.80      0.81      2704
#           1       0.23      0.25      0.24       651
#
# avg / total       0.70      0.69      0.70      3355
#
#
# Accuracy Score:
# 0.690611028316
#
# Confusion Matrix:
# [[2155  549]
#  [ 489  162]]
#
# ##############################################
#
# Conducting parameter optimization for: KNeighborsClassifier()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: KNeighborsClassifier() using Random Search with score: log_loss.
#
# RandomSearchCV took 57.32 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -1.192 (std: 0.134)
# Parameters: {'n_neighbors': 7, 'metric': 'manhattan', 'weights': 'uniform', 'leaf_size': 30, 'algorithm': 'kd_tree'}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: KNeighborsClassifier() using Grid Search with score: log_loss.
#
# GridSearchCV took 32.58 seconds for 46 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -0.603 (std: 0.004)
# Parameters: {'n_neighbors': 94, 'metric': 'manhattan', 'weights': 'uniform', 'algorithm': 'brute'}
#
#
# Evaluating KNeighborsClassifier().
#
# Training and evaluating...
#
# ########
# Evaluating: KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='manhattan',
#            metric_params=None, n_neighbors=7, p=2, weights='uniform')
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.84      0.70      0.76      2704
#           1       0.26      0.44      0.33       651
#
# avg / total       0.73      0.65      0.68      3355
#
#
# Accuracy Score:
# 0.650074515648
#
# Confusion Matrix:
# [[1897  807]
#  [ 367  284]]
#
# ########
# Evaluating: KNeighborsClassifier(algorithm='brute', leaf_size=30, metric='manhattan',
#            metric_params=None, n_neighbors=94, p=2, weights='uniform')
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.82      0.96      0.88      2704
#           1       0.41      0.12      0.18       651
#
# avg / total       0.74      0.80      0.75      3355
#
#
# Accuracy Score:
# 0.796721311475
#
# Confusion Matrix:
# [[2598  106]
#  [ 576   75]]
#
# ##############################################
#
# Conducting parameter optimization for: MultilayerPerceptronClassifier()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: MultilayerPerceptronClassifier() using Random Search with score: log_loss.
#
# RandomSearchCV took 859.05 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -1.123 (std: 0.130)
# Parameters: {'warm_start': True, 'alpha': 0.01068112140297313, 'learning_rate': 'constant', 'eta0': 0.09443336374278448, 'n_hidden': 3}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: MultilayerPerceptronClassifier() using Grid Search with score: log_loss.
#
# GridSearchCV took 369.69 seconds for 48 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -0.594 (std: 0.006)
# Parameters: {'warm_start': True, 'alpha': 10.0, 'learning_rate': 'constant', 'eta0': 10.0, 'n_hidden': 4}
#
#
# Evaluating MultilayerPerceptronClassifier().
#
# Training and evaluating...
#
# ########
# Evaluating: MultilayerPerceptronClassifier(activation='tanh', algorithm='l-bfgs',
#                 alpha=0.010681121403, batch_size=200, eta0=0.0944333637428,
#                 learning_rate='constant', max_iter=200, n_hidden=3,
#                 power_t=0.25, random_state=None, shuffle=False, tol=1e-05,
#                 verbose=False, warm_start=True)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.90      0.06      0.11      2704
#           1       0.20      0.97      0.33       651
#
# avg / total       0.76      0.23      0.15      3355
#
#
# Accuracy Score:
# 0.234277198212
#
# Confusion Matrix:
# [[ 152 2552]
#  [  17  634]]
#
# ########
# Evaluating: MultilayerPerceptronClassifier(activation='tanh', algorithm='l-bfgs',
#                 alpha=10.0, batch_size=200, eta0=10.0,
#                 learning_rate='constant', max_iter=200, n_hidden=4,
#                 power_t=0.25, random_state=None, shuffle=False, tol=1e-05,
#                 verbose=False, warm_start=True)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.82      0.72      0.77      2704
#           1       0.23      0.34      0.27       651
#
# avg / total       0.70      0.65      0.67      3355
#
#
# Accuracy Score:
# 0.645305514158
#
# Confusion Matrix:
# [[1941  763]
#  [ 427  224]]
#
# ##############################################
#
# Conducting parameter optimization for: SVC()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: SVC() using Random Search with score: log_loss.
# C:\Python27\lib\site-packages\sklearn\svm\base.py:209: ConvergenceWarning: Solver terminated early (max_iter=250000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#   % self.max_iter, ConvergenceWarning)
# C:\Python27\lib\site-packages\sklearn\svm\base.py:209: ConvergenceWarning: Solver terminated early (max_iter=250000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#   % self.max_iter, ConvergenceWarning)
#
# RandomSearchCV took 1452.53 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -0.594 (std: 0.003)
# Parameters: {'kernel': 'rbf', 'C': 134.16079283844925, 'verbose': False, 'degree': 4, 'max_iter': 250000, 'probability': True, 'random_state': 10, 'cache_size': 1000, 'gamma': 0.0, 'class_weight': 'auto'}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: SVC() using Grid Search with score: log_loss.
#
# GridSearchCV took 1099.74 seconds for 57 candidate parameter settings.
# Model with rank: 1
# Mean validation log_loss score: -0.593 (std: 0.005)
# Parameters: {'kernel': 'linear', 'C': 1.0, 'verbose': False, 'probability': True, 'max_iter': 250000, 'cache_size': 1000, 'class_weight': 'auto'}
#
#
# Evaluating SVC().
#
# Training and evaluating...
#
# ########
# Evaluating: SVC(C=134.160792838, cache_size=1000, class_weight='auto', coef0=0.0,
#   degree=4, gamma=0.0, kernel='rbf', max_iter=250000, probability=True,
#   random_state=10, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
# C:\Python27\lib\site-packages\sklearn\metrics\classification.py:958: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#
#   'precision', 'predicted', average, warn_for)
#           0       0.00      0.00      0.00      2704
#           1       0.19      1.00      0.33       651
#
# avg / total       0.04      0.19      0.06      3355
#
#
# Accuracy Score:
# 0.194038748137
#
# Confusion Matrix:
# [[   0 2704]
#  [   0  651]]
#
# ########
# Evaluating: SVC(C=1.0, cache_size=1000, class_weight='auto', coef0=0.0, degree=3,
#   gamma=0.0, kernel='linear', max_iter=250000, probability=True,
#   random_state=None, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.81      0.88      0.85      2704
#           1       0.25      0.17      0.20       651
#
# avg / total       0.71      0.74      0.72      3355
#
#
# Accuracy Score:
# 0.740983606557
#
# Confusion Matrix:
# [[2376  328]
#  [ 541  110]]
#
# ##############################################
# Binary-class runtime: 3974.94899988 s
#
# #############
#
# Processing CSV file...
#
# #############
#
# Dataframe is being built...
# Dataframe is being built...
# Values are being imputed in dataframe...
# ((3564, 43), (3355, 43))
# Processing features...
# Converting some features to indicator variables...
# Processing features...
# Converting some features to indicator variables...
#
# ################
#
# Now generating model results...
#
# ################
#
#
# Conducting parameter optimization for: LogisticRegression()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: LogisticRegression() using Random Search with score: accuracy.
#
# RandomSearchCV took 72.85 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.444 (std: 0.011)
# Parameters: {'penalty': 'l1', 'multi_class': 'multinomial', 'C': 4.437901673440723, 'solver': 'lbfgs', 'class_weight': 'auto'}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: LogisticRegression() using Grid Search with score: accuracy.
#
# GridSearchCV took 9.53 seconds for 6 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.451 (std: 0.012)
# Parameters: {'penalty': 'l1', 'multi_class': 'multinomial', 'C': 0.10000000000000001, 'solver': 'lbfgs', 'class_weight': 'auto'}
#
#
# Evaluating LogisticRegression().
#
# Training and evaluating...
#
# ########
# Evaluating: LogisticRegression(C=4.43790167344, class_weight='auto', dual=False,
#           fit_intercept=True, intercept_scaling=1, max_iter=100,
#           multi_class='multinomial', penalty='l1', random_state=None,
#           solver='lbfgs', tol=0.0001, verbose=0)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.89      0.00      0.01      2097
#        Grad       0.25      0.14      0.18       651
#       Trans       0.18      0.90      0.30       607
#
# avg / total       0.64      0.19      0.10      3355
#
#
# Accuracy Score:
# 0.192846497765
#
# Confusion Matrix:
# [[   8  217 1872]
#  [   0   93  558]
#  [   1   60  546]]
#
# ########
# Evaluating: LogisticRegression(C=0.10000000000000001, class_weight='auto', dual=False,
#           fit_intercept=True, intercept_scaling=1, max_iter=100,
#           multi_class='multinomial', penalty='l1', random_state=None,
#           solver='lbfgs', tol=0.0001, verbose=0)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.76      0.02      0.03      2097
#        Grad       0.22      0.27      0.24       651
#       Trans       0.19      0.78      0.30       607
#
# avg / total       0.55      0.20      0.12      3355
#
#
# Accuracy Score:
# 0.202086438152
#
# Confusion Matrix:
# [[  34  475 1588]
#  [   6  173  472]
#  [   5  131  471]]
#
# ##############################################
#
# Conducting parameter optimization for: KNeighborsClassifier()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: KNeighborsClassifier() using Random Search with score: accuracy.
#
# RandomSearchCV took 59.29 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.493 (std: 0.019)
# Parameters: {'n_neighbors': 2, 'metric': 'euclidean', 'weights': 'uniform', 'leaf_size': 37, 'algorithm': 'ball_tree'}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: KNeighborsClassifier() using Grid Search with score: accuracy.
#
# GridSearchCV took 33.46 seconds for 46 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.529 (std: 0.013)
# Parameters: {'n_neighbors': 78, 'metric': 'manhattan', 'weights': 'uniform', 'algorithm': 'brute'}
#
#
# Evaluating KNeighborsClassifier().
#
# Training and evaluating...
#
# ########
# Evaluating: KNeighborsClassifier(algorithm='ball_tree', leaf_size=37, metric='euclidean',
#            metric_params=None, n_neighbors=2, p=2, weights='uniform')
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.61      0.56      0.58      2097
#        Grad       0.18      0.39      0.25       651
#       Trans       0.17      0.01      0.02       607
#
# avg / total       0.44      0.43      0.42      3355
#
#
# Accuracy Score:
# 0.425931445604
#
# Confusion Matrix:
# [[1166  904   27]
#  [ 383  255   13]
#  [ 378  221    8]]
#
# ########
# Evaluating: KNeighborsClassifier(algorithm='brute', leaf_size=30, metric='manhattan',
#            metric_params=None, n_neighbors=78, p=2, weights='uniform')
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.67      0.73      0.69      2097
#        Grad       0.27      0.43      0.33       651
#       Trans       0.33      0.02      0.04       607
#
# avg / total       0.53      0.54      0.51      3355
#
#
# Accuracy Score:
# 0.541579731744
#
# Confusion Matrix:
# [[1525  554   18]
#  [ 365  280    6]
#  [ 402  193   12]]
#
# ##############################################
#
# Conducting parameter optimization for: MultilayerPerceptronClassifier()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: MultilayerPerceptronClassifier() using Random Search with score: accuracy.
#
# RandomSearchCV took 964.04 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.505 (std: 0.022)
# Parameters: {'warm_start': True, 'alpha': 0.006826851947907079, 'learning_rate': 'constant', 'eta0': 0.6980099398043059, 'n_hidden': 2}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: MultilayerPerceptronClassifier() using Grid Search with score: accuracy.
#
# GridSearchCV took 350.98 seconds for 48 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.543 (std: 0.014)
# Parameters: {'warm_start': True, 'alpha': 10.0, 'learning_rate': 'constant', 'eta0': 0.01, 'n_hidden': 4}
#
#
# Evaluating MultilayerPerceptronClassifier().
#
# Training and evaluating...
#
# ########
# Evaluating: MultilayerPerceptronClassifier(activation='tanh', algorithm='l-bfgs',
#                 alpha=0.00682685194791, batch_size=200,
#                 eta0=0.698009939804, learning_rate='constant',
#                 max_iter=200, n_hidden=2, power_t=0.25, random_state=None,
#                 shuffle=False, tol=1e-05, verbose=False, warm_start=True)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.70      0.25      0.37      2097
#        Grad       0.25      0.25      0.25       651
#       Trans       0.20      0.63      0.30       607
#
# avg / total       0.52      0.32      0.33      3355
#
#
# Accuracy Score:
# 0.32041728763
#
# Confusion Matrix:
# [[ 528  378 1191]
#  [ 101  162  388]
#  [ 122  100  385]]
#
# ########
# Evaluating: MultilayerPerceptronClassifier(activation='tanh', algorithm='l-bfgs',
#                 alpha=10.0, batch_size=200, eta0=0.01,
#                 learning_rate='constant', max_iter=200, n_hidden=4,
#                 power_t=0.25, random_state=None, shuffle=False, tol=1e-05,
#                 verbose=False, warm_start=True)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.71      0.02      0.04      2097
#        Grad       0.24      0.20      0.22       651
#       Trans       0.18      0.83      0.30       607
#
# avg / total       0.52      0.20      0.12      3355
#
#
# Accuracy Score:
# 0.202384500745
#
# Confusion Matrix:
# [[  46  318 1733]
#  [   6  129  516]
#  [  13   90  504]]
#
# ##############################################
#
# Conducting parameter optimization for: SVC()...
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: SVC() using Random Search with score: accuracy.
# C:\Python27\lib\site-packages\sklearn\svm\base.py:209: ConvergenceWarning: Solver terminated early (max_iter=250000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#   % self.max_iter, ConvergenceWarning)
# C:\Python27\lib\site-packages\sklearn\svm\base.py:209: ConvergenceWarning: Solver terminated early (max_iter=250000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#   % self.max_iter, ConvergenceWarning)
#
# RandomSearchCV took 2452.31 seconds for 50 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.511 (std: 0.001)
# Parameters: {'kernel': 'sigmoid', 'C': 547.0062092984043, 'verbose': False, 'degree': 5, 'max_iter': 250000, 'probability': True, 'random_state': 10, 'cache_size': 1000, 'gamma': 0.0, 'class_weight': 'auto'}
#
#
# Retrieving parameter grid...
#
# Optimizing parameters for: SVC() using Grid Search with score: accuracy.
#
# GridSearchCV took 1416.93 seconds for 57 candidate parameter settings.
# Model with rank: 1
# Mean validation accuracy score: 0.511 (std: 0.001)
# Parameters: {'kernel': 'sigmoid', 'C': 10.0, 'verbose': False, 'probability': True, 'max_iter': 250000, 'cache_size': 1000, 'gamma': 10.0, 'class_weight': 'auto'}
#
#
# Evaluating SVC().
#
# Training and evaluating...
#
# ########
# Evaluating: SVC(C=547.006209298, cache_size=1000, class_weight='auto', coef0=0.0,
#   degree=5, gamma=0.0, kernel='sigmoid', max_iter=250000, probability=True,
#   random_state=10, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.63      1.00      0.77      2097
#        Grad       0.00      0.00      0.00       651
#       Trans       0.00      0.00      0.00       607
#
# avg / total       0.39      0.63      0.48      3355
#
#
# Accuracy Score:
# 0.625037257824
#
# Confusion Matrix:
# [[2097    0    0]
#  [ 651    0    0]
#  [ 607    0    0]]
#
# ########
# Evaluating: SVC(C=10.0, cache_size=1000, class_weight='auto', coef0=0.0, degree=3,
#   gamma=10.0, kernel='sigmoid', max_iter=250000, probability=True,
#   random_state=None, shrinking=True, tol=0.001, verbose=False)
#
# Classification Report:
#              precision    recall  f1-score   support
#
#        Drop       0.62      1.00      0.77      2097
#        Grad       0.00      0.00      0.00       651
#       Trans       0.00      0.00      0.00       607
#
# avg / total       0.39      0.62      0.48      3355
#
#
# Accuracy Score:
# 0.624441132638
#
# Confusion Matrix:
# [[2095    2    0]
#  [ 651    0    0]
#  [ 607    0    0]]
#
# ##############################################
# ---BinaryClass---
# most frequent:
# 0.805961251863
#              precision    recall  f1-score   support
#
#           0       0.81      1.00      0.89      2704
#           1       0.00      0.00      0.00       651
#
# avg / total       0.65      0.81      0.72      3355
#
# stratified:
# 0.62086438152
#              precision    recall  f1-score   support
#
#           0       0.81      0.69      0.74      2704
#           1       0.20      0.32      0.25       651
#
# avg / total       0.69      0.62      0.65      3355
#
# [[1861  843]
#  [ 441  210]]
# ---MultiClass---
# most frequent:
# 0.625037257824
#              precision    recall  f1-score   support
#
#        Drop       0.63      1.00      0.77      2097
#        Grad       0.00      0.00      0.00       651
#       Trans       0.00      0.00      0.00       607
#
# avg / total       0.39      0.63      0.48      3355
#
# stratified:
# 0.401490312966
#              precision    recall  f1-score   support
#
#        Drop       0.62      0.50      0.55      2097
#        Grad       0.18      0.28      0.22       651
#       Trans       0.17      0.18      0.17       607
#
# avg / total       0.45      0.40      0.42      3355
#
# [[1056  659  382]
#  [ 337  184  130]
#  [ 319  181  107]]
# Total runtime: 9378.67700005 s
#
# Process finished with exit code 0