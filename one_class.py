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
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from multiprocessing.dummy import Pool as ThreadPool
from operator import itemgetter
from sklearn.externals import joblib

# Begin timer for output
start_main = time.time()

# Set variable for a file in the current path for generating logs
out_txt = open('output.txt', "wb")

# Setup list for column order of output dataframe
column_list = ['model', 'kernel', 'degree', 'gamma', 'probability',
               'C', 'penalty', 'solver', 'logisticregression__C',
               'logisticregression__penalty', 'logisticregression__solver',
               'perceptron__penalty', 'perceptron__alpha',
               'perceptron__eta0', 'rbm__learning_rate',
               'rbm__n_components', 'leaf_size', 'metric', 'algorithm',
               'n_neighbors', 'weights', 'search type', 'optimization score function',
               'optimization score', 'Accuracy', 'Recall', 'F1 Score', 'Precision',
               'Average Precision', 'ROC AUC']


def process_csv2(location):

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
        df = df.replace(to_replace='Transferred', value='Dropped')
        df = df.replace(to_replace='Dropped', value=0)
        df = df.replace(to_replace='Graduated', value=1)

        return df

    def impute_missing_values(df):
        ########
        # Takes in a dataframe and imputes missing values
        ########
        print('Values are being imputed in dataframe...')
        df = df.fillna(0)
        # Impute numerical variables with mean
        df = df.fillna(df.mean())
        # Impute categorical variables with mode
        df = df.apply(lambda i: i.fillna(i.mode()[0]))
        return df

    # def process_features(df):
        #######
        # Takes in a dataframe and converts categorical variables into indicator variables
        # and rescales numerical variables between -1 and 1
        ########
        # Split dataset in two, one with all numerical features and one with all categorical features
        # print('Processing features...')
        # num_df = df.select_dtypes(include=['float64'])
        # cat_df = df.select_dtypes(include=['object'])
        # # Convert categorical features into indicator variables
        # if len(cat_df.columns) > 0:
        #     cat_df = convert_to_indicators(cat_df)
        # # Rescale numerical features between -1 and 1
        # if len(num_df.columns) > 0:
        #     num_df = ((1/(num_df.max()-num_df.min()))*(2*num_df-num_df.max()-num_df.min()))
        ### Since data was preprocessed
        # Rescale between -1 and 1
        # df = ((1/(df.max()-df.min()))*(2*df-df.max()-df.min()))
        # # Standardize data
        # df = (df - df.mean())/df.std()
        # # Rescale between 0 and 1
        # df = (df - df.min())/(df.max()-df.min())
        # # Recombine categorical and numerical feature into one dataframe
        # df = num_df.join(cat_df)
        # Replace NaN's that were caused by division by 0 when rescaling with 0's
        # This occurs when all values are 0 (eg. indicator variables)
        # df = df.fillna(0)

        # return df

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

    def transform_and_split(features, labels):
        ########
        # Takes in two dataframes for the features and labels of a dataset and
        # outputs a dictionary with training and keys relating to training testing sets for each
        ########
        print('Performing prelimianry datasplit')
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        scaler2 = MinMaxScaler()
        scaler2.fit(x_train)
        x_train = scaler2.transform(x_train)
        x_test = scaler2.transform(x_test)

        data_dict = {'x_test': x_test, 'x_train': x_train,
                     'y_test': y_test, 'y_train': y_train}
        return data_dict

    # Create a dataframe from a provided path
    data = build_df(location)
    # Separate dataframe into labels and features
    y = data.pop(data.columns[len(data.columns)-1])
    x = data

    return transform_and_split(x, y)


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
                c_range = 10.0 ** np.arange(-2, 3)
                # print 'Getting Parameter grid...'
                # out_txt.write('Getting Parameter grid...')
                gamma_range = 10.0 ** np.arange(-3, 4)
                # neighbor_range = np.arange(2, points, step=5)
                # leaf_range = np.arange(10, points, step=5)
                neighbor_range = np.arange(2, 100, step=2)
                leaf_range = np.arange(5, 65, step=5)
                alpha_range = 10.0**np.arange(-3, 3)
                eta_range = 10.0**np.arange(-3, 3)
                learning_rate_range = 10.0**np.arange(-3, 1)
                component_range = 2**np.arange(0, 9, step=4)
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
                                                               'metric': ['euclidean', 'manhattan']},
                                                              {'n_neighbors': neighbor_range,
                                                               'weights': ['uniform'],
                                                               'algorithm': ['ball_tree', 'kd_tree'],
                                                               'metric': ['euclidean', 'manhattan'],
                                                               'leaf_size': leaf_range}],
                                   'LogisticRegression()': [{'penalty': ['l1', 'l2'],
                                                             'C': c_range,
                                                             'class_weight': ['auto'],
                                                             'multi_class': ['multinomial'],
                                                             'solver': ['newton-cg', 'lbfgs']}],
                                   'ANNLogisticRegression()': [{'annlogisticregression__penalty': ['l1', 'l2'],
                                                                'annlogisticregression__C': c_range,
                                                                'annlogisticregression__class_weight': ['auto'],
                                                                'annlogisticregression__multi_class': ['multinomial'],
                                                                'annlogisticregression__solver': ['newton-cg', 'lbfgs']}],
                                   'Perceptron()': [{'perceptron__penalty': ['l1', 'l2', 'elasticnet', None],
                                                     'perceptron__alpha': alpha_range,
                                                     'perceptron__eta0': eta_range,
                                                     'perceptron__class_weight': ['auto'],
                                                     'perceptron__warm_start': [True],
                                                     'rbm__learning_rate': learning_rate_range,
                                                     'rbm__n_components': component_range}]}
                    return grid_params[model]
                else:
                    rand_params = {'SVC()': {'C': stats.expon(scale=300),
                                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                             'degree': [3, 4, 5, 6, 7, 8],
                                             'gamma': stats.expon(scale=1/3),
                                             'random_state': [10],
                                             'probability': [True],
                                             'cache_size': [1000],
                                             'class_weight': ['auto'],
                                             'max_iter': [250000],
                                             'verbose': [False]},
                                   'KNeighborsClassifier()': {'n_neighbors': stats.randint(low=2, high=8),
                                                              'weights': ['uniform', 'distance'],
                                                              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                                              'metric': ['euclidean', 'manhattan'],
                                                              'leaf_size': stats.randint(low=10, high=60)},
                                   'LogisticRegression()': {'penalty': ['l1', 'l2'],
                                                            'C': stats.expon(scale=300),
                                                            'class_weight': ['auto'],
                                                            'multi_class': ['multinomial'],
                                                            'solver': ['newton-cg', 'lbfgs']},
                                   'ANNLogisticRegression()': {'annlogisticregression__penalty': ['l1', 'l2'],
                                                               'annlogisticregression__C': c_range,
                                                               'annlogisticregression__class_weight': ['auto'],
                                                               'annlogisticregression__multi_class': ['multinomial'],
                                                               'annlogisticregression__solver': ['newton-cg', 'lbfgs']},
                                   'Perceptron()': {'perceptron__penalty': ['l1', 'l2', 'elasticnet', None],
                                                    'perceptron__alpha': stats.expon(scale=.005),
                                                    'perceptron__eta0': stats.expon(scale=.8),
                                                    'perceptron__class_weight': ['auto'],
                                                    'perceptron__warm_start': [True],
                                                    'rbm__learning_rate': learning_rate_range,
                                                    'rbm__n_components': stats.randint(1, 257)}}
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
                                                    cv=5, scoring=cur_score, n_jobs=3, n_iter=300)
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
                                                    cv=5, scoring=cur_score, n_jobs=3, n_iter=300)
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

        scores = ['accuracy', 'log_loss', 'recall', 'roc_auc']
        out_list = []

        # def main(cur_score, out_list):
        for cur_score in scores:
            if cur_model[1] == "Perceptron()":
                if cur_score != 'log_loss':
                    opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, True)
                    out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                     opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
                    opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                    out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                     opt_model.best_score_, opt_model.best_params_, 'RandomSearch'])
            elif cur_model[1] == "ANNLogisticRegression()":
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
            else:
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, True)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'GridSearch'])
                opt_model = optimize_params(cur_model, cur_score, x_dev, y_dev, False)
                out_list.append([cur_model[1], cur_score, opt_model.best_estimator_,
                                 opt_model.best_score_, opt_model.best_params_, 'RandomSearch'])

        # multithread(main, scores, out_list, threads=1)

        return out_list

    def train_and_eval(x_train, y_train, x_test, y_test, model, param_result):
        print('\nTraining and evaluating...')

        def get_kfold_scores(cur_model, score, score_name, result):
            # try:
            print('\n10-Fold CV ' + score_name + ' Score:')
            scores = cross_val_score(estimator=cur_model, X=x_test, y=y_test, cv=5,
                                     scoring=score, n_jobs=3)
            print(score_name + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            result.update({score_name: scores.mean()})
            # except:

        all_results = []
        for result_list in param_result:

            print('\n########\nEvaluating: ' + str(result_list[2]))
            opt_model = result_list[2]
            result = {'model': result_list[0][0:-2], 'optimization score function': result_list[1],
                      'optimization score': result_list[3], 'search type': result_list[5]}
            result.update(result_list[4])

            # result = {'model': str(result_list[2])[0:str(result_list[2]).find('(')]}
            avail_scores = [['accuracy', 'Accuracy'],
                            ['average_precision', 'Average Precision'],
                            ['f1', 'F1 Score'],
                            ['precision', 'Precision'],
                            ['recall', 'Recall'],
                            ['roc_auc', 'ROC AUC']]
            #### vv Only metrics available for multiclass classification
            # avail_scores = [['accuracy', 'Accuracy'],
            #                 ['f1_weighted', 'F1 Score'],
            #                 ['precision_weighted', 'Precision'],
            #                 ['recall_weighted', 'Recall']]
            for this_score in avail_scores:
                get_kfold_scores(cur_model=opt_model, score=this_score[0], score_name=this_score[1], result=result)

            all_results.append(result)

            ##### Not using k-fold cross validation
            opt_model.fit(x_train, y_train)
            joblib.dump(opt_model, 'one_class-' +
                        str(result_list[0][0:-2]) +
                        '-' + str(result_list[1]) +
                        '-' + str(result_list[5]) + '.pkl')
            # y_pred = opt_model.predict(x_test)
            # print('\nClassification Report:')
            # print(metrics.classification_report(y_test, y_pred))
            # print('\nAccuracy Score:')
            # print(metrics.accuracy_score(y_test, y_pred))
            # print('\nConfusion Matrix:')
            # print(metrics.confusion_matrix(y_test, y_pred))
            # # print('\nF1-Score:')
            # # print(metrics.f1_score(y_test, y_pred))
            # print('\nHamming Loss:')
            # print(metrics.hamming_loss(y_test, y_pred))
            # print('\nJaccard Similarity:')
            # print(metrics.jaccard_similarity_score(y_test, y_pred))
            # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # # print('\nLog Loss:')
            # # print(metrics.log_loss(y_test, y_pred))
            # #vvv multiclass not supported
            # # print('\nMatthews Correlation Coefficient:')
            # # print(metrics.matthews_corrcoef(y_test, y_pred))
            # print('\nPrecision:')
            # print(metrics.precision_score(y_test, y_pred))
            # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
            # # print('\nRecall:')
            # # print(metrics.recall(y_test, y_pred))
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

model_list = [[KNeighborsClassifier(), 'KNeighborsClassifier()'],
              [LogisticRegression(), 'ANNLogisticRegression()'],
              [Perceptron(), 'Perceptron()'],
              [LogisticRegression(), 'LogisticRegression()'],
              [SVC(), 'SVC()']]

# Generate df from file in path and split into labels/features
# Then generate model results for objects in model_list
if __name__ == '__main__':

    data = process_csv('Trial_Data.csv')
    results = gen_model_results(data, model_list, out_txt)
    df = pd.DataFrame(results)
    df = df[column_list]
    pd.DataFrame.to_csv(df, 'results_dataframe(one_class).csv')
    print('Total runtime: ' + str(time.time() - start_main) + ' s')

    # if sys.version_info < (3, 0):
    #     raw_input('Press Enter to close:')
    # else:
    #     input('Press Enter to close:')

# # To-Do
# # Need to reassess neural net implementation and bring into main