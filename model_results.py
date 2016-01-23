__author__ = 'Ricardo'

from abc import ABCMeta, abstractmethod

import time

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import numpy as np

from scipy import stats

from operator import itemgetter


def param_optimization(model, x_dev, y_dev, output):
    """

    :param cur_model:
    :param x_dev:
    :param y_dev:
    :param output:
    :return:
    """

    print('\nConducting parameter optimization for: ' + model[1] + '...\n')

    def get_param_grid(cur_model, points, rand):
        print('\nRetrieving parameter grid...')
        try:
            c_range = 10.0 ** np.arange(-2, 3)
            # print 'Getting Parameter grid...'
            # out_txt.write('Getting Parameter grid...')
            gamma_range = [0, .01, .1, .3]
            # neighbor_range = np.arange(2, points, step=5)
            # leaf_range = np.arange(10, points, step=5)
            neighbor_range = np.arange(2, 17, step=5)
            leaf_range = np.arange(10, 60, step=5)
            if not rand:
                grid_params = {'SVC()': [{'C': c_range,
                                          'kernel': ['poly'],
                                          'degree': [3, 5, 8],
                                          'gamma': gamma_range,
                                          'probability': [True],
                                          'class_weight': ['auto', None]},
                                         {'C': c_range,
                                          'kernel': ['rbf', 'sigmoid'],
                                          'gamma': gamma_range,
                                          'probability': [True],
                                          'class_weight': ['auto', None]},
                                         {'C': c_range,
                                          'kernel': ['linear'],
                                          'random_state': [10],
                                          'probability': [True],
                                          'class_weight': ['auto', None]}],
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
                                                         'class_weight': [None, 'auto']}]}
                return grid_params[cur_model]
            else:
                rand_params = {'SVC()': {'C': stats.expon(scale=300),
                                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                         'degree': [3, 4, 5, 6, 7, 8],
                                         'gamma': stats.expon(scale=1/3),
                                         'random_state': [10],
                                         'probability': [True],
                                         'class_weight': ['auto', None]},
                               'KNeighborsClassifier()': {'n_neighbors': stats.randint(low=2, high=20),
                                                          'weights': ['uniform', 'distance'],
                                                          'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                                          'metric': ['euclidean', 'manhattan'],
                                                          'leaf_size': stats.randint(low=10, high=60)},
                               'LogisticRegression()': {'penalty': ['l1', 'l2'],
                                                        'C': stats.expon(scale=300),
                                                        'class_weight': [None, 'auto']}}
                return rand_params[cur_model]
        except:
            print('could not get parameter grid')

    def optimize_params(cur_model, cur_score, x, y, rand):

        def report(grid_scores, n_top=1):
            top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
            for i, score in enumerate(top_scores):
                print("Model with rank: {0}".format(i + 1))
                print("Mean validation " + cur_score +
                      " score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score,
                                                              np.std(score.cv_validation_scores)))
                print("Parameters: {0}".format(score.parameters))
                print("")

        if not rand:
            start = time.time()
            param_grid = get_param_grid(model[1], int(y.size * 0.65), rand)
            print('\nOptimizing parameters for: ' + model[1] +
                  ' using Grid Search with score: ' + cur_score + '.')
            grid_model = GridSearchCV(model[0], param_grid, cv=5,
                                      scoring=cur_score, n_jobs=2)
            grid_model.fit(x, y)
            print("\nGridSearchCV took %.2f seconds for %d candidate parameter settings."
                  % (time.time() - start, len(grid_model.grid_scores_)))
            report(grid_model.grid_scores_)
            return grid_model
        else:
            start = time.time()
            param_grid = get_param_grid(cur_model[1], int(y.size * 0.65), rand)
            print('\nOptimizing parameters for: ' + model[1] +
                  ' using Random Search with score: ' + cur_score + '.')
            rand_model = RandomizedSearchCV(model[0],
                                            param_distributions=param_grid,
                                            n_jobs=2, n_iter=75)
            rand_model.fit(x, y)
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

    scores = ['accuracy', 'log_loss', 'recall']
    out_list = []

    for score in scores:
        opt_model = optimize_params(model, score, x_dev, y_dev, False)
        out_list.append([model, score, opt_model.best_estimator_])
        opt_model = optimize_params(model, score, x_dev, y_dev, True)
        out_list.append([model, score, opt_model.best_estimator_])

    return out_list


def train_and_eval(x_train, y_train, x_test, y_test, param_result):

    print('\nTraining and evaluating...')

    def get_kfold_scores(cur_model, score, score_name):

        print('\n10-Fold CV ' + score_name + ' Score:')

        scores = cross_val_score(estimator=cur_model, X=x_train, y=y_train,
                                 cv=5, scoring=score, n_jobs=2)
        print(score_name + " : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores.mean()

    result = {}

    for result_list in param_result:

        print('\n########\nEvaluating: ' + str(result_list[2]))

        # avail_scores = [['accuracy', 'Accuracy'],
        #                 ['average_precision', 'Average Precision'],
        #                 ['f1', 'F1 Score'],
        #                 ['precision', 'Precision'],
        #                 ['recall', 'Recall'],
        #                 ['roc_auc', 'ROC AUC']]
        #### vv Only metrics available for multiclass classification
        avail_scores = [['accuracy', 'Accuracy'],
                        ['f1', 'F1 Score'],
                        ['precision', 'Precision'],
                        ['recall', 'Recall']]
        opt_model = result_list[2]

        scores = {}
        for this_score in avail_scores:
            scores[this_score] = get_kfold_scores(cur_model=opt_model,
                                                  score=this_score[0],
                                                  score_name=this_score[1])
        result[result_list[2]] = scores

        ##### Not using k-fold cross validation
        opt_model.fit(x_train, y_train)
        y_pred = opt_model.predict(x_test)

        # print('\nClassification Report:')
        # print(metrics.classification_report(y_test, y_pred))
        # print('\nAccuracy Score:')
        # print(metrics.accuracy_score(y_test, y_pred))
        # print('\nF1-Score:')
        # print(metrics.f1_score(y_test, y_pred))

        print('\nConfusion Matrix:')
        print(metrics.confusion_matrix(y_test, y_pred))
        print('\nHamming Loss:')
        print(metrics.hamming_loss(y_test, y_pred))
        print('\nJaccard Similarity:')
        print(metrics.jaccard_similarity_score(y_test, y_pred))

        # #vvv Not supported due to ValueError: y_true and y_pred have different number of classes 3, 2
        # # print('\nLog Loss:')
        # # print(metrics.log_loss(y_test, y_pred))
        # # print('\nRecall:')
        # # print(metrics.recall(y_test, y_pred))

        # #vvv multiclass not supported
        # # print('\nMatthews Correlation Coefficient:')
        # # print(metrics.matthews_corrcoef(y_test, y_pred))
        ## print('\nPrecision:')
        # print(metrics.precision_score(y_test, y_pred))
    return result


def generate_results(data_dict, model_list, output):

    print('\n################')
    print('\nNow generating model results...')
    print('\n################\n')

    results = []

    x_train, x_dev, y_train, y_dev = train_test_split(data_dict['x_train'],
                                                      data_dict['y_train'],
                                                      test_size=0.5,
                                                      random_state=33)
    for model in model_list:

        print('\nEvaluating ' + model[1] + '.')

        cur_result = param_optimization(model, x_dev, y_dev, output)
        results.append(train_and_eval(x_train, y_train,
                                      data_dict['x_test'],
                                      data_dict['y_test'],
                                      cur_result))
        print('\n##############################################')

    return results