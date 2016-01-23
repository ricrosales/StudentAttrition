import pandas
import time
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

def build_df(csv_location):
    # Convert csv to df
    db = pandas.read_csv(csv_location)
    db.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
    return db

def evaluate_cross_validation(clf, x, y, k, out_txt):
    # Create a k-fold cross validation iterator
    # write results of cross validation to out_txt
    out_txt.write('\n5-Fold Cross Validation: ')
    print('\n5-Fold Cross Validation: ')
    cv = KFold(len(y), k, shuffle=True, random_state=0)
    # By default the score used is the one returned by score method of the estimator accuracy
    scores = cross_val_score(clf, x, y, cv=cv)
    out_txt.write('\nCross Validation Score: ' + '\n' + str(scores))
    out_txt.write('\n' + str('Mean Score: {0:.3f} (+/-{1:.3f})'.format(np.mean(scores), sem(scores))))
    print('\nCross Validation Score: ' + '\n' + str(scores))
    print('\n' + str('Mean Score: {0:.3f} (+/-{1:.3f})'.format(np.mean(scores), sem(scores))))



def train_and_evaluate(clf, x_train, x_test, y_train, y_test, out_txt):
    clf.fit(x_train, y_train)
    print('\n\n************************************')
    print('\n\n' + str(clf))
    print('\nAccuracy on training set: ')
    print('\n' + str(clf.score(x_train, y_train)))
    print('\nAccuracy on testing set: ')
    print('\n' + str(clf.score(x_test, y_test)))
    out_txt.write('\n\n************************************')
    out_txt.write('\n\n' + str(clf))
    out_txt.write('\nAccuracy on training set: ')
    out_txt.write('\n' + str(clf.score(x_train, y_train)))
    out_txt.write('\nAccuracy on testing set: ')
    out_txt.write('\n' + str(clf.score(x_test, y_test)))
    y_pred = clf.predict(x_test)
    out_txt.write('\nClassification Report: ')
    out_txt.write('\n' + str(metrics.classification_report(y_test, y_pred)))
    out_txt.write('\nConfusion Matrix: ')
    out_txt.write('\n' + str(metrics.confusion_matrix(y_test, y_pred)))
    return clf.score(x_test, y_test), str(clf)


models = [SVR(kernel='linear'), SVR(kernel='poly'),
          SVR(kernel='rbf'), SVR(kernel='sigmoid'), NuSVR(kernel='linear'), NuSVR(kernel='poly'),
          NuSVR(kernel='rbf'), NuSVR(kernel='sigmoid'),
          KNeighborsRegressor(weights='distance', algorithm='auto'),
          KNeighborsRegressor(weights='uniform', algorithm='auto'),
          KNeighborsRegressor(weights='distance', algorithm='ball_tree'),
          KNeighborsRegressor(weights='distance', algorithm='ball_tree'),
          KNeighborsRegressor(weights='distance', algorithm='kd_tree'),
          KNeighborsRegressor(weights='distance', algorithm='kd_tree'),
          KNeighborsRegressor(weights='uniform', algorithm='brute'),
          KNeighborsRegressor(weights='distance', algorithm='brute'),
          KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='auto'),
          KNeighborsRegressor(n_neighbors=8, weights='uniform', algorithm='auto'),
          KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='ball_tree'),
          KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='ball_tree'),
          KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='kd_tree'),
          KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='kd_tree'),
          KNeighborsRegressor(n_neighbors=8, weights='uniform', algorithm='brute'),
          KNeighborsRegressor(n_neighbors=8, weights='distance', algorithm='brute'),
          RadiusNeighborsRegressor(weights='distance', algorithm='auto'),
          RadiusNeighborsRegressor(weights='uniform', algorithm='auto'),
          RadiusNeighborsRegressor(weights='distance', algorithm='ball_tree'),
          RadiusNeighborsRegressor(weights='distance', algorithm='ball_tree'),
          RadiusNeighborsRegressor(weights='distance', algorithm='kd_tree'),
          RadiusNeighborsRegressor(weights='distance', algorithm='kd_tree'),
          RadiusNeighborsRegressor(weights='uniform', algorithm='brute'),
          RadiusNeighborsRegressor(weights='distance', algorithm='brute'),
          RadiusNeighborsRegressor(radius=0.5, weights='distance', algorithm='auto'),
          RadiusNeighborsRegressor(radius=0.5, weights='uniform', algorithm='auto'),
          RadiusNeighborsRegressor(radius=0.5, weights='distance', algorithm='ball_tree'),
          RadiusNeighborsRegressor(radius=0.5, weights='distance', algorithm='ball_tree'),
          RadiusNeighborsRegressor(radius=0.5, weights='distance', algorithm='kd_tree'),
          RadiusNeighborsRegressor(radius=0.5, weights='distance', algorithm='kd_tree'),
          RadiusNeighborsRegressor(radius=0.5, weights='uniform', algorithm='brute'),
          RadiusNeighborsRegressor(radius=0.5, weights='distance', algorithm='brute'),
          DecisionTreeRegressor(splitter='best'), DecisionTreeRegressor(splitter='random'),
          AdaBoostRegressor(base_estimator='SVR', loss='linear'),
          AdaBoostRegressor(base_estimator='SVR', loss='square'),
          AdaBoostRegressor(base_estimator='SVR', loss='exponential'),
          AdaBoostRegressor(loss='linear'), AdaBoostRegressor(loss='square'),
          AdaBoostRegressor(loss='exponential'), BaggingRegressor(), BaggingRegressor(base_estimator='SVR'),
          ExtraTreesRegressor(n_estimators=5), ExtraTreesRegressor(n_estimators=10),
          ExtraTreesRegressor(n_estimators=15), GradientBoostingRegressor(loss='ls'),
          GradientBoostingRegressor(loss='lad'), GradientBoostingRegressor(loss='huber'),
          GradientBoostingRegressor(loss='quantile')]

# Begin timer for output
start = time.time()
# Set out_txt
out_txt = open('output2.txt', "wb")
# Generate df and split into labels/features
df = build_df('Vol Forecast.csv')
labels = df.pop('pred_day').values
features = df.values
# Split dataset into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)
# Standardize dataset
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# Define variables to rank models
best_model = ''
last_score = 0
# Master loop
# Loop through model list
# Train and evaluate ech model
# Rank models and write results to output
for model in models:
    try:
        cur_score, cur_model = train_and_evaluate(model, x_train, x_test, y_train, y_test, out_txt)
        if cur_score > last_score:
            best_model = cur_model
        evaluate_cross_validation(model, x_train, y_train, 5, out_txt)
    except:
        print str(model) + '\n\n The above model had an error. It was not considered'
        out_txt.write(str(model) + '\n\n The above model had an error. It was not considered')
out_txt.write('\n\n*************************')
out_txt.write('\n\nThe best model is: \n' + best_model)
end = time.time()
out_txt.write("\n\nTime to run: " + str(end - start))