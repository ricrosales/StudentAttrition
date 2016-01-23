import pandas
import time

def build_df(csv_location):
    db = pandas.read_csv(csv_location)
    db.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
    return db

df = build_df('Trial_Data.csv')
labels = df.pop('exit_status').values
features = df.values

# Training the SVM

# Split dataset into training and test dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)
# Define function to evaluate k-fold cross validation
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import numpy as np

def evaluate_cross_validation(clf, x, y, k, out_txt):
    # Create a k-fold cross validation iterator
    out_txt.write('\n5-Fold Cross Validation: ')
    cv = KFold(len(y), k, shuffle=True, random_state=0)
    # By default the score used is the one returned by score method of the estimator accuracy
    scores = cross_val_score(clf, x, y, cv=cv)
    out_txt.write('\nCross Validation Score: ' + '\n' + str(scores))
    out_txt.write('\n' + str('Mean Score: {0:.3f} (+/-{1:.3f})'.format(np.mean(scores), sem(scores))))

from sklearn import metrics
from sklearn import grid_search
def train_and_evaluate(clf, x_train, x_test, y_train, y_test, out_txt):
    parameters = {'n_estimators': [25, 300], 'learning_rate': [0.001, 1.0]}
    new_clf = grid_search.GridSearchCV(clf, parameters)
    new_clf.fit(x_train, y_train)
    out_txt.write('\n\n************************************')
    out_txt.write('\n\n' + str(clf))
    out_txt.write('\nAccuracy on training set: ')
    out_txt.write('\n' + str(new_clf.score(x_train, y_train)))
    out_txt.write('\nAccuracy on testing set: ')
    out_txt.write('\n' + str(new_clf.score(x_test, y_test)))
    y_pred = new_clf.predict(x_test)
    out_txt.write('\nClassification Report: ')
    out_txt.write('\n' + str(metrics.classification_report(y_test, y_pred)))
    out_txt.write('\nConfusion Matrix: ')
    out_txt.write('\n' + str(metrics.confusion_matrix(y_test, y_pred)))
    return new_clf.score(x_test, y_test), str(new_clf)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

models = [AdaBoostClassifier()]

out_txt = open('output2.txt', "wb")
start = time.time()
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
best_model = ''
last_score = 0
for model in models:
    cur_score, cur_model = train_and_evaluate(model, x_train, x_test, y_train, y_test, out_txt)
    if cur_score > last_score:
        best_model = cur_model
    last_score = cur_score
    evaluate_cross_validation(model, x_train, y_train, 5, out_txt)
out_txt.write('\n\n*************************')
out_txt.write('\n\nThe best model is: \n' + best_model)
end = time.time()
out_txt.write("\n\nTime to run: " + str(end - start))