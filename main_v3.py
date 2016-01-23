__author__ = 'Ricardo'

import time
from process_csv import process_csv
from model_results import generate_results
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

start = time.time()

file_path = 'C:/Users/Ricardo/PycharmProjects/Student Attrition/Trial_Data(50).csv'
out_txt = open('output.txt', "wb")
data = process_csv(file_path)

# Need to test compatibility with Neural Net
model_list = [[SVC(), 'SVC()'],
              [KNeighborsClassifier(), 'KNeighborsClassifier()'],
              [LogisticRegression(), 'LogisticRegression()']]

# Generate df from file in path and split into labels/features
# Then generate model results for objects in model_list
if __name__ == '__main__':

    print(generate_results(data, model_list, out_txt))

    print('Total runtime: ' + str(time.time() - start) + ' s')

    # if sys.version_info < (3, 0):
    #     raw_input('Press Enter to close:')
    # else:
    #     input('Press Enter to close:')