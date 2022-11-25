from array import array
from re import sub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import itertools
import sys
import multiprocessing
from tqdm import tqdm #tqmd progress bar

# function for getting all possible combinations of a list
from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def ml(df,subset, rndm): #, svm=True, lr=True, knn=True, dt=True):
    global i
    global best_acc
    # select features
    feature_df = df[list(subset)]
    X = np.asarray(feature_df)
    # select target variable
    df['CAD'] = df['CAD'].astype('int')
    y = np.asarray(df['CAD'])

    ## Train/Test split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=rndm, shuffle=True)

    # #################################
    # #### Support Vector Machines ####
    # #################################
    # # print("\nSupport Vector Machines\n")
    # alg="svm"
    # ## Modelling
    # rbf = svm.SVC(kernel='rbf')
    # rbf.fit(X_train, y_train)

    # ## Prediction
    # yhat = rbf.predict(X_test)

    # return metrics.accuracy_score(y_test, yhat)

    # if metrics.accuracy_score(y_test, yhat) > best_acc:
    #     best_acc=metrics.accuracy_score(y_test, yhat)
    #     print("new best accuracy: ", best_acc)
    #     print("algorithm: ", alg)
    #     print("features: ", subset)

    ##########################
    # ### Linear Regression ####
    # ##########################
    # # print("\nLinear Regression\n")
    # alg="lr"
    # # train a Logistic Regression Model using the train_x you created and the train_y created previously
    # regr = linear_model.LinearRegression()
    # X = np.asarray(feature_df)
    # y = np.asarray(df['CAD'])
    # regr.fit(X, y)

    # test_x = np.asarray(feature_df)
    # test_y = np.asarray(df['CAD'])
    # # find the predictions using the model's predict function and the test_x data
    # test_y_ = regr.predict(test_x)
    # # convert test_y_ to boolean
    # bool_test_y_ = []
    # for y in test_y_:
    #     if y < 0.5:
    #         bool_test_y_.append(0)
    #     else:
    #         bool_test_y_.append(1)

    # return metrics.accuracy_score(test_y, bool_test_y_)

    # if metrics.accuracy_score(test_y, bool_test_y_) > best_acc:
    #     best_acc=metrics.accuracy_score(test_y, bool_test_y_)
    #     print("new best accuracy: ", best_acc)
    #     print("algorithm: ", alg)
    #     print("features: ", subset)

    # ############################
    # ### K Nearest Neighbors ####
    # ############################
    # # print("\nK Nearest Neighbors\n")
    # alg="knn"
    # # train a Logistic Regression Model using the train_x you created and the train_y created previously
    # k = 20 #TODO k=13 when testing with doctor, 20 w/o
    # neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    # # print("neigh :", neigh)

    # # Prediction
    # n_yhat = neigh.predict(X_test)

    # # print("KNN with n {a}, try {b}: {c}%".format(a = k, b = i, c = metrics.accuracy_score(y_test, n_yhat)))
    # return metrics.accuracy_score(y_test, n_yhat)

    # if metrics.accuracy_score(y_test, n_yhat) > best_acc:
    #     best_acc=metrics.accuracy_score(y_test, n_yhat)
    #     print("new best accuracy: ", best_acc)
    #     print("algorithm: ", alg)
    #     print("features: ", subset)

    # ########################
    # #### Decision Trees ####
    # ########################
    # # print("\nDecision Trees\n")
    # alg="dt"
    # ## Modelling
    # # We will first create an instance of the DecisionTreeClassifier called drugTree
    # # Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
    # drugTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    # drugTree # it shows the default parameters
    # # fit the data with the training feature matrix X_train and training response vector y_train
    # drugTree.fit(X_train,y_train)

    # ## Prediction
    # predTree = drugTree.predict(X_test)

    # return metrics.accuracy_score(y_test, predTree)

    # if metrics.accuracy_score(y_test, predTree) > best_acc:
    #     best_acc=metrics.accuracy_score(y_test, predTree)
    #     print("new best accuracy: ", best_acc)
    #     print("algorithm: ", alg)
    #     print("features: ", subset)

    ######################
    ### Random Forest ####
    ######################
    # print("\nRandom Forest\n")
    alg="rf"
    # train a Logistic Regression Model using the train_x you created and the train_y created previously
    rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60).fit(X_train,y_train)

    # Prediction
    n_yhat = rndF.predict(X_test)
    # print("Random Forest accuracy: {c}%".format(c = metrics.accuracy_score(y_test, n_yhat)))
    return metrics.accuracy_score(y_test, n_yhat)

#     # # if metrics.accuracy_score(y_test, n_yhat) > best_acc:
#     # #     best_acc=metrics.accuracy_score(y_test, n_yhat)
#     # #     print("new best accuracy: ", best_acc)
#     # #     print("algorithm: ", alg)
#     # #     print("features: ", subset)

    # ##################
    # ### ADA Boost ####
    # ##################
    # # print("\nADA Boost\n")
    # alg="ada"
    # # train a Logistic Regression Model using the train_x you created and the train_y created previously
    # ada = AdaBoostClassifier(n_estimators=30, random_state=0).fit(X_train,y_train)

    # # Prediction
    # n_yhat = ada.predict(X_test)

    # # print("Ada Boost accuracy: {c}%".format(c = metrics.accuracy_score(y_test, n_yhat)))
    # return metrics.accuracy_score(y_test, n_yhat)

    # if metrics.accuracy_score(y_test, n_yhat) > best_acc:
    #     best_acc=metrics.accuracy_score(y_test, n_yhat)
    #     print("new best accuracy: ", best_acc)
    #     print("algorithm: ", alg)
    #     print("features: ", subset)

    # if i % 1000 == 0: #13400 == 0:
    #     # prog_bar.update(13400)
    #     # print_progress_bar(i, total=total, label="total progress\n")
    #     print("PROGRESS: {a:5.2f}%".format(a =float(i)/float(total)))

# create a dataframe from the data and read it
df = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/src/cad_dset.csv')
test = ['known CAD', 'previous PCI', 'Diabetes', 'Smoking',
       'Arterial Hypertension', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS',
       'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN',
       'RST ECG', 'male', 'Overweight', '<40', 'CNN_Healthy',
       'Doctor: Healthy']

# doctor's decision's accuracy
doctor = np.asarray(df['Doctor: Healthy'])
true = np.asarray(df['HEALTHY'])
# initialize best accuracy as the doctor's accuracy
doc_acc=metrics.accuracy_score(true, doctor)
best_acc = 80.0
print("doctor's accuracy: ", doc_acc)

iterations = 200

from joblib import Parallel, delayed
# knn_acc = Parallel(n_jobs=-1)(delayed(ml)(df, test, rndm=i) for i in range(0,iterations))
# total_knn_acc = sum(knn_acc)
# print("avg algorithm acc: {a:5.2f}%".format(a = total_knn_acc/iterations*100))

# # with tqdm(total=iterations) as prog_bar:
# for i in range(0,iterations):
#     ml(df, knn_SFS_doc, rndm=i)
#         # if i % 1000:
#         #     prog_bar.update(1000)

# print("avg knn acc: {a:5.2f}%".format(a = total_knn_acc/iterations*100))

no_doc_gen_rdnF_60_none = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN',
       'RST ECG', 'male', '40-50']

knn_acc = Parallel(n_jobs=-1)(delayed(ml)(df, no_doc_gen_rdnF_60_none, rndm=i) for i in range(0,iterations))
total_knn_acc = sum(knn_acc)
print("avg algorithm acc: {a:5.2f}%".format(a = total_knn_acc/iterations*100))

print('\a')
