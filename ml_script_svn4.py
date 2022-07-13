from array import array
from enum import auto
from re import sub
from turtle import backward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import itertools
import sys
import multiprocessing
from tqdm import tqdm #tqmd progress bar

from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/src/cad_dset.csv')
# print(data.columns)
# print(data.values)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x = dataframe.drop(['ID','female','CNN_CAD','Doctor: CAD','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_CAD','Doctor: CAD', 'Doctor: Healthy','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x = x_nodoc
# print("x:\n",x.columns)
y = dataframe['CAD'].astype(int)
# print("y:\n",y)
svm = svm.SVC(kernel='rbf')
lr = linear_model.LinearRegression()
dt = DecisionTreeClassifier()
rndF = RandomForestClassifier(max_depth=2, random_state=0)
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
knn = KNeighborsClassifier(n_neighbors=13)
selector = GeneticSelectionCV(
    estimator=svm,
    cv=10,
    verbose=2,
    scoring="accuracy", 
    max_features=27, #TODO change to 28 when testing with doctor
    n_population=100,
    crossover_proba=0.8,
    mutation_proba=0.8,
    n_generations=1000,
    crossover_independent_proba=0.8,
    mutation_independent_proba=0.4,
    tournament_size=5,
    n_gen_no_change=60,
    caching=True,
    n_jobs=-1)
selector = selector.fit(x, y)
n_yhat = selector.predict(x)
print("Genetic Feature Selection:", x.columns[selector.support_])
print("Genetic Accuracy Score: ", selector.score(x, y))
print("Testing Accuracy: ", metrics.accuracy_score(y, n_yhat))


#######################################
#### Best Result So Far - w Doctor ####
#######################################
X = x
doc_gen_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking',
              'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
              'Chronic Kindey Disease', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS',
              'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight',
              '<40', '50-60', 'Doctor: Healthy'] # knn (n=13) features from genetic selection 83,89% -> cv-10: 83.01% / manual: 80.29%
doc_gen_dt = ['previous AMI', 'previous CABG', 'Arterial Hypertension', 'Angiopathy',
             'Chronic Kindey Disease', 'Family History of CAD', 'ANGINA LIKE',
             'male', '<40', 'Doctor: Healthy'] # dt 83,89% -> cv-10: 81,08% / manual: 79,84%
doc_gen_rdnF = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'ANGINA LIKE',
               'DYSPNOEA ON EXERTION', 'RST ECG', 'male', 'normal_weight', '50-60',
               '>60', 'Doctor: Healthy'] # rndF 81,96% -> cv-10: 79,16% / manual: 79,32%
doc_gen_svm = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG',
              'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension',
              'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE',
              'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese',
              '<40', '40-50', '50-60', 'CNN_Healthy', 'Doctor: Healthy'] # svm 87,39% -> cv-10: 81,09% / manual: 79,83%
doc_gen_ada = ['known CAD', 'previous CABG', 'Arterial Hypertension', 'Angiopathy',
              'ASYMPTOMATIC', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION',
              'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese',
              'normal_weight', '<40', 'Doctor: Healthy'] # ada 82.66% -> cv-10: 77,24% / manual: 79,73%

doc_SFS_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 
              'ANGINA LIKE', 'male', 'Overweight', '40-50', '>60', 'Doctor: Healthy'] # 82,67% -> cv-10: 82,66% /  manual: 79,36%
doc_SFS_knn_fwd = ['previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Arterial Hypertension', 'Angiopathy', 'Chronic Kindey Disease', 'Family History of CAD',
                  'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight', '<40', '40-50', '>60', 'Doctor: Healthy'] # 82,49% -> cv-10: 82,14% /  manual: 78,92%
doc_sfs_ada = ['known CAD', 'previous CABG', 'Angiopathy', 'ASYMPTOMATIC', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 
               'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', '<40', 'CNN_Healthy', 'Doctor: Healthy'] # 81,79% -> cv-10: 79,69% /  manual: 79,57% (w 1000 iterations)
doc_sfs_ada_fwd = ['previous CABG', 'Dislipidemia', 'Angiopathy', 'male', 'Doctor: Healthy'] # 80,74% -> cv-10: 80,74% /  manual: 80,26% (w 1000 iterations)
doc_sfs_rndF = ['Smoking', 'Angiopathy', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'RST ECG', 'normal_weight', '<40', '40-50', '>60', 'Doctor: Healthy'] # 79,86% -> cv-10: 75,48% /  manual: 78,82% (w 1000 iterations)
doc_sfs_rndF_fwd = ['known CAD', 'previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes', 
                   'Smoking', 'Chronic Kindey Disease', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '>60', 'Doctor: Healthy'] # 80,74% -> cv-10: 77,23% /  manual: 79,28% (w 1000 iterations)
#########################################
#### Best Result So Far - w/o Doctor ####
#########################################
X2 = x_nodoc
no_doc_gen_knn = [] # knn (n=13) features from genetic selection 83,89% -> cv-10: 83.01% / manual: 80.29%
no_doc_gen_dt = [] # % -> cv-10: % / manual: %
no_doc_gen_rdnF = [] # % -> cv-10: % / manual: %
no_doc_gen_svm = ['previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes',
       'Smoking', 'Chronic Kindey Disease', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE',
       'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG',
       'male', '50-60', 'CNN_Healthy'] # svm 81,43% -> cv-10: % / manual: %
no_doc_gen_ada = [] # % -> cv-10: % / manual: %

no_doc_SFS_svm = [] # % -> cv-10: % /  manual: %
no_doc_SFS_svm_fwd = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '40-50'] # 78,29% -> cv-10: % /  manual: %
no_doc_SFS_dt = [] # % -> cv-10: % /  manual: %
no_doc_SFS_dt_fwd = [] # % -> cv-10: % /  manual: %
no_doc_SFS_knn = [] # % -> cv-10: % /  manual: %
no_doc_SFS_knn_fwd = [] # % -> cv-10: % /  manual: %
no_doc_sfs_ada = [] # % -> cv-10: % /  manual: % (w 1000 iterations)
no_doc_sfs_ada_fwd = [] # % -> cv-10: % /  manual: % (w 1000 iterations)
no_doc_sfs_rndF = [] # % -> cv-10: % /  manual: % (w 1000 iterations)
no_doc_sfs_rndF_fwd = [] #  % -> cv-10: % /  manual: % (w 1000 iterations)

# for feature in x_nodoc.columns:
#     if feature in doc_sfs_rndF_fwd:
#         pass
#     else:
#         X2 = X2.drop(feature, axis=1)

# sel = rndF.fit(X2, y)
# n_yhat = sel.predict(X2)
# print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat))) # 83,54%

# # cross-validate result(s) 10fold
# cv_results = cross_validate(knn, X2, y, cv=10)
# # sorted(cv_results.keys())
# print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score']))) # 82,32%



## By running the following loop we found out knn algorithm  gives best results for n=13
## best features: ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 
## 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'male', 'Overweight', '40-50', 
## '>60', 'Doctor: Healthy']
# best_acc=0
# for n in range (20,1,-1):
#     knn = KNeighborsClassifier(n_neighbors=n)
#     sfs1 = SFS(knn,
#            k_features="best",
#            forward=False,
#            floating=False, 
#            verbose=0,
#            scoring='accuracy',
#            cv=10,
#            n_jobs=-1)

#     sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
#     acc = sfs1.k_score_
#     if acc > best_acc:
#         best_acc = acc
#         print("n: ", n)
#         print("score: ", sfs1.k_score_)
#         print("beast features: ", sfs1.k_feature_names_)
#     # # print("features: ", sfs1.subsets_)
#     # print("score: ", sfs1.k_score_)
#     # # print("beast features: ", sfs1.k_feature_idx_)
#     # print("beast features: ", sfs1.k_feature_names_)

sfs1 = SFS(svm,
           k_features="best",
           forward=False,
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=10,
           n_jobs=-1)

sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
# print("features: ", sfs1.subsets_)
print("SFS Accuracy Score: ", sfs1.k_score_)
# print("beast features: ", sfs1.k_feature_idx_)
print("SFS best features: ", sfs1.k_feature_names_)
print("end")

# function for getting all possible combinations of a list
from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def ml(df,subset):
    global i
    global best_acc
    # select features
    feature_df = df[list(subset)]
    X = np.asarray(feature_df)
    # select target variable
    df['CAD'] = df['CAD'].astype('int')
    y = np.asarray(df['CAD'])

    ## Train/Test split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4, shuffle=True)

    #################################
    #### Support Vector Machines ####
    #################################
    # print("\nSupport Vector Machines\n")
    alg="svm"
    ## Modelling
    rbf = svm.SVC(kernel='rbf')
    rbf.fit(X_train, y_train)

    ## Prediction
    yhat = rbf.predict(X_test) 

    if metrics.accuracy_score(y_test, yhat) > best_acc:
        best_acc=metrics.accuracy_score(y_test, yhat)
        print("new best accuracy: ", best_acc)
        print("algorithm: ", alg)
        print("features: ", subset)

    ###########################
    #### Linear Regression ####
    ###########################
    # print("\nLinear Regression\n")
    alg="lr"
    # train a Logistic Regression Model using the train_x you created and the train_y created previously
    regr = linear_model.LinearRegression()
    X = np.asarray(feature_df)
    y = np.asarray(df['CAD'])
    regr.fit(X, y)

    test_x = np.asarray(feature_df)
    test_y = np.asarray(df['CAD'])
    # find the predictions using the model's predict function and the test_x data
    test_y_ = regr.predict(test_x)
    # convert test_y_ to boolean
    bool_test_y_ = []
    for y in test_y_:
        if y < 0.5:
            bool_test_y_.append(0)
        else:
            bool_test_y_.append(1)

    if metrics.accuracy_score(test_y, bool_test_y_) > best_acc:
        best_acc=metrics.accuracy_score(test_y, bool_test_y_)
        print("new best accuracy: ", best_acc)
        print("algorithm: ", alg)
        print("features: ", subset)

    #############################
    #### K Nearest Neighbors ####
    #############################
    # print("\nK Nearest Neighbors\n")
    alg="knn"
    # train a Logistic Regression Model using the train_x you created and the train_y created previously
    ## Training
    # with k=4
    k = 7
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    # print("neigh :", neigh)

    ## Prediction
    n_yhat = neigh.predict(X_test)

    if metrics.accuracy_score(y_test, n_yhat) > best_acc:
        best_acc=metrics.accuracy_score(y_test, n_yhat)
        print("new best accuracy: ", best_acc)
        print("algorithm: ", alg)
        print("features: ", subset)

    ########################
    #### Decision Trees ####
    ########################
    # print("\nDecision Trees\n")
    alg="dt"
    ## Modelling
    # We will first create an instance of the DecisionTreeClassifier called drugTree
    # Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
    drugTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    drugTree # it shows the default parameters
    # fit the data with the training feature matrix X_train and training response vector y_train
    drugTree.fit(X_train,y_train)

    ## Prediction
    predTree = drugTree.predict(X_test)

    if metrics.accuracy_score(y_test, predTree) > best_acc:
        best_acc=metrics.accuracy_score(y_test, predTree)
        print("new best accuracy: ", best_acc)
        print("algorithm: ", alg)
        print("features: ", subset)

    if i % 1000 == 0: #13400 == 0:
        # prog_bar.update(13400)
        # print_progress_bar(i, total=total, label="total progress\n")
        print("PROGRESS: {a:5.2f}%".format(a =float(i)/float(total)))

# create a dataframe from the data and read it
df = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/src/cad_dset.csv')
test = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
'Chronic Kindey Disease','Family History of CAD','ASYMPTOMATIC','ATYPICAL SYMPTOMS','ANGINA LIKE','DYSPNOEA ON EXERTION','INCIDENT OF PRECORDIAL PAIN','RST ECG','male','Overweight',
'Obese','normal_weight','<40','40-50','50-60','>60','CNN_Healthy']

# doctor's decision's accuracy
doctor = np.asarray(df['Doctor: Healthy'])
true = np.asarray(df['HEALTHY'])
# initialize best accuracy as the doctor's accuracy
doc_acc=metrics.accuracy_score(true, doctor)
best_acc = doc_acc
print("doctor's accuracy: ", doc_acc)

i=0
total = 134217728

print('\a')
