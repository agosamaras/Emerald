from array import array
from enum import auto
from re import sub
from turtle import backward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
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

# function for printing each component of confusion matrix
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

data = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/CAD/src/cad_dset.csv')
# print(data.columns)
# print(data.values)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD', 'Doctor: Healthy','HEALTHY','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
# print("x:\n",x.columns)
y = dataframe['CAD'].astype(int)
# print("y:\n",y)

# ml algorithms initialization
svm = svm.SVC(kernel='rbf')
lr = linear_model.LinearRegression()
dt = DecisionTreeClassifier()
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60) #TODO n_estimators=80 when testing with doctor, 60 w/o doctor
ada = AdaBoostClassifier(n_estimators=30, random_state=0) #TODO n_estimators=150 when testing with doctor, 30 w/o doctor
knn = KNeighborsClassifier(n_neighbors=20) #TODO n_neighbors=13 when testing with doctor, 20 w/o doctor



#################################
#### Best Results - w Doctor ####
#################################
doc_svm = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking',
           'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 'ASYMPTOMATIC',
           'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'RST ECG', 'male', '40-50',
           'Doctor: Healthy'] # svm 86,51% -> cv-10: 82,66%
doc_dt = ['previous AMI', 'previous CABG', 'Arterial Hypertension', 'Angiopathy',
          'Chronic Kindey Disease', 'Family History of CAD', 'ANGINA LIKE',
          'male', '<40', 'Doctor: Healthy'] # dt 83,89% -> cv-10: 82,14%
doc_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking',
           'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
           'Chronic Kindey Disease', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS',
           'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight',
           '<40', '50-60', 'Doctor: Healthy'] # knn (n=13) features from genetic selection 83,89% -> cv-10: 83.01%
doc_ada = ['known CAD', 'previous AMI', 'Diabetes', 'Family History of CAD', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese', '<40', 'Doctor: Healthy'] # 81,96% -> cv-10: 81,62%
doc_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease', 'ANGINA LIKE', 'RST ECG', 'male', '<40', 'Doctor: Healthy'] # rndF 84,41% -> cv-10: 83,02%


###################################
#### Best Results - w/o Doctor ####
###################################
no_doc_svm = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '40-50'] # 78,29% -> cv-10: 78,29%
no_doc_dt = ['known CAD', 'previous PCI', 'previous STROKE', 'Diabetes', 'Family History of CAD', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '<40', '40-50'] # 78,29% -> cv-10: 77,07%
no_doc_knn = ['known CAD', 'previous CABG', 'Diabetes', 'Angiopathy', 'Chronic Kindey Disease', 'ATYPICAL SYMPTOMS', 'INCIDENT OF PRECORDIAL PAIN', 'male', '40-50', '50-60', '>60'] # knn (n=20) 76,89% -> cv-10: 76,89%
no_doc_ada_30 = ['known CAD', 'Diabetes', 'Angiopathy', 'Family History of CAD', 'ATYPICAL SYMPTOMS', 'male', '40-50'] # 73,56 / 76,54
no_doc_rdnF_60_none = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '40-50'] # 79,33 / 77,59
#######################################

x = x_nodoc #TODO ucommment when running w/o doctor
X = x
sel_features = no_doc_rdnF_60_none
sel_alg = rndF

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass    
    else:
        X = X.drop(feature, axis=1)

sel = sel_alg.fit(X, y)
n_yhat = sel.predict(X)
print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))

# cross-validate result(s) 10fold
cv_results = cross_validate(sel_alg, X, y, cv=10)
# sorted(cv_results.keys())
print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])))
print("metrics:\n", metrics.classification_report(y, n_yhat, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn'))
print("f1_score: ", metrics.f1_score(y, n_yhat, average='weighted'))
print("jaccard_score: ", metrics.jaccard_score(y, n_yhat,pos_label=1))
print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))

print("###### XAI ######")
print(sel.feature_names_in_) # feature names
print(sel.feature_importances_) # feature importance
for name, importance in zip(sel.feature_names_in_,sel.feature_importances_):
   print(f"{name : <50}{importance:1.4f}")
   # print(f"{importance:1.4f}")

# # ONLY for SVM & KNN
# from sklearn.inspection import permutation_importance
# perm_importance = permutation_importance(sel, X, y)
# for name, importance in zip(sel.feature_names_in_,perm_importance.importances_mean):
#    print(f"{name : <50}{importance:1.4f}")
#    # print(f"{importance:1.4f}")
