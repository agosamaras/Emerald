from array import array
from enum import auto
from re import sub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import xgboost
from sklearn import tree
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import joblib

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

data_path = 'cad_dset_multi.csv'
data = pd.read_csv(data_path)
dataframe = pd.DataFrame(data.values, columns=data.columns)
x = dataframe.drop(['ID','female','CNN_CAD','Doctor: CAD','output','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_CAD','Doctor: CAD', 'Expert Diagnosis Binary','output','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
y = dataframe['output'].astype(int)

# ml algorithms initialization
svm = svm.SVC(kernel='rbf')
lr = linear_model.LinearRegression()
dt = DecisionTreeClassifier()
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60)
ada = AdaBoostClassifier(n_estimators=150, random_state=0)
knn = KNeighborsClassifier(n_neighbors=20) #TODO n_neighbors=13 when testing with doctor, 20 w/o doctor
tab = TabPFNClassifier(device='cpu', N_ensemble_configurations=26)
xgb = xgboost.XGBRegressor(objective="binary:hinge", random_state=42) # 68,48%
light = LGBMClassifier(objective='binary', random_state=5, n_estimators=80, n_jobs=-1) # 72,16% / 80 -> 78,291
catb = CatBoostClassifier(n_estimators=79, learning_rate=0.1, verbose=False)

# doc/no_doc parameterization
sel_alg = catb
x = x_nodoc #TODO comment when testing with doctor
X = x


doc_gen_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kidney Disease',
       'ANGINA LIKE', 'ECG', 'SEX', 'Age: under 40', 'CNN', 'Expert Diagnosis Binary']

no_doc_sfs_catb = ['known CAD', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 
                   'Angiopathy', 'Chronic Kidney Disease', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 
                   'ECG', 'SEX', 'Obese', 'Age: 40-50', 'Age: 50-60', 'CNN']

sel_features = no_doc_sfs_catb

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass
    else:
        X = X.drop(feature, axis=1)
print(X.columns)

est = sel_alg.fit(X, y)
n_yhat = est.predict(X)

# Save the trained model to a file
joblib.dump(est, f'{sel_alg}_multimodal_model.joblib')

print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)
print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
scoring = {
    'sensitivity': metrics.make_scorer(metrics.recall_score),
    'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
}
print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)

# print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
# print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))
