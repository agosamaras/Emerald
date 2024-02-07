from array import array
from enum import auto
from re import sub
from turtle import backward
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
import itertools
import sys
import multiprocessing
import joblib
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

data_path = '/d/Σημειώσεις/PhD - EMERALD/1. CAD/src/cad_dset.csv'
# data_path = '/mnt/c/Users/samar/Documents/PhD - EMERALD/Extras/Parathyroid/input_data.csv'
data = pd.read_csv(data_path)
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
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60)
ada = AdaBoostClassifier(n_estimators=150, random_state=0)
knn = KNeighborsClassifier(n_neighbors=20) #TODO n_neighbors=13 when testing with doctor, 20 w/o doctor
tab = TabPFNClassifier(device='cpu', N_ensemble_configurations=26)
xgb = xgboost.XGBRegressor(objective="binary:hinge", random_state=42) # 68,48%
light = LGBMClassifier(objective='binary', random_state=5, n_estimators=80, n_jobs=-1) # 72,16% / 80 -> 78,291
catb = CatBoostClassifier(n_estimators=79, learning_rate=0.1, verbose=False)

# doc/no_doc parameterization
sel_alg = rndF
# x = x_nodoc #TODO comment when testing with doctor
X = x

# selected_features = catb.select_features(
#                 X,
#                 y,
#                 # eval_set=None,
#                 features_for_select=X.columns.values,
#                 num_features_to_select=10,
#                 algorithm='RecursiveByShapValues',
#                 # steps=None,
#                 shap_calc_type='Regular',
#                 train_final_model=True,
#                 verbose=1,
#                 logging_level=None,
#                 plot=False,
#                 log_cout=sys.stdout,
#                 log_cerr=sys.stderr)
# print("now: ", selected_features['selected_features_names'])

#############################################
#### Genetic Algorithm Feature Selection ####
#############################################

# for i in range (0,3):
#     print("run no ", i, ":")
#     selector = GeneticSelectionCV(
#         estimator=sel_alg,
#         cv=10,
#         verbose=2,
#         scoring="accuracy", 
#         max_features=26, #TODO change to 27 when testing with doctor, 26 without
#         n_population=100,
#         crossover_proba=0.8,
#         mutation_proba=0.8,
#         n_generations=200,
#         crossover_independent_proba=0.8,
#         mutation_independent_proba=0.4,
#         tournament_size=5,
#         n_gen_no_change=60,
#         caching=True,
#         n_jobs=-1)
#     selector = selector.fit(x, y)
#     n_yhat = selector.predict(x)
#     sel_features = x.columns[selector.support_]
#     print("Genetic Feature Selection:", x.columns[selector.support_])
#     print("Genetic Accuracy Score: ", selector.score(x, y))
#     print("Testing Accuracy: ", metrics.accuracy_score(y, n_yhat))

#     ###############
#     #### CV-10 ####
#     ###############
#     x = x_nodoc
#     for feature in x.columns:
#         if feature in sel_features:
#             pass
#         else:
#             X = X.drop(feature, axis=1)
    
#     print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)

#######################################
#### Best Result So Far - w Doctor ####
#######################################
doc_gen_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking',
              'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
              'Chronic Kindey Disease', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS',
              'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight',
              'u40', '50b60', 'Doctor: Healthy'] # knn (n=13) features from genetic selection 83,89% -> cv-10: 83.01% / manual: 80.36%
doc_gen_dt = ['previous AMI', 'previous CABG', 'Arterial Hypertension', 'Angiopathy',
             'Chronic Kindey Disease', 'Family History of CAD', 'ANGINA LIKE',
             'male', 'u40', 'Doctor: Healthy'] # dt 83,89% -> cv-10: 81,96% / manual: 79,84%
doc_gen_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease',
       'ANGINA LIKE', 'RST ECG', 'male', 'u40', 'Doctor: Healthy'] # rndF 84,41% -> cv-10: 83,02% / manual: 81,13% # good results, small feature set each time
doc_gen_svm = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking',
       'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 'ASYMPTOMATIC',
       'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'RST ECG', 'male', '40b50',
       'Doctor: Healthy'] # svm 86,51% -> cv-10: 82,66% / manual:
doc_gen_ada_150 = ['known CAD', 'previous STROKE', 'Diabetes', 'Family History of CAD',
       'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION',
       'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese',
       'u40', 'Doctor: Healthy'] # ada 81,79% -> cv-10: 80,03

doc_SFS_svm = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking', 'Dislipidemia', 
              'Chronic Kindey Disease', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 
              'ANGINA LIKE', 'RST ECG', 'male', 'Overweight', 'Obese', 'u40', '40b50', 'Doctor: Healthy'] # 82,13% -> cv-10: 82,13% / manual: 79,99%
doc_SFS_svm_fwd = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 
                  'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 
                  'Family History of CAD', 'ASYMPTOMATIC', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 
                  'RST ECG', 'male', 'Overweight', 'Obese', 'u40', '40b50', 'o60', 'Doctor: Healthy'] # 81,78% -> cv-10: 81,78% / manual: 79,8%
doc_doc_SFS_dt = ['known CAD', 'Smoking', 'Arterial Hypertension', 'Overweight', 'Doctor: Healthy'] # 79,69% -> cv-10: 78,63% / manual: 77,98%
doc_doc_SFS_dt_fwd = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Arterial Hypertension', 
                     'Chronic Kindey Disease', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight', 'u40', 'Doctor: Healthy'] # 82,32% -> cv-10: 78,81% / manual: 79,1%
doc_SFS_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 
              'ANGINA LIKE', 'male', 'Overweight', '40b50', 'o60', 'Doctor: Healthy'] # knn (n=13) 82,67% -> cv-10: 82,66% /  manual: 79,36%
doc_SFS_knn_fwd = ['previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Arterial Hypertension', 'Angiopathy', 'Chronic Kindey Disease', 'Family History of CAD',
                  'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight', 'u40', '40b50', 'o60', 'Doctor: Healthy'] # knn (n=13) 82,49% -> cv-10: 82,14% /  manual: 78,92%
doc_sfs_ada = ['known CAD', 'previous AMI', 'Diabetes', 'Family History of CAD', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese', 'u40', 'Doctor: Healthy'] # 81,96% -> cv-10: 81,62%
doc_sfs_ada_fwd = ['previous CABG', 'Dislipidemia', 'Angiopathy', 'male', 'Doctor: Healthy'] # 80,74% -> cv-10: 80,74% /  manual: 80,26% (w 1000 iterations)
doc_sfs_rndF = ['Smoking', 'Angiopathy', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'RST ECG', 'normal_weight', 'u40', '40b50', 'o60', 'Doctor: Healthy'] # 79,86% -> cv-10: 75,48% /  manual: 78,82% (w 1000 iterations)
doc_sfs_rndF_fwd = ['known CAD', 'previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes', 
                   'Smoking', 'Chronic Kindey Disease', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'o60', 'Doctor: Healthy'] # 80,74% -> cv-10: 77,23% /  manual: 79,28% (w 1000 iterations)
#########################################
#### Best Result So Far - w/o Doctor ####
#########################################
no_doc_gen_rdnF_60_none = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN',
       'RST ECG', 'male', '40b50'] # 79,33 / 77,59 -> 76,62
no_doc_gen_ada_30 = ['known CAD', 'Diabetes', 'Angiopathy', 'Family History of CAD',
       'ATYPICAL SYMPTOMS', 'male', '40b50'] # 73,56 / 76,54 -> 76,05
no_doc_gen_knn = ['known CAD', 'previous CABG', 'Diabetes', 'Smoking',
       'Chronic Kindey Disease', 'Family History of CAD', 'ASYMPTOMATIC',
       'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'RST ECG', 'male', 'Obese',
       '40b50', '50b60', 'CNN_Healthy'] # knn (n=20) features from genetic selection 77,58% -> cv-10: 76,37% / manual: 75,32%
no_doc_gen_dt = ['known CAD', 'previous PCI', 'Arterial Hypertension', 'Angiopathy',
       'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'normal_weight', '40b50',
       '50b60', 'CNN_Healthy'] # 82,49% -> cv-10: 73,04% / manual: 71,89%
no_doc_gen_svm = ['known CAD', 'previous CABG', 'Diabetes', 'Smoking', 'ASYMPTOMATIC',
       'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN',
       'RST ECG', 'male', '40b50', '50b60', 'CNN_Healthy'] # svm 82,31% -> cv-10: 78,12% / manual: 75,7%
no_doc_gen_tab = [] # %
no_doc_gen_xgb = [] # %
no_doc_gen_light = [] # %
no_doc_gen_catb = ['known CAD', 'previous PCI', 'Diabetes', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', '40', '60'] # 76,89%
no_doc_catb  = ['known CAD', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'ANGINA LIKE', 'RST ECG', 'male', 'Overweight'] # 73,84%
no_doc_catb2 = ['known CAD', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'ANGINA LIKE', 'RST ECG', 'male', 'Overweight']

no_doc_SFS_svm = ['known CAD', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'normal_weight', '40b50'] # 77,77% -> cv-10: 77,77% /  manual: 75,53%
no_doc_SFS_svm_fwd = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '40b50'] # 78,29% -> cv-10: 78,29% /  manual: 76,82%
no_doc_SFS_dt = ['known CAD', 'Diabetes', 'Smoking', 'Chronic Kindey Disease', 'Family History of CAD', 'male', '40b50', '50b60'] # 76,72% -> cv-10: 76,01% /  manual: 75,41%
no_doc_SFS_dt_fwd = ['known CAD', 'previous PCI', 'previous STROKE', 'Diabetes', 'Family History of CAD', 'DYSPNOEA ON EXERTION', 
                    'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'u40', '40b50'] # 78,29% -> cv-10: 78,294% /  manual: 73,85%
no_doc_SFS_knn = ['known CAD', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Chronic Kindey Disease', 'ATYPICAL SYMPTOMS', 'male'] # knn (n=20) 76,54% -> cv-10: 76,54% /  manual: %
no_doc_SFS_knn_fwd = ['known CAD', 'previous CABG', 'Diabetes', 'Angiopathy', 'Chronic Kindey Disease', 'ATYPICAL SYMPTOMS', 'INCIDENT OF PRECORDIAL PAIN', 'male', '40b50', '50b60', 'o60'] # knn (n=20) 76,89% -> cv-10: 76,89% /  manual: %
no_doc_sfs_ada = ['known CAD', 'previous CABG', 'Diabetes', 'Dislipidemia', 'Family History of CAD', 'ASYMPTOMATIC', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 
                 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '40b50'] # 77,07% -> cv-10: 75,49% / manual: 75,24%
no_doc_sfs_ada_fwd = ['known CAD', 'previous CABG', 'Diabetes', 'Angiopathy', 'Family History of CAD', 'ATYPICAL SYMPTOMS', 'male'] # 77,07% -> cv-10: 75,49% / manual: 76,30%
no_doc_sfs_rndF = ['known CAD', 'Diabetes', 'DYSPNOEA ON EXERTION', 'male', 'Overweight', 'Obese'] # 75,84% -> cv-10: 74,09% / manual: 75,25%
no_doc_sfs_rndF_fwd = ['known CAD', 'previous STROKE', 'Diabetes', 'Angiopathy', 'ATYPICAL SYMPTOMS', 'male'] #  76,37% -> cv-10: 76,54% / manual: 74,9%
no_doc_sfs_tab = [] # 75,84% -> cv-10: 74,09% / manual: 75,25%
no_doc_sfs_tab_fwd = [] #  76,37% -> cv-10: 76,54%
no_doc_sfs_xgb = ['known CAD', 'previous AMI', 'previous PCI', 'Diabetes', 'Arterial Hypertension', 'Angiopathy', 'Chronic Kindey Disease', 'ATYPICAL SYMPTOMS', 'male'] # 76.9%
no_doc_sfs_xgb_fwd = ['known CAD', 'previous PCI', 'Diabetes', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '40b50'] #  77,6%
no_doc_sfs_light = ['previous AMI', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking', 'ASYMPTOMATIC', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'RST ECG', 'male', 'Obese', '40b50', '50b60', '60'] # 77.59%
no_doc_sfs_light_fwd = ['known CAD', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Angiopathy', 'Chronic Kindey Disease', 'ASYMPTOMATIC', 
                        'ATYPICAL SYMPTOMS', 'INCIDENT OF PRECORDIAL PAIN', 'male', '40b50'] # 78,12%
no_doc_sfs_catb = ['known CAD', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Angiopathy', 'Chronic Kindey Disease', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Obese', '40b50', '50b60'] # 78,82%
#######################################
doc_SFS_svm_plus = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 
              'Chronic Kindey Disease', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 
              'ANGINA LIKE', 'RST ECG', 'male', 'Overweight', 'Obese', 'u40', '40b50', '50b60','o60', 'Doctor: Healthy'] # svm 85,81% -> cv-10: 81,78%
doc_gen_dt_plus = ['known CAD','previous AMI', 'previous CABG', 'Arterial Hypertension', 'Angiopathy',
             'Chronic Kindey Disease', 'Family History of CAD', 'ANGINA LIKE',
             'male', 'u40', '40b50', '50b60','o60', 'Doctor: Healthy', 'Diabetes', 'Smoking', 'Dislipidemia'] # dt 90,37% -> cv-10: 76,19%
doc_gen_knn_plus = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking',
              'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
              'Chronic Kindey Disease', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS',
              'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight',
              'u40', '40b50', '50b60','o60', 'Doctor: Healthy', 'Family History of CAD'] # knn 81,44% -> cv-10: 78.64%
doc_gen_rdnF_80_none_plus = ['known CAD', 'previous PCI', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 'Chronic Kindey Disease',
       'ANGINA LIKE', 'RST ECG', 'male', 'u40', '40b50', '50b60','o60', 'Doctor: Healthy', 'Family History of CAD'] # rndF 90,37% -> cv-10: 76,01%
doc_gen_ada_150_plus = ['known CAD', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension',
       'Dislipidemia', 'Family History of CAD', 'ASYMPTOMATIC',
       'ATYPICAL SYMPTOMS', 'DYSPNOEA ON EXERTION', 'RST ECG', 'male', 'Obese',
       'u40', '40b50', '50b60','o60', 'CNN_Healthy', 'Doctor: Healthy'] # ada 79,51% -> cv-10: 78,82
###
random_forest_no_doc_plus = ['known CAD', 'previous PCI', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 'Family History of CAD', 'INCIDENT OF PRECORDIAL PAIN',
                 'RST ECG', 'male', 'u40', '40b50', '50b60','o60'] # 87,92 -> cv-10: 73,21
svm_no_doc_plus = ['known CAD', 'previous PCI', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 'Family History of CAD', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'u40', '40b50', '50b60','o60'] # 79,33% -> cv-10: 75,84%
decision_tree_no_doc_plus = ['known CAD', 'previous PCI', 'previous STROKE', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 'Family History of CAD', 'DYSPNOEA ON EXERTION', 
                 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'u40', '40b50', '50b60','o60'] # 88,79% -> cv-10: 68,13%
knn_no_doc_plus = ['known CAD', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 'Family History of CAD', 'Chronic Kindey Disease', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 
       'DYSPNOEA ON EXERTION', 'RST ECG', 'male', 'Obese', 'u40', '40b50', '50b60','o60', 'CNN_Healthy'] # 73,91% -> cv-10: 70,58%
ada_no_doc_plus = ['known CAD', 'Diabetes', 'Smoking','Arterial Hypertension', 'Dislipidemia', 'Family History of CAD', 'Angiopathy',
       'ATYPICAL SYMPTOMS', 'male', 'u40', '40b50', '50b60','o60'] # 76,53 -> cv-10: 75,83
#######################################
sel_features = doc_gen_rdnF_80_none

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass
    else:
        X = X.drop(feature, axis=1)

est = sel_alg.fit(X, y)
n_yhat = est.predict(X)

# Save the trained model to a file
joblib.dump(est, f'{sel_alg}_model.joblib')

print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)
print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
print("f1_score: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).mean() * 100)
print("f1_score STD: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).std() * 100)
print("jaccard_score: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).mean() * 100)
print("jaccard_score STD: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).std() * 100)
scoring = {
    'sensitivity': metrics.make_scorer(metrics.recall_score),
    'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
}
print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)

print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))

# # By running the following loop we found out knn algorithm  gives best results for n=13
# # best features: ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 
# # 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'male', 'Overweight', '40b50', 
# # 'o60', 'Doctor: Healthy']
# best_acc=0
# for n in range (40,1,-1):
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
#         # print("features: ", sfs1.subsets_)
#         # print("beast features: ", sfs1.k_feature_idx_)

###############################
#### SFS Feature Selection ####
###############################
# X = x_nodoc
# sfs1 = SFS(sel_alg,
#            k_features="best",
#            forward=False,
#            floating=False, 
#            verbose=0,
#            scoring='accuracy',
#            cv=10,
#            n_jobs=-1)

# sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
# # print("features: ", sfs1.subsets_)
# print("SFS Accuracy Score: ", sfs1.k_score_)
# # print("beast features: ", sfs1.k_feature_idx_)
# print("SFS best features: ", sfs1.k_feature_names_)

# sel_features = sfs1.k_feature_names_

# ###############
# #### CV-10 ####
# ###############
# for feature in x.columns:
#     if feature in sel_features:
#         pass
#     else:
#         X = X.drop(feature, axis=1)

# # cross-validate result(s) 10fold
# print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)

# print("\n\n#### SFS Fwd ####")
# X = x_nodoc
# sfs1 = SFS(sel_alg,
#            k_features="best",
#            forward=True,
#            floating=False, 
#            verbose=0,
#            scoring='accuracy',
#            cv=10,
#            n_jobs=-1)

# sfs1 = sfs1.fit(x, y, custom_feature_names=x.columns)
# # print("features: ", sfs1.subsets_)
# print("SFS Accuracy Score: ", sfs1.k_score_)
# # print("beast features: ", sfs1.k_feature_idx_)
# print("SFS best features: ", sfs1.k_feature_names_)

# sel_features = sfs1.k_feature_names_

# ###############
# #### CV-10 ####
# ###############
# for feature in x.columns:
#     if feature in sel_features:
#         pass
#     else:
#         X = X.drop(feature, axis=1)

# # cross-validate result(s) 10fold
# print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)

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
df = pd.read_csv(data_path)
test = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
'Chronic Kindey Disease','Family History of CAD','ASYMPTOMATIC','ATYPICAL SYMPTOMS','ANGINA LIKE','DYSPNOEA ON EXERTION','INCIDENT OF PRECORDIAL PAIN','RST ECG','male','Overweight',
'Obese','normal_weight','u40','40b50','50b60','o60','CNN_Healthy']

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
