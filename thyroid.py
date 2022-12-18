import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from genetic_selection import GeneticSelectionCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from typing import Tuple
import copy as cp
import seaborn as sns
import shap

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

# function to store predictions for each fold of Repeated Stratified KFold
def val_predict(model, kfold : RepeatedStratifiedKFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X,y):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba

# function for printing confusion matrix
def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = metrics.confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()

data = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/Extras/Parathyroid/input_data2.csv')

# print(data.columns)
# print(data.values)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['MULTIGLAND'] = data.MULTIGLAND
x = dataframe.drop(['MULTIGLAND'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
# print("x:\n",x.columns)
y = dataframe['MULTIGLAND'].astype(int)
# print("y:\n",y)

# ml algorithms initialization
svm = svm.SVC(kernel='rbf') # Avg CV-10 Testing Accuracy: 81.52% / '0recall': 0.9578947368421052 / '1recall': 0.7083333333333334
lr = linear_model.LinearRegression()
dt = DecisionTreeClassifier() # Avg CV-10 Testing Accuracy: 84.92%% / '0recall':  0.9789473684210527 / '1recall': 0.75
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60) # Avg CV-10 Testing Accuracy: 85.76% / '0recall': 0.968421052631579 / '1recall': 0.8800438596491228
ada = AdaBoostClassifier(n_estimators=150, random_state=0) # Avg CV-10 Testing Accuracy: 79.92% / '0recall': 0.9157894736842105 / '1recall': 0.5416666666666666
knn = KNeighborsClassifier(n_neighbors=3) # Avg CV-10 Testing Accuracy: 79.92% / '0recall': 0.9578947368421052 / '1recall':  0.5833333333333334

#############################################
#### Genetic Algorithm Feature Selection ####
#############################################

# for i in range (0,3):
#     X = x
#     print("run no ", i, ":")
#     selector = GeneticSelectionCV(
#         estimator=ada,
#         cv=10,
#         verbose=2,
#         scoring="accuracy", 
#         max_features=27, #TODO change to 27 when testing with doctor, 26 without
#         n_population=100,
#         crossover_proba=0.8,
#         mutation_proba=0.8,
#         n_generations=1000,
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
#     for feature in x.columns:
#         if feature in sel_features:
#             pass
#         else:
#             X = X.drop(feature, axis=1)

#     sel_fit = rndF.fit(X, y)
#     n_yhat = sel_fit.predict(X)
#     print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))
    
#     # cross-validate result(s) 10fold
#     cv_results = cross_validate(svm, X, y, cv=10)
#     # sorted(cv_results.keys())
#     print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])))


############################
#### Best Result So Far ####
############################
gen_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking',
              'Arterial Hypertension', 'Dislipidemia', 'Angiopathy',
              'Chronic Kindey Disease', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS',
              'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight',
              '<40', '50-60', 'Doctor: Healthy'] # knn (n=13) features from genetic selection 83,89% -> cv-10: 83.01% / manual: 80.36%
gen_dt = ['previous AMI', 'previous CABG', 'Arterial Hypertension', 'Angiopathy',
             'Chronic Kindey Disease', 'Family History of CAD', 'ANGINA LIKE',
             'male', '<40', 'Doctor: Healthy'] # dt 83,89% -> cv-10: 81,96% / manual: 79,84%
gen_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease',
       'ANGINA LIKE', 'RST ECG', 'male', '<40', 'Doctor: Healthy'] # rndF 84,41% -> cv-10: 83,02% / manual: 81,13% # good results, small feature set each time
gen_svm = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking',
       'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 'ASYMPTOMATIC',
       'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'RST ECG', 'male', '40-50',
       'Doctor: Healthy'] # svm 86,51% -> cv-10: 82,66% / manual:
gen_ada_150 = ['known CAD', 'previous STROKE', 'Diabetes', 'Family History of CAD',
       'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION',
       'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese',
       '<40', 'Doctor: Healthy'] # ada 81,79% -> cv-10: 80,03

SFS_svm = ['known CAD', 'previous PCI', 'previous CABG', 'Diabetes', 'Smoking', 'Dislipidemia', 
              'Chronic Kindey Disease', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 
              'ANGINA LIKE', 'RST ECG', 'male', 'Overweight', 'Obese', '<40', '40-50', 'Doctor: Healthy'] # 82,13% -> cv-10: 82,13% / manual: 79,99%
SFS_svm_fwd = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 
                  'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 
                  'Family History of CAD', 'ASYMPTOMATIC', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 
                  'RST ECG', 'male', 'Overweight', 'Obese', '<40', '40-50', '>60', 'Doctor: Healthy'] # 81,78% -> cv-10: 81,78% / manual: 79,8%
SFS_dt = ['known CAD', 'Smoking', 'Arterial Hypertension', 'Overweight', 'Doctor: Healthy'] # 79,69% -> cv-10: 78,63% / manual: 77,98%
SFS_dt_fwd = ['known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Arterial Hypertension', 
                     'Chronic Kindey Disease', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight', '<40', 'Doctor: Healthy'] # 82,32% -> cv-10: 78,81% / manual: 79,1%
SFS_knn = ['known CAD', 'previous AMI', 'previous CABG', 'Diabetes', 'Smoking', 'Arterial Hypertension', 'Dislipidemia', 'Angiopathy', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 
              'ANGINA LIKE', 'male', 'Overweight', '40-50', '>60', 'Doctor: Healthy'] # knn (n=13) 82,67% -> cv-10: 82,66% /  manual: 79,36%
SFS_knn_fwd = ['previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Arterial Hypertension', 'Angiopathy', 'Chronic Kindey Disease', 'Family History of CAD',
                  'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'male', 'Overweight', '<40', '40-50', '>60', 'Doctor: Healthy'] # knn (n=13) 82,49% -> cv-10: 82,14% /  manual: 78,92%
sfs_ada = ['known CAD', 'previous AMI', 'Diabetes', 'Family History of CAD', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Overweight', 'Obese', '<40', 'Doctor: Healthy'] # 81,96% -> cv-10: 81,62%
sfs_ada_fwd = ['previous CABG', 'Dislipidemia', 'Angiopathy', 'male', 'Doctor: Healthy'] # 80,74% -> cv-10: 80,74% /  manual: 80,26% (w 1000 iterations)
sfs_rndF = ['Smoking', 'Angiopathy', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'RST ECG', 'normal_weight', '<40', '40-50', '>60', 'Doctor: Healthy'] # 79,86% -> cv-10: 75,48% /  manual: 78,82% (w 1000 iterations)
sfs_rndF_fwd = ['known CAD', 'previous AMI', 'previous CABG', 'previous STROKE', 'Diabetes', 
                   'Smoking', 'Chronic Kindey Disease', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', '>60', 'Doctor: Healthy'] # 80,74% -> cv-10: 77,23% /  manual: 79,28% (w 1000 iterations)
############################
X = x
# sel_features = gen_knn  
sel_alg = rndF

################################
### Drop unnecessary fields ####
################################
# for feature in x.columns:
#     if feature in sel_features:
#         pass
#     else:
#         X = X.drop(feature, axis=1)

sel_fit = sel_alg.fit(X, y)
n_yhat = sel_fit.predict(X)
n_yhat = cross_val_predict(sel_alg, X, y)
print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))

#############
### CV-10 ###
#############
# cross-validate result(s) 10fold
# print(metrics.get_scorer_names())
# cv_results = cross_validate(sel_alg, X, y, cv=10, scoring=('accuracy'))
cv_results = cross_validate(sel_alg, X, y, cv=10)
# sorted(cv_results.keys())
print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])))
print("metrics:\n", metrics.classification_report(y, n_yhat, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn'))
print("f1_score: ", metrics.f1_score(y, n_yhat, average='weighted'))
print("jaccard_score: ", metrics.jaccard_score(y, n_yhat,pos_label=1))
print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))
plot_confusion_matrix(y, n_yhat, [0, 1])

###########
### XAI ###
###########
shap_values = shap.TreeExplainer(sel_alg).shap_values(X)
shap.summary_plot(shap_values[0], X)
for feature in X.columns:
    print(feature)
    shap.dependence_plot(feature, shap_values[0], X)

# #############
# ### SMOTE ###
# #############
# print("\n\n### SMOTE ####")
# # print(metrics.get_scorer_names())
# steps = [('over', SMOTE()), ('model', sel_alg)]
# pipeline = Pipeline(steps=steps)
# # evaluate pipeline
# for scoring in["accuracy", "roc_auc", "f1", "jaccard"]:
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
#     scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
#     print("Model", scoring, " mean=", scores.mean() , "stddev=", scores.std())


# print("\n\n### SMOTE No 2####")
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
# actual_classes, predicted_classes, predicted_prob = val_predict(sel_alg, cv, X.to_numpy(), y.to_numpy())

# prob_sum = np.sum(predicted_prob, axis = 0)
# print("Avg '0' Predicting Accuracy: {a:5.2f}%".format(a = 100*prob_sum[0]/len(predicted_prob)))
# print("Avg '1' Predicting Accuracy: {a:5.2f}%".format(a = 100*prob_sum[1]/len(predicted_prob)))
# print("Avg Predicting Accuracy: {a:5.2f}%".format(a = 100*prob_sum[0]/len(predicted_prob)*285/357+ 100*prob_sum[1]/len(predicted_prob)*72/357))
# plot_confusion_matrix(actual_classes, predicted_classes, [0, 1])


# # By running the following loop we found out knn algorithm  gives best results for n=3
# best_acc=0
# for n in range (300,1,-1):
#     rndF1 = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=n)
#     cv_results = cross_validate(rndF1, X, y, cv=10)

#     acc = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])
#     if acc > best_acc:
#         best_acc = acc
#         print("n: ", n)
#         print("score: ", acc)

###############################
#### SFS Feature Selection ####
###############################
# sfs1 = SFS(ada,
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
# print("end")

# function for getting all possible combinations of a list
from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

print('\a')
