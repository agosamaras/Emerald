import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
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
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
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
    # initialization
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

# function to print SHAP values and plots
def xai(model, X, val):
    shap_values = shap.TreeExplainer(model).shap_values(X)
    shap.summary_plot(shap_values[val], X)
    for feature in X.columns:
        print(feature)
        shap.dependence_plot(feature, shap_values[0], X)
    # shap.force_plot(explainer.expected_value, shap_values, X)
    p = shap.force_plot(shap.TreeExplainer(model).expected_value, shap_values, X, matplotlib = True, show = False)
    plt.savefig('tmp.svg')
    plt.close()
    return(p)

def xai_svm(model, X, idx):
    explainer = shap.KernelExplainer(model.predict, X.values[idx])
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    # shap.summary_plot(shap_values, X, plot_type='violin')
    for feature in X.columns:
        print(feature)
        shap.dependence_plot(feature, shap_values, X)
    shap.force_plot(explainer.expected_value, shap_values, X)

# function for random oversampling predict
def rndm_osmpl(model, kfold, X, y) -> Tuple[np.array, np.array, np.array]:
    # initialization
    y_pred = []
    y_actual = []
    oversample = RandomOverSampler(sampling_strategy='not majority')
    model_ = cp.deepcopy(model)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for train_index, test_index in kfold.split(X.values, y.values):
        # Split the data into train and test sets for the current fold
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        X_over, y_over = oversample.fit_resample(X_train, y_train)
        est = model_.fit(X_over, y_over)
        n_yhat = est.predict(X_test)

        # xai(est, X_train, 0)
        xai_svm(est, X, train_index)

        tpa, fpa, tna, fna = perf_measure(y_test, n_yhat)
        tp = tp + tpa
        fp = fp + fpa
        tn = tn + tna
        fn = fn + fna
        y_pred.extend(n_yhat)
        y_actual.extend(y_test)

    # print("TP/FP/TN/FN: ", tp, fp, tn, fn)
    # print("confusion matrix:\n", metrics.confusion_matrix(y_actual, y_pred, labels=[0,1]))
    plot_confusion_matrix(y_actual, y_pred, [0, 1])
    report = metrics.classification_report(y_actual, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn')
    print("metrics:\n", report)
    roc = metrics.roc_auc_score(y_actual, y_pred)
    # print("roc:\n", roc)
    # fpr, tpr, _ = metrics.roc_curve(y_actual, y_pred)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc, estimator_name='SVM')
    # display.plot()
    # plt.show()
    return report["0"]["recall"], report["1"]["recall"], report["accuracy"], roc

# function for printing confusion matrix
def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):
    matrix = metrics.confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()

def plot_2D_confusion_matrix_by_values(tp, fp, tn, fn, sorted_labels : list):
    matrix = [[tn, fp,],[fn, tp]] 
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()

# data_path = '/d/Σημειώσεις/PhD - EMERALD/Extras/Parathyroid/input_data2.csv'
data_path = '/mnt/c/Users/samar/Documents/PhD - EMERALD/Extras/Parathyroid/input_data.csv'
data = pd.read_csv(data_path)
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
X = x
sel_alg = svm

################################
### Drop unnecessary fields ####
################################
# for feature in x.columns:
#     if feature in sel_features:
#         pass
#     else:
#         X = X.drop(feature, axis=1)

sel_fit = sel_alg.fit(X, y)
n_yhat = cross_val_predict(sel_alg, X, y)
print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))

# #############
# ### CV-10 ###
# #############
# # cross-validate result(s) 10fold
# # print(metrics.get_scorer_names())
# # cv_results = cross_validate(sel_alg, X, y, cv=10, scoring=('accuracy'))
# cv_results = cross_validate(sel_alg, X, y, cv=10)
# # sorted(cv_results.keys())
# print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])))
# print("metrics:\n", metrics.classification_report(y, n_yhat, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn'))
# print("f1_score: ", metrics.f1_score(y, n_yhat, average='weighted'))
# print("jaccard_score: ", metrics.jaccard_score(y, n_yhat,pos_label=1))
# print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
# print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))
# plot_confusion_matrix(y, n_yhat, [0, 1])


##############
### RANDOM ###
##############
cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
rndm_osmpl(sel_alg, cv, X, y)

# # Loop to find the optimal ML model
# models = [svm, dt, knn, rndF, ada]
# runs = 50
# best_0 = 0
# best_1 = 0
# best_acc = 0
# best_roc = 0

# for model in models:
#     tot_0 = 0
#     tot_1 = 0
#     tot_acc = 0
#     tot_roc = 0
#     for i in range (0,runs):
#         recall_0, recall_1, acc, roc = rndm_osmpl(model, cv, X, y)
#         tot_0 += recall_0
#         tot_1 += recall_1
#         tot_acc += acc
#         tot_roc += roc
#     avg_0 = tot_0/runs
#     avg_1 = tot_1/runs
#     avg_acc = tot_acc/runs
#     avg_roc = tot_roc/runs
#     # Smaller CLass Accuracy based selection
#     if avg_1 > best_1:
#         best_1 = avg_1
#         print(model, ": ", best_1, " ## acc: ", avg_acc)
#     elif avg_1 == best_1:
#         if avg_0 > best_0:
#             best_1 = avg_1
#             print(model, ": ", best_1, " ## acc: ", avg_acc)
#     # # ROC_AUC based selection
#     # if avg_roc > best_roc:
#     #     best_roc = avg_roc
#     #     print(model, ": ", best_roc, " ## avg_acc: ", avg_acc, " ## avg_0: ", avg_0, " ## avg_1: ", avg_1)
#     if avg_acc > best_acc:
#         best_acc = avg_acc
#         print(model, ": ", avg_acc, " ## avg_0: ", avg_0, " ## avg_1: ", avg_1)


# X2 = X.to_numpy()
# y2 = y.to_numpy()
# X = X2
# y = y2
# oversample = RandomOverSampler(sampling_strategy='not majority')
# for train_index, test_index in cv.split(X, y):
#     # Split the data into train and test sets for the current fold
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     X_over, y_over = oversample.fit_resample(X, y)
#     model = sel_alg
#     model.fit(X_over, y_over)
#     n_yhat = sel_fit.predict(X_over)
#     print("confusion matrix:\n", metrics.confusion_matrix(y_over, n_yhat, labels=[0,1]))
#     score = model.score(X_over, y_over)
#     scores.append(score)
#     print("acc: ", score)
# # Calculate the mean evaluation score
# mean_score = np.mean(scores)
# print(f'Mean score with random replications: {mean_score:.3f}')


# ###########
# ### XAI ###
# ###########
# shap_values = shap.TreeExplainer(sel_alg).shap_values(X)
# shap.summary_plot(shap_values[0], X)
# for feature in X.columns:
#     print(feature)
#     shap.dependence_plot(feature, shap_values[0], X)

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
