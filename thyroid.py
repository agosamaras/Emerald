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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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

from scipy.stats import ttest_rel

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

def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        Notes
        -----
        Based on https://github.com/AllenDowney/CompStats/blob/master/effect_size.ipynb
    """
    group1, group2 = X[y==0], X[y==1]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = group1.shape[0], group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d

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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[val], X)
    # for feature in X.columns:
    #     print(feature)
    #     shap.dependence_plot(feature, shap_values[0], X)
    ###
    # sv = explainer(X)
    # exp = shap.Explanation(sv[:,:,1], sv.base_values[:,1], X, feature_names=X.columns)
    # idx_mg = 2 # datapoint to explain (MG)
    # idx_s = 9 # datapoint to explain (S)
    # shap.waterfall_plot(exp[idx_mg])
    # shap.waterfall_plot(exp[idx_s])
    ###
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0,:], matplotlib=True)
    # shap.force_plot(explainer.expected_value[1], shap_values[0][0], X.iloc[0,:], matplotlib=True)
    return shap_values

def xai_svm(model, X, idx):
    explainer = shap.KernelExplainer(model.predict, X.values[idx])
    shap_values = explainer.shap_values(X)
    print(shap_values.shape)
    shap.summary_plot(shap_values, X)
    # shap.summary_plot(shap_values, X, plot_type='violin')
    # for feature in X.columns:
    #     print(feature)
    #     shap.dependence_plot(feature, shap_values, X)
    # idx_mg = 2 # datapoint to explain (MG)
    # idx_s = 9 # datapoint to explain (S)
    # sv = explainer.shap_values(X.loc[[idx_mg]])
    # exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_mg]].values, feature_names=X.columns)
    # shap.waterfall_plot(exp[0])
    # sv = explainer.shap_values(X.loc[[idx_s]])
    # exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_s]].values, feature_names=X.columns)
    # shap.waterfall_plot(exp[0])
    # shap.force_plot(explainer.expected_value, shap_values[idx_mg,:], X.iloc[idx_mg,:], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[idx_s,:], X.iloc[idx_s,:], matplotlib=True)
    return shap_values

# function to print SHAP values and plots for CatBoost
def xai_cat(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    ###
    idx_healthy = 2 # datapoint to explain (healthy)
    idx_cad = 9 # datapoint to explain (CAD)
    sv = explainer.shap_values(X.loc[[idx_healthy]])
    exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_healthy]].values, feature_names=X.columns)
    shap.waterfall_plot(exp[0])
    sv = explainer.shap_values(X.loc[[idx_cad]]) # CAD
    exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_cad]].values, feature_names=X.columns)
    shap.waterfall_plot(exp[0])
    ###
    shap.summary_plot(shap_values, X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # shap.summary_plot(shap_values, X, plot_type='violin')
    # # for feature in X.columns:
    # #     print(feature)
    # #     shap.dependence_plot(feature, shap_values, X)
    # shap.force_plot(explainer.expected_value, shap_values[idx_healthy,:], X.iloc[idx_healthy,:], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[idx_cad,:], X.iloc[idx_cad,:], matplotlib=True)

# function to print SHAP values and plots for lightGBM
def xai_light(model, X):
    # ref: https://www.kaggle.com/code/kaanboke/catboost-lightgbm-xgboost-explained-by-shap/notebook
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    ###
    idx_s = 0 # datapoint to explain (healthy)
    idx_mg = 1 # datapoint to explain (CAD)
    sv = explainer(X)
    exp = shap.Explanation(sv.values[:,:,0], sv.base_values[:,1], data=X.values, feature_names=X.columns)
    shap.waterfall_plot(exp[idx_s])
    shap.waterfall_plot(exp[idx_mg])
    ###    
    fig = plt.subplots(figsize=(6,6),dpi=200)
    ax = shap.summary_plot(shap_values[1], X,plot_type="dot")
    plt.show()
    fig = plt.subplots(figsize=(6,6),dpi=200)
    ax_2= shap.decision_plot(explainer.expected_value[1], shap_values[1][1], X.iloc[[1]],link= "logit")
    plt.show()

# def print_metrics(sel_alg, X, y):
#     print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)
#     print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
#     print("f1_score: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).mean() * 100)
#     print("f1_score STD: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).std() * 100)
#     print("jaccard_score: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).mean() * 100)
#     print("jaccard_score STD: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).std() * 100)
#     scoring = {
#         'sensitivity': metrics.make_scorer(metrics.recall_score),
#         'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
#     }
#     print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
#     print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
#     print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
#     print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)


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

        # Calculate accuracy for the current fold
        accuracy = metrics.accuracy_score(y_test, n_yhat)
        print(f'Accuracy for current fold: {accuracy}')

        # xai(est, X_train, 0)
        # xai_svm(est, X, train_index)
        # xai_cat(est, X)
        # xai_light(est, X)

        # effect_sizes = cohen_effect_size(X.loc[train_index], y.loc[train_index])
        # effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(40).index)[::-1].plot.barh(figsize=(6, 10))
        # plt.title('Features with the largest effect sizes')
        # plt.show()

        tpa, fpa, tna, fna = perf_measure(y_test, n_yhat)
        tp = tp + tpa
        fp = fp + fpa
        tn = tn + tna
        fn = fn + fna
        y_pred.extend(n_yhat)
        y_actual.extend(y_test)

    # print("TP/FP/TN/FN: ", tp, fp, tn, fn)
    # print("confusion matrix:\n", metrics.confusion_matrix(y_actual, y_pred, labels=[0,1]))
    # plot_confusion_matrix(y_actual, y_pred, [0, 1])
    report = metrics.classification_report(y_actual, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn')
    # print("metrics:\n", report)
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

data_path = '/d/Σημειώσεις/PhD - EMERALD/3. Extras/Parathyroid/input_data.extra.csv'
# data_path = '/mnt/c/Users/samar/Documents/PhD - EMERALD/Extras/Parathyroid/input_data.csv'
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
svmn = svm.SVC(kernel='rbf') # Avg CV-10 Testing Accuracy: 81.52% / '0recall': 0.9578947368421052 / '1recall': 0.7083333333333334
lr = linear_model.LinearRegression()
dt = DecisionTreeClassifier() # Avg CV-10 Testing Accuracy: 84.92%% / '0recall':  0.9789473684210527 / '1recall': 0.75
# metrics: {'0': {'precision': 0.9230769230769231, 'recall': 0.8842105263157894, 'f1-score': 0.9032258064516129, 'support': 95}, '1': {'precision': 0.6071428571428571, 'recall': 0.7083333333333334, 'f1-score': 0.6538461538461539, 'support': 24}, 'accuracy': 0.8487394957983193, 'macro avg': {'precision': 0.7651098901098901, 'recall': 0.7962719298245614, 'f1-score': 0.7785359801488834, 'support': 119}, 'weighted avg': {'precision': 0.8593591282666913, 'recall': 0.8487394957983193, 'f1-score': 0.8529307504639573, 'support': 119}}
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=60) # Avg CV-10 Testing Accuracy: 85.76% / '0recall': 0.968421052631579 / '1recall': 0.8800438596491228
# metrics: {'0': {'precision': 0.9213483146067416, 'recall': 0.8631578947368421, 'f1-score': 0.891304347826087, 'support': 95}, '1': {'precision': 0.5666666666666667, 'recall': 0.7083333333333334, 'f1-score': 0.6296296296296297, 'support': 24}, 'accuracy': 0.8319327731092437, 'macro avg': {'precision': 0.7440074906367041, 'recall': 0.7857456140350878, 'f1-score': 0.7604669887278583, 'support': 119}, 'weighted avg': {'precision': 0.8498158814087432, 'recall': 0.8319327731092437, 'f1-score': 0.8385296147444485, 'support': 119}}
ada = AdaBoostClassifier(n_estimators=150, random_state=0) # Avg CV-10 Testing Accuracy: 79.92% / '0recall': 0.9157894736842105 / '1recall': 0.5416666666666666
# metrics: {'0': {'precision': 0.9310344827586207, 'recall': 0.8526315789473684, 'f1-score': 0.8901098901098902, 'support': 95}, '1': {'precision': 0.5625, 'recall': 0.75, 'f1-score': 0.6428571428571429, 'support': 24}, 'accuracy': 0.8319327731092437, 'macro avg': {'precision': 0.7467672413793103, 'recall': 0.8013157894736842, 'f1-score': 0.7664835164835165, 'support': 119}, 'weighted avg': {'precision': 0.8567082005215879, 'recall': 0.8319327731092437, 'f1-score': 0.8402437898236218, 'support': 119}}
knn = KNeighborsClassifier(n_neighbors=3) # Avg CV-10 Testing Accuracy: 79.92% / '0recall': 0.9578947368421052 / '1recall':  0.5833333333333334
# metrics: {'0': {'precision': 0.8787878787878788, 'recall': 0.9157894736842105, 'f1-score': 0.8969072164948454, 'support': 95}, '1': {'precision': 0.6, 'recall': 0.5, 'f1-score': 0.5454545454545454, 'support': 24}, 'accuracy': 0.8319327731092437, 'macro avg': {'precision': 0.7393939393939394, 'recall': 0.7078947368421052, 'f1-score': 0.7211808809746953, 'support': 119}, 'weighted avg': {'precision': 0.8225617519735166, 'recall': 0.8319327731092437, 'f1-score': 0.8260260055287345, 'support': 119}}
catb = CatBoostClassifier(n_estimators=6, learning_rate=0.1, verbose=False)
light = LGBMClassifier(objective='binary', random_state=5, n_estimators=29, n_jobs=-1) # 93/5

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
#     cv_results = cross_validate(svmn, X, y, cv=10)
#     # sorted(cv_results.keys())
#     print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*sum(cv_results['test_score'])/len(cv_results['test_score'])))


############################
X = x
sel_alg = light

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
# print(f"n_yhat: {n_yhat}")
print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))

# metrics w/o oversampling
# report = metrics.classification_report(y, n_yhat, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn')
# print("metrics:\n", report)

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
cv10 = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

# from sklearn.model_selection import cross_val_score
# # Perform cross-validation
# scores = cross_val_score(sel_alg, X, y, cv=cv10)  # cv=5 for 5-fold cross-validation
# # Print accuracy for each fold
# for i, score in enumerate(scores, 1):
#     print(f"Accuracy for fold {i}: {score}")

# infinite run:
# try:
#     while True:
#         rndm_osmpl(sel_alg, cv, X, y)
# except KeyboardInterrupt:
#     print("Oh! you pressed CTRL + C.")
#     print("Program interrupted.")
# single run:
rndm_osmpl(sel_alg, cv10, X, y)

# # Loop to find the optimal ML model
# models = [svmn, dt, knn, rndF, ada]
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
