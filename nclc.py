import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import xgboost
from sklearn import tree
import shap
from sklearn.tree import DecisionTreeClassifier
import sys
import seaborn as sns

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

# function to print SHAP values and plots for tree based algorithms
def xai(model, X, idx):
    explainer = shap.KernelExplainer(model.predict, X.values[idx])
    shap_values = explainer.shap_values(X)
    print(shap_values.shape)
    # shap.summary_plot(shap_values, X)
    # shap.summary_plot(shap_values, X, plot_type='violin')
    # for feature in X.columns:
    #     print(feature)
    #     shap.dependence_plot(feature, shap_values, X)
    # idx_ben = 4 # datapoint to explain (benign)
    # idx_mal = 1 # datapoint to explain (malignant)
    # sv = explainer.shap_values(X.loc[[idx_ben]])
    # exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_ben]].values, feature_names=X.columns)
    # shap.waterfall_plot(exp[0])
    # sv = explainer.shap_values(X.loc[[idx_mal]])
    # exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[idx_mal]].values, feature_names=X.columns)
    # shap.waterfall_plot(exp[0])
    # shap.force_plot(explainer.expected_value, shap_values[idx_ben,:], X.iloc[idx_ben,:], matplotlib=True)
    # shap.force_plot(explainer.expected_value, shap_values[idx_mal,:], X.iloc[idx_mal,:], matplotlib=True)
    ###
    # shap.decision_plot(0, shap_values, X.loc[idx])
    # shap.decision_plot(0, shap_values[idx_ben], X.loc[idx[idx_ben]], highlight=0)
    # shap.decision_plot(0, shap_values[idx_mal], X.loc[idx[idx_mal]], highlight=0)
    return shap_values

data_path = '/d/Σημειώσεις/PhD - EMERALD/2. NSCLC/Input Data/stats.csv'
data = pd.read_csv(data_path, na_filter = False)
# print(data.columns)
# print(data.values)
dataframe = pd.DataFrame(data.values, columns=data.columns)
# dataframe['BENIGN'] = data.BENIGN
x = dataframe.drop(['BENIGN'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
# print("x:\n",x.columns)
# y = dataframe['BENIGN'].astype(int)
y = dataframe['BENIGN']
# print("y:\n",y)

# ml algorithms initialization
svmc = svm.SVC(kernel='rbf') # 77.87%, STD 11.373
dt = DecisionTreeClassifier() # 91.47%, STD 7.523
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=36) # 93.1%, STD 6.231 [*]
ada = AdaBoostClassifier(n_estimators=30, random_state=0) # 94.33%, STD 6.3 [*]
knn = KNeighborsClassifier(n_neighbors=7) # 73.27%, STD 7.177
tab = TabPFNClassifier(device='cpu', N_ensemble_configurations=8) # 91.87%, STD 7.414 [*]
xgb = xgboost.XGBRegressor(objective="binary:hinge", random_state=42) # 88.93%, STD 5.446
light = LGBMClassifier(objective='binary', random_state=5, n_estimators=25, n_jobs=-1) # 92,27 6,365 [*]
catb = CatBoostClassifier(n_estimators=79, learning_rate=0.1, verbose=False) # 92.25%, STD 6.42

sel_alg = light
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

# sel_features = x_nodoc

# ##############
# ### CV-10 ####
# ##############
# for feature in x.columns:
#     if feature in sel_features:
#         pass
#     else:
#         X = X.drop(feature, axis=1)

# ######################################
# ### Hyperparameter Selection Loop ####
# ######################################
# best_acc = 0 # 91.36 # SUV classification
# best_std = 100
# for i in range(1,100):
#    sel = TabPFNClassifier(device='cpu', N_ensemble_configurations=i)
#    print(i)
#    model = cross_val_score(sel, X, y, scoring='accuracy', cv = 10)
#    acc = model.mean() * 100
#    std = model.std() * 100
#    if (acc > best_acc) or (acc == best_acc and std < best_std):
#       best_acc = acc
#       best_std = std
#       print("cv-10 accuracy: ", acc)
#       print("cv-10 accuracy STD: ", std)

# sys.exit()

# est = sel_alg.fit(X, y)
# n_yhat = cross_val_predict(sel_alg, X, y, cv=10)

# print("cv-10 accuracy: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).mean() * 100)
# print("cv-10 accuracy STD: ", cross_val_score(sel_alg, X, y, scoring='accuracy', cv = 10).std() * 100)
# print("f1_score: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).mean() * 100)
# print("f1_score STD: ", cross_val_score(sel_alg, X, y, scoring='f1', cv = 10).std() * 100)
# print("jaccard_score: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).mean() * 100)
# print("jaccard_score STD: ", cross_val_score(sel_alg, X, y, scoring='jaccard', cv = 10).std() * 100)
# scoring = {
#     'sensitivity': metrics.make_scorer(metrics.recall_score),
#     'specificity': metrics.make_scorer(metrics.recall_score,pos_label=0)
# }
# print("sensitivity: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).mean() * 100)
# print("sensitivity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['sensitivity'], cv = 10).std() * 100)
# print("specificity: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).mean() * 100)
# print("specificity STD: ", cross_val_score(sel_alg, X, y, scoring=scoring['specificity'], cv = 10).std() * 100)

# print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
# print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))

# # Multiple ROCs
# # fit AdaBoost model and plot ROC curve
# n_yhat = cross_val_predict(ada, X, y, cv=10)
# fpr, tpr, _ = metrics.roc_curve(y, n_yhat)
# auc = round(metrics.roc_auc_score(y, n_yhat), 4)
# plt.plot(fpr,tpr,label="AdaBoost, AUC="+str(auc))

# # fit Random Forest model and plot ROC curve
# n_yhat = cross_val_predict(rndF, X, y, cv=10)
# fpr, tpr, _ = metrics.roc_curve(y, n_yhat)
# auc = round(metrics.roc_auc_score(y, n_yhat), 4)
# plt.plot(fpr,tpr,label="Random Forest, AUC="+str(auc))

# # fit LightGBM model and plot ROC curve
# n_yhat = cross_val_predict(light, X, y, cv=10)
# fpr, tpr, _ = metrics.roc_curve(y, n_yhat)
# auc = round(metrics.roc_auc_score(y, n_yhat), 4)
# plt.plot(fpr,tpr,label="LightGBM, AUC="+str(auc))

# # fit TabPFN model and plot ROC curve
# n_yhat = cross_val_predict(tab, X, y, cv=10)
# fpr, tpr, _ = metrics.roc_curve(y, n_yhat)
# auc = round(metrics.roc_auc_score(y, n_yhat), 4)
# plt.plot(fpr,tpr,label="TabPFN, AUC="+str(auc))

# # add plot config
# plt.title('Comparison of Models (ROC curves)')
# plt.legend()
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# Multiple CMs
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
sns.set(font_scale=2.0)
# fit AdaBoost model and plot CM
n_yhat = cross_val_predict(ada, X, y, cv=10)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y, n_yhat, display_labels=['Benign', 'Malignant'])
disp.plot(ax=axes[0][0], xticks_rotation=45, cmap='Blues')
disp.ax_.set_title('AdaBoost')
disp.im_.colorbar.remove()

# fit Random Forest model and plot CM
n_yhat = cross_val_predict(rndF, X, y, cv=10)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y, n_yhat, display_labels=['Benign', 'Malignant'])
disp.plot(ax=axes[0][1], xticks_rotation=45, cmap='Blues')
disp.ax_.set_title('Random Forest')
disp.im_.colorbar.remove()

# fit LightGBM model and plot CM
n_yhat = cross_val_predict(light, X, y, cv=10)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y, n_yhat, display_labels=['Benign', 'Malignant'])
disp.plot(ax=axes[1][0], xticks_rotation=45, cmap='Blues')
disp.ax_.set_title('LightGBM')
disp.im_.colorbar.remove()

# fit TabPFN model and plot CM
n_yhat = cross_val_predict(tab, X, y, cv=10)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y, n_yhat, display_labels=['Benign', 'Malignant'])
disp.plot(ax=axes[1][1], xticks_rotation=45, cmap='Blues')
disp.ax_.set_title('TabPFN')
disp.im_.colorbar.remove()

fig.subplots_adjust(wspace=0.10, hspace=0.45)
fig.colorbar(disp.im_, ax=axes)
fig.suptitle('Comparison of Models (Confusion Matrices)')
plt.show()

# xai(est, X, X.index)

#######
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


print('\a')
