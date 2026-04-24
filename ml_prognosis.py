from array import array
from enum import auto
from re import sub
from turtle import backward
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier
import xgboost
import shap
import joblib
from scipy.stats import randint, uniform

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

def tune_hyperparameters(model, param_grid, X, y, model_name, cv=10, n_iter=None):
    """Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
    
    Parameters
    ----------
    model : estimator
        The model to tune.
    param_grid : dict
        Parameter grid for tuning.
    X : array-like
        Feature data.
    y : array-like
        Target data.
    model_name : str
        Name of the model for printing.
    cv : int
        Number of cross-validation folds.
    n_iter : int or None
        Number of iterations for RandomizedSearchCV. If None, uses GridSearchCV.
    
    Returns
    -------
    best_model : estimator
        Best model found.
    best_params : dict
        Best parameters found.
    best_score : float
        Best score found.
    """
    print(f"\n{'='*60}")
    print(f"Tuning hyperparameters for {model_name}...")
    print(f"{'='*60}")
    
    if n_iter is not None:
        # Use RandomizedSearchCV for large parameter spaces
        search = RandomizedSearchCV(
            model, 
            param_grid, 
            n_iter=n_iter, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # Use GridSearchCV for smaller parameter spaces
        search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
    
    search.fit(X, y)
    
    print(f"\nBest parameters for {model_name}: {search.best_params_}")
    print(f"Best CV-{cv} accuracy: {search.best_score_ * 100:.2f}%")
    
    return search.best_estimator_, search.best_params_, search.best_score_

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

# function for printing each feature's SHAP value
def shapley_feature_ranking(shap_values, X):
    """Calculates the SHAP value of each feature.
    
        Parameters
        ----------
        shap_values : array-like, shape = [n_samples, n_features]
            vector, where n_samples in the number of samples and
            n_features is the number of features.
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        Returns
        -------
        pd.DataFrame [n_features, 2]
            Dataframe containing feature names and according SHAP value.
    """
    feature_order = np.argsort(np.mean(shap_values, axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(shap_values, axis=0)[i] for i in feature_order
            ][::-1],
        }
    )

# function to print SHAP values and plots
def xai(model, X, val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    ###
    sv = explainer(X)
    exp = shap.Explanation(sv[:,:,1], sv.base_values[:,1], X, feature_names=X.columns)
    idx_healthy = 2 # datapoint to explain (healthy)
    idx_cad = 9 # datapoint to explain (CAD)
    shap.waterfall_plot(exp[idx_healthy])
    shap.waterfall_plot(exp[idx_cad])
    ###

    shap.summary_plot(shap_values[val], X)
    # shap.summary_plot(shap_values[0], X, plot_type="bar")
    shap.summary_plot(shap_values[0], X, plot_type='violin')
    for feature in X.columns:
        print(feature)
        shap.dependence_plot(feature, shap_values[0], X)
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0,:], matplotlib=True)
    shap.force_plot(explainer.expected_value[1], shap_values[0][0], X.iloc[0,:], matplotlib=True)

    ###
    shap_rank = shapley_feature_ranking(shap_values[0], X)
    shap_rank.sort_values(by="importance", ascending=False)
    print(shap_rank)

def xai_svm(model, X, idx):
    explainer = shap.KernelExplainer(model.predict, X.values[idx])
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
    shap.summary_plot(shap_values, X, plot_type='violin')
    for feature in X.columns:
        print(feature)
        shap.dependence_plot(feature, shap_values, X)
    shap.force_plot(explainer.expected_value, shap_values[idx_healthy,:], X.iloc[idx_healthy,:], matplotlib=True)
    shap.force_plot(explainer.expected_value, shap_values[idx_cad,:], X.iloc[idx_cad,:], matplotlib=True)

    ###
    shap_rank = shapley_feature_ranking(shap_values, X)
    shap_rank.sort_values(by="importance", ascending=False)
    print(shap_rank)

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
    shap.summary_plot(shap_values, X, plot_type='violin')
    # for feature in X.columns:
    #     print(feature)
    #     shap.dependence_plot(feature, shap_values, X)
    shap.force_plot(explainer.expected_value, shap_values[idx_healthy,:], X.iloc[idx_healthy,:], matplotlib=True)
    shap.force_plot(explainer.expected_value, shap_values[idx_cad,:], X.iloc[idx_cad,:], matplotlib=True)

    ###
    shap_rank = shapley_feature_ranking(shap_values, X)
    shap_rank.sort_values(by="importance", ascending=False)
    print(shap_rank)

data = pd.read_csv('/mnt/d/Σημειώσεις/PhD - EMERALD/1. CAD/src/cad_dset.csv')
# print(data.columns)
# print(data.values)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x_prognosis = dataframe.drop(['ID','female','Arterial Hypertension','Dislipidemia','Angiopathy','ASYMPTOMATIC','ATYPICAL SYMPTOMS','ANGINA LIKE','DYSPNOEA ON EXERTION','INCIDENT OF PRECORDIAL PAIN','RST ECG','CNN_Healthy','CNN_CAD','Doctor: CAD', 'Doctor: Healthy','HEALTHY','CAD'], axis=1)

y = dataframe['CAD'].astype(int)
# print("y:\n",y)

x = x_prognosis #TODO ucommment when running prognosis
X = x
# print("x:\n",x.columns)

#TODO uncomment for hyperparameter tuning
############################################
### HYPERPARAMETER TUNING FOR ALL MODELS ###
############################################

# SVM hyperparameters
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

# Decision Tree hyperparameters
dt_params = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest hyperparameters
rf_params = {
    'n_estimators': [50, 80, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# AdaBoost hyperparameters
ada_params = {
    'n_estimators': [30, 50, 100, 150],
    'learning_rate': [0.5, 1.0, 1.5],
    'algorithm': ['SAMME', 'SAMME.R']
}

# KNN hyperparameters
knn_params = {
    'n_neighbors': [3, 5, 10, 13, 20, 25],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# TabPFN hyperparameters
tabpfn_params = {
    'N_ensemble_configurations': [10, 16, 26],
    'device': ['cpu']
}

# XGBoost hyperparameters
xgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0]
}

# LightGBM hyperparameters
lgb_params = {
    'n_estimators': [50, 80, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_leaves': [20, 31, 50]
}

# CatBoost hyperparameters
catb_params = {
    'n_estimators': [50, 79, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10]
}

# Perform hyperparameter tuning for each model
tuned_models = {}

# SVM tuning
print("\n" + "="*60)
print("STARTING HYPERPARAMETER TUNING FOR ALL MODELS")
print("="*60)
svm_tuned, svm_best_params, svm_best_score = tune_hyperparameters(
    svm.SVC(), svm_params, X, y, "SVM", cv=10
)
tuned_models['SVM'] = (svm_tuned, svm_best_score, svm_best_params)

# Decision Tree tuning
dt_tuned, dt_best_params, dt_best_score = tune_hyperparameters(
    DecisionTreeClassifier(random_state=0), dt_params, X, y, "Decision Tree", cv=10
)
tuned_models['DecisionTree'] = (dt_tuned, dt_best_score, dt_best_params)

# Random Forest tuning
rf_tuned, rf_best_params, rf_best_score = tune_hyperparameters(
    RandomForestClassifier(random_state=0), rf_params, X, y, "Random Forest", cv=10
)
tuned_models['RandomForest'] = (rf_tuned, rf_best_score, rf_best_params)

# AdaBoost tuning
ada_tuned, ada_best_params, ada_best_score = tune_hyperparameters(
    AdaBoostClassifier(random_state=0), ada_params, X, y, "AdaBoost", cv=10
)
tuned_models['AdaBoost'] = (ada_tuned, ada_best_score, ada_best_params)

# KNN tuning
knn_tuned, knn_best_params, knn_best_score = tune_hyperparameters(
    KNeighborsClassifier(), knn_params, X, y, "KNN", cv=10
)
tuned_models['KNN'] = (knn_tuned, knn_best_score, knn_best_params)

# TabPFN tuning (using GridSearchCV since parameter space is small)
tabpfn_tuned, tabpfn_best_params, tabpfn_best_score = tune_hyperparameters(
    TabPFNClassifier(device='cpu'), tabpfn_params, X, y, 
    "TabPFN", cv=10
)
tuned_models['TabPFN'] = (tabpfn_tuned, tabpfn_best_score, tabpfn_best_params)

# XGBoost tuning (using RandomizedSearchCV for efficiency)
xgb_tuned, xgb_best_params, xgb_best_score = tune_hyperparameters(
    xgboost.XGBRegressor(objective="binary:hinge", random_state=42), xgb_params, X, y, 
    "XGBoost", cv=10, n_iter=20
)
tuned_models['XGBoost'] = (xgb_tuned, xgb_best_score, xgb_best_params)

# CatBoost tuning
catb_tuned, catb_best_params, catb_best_score = tune_hyperparameters(
    CatBoostClassifier(verbose=False), catb_params, X, y, 
    "CatBoost", cv=10
)
tuned_models['CatBoost'] = (catb_tuned, catb_best_score, catb_best_params)

# Print summary of all models
print(f"\n{'='*60}")
print("HYPERPARAMETER TUNING SUMMARY - ALL MODELS")
print(f"{'='*60}")
results_summary = []
for model_name, (model, score, params) in tuned_models.items():
    print(f"{model_name:20} - CV-10 Accuracy: {score * 100:6.2f}%")
    results_summary.append((model_name, score, params))

# Sort by accuracy
results_summary.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*60}")
print("RANKING OF ALL MODELS")
print(f"{'='*60}")
for idx, (model_name, score, params) in enumerate(results_summary, 1):
    print(f"{idx}. {model_name:20} - CV-10 Accuracy: {score * 100:6.2f}%")

# Select the best model
best_model_name = results_summary[0][0]
sel_alg, best_score, best_params = tuned_models[best_model_name]
print(f"\n{'='*60}")
print(f"BEST MODEL SELECTED: {best_model_name}")
print(f"CV-10 Accuracy: {best_score * 100:.2f}%")
print(f"Best Parameters: {best_params}")
print(f"{'='*60}")

# Save all tuned models
print("\nSaving all tuned models...")
for model_name, (model, score, params) in tuned_models.items():
    model_file = f"tuned_{model_name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, model_file)
    print(f"  Saved: {model_file}")

# Save results summary
results_file = "hyperparameter_tuning_results.txt"
with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("HYPERPARAMETER TUNING RESULTS - ALL MODELS\n")
    f.write("="*60 + "\n\n")
    for idx, (model_name, score, params) in enumerate(results_summary, 1):
        f.write(f"{idx}. {model_name}\n")
        f.write(f"   CV-10 Accuracy: {score * 100:.2f}%\n")
        f.write(f"   Best Parameters: {params}\n\n")
print(f"  Saved: {results_file}")

####################
### FIXED MODELS ###
####################
#load tuned models from previous tuning
rf = joblib.load('tuned_randomforest_model.joblib')
cat = joblib.load('tuned_catboost_model.joblib')
ada = joblib.load('tuned_adaboost_model.joblib')
xg = joblib.load('tuned_xgboost_model.joblib')
tab = joblib.load('tuned_tabpfn_model.joblib')
knn = joblib.load('tuned_knn_model.joblib')

for sel_alg in [rf, cat, ada, xg, tab, knn]:
    print(f"\n\nmodel: {sel_alg.__class__.__name__}")
    #############################
    ### CV-10 with BEST MODEL ###
    #############################

    est = sel_alg
    # n_yhat = est.predict(X)
    n_yhat = cross_val_predict(est, X, y, cv=10)
    print("Testing Accuracy: {a:5.2f}%".format(a = 100*metrics.accuracy_score(y, n_yhat)))

    # cross-validate result(s) 10fold
    cv_results = cross_validate(sel_alg, X, y, cv=10)
    # sorted(cv_results.keys())
    avg_cv_accuracy = sum(cv_results['test_score'])/len(cv_results['test_score'])
    std_cv_accuracy = np.std(cv_results['test_score'])
    print("Avg CV-10 Testing Accuracy: {a:5.2f}%".format(a = 100*avg_cv_accuracy))
    print("Std CV-10 Testing Accuracy: {s:5.2f}%".format(s = 100*std_cv_accuracy))
    
    # Calculate sensitivity and specificity for each CV fold
    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
    sensitivities = []
    specificities = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        est_fold = sel_alg.__class__(**sel_alg.get_params())
        est_fold.fit(X_train, y_train)
        y_pred_fold = est_fold.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred_fold, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    avg_sensitivity = np.mean(sensitivities)
    std_sensitivity = np.std(sensitivities)
    avg_specificity = np.mean(specificities)
    std_specificity = np.std(specificities)
    print("Avg CV-10 Sensitivity: {a:5.2f}% (±{s:5.2f}%)".format(a=100*avg_sensitivity, s=100*std_sensitivity))
    print("Avg CV-10 Specificity: {a:5.2f}% (±{s:5.2f}%)".format(a=100*avg_specificity, s=100*std_specificity))
    
    print("metrics:\n", metrics.classification_report(y, n_yhat, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=True, zero_division='warn'))
    print("f1_score: ", metrics.f1_score(y, n_yhat, average='weighted'))
    print("jaccard_score: ", metrics.jaccard_score(y, n_yhat,pos_label=1))
    print("confusion matrix:\n", metrics.confusion_matrix(y, n_yhat, labels=[0,1]))
    print("TP/FP/TN/FN: ", perf_measure(y, n_yhat))

# print("###### XAI ######")
# # print(est.feature_names_in_) # feature names
# # print(est.feature_importances_) # feature importance
# for name, importance in zip(est.feature_names_in_,est.feature_importances_):
#    print(f"{name : <50}{importance:1.4f}")
#    # print(f"{importance:1.4f}")

# # ONLY for SVM & KNN
# from sklearn.inspection import permutation_importance
# perm_importance = permutation_importance(est, X, y)
# for name, importance in zip(est.feature_names_in_,perm_importance.importances_mean):
#    print(f"{name : <50}{importance:1.4f}")
#    # print(f"{importance:1.4f}")


# print("###### SHAP ######")
# # print('Number of features %d' % len(est.feature_names_in_))
# effect_sizes = cohen_effect_size(X, y)
# effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(40).index)[::-1].plot.barh(figsize=(6, 10))
# plt.title('Features with the largest effect sizes')
# plt.show()

# xai(rf, X, 0)
# xai_svm(est, X, pd.core.indexes.range.RangeIndex(start=0, stop=2, step=1))
# xai_svm(est, X, X.index)

# xai_cat(cat, X)

###### RESULTS ######
# model: RandomForestClassifier
# Testing Accuracy: 76.18%
# Avg CV-10 Testing Accuracy: 76.19%
# Std CV-10 Testing Accuracy:  4.61%
# Avg CV-10 Sensitivity: 67.43% (±10.00%)
# Avg CV-10 Specificity: 82.96% (± 5.95%)
# metrics:
#  {'0': {'precision': 0.7679083094555874, 'recall': 0.8297213622291022, 'f1-score': 0.7976190476190476, 'support': 323.0}, '1': {'precision': 0.7522522522522522, 'recall': 0.6733870967741935, 'f1-score': 0.7106382978723403, 'support': 248.0}, 'accuracy': 0.7618213660245184, 'macro avg': {'precision': 0.7600802808539198, 'recall': 0.7515542295016479, 'f1-score': 0.754128672745694, 'support': 571.0}, 'weighted avg': {'precision': 0.7611084807578167, 'recall': 0.7618213660245184, 'f1-score': 0.7598410687448209, 'support': 571.0}}
# f1_score:  0.7598410687448209
# jaccard_score:  0.5511551155115512
# confusion matrix:
#  [[268  55]
#  [ 81 167]]
# TP/FP/TN/FN:  (167, 55, 268, 81)

# model: CatBoostClassifier
# Testing Accuracy: 76.18%
# Avg CV-10 Testing Accuracy: 76.19%
# Std CV-10 Testing Accuracy:  4.55%
# Avg CV-10 Sensitivity: 67.42% (±10.04%)
# Avg CV-10 Specificity: 82.96% (± 5.06%)
# metrics:
#  {'0': {'precision': 0.7679083094555874, 'recall': 0.8297213622291022, 'f1-score': 0.7976190476190476, 'support': 323.0}, '1': {'precision': 0.7522522522522522, 'recall': 0.6733870967741935, 'f1-score': 0.7106382978723403, 'support': 248.0}, 'accuracy': 0.7618213660245184, 'macro avg': {'precision': 0.7600802808539198, 'recall': 0.7515542295016479, 'f1-score': 0.754128672745694, 'support': 571.0}, 'weighted avg': {'precision': 0.7611084807578167, 'recall': 0.7618213660245184, 'f1-score': 0.7598410687448209, 'support': 571.0}}
# f1_score:  0.7598410687448209
# jaccard_score:  0.5511551155115512
# confusion matrix:
#  [[268  55]
#  [ 81 167]]
# TP/FP/TN/FN:  (167, 55, 268, 81)


# model: AdaBoostClassifier
# Testing Accuracy: 76.01%
# Avg CV-10 Testing Accuracy: 76.01%
# Std CV-10 Testing Accuracy:  3.72%
# Avg CV-10 Sensitivity: 70.63% (± 8.67%)
# Avg CV-10 Specificity: 80.18% (± 5.62%)
# metrics:
#  {'0': {'precision': 0.7801204819277109, 'recall': 0.8018575851393189, 'f1-score': 0.7908396946564886, 'support': 323.0}, '1': {'precision': 0.7322175732217573, 'recall': 0.7056451612903226, 'f1-score': 0.7186858316221765, 'support': 248.0}, 'accuracy': 0.7600700525394045, 'macro avg': {'precision': 0.7561690275747341, 'recall': 0.7537513732148208, 'f1-score': 0.7547627631393325, 'support': 571.0}, 'weighted avg': {'precision': 0.7593150154494683, 'recall': 0.7600700525394045, 'f1-score': 0.7595014143893968, 'support': 571.0}}
# f1_score:  0.7595014143893968
# jaccard_score:  0.5608974358974359
# confusion matrix:
#  [[259  64]
#  [ 73 175]]
# TP/FP/TN/FN:  (175, 64, 259, 73)


# model: XGBRegressor
# Testing Accuracy: 75.48%
# Avg CV-10 Testing Accuracy: -2.31%
# Std CV-10 Testing Accuracy: 10.92%
# Avg CV-10 Sensitivity: 68.25% (±11.23%)
# Avg CV-10 Specificity: 80.82% (± 5.44%)
# metrics:
#  {'0': {'precision': 0.7699115044247787, 'recall': 0.8080495356037152, 'f1-score': 0.7885196374622356, 'support': 323.0}, '1': {'precision': 0.7327586206896551, 'recall': 0.6854838709677419, 'f1-score': 0.7083333333333333, 'support': 248.0}, 'accuracy': 0.7548161120840631, 'macro avg': {'precision': 0.7513350625572169, 'recall': 0.7467667032857286, 'f1-score': 0.7484264853977844, 'support': 571.0}, 'weighted avg': {'precision': 0.7537750505433239, 'recall': 0.7548161120840631, 'f1-score': 0.753692661238124, 'support': 571.0}}
# f1_score:  0.753692661238124
# jaccard_score:  0.5483870967741935
# confusion matrix:
#  [[261  62]
#  [ 78 170]]
# TP/FP/TN/FN:  (170, 62, 261, 78)


# model: TabPFNClassifier
# Testing Accuracy: 75.66%
# Avg CV-10 Testing Accuracy: 75.66%
# Std CV-10 Testing Accuracy:  3.99%
# Avg CV-10 Sensitivity: 68.62% (± 8.73%)
# Avg CV-10 Specificity: 81.10% (± 5.33%)
# metrics:
#  {'0': {'precision': 0.7705882352941177, 'recall': 0.8111455108359134, 'f1-score': 0.7903469079939669, 'support': 323.0}, '1': {'precision': 0.7359307359307359, 'recall': 0.6854838709677419, 'f1-score': 0.7098121085594989, 'support': 248.0}, 'accuracy': 0.7565674255691769, 'macro avg': {'precision': 0.7532594856124268, 'recall': 0.7483146909018277, 'f1-score': 0.7500795082767329, 'support': 571.0}, 'weighted avg': {'precision': 0.7555355910872549, 'recall': 0.7565674255691769, 'f1-score': 0.7553685712868774, 'support': 571.0}}
# f1_score:  0.7553685712868774
# jaccard_score:  0.5501618122977346
# confusion matrix:
#  [[262  61]
#  [ 78 170]]
# TP/FP/TN/FN:  (170, 61, 262, 78)


# model: KNeighborsClassifier
# Testing Accuracy: 72.50%
# Avg CV-10 Testing Accuracy: 72.51%
# Std CV-10 Testing Accuracy:  3.82%
# Avg CV-10 Sensitivity: 56.50% (± 9.73%)
# Avg CV-10 Specificity: 84.85% (± 6.67%)
# metrics:
#  {'0': {'precision': 0.7172774869109948, 'recall': 0.848297213622291, 'f1-score': 0.7773049645390071, 'support': 323.0}, '1': {'precision': 0.7407407407407407, 'recall': 0.5645161290322581, 'f1-score': 0.6407322654462242, 'support': 248.0}, 'accuracy': 0.7250437828371279, 'macro avg': {'precision': 0.7290091138258677, 'recall': 0.7064066713272745, 'f1-score': 0.7090186149926156, 'support': 571.0}, 'weighted avg': {'precision': 0.727468182094492, 'recall': 0.7250437828371279, 'f1-score': 0.7179879253533501, 'support': 571.0}}
# f1_score:  0.7179879253533501
# jaccard_score:  0.4713804713804714
# confusion matrix:
#  [[274  49]
#  [108 140]]
# TP/FP/TN/FN:  (140, 49, 274, 108)