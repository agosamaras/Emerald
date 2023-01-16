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
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
import shap

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
    idx = 540 # datapoint to explain
    shap.waterfall_plot(exp[idx])
    ###

    shap.summary_plot(shap_values[val], X)
    # shap.summary_plot(shap_values[0], X, plot_type="bar")
    shap.summary_plot(shap_values[0], X, plot_type='violin')
    # for feature in X.columns:
    #     print(feature)
    #     shap.dependence_plot(feature, shap_values[0], X)
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0,:], matplotlib=True)

    ###
    shap_rank = shapley_feature_ranking(shap_values[0], X)
    shap_rank.sort_values(by="importance", ascending=False)
    print(shap_rank)

def xai_svm(model, X, idx):
    explainer = shap.KernelExplainer(model.predict, X.values[idx])
    shap_values = explainer.shap_values(X)
    ###
    index = 2 # datapoint to explain (healthy)
    sv = explainer.shap_values(X.loc[[index]])
    exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[index]].values, feature_names=X.columns)
    sv = explainer.shap_values(X.loc[[9]]) # CAD
    exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[9]].values, feature_names=X.columns)
    shap.waterfall_plot(exp[0])
    ###
    shap.summary_plot(shap_values, X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    shap.summary_plot(shap_values, X, plot_type='violin')
    for feature in X.columns:
        print(feature)
        shap.dependence_plot(feature, shap_values, X)
    shap.force_plot(explainer.expected_value, shap_values[index,:], X.iloc[index,:], matplotlib=True)
    shap.force_plot(explainer.expected_value, shap_values[9,:], X.iloc[9,:], matplotlib=True)

    ###
    shap_rank = shapley_feature_ranking(shap_values, X)
    shap_rank.sort_values(by="importance", ascending=False)
    print(shap_rank)

data = pd.read_csv('/d/Σημειώσεις/PhD - EMERALD/1. CAD/src/cad_dset.csv')
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
sel_features = X.columns
sel_alg = knn

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass    
    else:
        X = X.drop(feature, axis=1)

est = sel_alg.fit(X, y)
# n_yhat = est.predict(X)
n_yhat = cross_val_predict(est, X, y, cv=10)
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


print("###### SHAP ######")
print('Number of features %d' % len(est.feature_names_in_))
effect_sizes = cohen_effect_size(X, y)
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(40).index)[::-1].plot.barh(figsize=(6, 10))
plt.title('Features with the largest effect sizes')
plt.show()

# xai(est, X, 0)
# xai_svm(est, X, pd.core.indexes.range.RangeIndex(start=0, stop=2, step=1))
xai_svm(est, X, X.index)


# summary plot
# waterfall (1 Cad, 1 Healthy)
# Dependence plot?
# cohen plot
# Μόνο χωρίς γιατρό
