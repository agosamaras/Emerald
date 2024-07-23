import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import shap
import joblib


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

# data_path = 'clinical_Data_ct.xlsx'
data_path = 'clinical_Data_pet.xlsx'
data = pd.read_excel(data_path, na_filter = False)
dataframe = pd.DataFrame(data.values, columns=data.columns)
x = dataframe.drop(['Output'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
y = dataframe['Output']

# ml algorithms initialization
rndF = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=17) #89.72 - 17
ada = AdaBoostClassifier(n_estimators=15, random_state=0) #88.64 - 15
catb = CatBoostClassifier(n_estimators=157, learning_rate=0.1, verbose=False) #90.82 - 157

sel_alg = rndF
X = x

print(X.columns)

est = sel_alg.fit(X, y)
n_yhat = cross_val_predict(sel_alg, X, y, cv=10)

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


# Save the trained model to a file
joblib.dump(est, f'{sel_alg}_model.joblib')

print('\a')
