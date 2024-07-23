import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Load and preprocess the dataset
data_path = 'cad_dset_multi.csv'
data = pd.read_csv(data_path)
dataframe = pd.DataFrame(data.values, columns=data.columns)
x = dataframe.drop(['ID','female','CNN_CAD','Doctor: CAD','output','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x_nodoc = dataframe.drop(['ID','female','CNN_CAD','Doctor: CAD', 'Expert Diagnosis Binary','output','CAD'], axis=1) # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
y = dataframe['output'].astype(int)

# Initialize classifiers
rndF = RandomForestClassifier(max_depth=None, random_state=0)
catb = CatBoostClassifier(learning_rate=0.1, verbose=False)

# Define the range of n_estimators to search over
n_estimators_range = range(10, 150, 10)

# Variables to store the best results
best_rndF_score = 0
best_rndF_n = 0
best_catb_score = 0
best_catb_n = 0

# Hyperparameter tuning for RandomForestClassifier
print("rndf")
for n in n_estimators_range:
    if n%10:
        print(n)
    rndF.set_params(n_estimators=n)
    scores = cross_val_score(rndF, x, y, scoring='accuracy', cv=10)
    mean_score = scores.mean()
    if mean_score > best_rndF_score:
        best_rndF_score = mean_score
        best_rndF_n = n

# Hyperparameter tuning for CatBoostClassifier
for n in n_estimators_range:
    if n%10:
        print(n)
    catb.set_params(n_estimators=n)
    scores = cross_val_score(catb, x, y, scoring='accuracy', cv=10)
    mean_score = scores.mean()
    if mean_score > best_catb_score:
        best_catb_score = mean_score
        best_catb_n = n

# Train the best RandomForest model on the full dataset and save it
rndF.set_params(n_estimators=best_rndF_n)
rndF.fit(x, y)
# joblib.dump(rndF, 'rndF_model.joblib')

# Train the best CatBoost model on the full dataset and save it
catb.set_params(n_estimators=best_catb_n)
catb.fit(x, y)
# joblib.dump(catb, 'catb_model.joblib')

# Print the best results
print(f"Best RandomForestClassifier n_estimators: {best_rndF_n}")
print(f"Best RandomForestClassifier cv-10 accuracy: {best_rndF_score * 100:.2f}%")
print(f"Best RandomForestClassifier cv-10 accuracy STD: {cross_val_score(rndF, x, y, scoring='accuracy', cv=10).std() * 100:.2f}%")

print(f"Best CatBoostClassifier n_estimators: {best_catb_n}")
print(f"Best CatBoostClassifier cv-10 accuracy: {best_catb_score * 100:.2f}%")
print(f"Best CatBoostClassifier cv-10 accuracy STD: {cross_val_score(catb, x, y, scoring='accuracy', cv=10).std() * 100:.2f}%")
