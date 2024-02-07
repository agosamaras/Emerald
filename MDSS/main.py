# import argparse
# import joblib
# import pandas as pd
# import numpy as np
# from PIL import Image
# from sklearn.preprocessing import StandardScaler
# from pathlib import Path

# cad_models = {
#     "knn": {"desc": "K Nearest Neighbors", "in": "Clinical data in form of csv - requires expert's yield",
#             "out": "Class & SHAP explanation"},
#     "CatBoost": {"desc": "Categorical Gradient Boosting", "in": "Clinical data in form of csv",
#                  "out": "Class & SHAP explanation"}
# }

# nsclc_models = ["adaboost"]

# def predict_cad(features):
#     # Load your CAD model (replace 'cad_model.pkl' with your actual file name)
#     model = joblib.load('cad_model.pkl')
#     # Perform any necessary preprocessing on features
#     # ...
#     # Make predictions
#     prediction = model.predict(features)
#     return prediction

# def predict_nsclc(features):
#     # Load your NSCLC model (replace 'nsclc_model.pkl' with your actual file name)
#     model = joblib.load('nsclc_model.pkl')
#     # Perform any necessary preprocessing on features
#     # ...
#     # Make predictions
#     prediction = model.predict(features)
#     return prediction

# def is_valid_filepath(directory_path):
#     path_obj = Path(directory_path)
#     return path_obj.is_dir()

# def preprocess_image(file_path):
#     image = Image.open(file_path)
#     image_array = np.array(image)
#     return image_array.reshape(-1)

# def preprocess_csv(file_path):
#     df = pd.read_csv(file_path)
#     features = df.to_numpy()
#     return features

# def choose_model(models, case):
#     cnt = 1
#     print(f"Choose the model for {case}:")
#     for model in models:
#         print(f"{cnt}. {model} - {models[model]['in']}")
#         cnt += 1

#     model_choice = int(input(f"Choose the model (1-{cnt-1}): "))
#     while not (1 <= model_choice <= cnt-1):
#         print(f"Invalid choice. Please pick a model choice from 1 to {cnt-1}.")
#         model_choice = int(input(f"Choose the model (1-{cnt-1}): "))

#     chosen_model = list(models.keys())[model_choice-1]
#     print(f"Selected model: {chosen_model} - {models[chosen_model]['desc']} - Returns {models[chosen_model]['out']}")
#     return chosen_model

# def main_menu():
#     print("\nMain Menu:")
#     print("1. Predict CAD")
#     print("2. Predict NSCLC")
#     print("3. Quit")

# def main():
#     while True:
#         main_menu()
#         choice = input("Enter your choice (1-3): ")

#         if choice == '1':
#             chosen_model = choose_model(cad_models, "CAD")
#             # Add code to get file path and make predictions here if needed
#             # file_path = input("Enter the filepath of the input data: ")
#             # features = preprocess_image(file_path)  # Modify based on your image preprocessing
#             # model_prediction = predict_cad(features)
#             # print(f"Prediction for CAD: {model_prediction}")
#         elif choice == '2':
#             file_path = input("Enter the filepath of the input data: ")
#             features = preprocess_csv(file_path)  # Modify based on your CSV data preprocessing
#             model_prediction = predict_nsclc(features)
#             print(f"Prediction for NSCLC: {model_prediction}")
#         elif choice == '3':
#             print("Quitting the program. Goodbye!")
#             break
#         else:
#             print("Invalid choice. Please enter a number between 1 and 3.")

# if __name__ == "__main__":
#     main()


import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import shap

from sklearn.ensemble import RandomForestClassifier

cad_models = {
    "RF": {"desc": "Random Forest", 
            "in": "Clinical data in form of csv (requires expert's yield)",
            "out": "Class & SHAP explanation"},
    "CatBoost": {"desc": "Categorical Gradient Boosting", 
                 "in": "Clinical data in form of csv",
                 "out": "Class & SHAP explanation"}
}

nsclc_models = ["adaboost"]

def predict_cad(features):
    # Load your CAD model (replace 'cad_model.pkl' with your actual file name)
    model = joblib.load('cad_model.pkl')
    # Perform any necessary preprocessing on features
    # ...
    # Make predictions
    prediction = model.predict(features)
    return prediction

def predict_nsclc(features):
    # Load your NSCLC model (replace 'nsclc_model.pkl' with your actual file name)
    model = joblib.load('nsclc_model.pkl')
    # Perform any necessary preprocessing on features
    # ...
    # Make predictions
    prediction = model.predict(features)
    return prediction

# function to print SHAP values and plots for tree ML models
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
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0,:], matplotlib=True)
    shap.force_plot(explainer.expected_value[1], shap_values[0][0], X.iloc[0,:], matplotlib=True)

def is_valid_filepath(directory_path):
    path_obj = Path(directory_path)
    return path_obj.is_dir()

def input_path():
    try_cnt = 0
    while True:
        data_path = input(f"Enter the path of the input data: ")
        if is_valid_filepath(data_path):
            return True, data_path
        else:
            if try_cnt > 1:
                return False, None
            print("Invalid path entered. Please enter a valid filepath.\n")
            try_cnt += 1

def preprocess_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array.reshape(-1)

def preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    features = df.to_numpy()
    return features

def choose_model(models, case):
    cnt = 1
    print(f"Choose the model for {case}:")
    for model in models:
        print(f"{cnt}. {model} - {models[model]['in']}")
        cnt += 1
    return cnt

def main_menu():
    print("\nMain Menu:")
    print("1. Predict CAD")
    print("2. Predict NSCLC")
    print("3. Quit")

def predict_cad_menu():
    cnt = choose_model(cad_models, "CAD")
    print(f"{cnt}. Back")
    return cnt

def predict_nsclc_menu():
    cnt = choose_model(nsclc_models, "NSCLC")
    print(f"{cnt}. Back")
    return cnt

def help_menu(models, cnt):   
    chosen_model = list(models.keys())[cnt-1]
    print(f"{models[chosen_model]['desc']}: Accepts {models[chosen_model]['in']} - Returns {models[chosen_model]['out']}")
    
    print("Procceed? (y/n)")

def main():

    while True:
        main_menu()
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            while True:
                cnt = predict_cad_menu()
                choice = int(input(f"Enter your choice (1-{cnt}): "))

                if choice == cnt:
                    break
                while True:
                    if choice in [1, cnt-1]:
                        help_menu(cad_models, choice)
                        while True:
                            choice = input()
                            if choice == 'y':
                                suc, data_path = input_path()
                                if not suc:
                                    return
                                print(f"model running: {data_path}")
                                # doc_gen_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease',
                                #                         'ANGINA LIKE', 'RST ECG', 'male', 'u40', 'Doctor: Healthy']
                                data = pd.read_csv(f"{data_path}/data.csv")
                                dataframe = pd.DataFrame(data.values, columns=data.columns)
                                # Load the saved Random Forest model
                                loaded_model = joblib.load('trained_models/RandomForestClassifier.joblib')
                                # Make predictions using the loaded model
                                predictions = loaded_model.predict(dataframe)
                                # Replace numeric predictions with labels
                                class_labels = {1: 'CAD', 0: 'HEALTHY'}
                                predicted_labels = [class_labels[prediction] for prediction in predictions]

                                # Display the predictions with labels
                                print("Predictions:", predicted_labels)
                                # # Display the predictions
                                # print("Predictions:", predictions)
                                xai(loaded_model, dataframe, 0)
                                return
                            elif choice == 'n':
                                break
                            else:
                               print("Please type 'y' or 'n'") 
                    else:
                        print(f"Invalid choice. Please pick a model choice from 1 to {cnt-1}.")
                    break

        elif choice == '2':
            while True:
                predict_nsclc_menu()
                choice = input("Enter your choice (1-2): ")

                if choice == '1':
                    file_path = input("Enter the filepath of the input data: ")
                    features = preprocess_csv(file_path)  # Modify based on your CSV data preprocessing
                    model_prediction = predict_nsclc(features)
                    print(f"Prediction for NSCLC: {model_prediction}")
                elif choice == '2':
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        elif choice == '3':
            print("Quitting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
