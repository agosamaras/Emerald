import streamlit as st
import pandas as pd
import numpy as np
import keras
from streamlit_option_menu import option_menu
import joblib
import shap
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from ultralytics import YOLO
import time
import os
import tempfile
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from openai import OpenAI
import mysql.connector
from mysql.connector import Error
import decimal

# App vars and functions
cad_features_dict = {
    "known CAD" : "Does the patient have previous CAD?",
    "previous AMI": "Does the patient have previous AMI?",
    "previous PCI": "Does the patient have previous PCI?",
    "previous CABG": "Does the patient have previous CABG?",
    "previous STROKE": "Does the patient have previous stroke?",
    "Diabetes": "Does the patient have diabetes?",
    "Smoking": "Does the patient smoke?",
    "Arterial Hypertension": "Does the patient have arterial hypertension?",
    "Dislipidemia": "Does the patient have dislipidemia?",
    "Angiopathy": "Does the patient have angiopathy?",
    "Chronic Kindey Disease": "Does the patient have chronic kidney disease?",
    "Family History of CAD": "Does the patient have a family history of CAD?",
    "ASYMPTOMATIC": "Is the patient asymptomatic?",
    "ATYPICAL SYMPTOMS": "Does the patient showcase atypical symptoms?",
    "ANGINA LIKE": "Are the symptoms angina-like?",
    "DYSPNOEA ON EXERTION": "Does the patient showcase dyspnoea on exertion?",
    "INCIDENT OF PRECORDIAL PAIN": "Does the patient showcase incidents of precordial pain?",
    "RST ECG": "Does the patient have RST ECG?",
    "male": "Is the patient male?",
    "CNN_Healthy": "Does the CNN report the patient as healthy?",
    "BMI": "What is the Body Mass Index of the patient?",
    "Overweight": "",
    "Obese": "",
    "normal_weight":"",
    # "Age: under 40":"",
    # "Age: 40-50":"",
    # "Age: 50-60":"",
    # "Age: over 60":"",
    # "Expert diagnosis:
    "u40":"",
    "40b50":"",
    "50b60":"",
    "o60":"",
    "Doctor: Healthy": "Does the doctor yield the patient as healthy? (This is a feature of outmost importance, as the expert's yield has been found to be extrmely important)",
}

cad_default_values = {
    "known CAD" : 0,
    "previous AMI": 0,
    "previous PCI": 0,
    "previous CABG": 0,
    "previous STROKE": 0,
    "Diabetes": 0,
    "Smoking": 0,
    "Arterial Hypertension": 0,
    "Dislipidemia": 0,
    "Angiopathy": 0,
    "Chronic Kindey Disease": 0,
    "Family History of CAD": 0,
    "ASYMPTOMATIC": 1,
    "ATYPICAL SYMPTOMS": 0,
    "ANGINA LIKE": 0,
    "DYSPNOEA ON EXERTION": 0,
    "INCIDENT OF PRECORDIAL PAIN": 0,
    "RST ECG": 0,
    "CNN_Healthy": 0,
    "BMI": 26.23,
    "Doctor: Healthy": None,
}

cad_mandatory_features = ["age", "weight"]

nsclc_features_dict = {
    "Age": "What is the age of the patient?",
    "BMI": "What is the Body Mass Index of the patient?",
    "GLU": "What is the GLU measurement of the patient?",
    "SUV": "What is the SUVmax measurement of the patient?",
    "Diameter": "What is the diameter of the SPN (in cm)?",
    "Location": "What is the location of the SPN?",
    "Type": "What is the type of the SPN?",
    "Limits": "How would you define the limits of the SPN?",
    "Gender": "Is the patient male?"
}

nsclc_mandatory_features = ["SUV", "Diameter"]

weight_cats = ["Overweight","Obese","normal_weight"]
age_cats = ["u40","40b50","50b60","o60"]

concept_names_list_cad_fcm = ["Gender", "Age", "BMI", "known CAD", "previous AMI", "previous PCI", "previous CABG",
                              "previous Stroke", "Diabetes", "Smoking", "Hypertension", "Dyslipidemia", "Angiopathy",
                              "Chronic Kidney Disease", "family history of CAD", "Asymptomatic", "Atypical Symptoms",
                              "Angina-like", "Dysponea οn exertion", "Incident of Precordial Pain", "ECG", "Expert Diagnosis yield"]

concept_names_list_nsclc_fcm = ["Age", "Gender", "BMI", "SUVmax", "Diameter", "Location of SPN inLeft Upper",
                                "Location of SPN in Right Upper Lobe", "Location of SPN in Left Lower Lobe",
                                "Location of SPN in Right Lower Lobe", "Location of SPN in Middle", "Location of SPN in Lingula",
                                "Type Solid", "Type Ground Class", "Type Consolidated", "Type Speckled", "Type Semi-Solid",
                                "Type calcified", "Type Cavitary", "Margins Spiculated", "Margins Lobulated", "Margins Well Defined",
                                "Margins Ill-Defined"]

client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)

st.set_page_config(layout = 'wide')

cad_features_db_mapping = {
    'male' : 'sex', 
    'age' : 'age', 
    'BMI' : 'bmi', 
    'known CAD' : 'known_cad', 
    'previous AMI' : 'previous_ami', 
    'previous PCI' : 'previous_pci', 
    'previous CABG' : 'previous_cabg', 
    'previous STROKE' : 'previous_stroke', 
    'Diabetes' : 'diabetes', 
    'Smoking' : 'smoking', 
    'Arterial Hypertension' : 'arterial_hypertension', 
    'Dislipidemia' : 'dyslipidemia', 
    'Angiopathy' : 'angiopathy', 
    'Chronic Kindey Disease' : 'chronic_kidney_disease', 
    'Family History of CAD' : 'fhcad', 
    'ASYMPTOMATIC' : 'asymptomatic', 
    'ATYPICAL SYMPTOMS' : 'atypical_symptoms', 
    'ANGINA LIKE' : 'angina_like', 
    'DYSPNOEA ON EXERTION' : 'dyspnoea_on_exertion', 
    'INCIDENT OF PRECORDIAL PAIN' : 'incident_of_precordial_pain', 
    'RST ECG' : 'rst_ecg', 
    #'' : 'expert_diagnosis_ischemia', 
    'Doctor: Healthy' : 'expert_diagnosis_binary', 
    #'' : 'cad_70', 
    #'' : 'defect_vessels'
}

nsclc_features_db_mapping = {
    'Gender': 'sex', 
    'Fdg': 'fdg_mci', 
    'Age': 'age', 
    # '': 'bw', 
    'BMI': 'bmi', 
    'GLU': 'glu', 
    'SUV': 'suv_max', 
    'Diameter': 'diameter_cm', 
    'Location': 'location', 
    'Type': 'type', 
    'Limits': 'limits', 
    # '': 'benign_malignant_susp', 
    #'': 'centroid_x', 
    #'': 'centroid_y', 
    #'': 'slice_number'
}

# MySQL functions
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="emerald",
            database="emerald_db"
        )
        if connection.is_connected():
            st.success("Successfully connected to the database")
        return connection
    except Error as e:
        st.error(f"The error '{e}' occurred")
        return None

def save_to_db(state_data, table="cad"):
    features_map = cad_features_db_mapping
    if table == "nsclc":
        features_map = nsclc_features_db_mapping
    # change feature names
    db_data = {features_map[key]:value for key,value in state_data.items() if key in features_map}
    print(f"UL db_data: {db_data}")

    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return None

    cursor = connection.cursor()
    placeholders = ', '.join(['%s'] * len(db_data))
    columns = ', '.join(db_data.keys())
    sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, list(db_data.values()))
    connection.commit()
    return cursor.lastrowid

# Example of how to use the function
# Assuming nlp_text, xai_img, and result are generated by the model
# nlp_text = "Sample NLP explanation text"
# xai_img = Image.new('RGB', (100, 100))  # Example PIL image
# result = "Sample model prediction result"
# save_results_to_db(nlp_text=nlp_text, xai_img=xai_img, result=result, table="nsclc")
def save_results_to_db(*, nlp_text=None, xai_img=None, result=None, table="cad"):
    print(f"nlp_text: {nlp_text}")
    print(f"xai_img: {xai_img}")
    print(f"result: {result}")
    print(f"table: {table}")
    # Check if entry_id is available
    if 'entry_id' not in st.session_state or st.session_state['entry_id'] is None:
        st.error("Entry ID is not available. Save features to the database first.")
        return

    entry_id = st.session_state['entry_id']
    prefix = r"F:\emerald_xai_img\cad" if table == "cad" else r"F:\emerald_xai_img\nsclc"
    xai_img_path = None

    # Save the xai_img to the specified path
    if xai_img is not None:
        xai_img_path = f"{prefix}\\{entry_id}.jpg"
        if isinstance(xai_img, np.ndarray):
            xai_img = Image.fromarray(xai_img)
        xai_img.save(xai_img_path)

    # Ensure result is converted to integer if it's a valid number
    try:
        result = int(result)
    except:
        st.error(f"Invalid result value: {result}({type(result)})")
        return

    # Connect to the database
    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return

    try:
        cursor = connection.cursor()
        sql = f"UPDATE {table} SET nlp_text = %s, xai_img_path = %s, dss_pred = %s WHERE id = %s"
        cursor.execute(sql, (nlp_text, xai_img_path, result, entry_id))
        connection.commit()
        st.success(f"Results saved to DB for entry ID: {entry_id}")
    except Error as e:
        st.error(f"The error '{e}' occurred while saving results to the database")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def load_from_db(entry_id, table="cad"): #TODO take into account the 3 new fields (xai_img, dss_pred, nlp_txt)
    features_map = cad_features_db_mapping
    if table == "nsclc":
        features_map = nsclc_features_db_mapping
    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return None

    cursor = connection.cursor(dictionary=True)
    sql = f"SELECT * FROM {table} WHERE id = %s"
    cursor.execute(sql, (entry_id,))
    record = cursor.fetchone()

    if record:
        # change feature names
        key_list = list(features_map.keys())
        val_list = list(features_map.values())
        db_data = {key_list[val_list.index(key)]:value for key,value in record.items() if key in features_map.values()}
        st.session_state["entry_id"] = entry_id
        print(f"DL db_data: {db_data}")
        return db_data
    return None

def load_results_to_db(entry_id, table="cad"):
    features_map = cad_features_db_mapping
    if table == "nsclc":
        features_map = nsclc_features_db_mapping
    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return None
    


    cursor = connection.cursor(dictionary=True)
    sql = f"SELECT nlp_text, xai_img_path, dss_pred FROM {table} WHERE id = %s"
    cursor.execute(sql, (entry_id,))
    record = cursor.fetchone()

    if record:
        # change feature names
        # key_list = list(features_map.keys())
        # val_list = list(features_map.values())
        # db_data = {key_list[val_list.index(key)]:value for key,value in record.items() if key in features_map.values()}
        # st.session_state["entry_id"] = entry_id
        print(f"DL db_data: {record}")
        return record
    return None

cad_features_db_mapping = {
    'male' : 'sex', 
    'age' : 'age', 
    'BMI' : 'bmi', 
    'known CAD' : 'known_cad', 
    'previous AMI' : 'previous_ami', 
    'previous PCI' : 'previous_pci', 
    'previous CABG' : 'previous_cabg', 
    'previous STROKE' : 'previous_stroke', 
    'Diabetes' : 'diabetes', 
    'Smoking' : 'smoking', 
    'Arterial Hypertension' : 'arterial_hypertension', 
    'Dislipidemia' : 'dyslipidemia', 
    'Angiopathy' : 'angiopathy', 
    'Chronic Kindey Disease' : 'chronic_kidney_disease', 
    'Family History of CAD' : 'fhcad', 
    'ASYMPTOMATIC' : 'asymptomatic', 
    'ATYPICAL SYMPTOMS' : 'atypical_symptoms', 
    'ANGINA LIKE' : 'angina_like', 
    'DYSPNOEA ON EXERTION' : 'dyspnoea_on_exertion', 
    'INCIDENT OF PRECORDIAL PAIN' : 'incident_of_precordial_pain', 
    'RST ECG' : 'rst_ecg', 
    #'' : 'expert_diagnosis_ischemia', 
    'Doctor: Healthy' : 'expert_diagnosis_binary', 
    #'' : 'cad_70', 
    #'' : 'defect_vessels'
}

nsclc_features_db_mapping = {
    'Gender': 'sex', 
    'Fdg': 'fdg_mci', 
    'Age': 'age', 
    # '': 'bw', 
    'BMI': 'bmi', 
    'GLU': 'glu', 
    'SUV': 'suv_max', 
    'Diameter': 'diameter_cm', 
    'Location': 'location', 
    'Type': 'type', 
    'Limits': 'limits', 
    # '': 'benign_malignant_susp', 
    #'': 'centroid_x', 
    #'': 'centroid_y', 
    #'': 'slice_number'
}

# MySQL functions
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="emerald",
            database="emerald_db"
        )
        if connection.is_connected():
            st.success("Successfully connected to the database")
        return connection
    except Error as e:
        st.error(f"The error '{e}' occurred")
        return None

def save_to_db(state_data, table="cad"):
    features_map = cad_features_db_mapping
    if table == "nsclc":
        features_map = nsclc_features_db_mapping
    # change feature names
    db_data = {features_map[key]:value for key,value in state_data.items() if key in features_map}
    print(f"UL db_data: {db_data}")

    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return None

    cursor = connection.cursor()
    placeholders = ', '.join(['%s'] * len(db_data))
    columns = ', '.join(db_data.keys())
    sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, list(db_data.values()))
    connection.commit()
    return cursor.lastrowid

# Example of how to use the function
# Assuming nlp_text, xai_img, and result are generated by the model
# nlp_text = "Sample NLP explanation text"
# xai_img = Image.new('RGB', (100, 100))  # Example PIL image
# result = "Sample model prediction result"
# save_results_to_db(nlp_text=nlp_text, xai_img=xai_img, result=result, table="nsclc")
def save_results_to_db(*, nlp_text=None, xai_img=None, result=None, table="cad"):
    print(f"nlp_text: {nlp_text}")
    print(f"xai_img: {xai_img}")
    print(f"result: {result}")
    print(f"table: {table}")
    # Check if entry_id is available
    if 'entry_id' not in st.session_state or st.session_state['entry_id'] is None:
        st.error("Entry ID is not available. Save features to the database first.")
        return

    entry_id = st.session_state['entry_id']
    prefix = r"F:\emerald_xai_img\cad" if table == "cad" else r"F:\emerald_xai_img\nsclc"
    xai_img_path = None

    # Save the xai_img to the specified path
    if xai_img is not None:
        xai_img_path = f"{prefix}\\{entry_id}.jpg"
        if isinstance(xai_img, np.ndarray):
            xai_img = Image.fromarray(xai_img)
        xai_img.save(xai_img_path)

    # Ensure result is converted to integer if it's a valid number
    try:
        result = int(result)
    except:
        st.error(f"Invalid result value: {result}({type(result)})")
        return

    # Connect to the database
    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return

    try:
        cursor = connection.cursor()
        sql = f"UPDATE {table} SET nlp_text = %s, xai_img_path = %s, dss_pred = %s WHERE id = %s"
        cursor.execute(sql, (nlp_text, xai_img_path, result, entry_id))
        connection.commit()
        st.success(f"Results saved to DB for entry ID: {entry_id}")
    except Error as e:
        st.error(f"The error '{e}' occurred while saving results to the database")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def load_from_db(entry_id, table="cad"): #TODO take into account the 3 new fields (xai_img, dss_pred, nlp_txt)
    features_map = cad_features_db_mapping
    if table == "nsclc":
        features_map = nsclc_features_db_mapping
    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return None

    cursor = connection.cursor(dictionary=True)
    sql = f"SELECT * FROM {table} WHERE id = %s"
    cursor.execute(sql, (entry_id,))
    record = cursor.fetchone()

    if record:
        # change feature names
        key_list = list(features_map.keys())
        val_list = list(features_map.values())
        db_data = {key_list[val_list.index(key)]:value for key,value in record.items() if key in features_map.values()}
        st.session_state["entry_id"] = entry_id
        print(f"DL db_data: {db_data}")
        return db_data
    return None

def load_results_to_db(entry_id, table="cad"):
    features_map = cad_features_db_mapping
    if table == "nsclc":
        features_map = nsclc_features_db_mapping
    connection = create_connection()
    if connection is None:
        st.error("Failed to connect to the database")
        return None
    


    cursor = connection.cursor(dictionary=True)
    sql = f"SELECT nlp_text, xai_img_path, dss_pred FROM {table} WHERE id = %s"
    cursor.execute(sql, (entry_id,))
    record = cursor.fetchone()

    if record:
        # change feature names
        # key_list = list(features_map.keys())
        # val_list = list(features_map.values())
        # db_data = {key_list[val_list.index(key)]:value for key,value in record.items() if key in features_map.values()}
        # st.session_state["entry_id"] = entry_id
        print(f"DL db_data: {record}")
        return record
    return None

def cad_pred_print(pred, score=None):
    pred = int(pred)
    if not pred:
        st.header(f"Results for CAD: :green[Healthy]")
    else:
        st.header(f"Results for CAD: :red[CAD]")
    if score:
        score = float(score)
        st.subheader(f"Confidence score: {score:.2f}")

def nsclc_pred_print(pred, score=None):
    if not pred:
        st.header(f"Results for NSCLC: :green[Benign]")
    else:
        st.header(f"Results for NSCLC: :red[Malignant]")
    if score:
        score = float(score)
        st.subheader(f"Confidence score: {score:.2f}")

def trigger_gpt(case, values, feature_names):    
    # Extract the values
    shap_values = values.values[0]
    base_values = values.base_values[0]
    data = values.data[0]
    
    # Combine them into a DataFrame
    df = pd.DataFrame({
        'feature_name': feature_names,
        'value': data,
        'shap_value': shap_values
    })
    
    # Sort by absolute shap_value
    df['abs_shap_value'] = df['shap_value'].abs()
    df = df.sort_values(by='abs_shap_value', ascending=False).drop(columns='abs_shap_value')
    df = df.reset_index(drop=True)
    
    # print(sorted_df.to_string(index=False))

    global client
    prompt = f"""
    You are a medical AI assistant tasked with explaining the results of an Medical Decision Support System prediction to medical professionals. 
    The given explanation is for a {case} classification case. 

    Here is table containing the feature names, their values and the SHAP values: 
    {df}
    The shap-values indicate how much each feature affected the prediction result. A positive shap-value indicates that the feature pushed the prediction 
    towards the {case} class and a negative shap-value indicates that the feature pushed the prediction towards the Healthy class. The order they appear on the
    table indicates the strength of the feature's influence.
    
    The results represent a specific patient and as such should they be analyzed, not in a general way.
    
    Interpret these findings in a comprehensive way that is easy for doctors to understand without being too technical.  
    Present the explanation in natural, professional language suitable for medical staff. As a doctor, you should be able to understand the results and
    present them to your fellow doctors. Be specific to the patient and the features that were analyzed and do not generalize.

    Ensure that the explanation is thorough and clear, emphasizing the most impactful factors influencing the risk assessment.

    When presenting, preserve the order of the features as they are in the table. Do not mention the shap-value, but analyze every feature.

    Example: if the first feature is "Doctor: Healthy" and has a shap-value of 0.5 and value of False, it means the Doctor deemed the patient as not
    healthy and this contributed much on the final prediction. So the text should print something like: "The doctor's diagnosis was the most important 
    feature indicating a significant impact on the CAD risk prediction for this patient. A diagnosis by the doctor as Healthy strongly influences a lower 
    risk of coronary artery disease." And use this format to explain the rest of the features.

    write 250 words.
    """

    print(prompt)


        # using the following mapping of feature names and questions that they indicate to understand what value=True and value=False mean:
        # "known CAD" : Does the patient have previous CAD? ,
        # "previous AMI": "Does the patient have previous AMI?",
        # "previous PCI": "Does the patient have previous PCI?",
        # "previous CABG": "Does the patient have previous CABG?",
        # "previous STROKE": "Does the patient have previous stroke?",
        # "Diabetes": "Does the patient have diabetes?",
        # "Smoking": "Does the patient smoke?",
        # "Arterial Hypertension": "Does the patient have arterial hypertension?",
        # "Dislipidemia": "Does the patient have dislipidemia?",
        # "Angiopathy": "Does the patient have angiopathy?",
        # "Chronic Kindey Disease": "Does the patient have chronic kidney disease?",
        # "Family History of CAD": "Does the patient have a family history of CAD?",
        # "ASYMPTOMATIC": "Is the patient asymptomatic?",
        # "ATYPICAL SYMPTOMS": "Does the patient showcase atypical symptoms?",
        # "ANGINA LIKE": "Are the symptoms angina-like?",
        # "DYSPNOEA ON EXERTION": "Does the patient showcase dyspnoea on exertion?",
        # "INCIDENT OF PRECORDIAL PAIN": "Does the patient showcase incidents of precordial pain?",
        # "RST ECG": "Does the patient have RST ECG?",
        # "male": "Is the patient male?",
        # "CNN_Healthy": "Does the CNN report the patient as healthy?",
        # "BMI": "What is the Body Mass Index of the patient?",
        # "Overweight": is the patient overweight?,
        # "Obese": is the patient Obese?,
        # "normal_weight": has the patient normal weight?,
        # "u40": is the patient under 40?,
        # "40b50": is the patient between 40 and 50?,
        # "50b60": is the patient between 50 and 60?,
        # "o60": is the patient over 60?,
        # "Doctor: Healthy": Does the doctor yield the patient as healthy?,

        # Your task is to interpret these findings in a comprehensive way that is easy for doctors to understand without being too technical. 
        # Focus on which features played a major role in the classification. Do not mention numbers and stats.

        # For each feature, explain its importance in the context of CAD risk and briefly discuss whether it is a logical risk factor for CAD. 
        # Present the explanation in natural, professional language suitable for medical staff.

        # Do not list the features by name, but rather discuss them in the context of the risk assessment, in order of descending impact. 
        # Values should be treated on their absolute shap value.

        # Ensure that the explanation is thorough and clear, emphasizing the most significant factors influencing the risk assessment.
        # """

    # Set the model to use
    model_engine = "gpt-3.5-turbo"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # Return the response
    print(chat_completion.choices[0].message.content)
    st.session_state['class_result'] = chat_completion.choices[0].message.content #TODO do this for all models
    # return str(chat_completion.choices[0].message)
    return

def spect_pred_print(classification_preds, preds):
    if (classification_preds[0, 0] == 1) :
        st.header(f"Results for CAD: :red[Infarction]")
        st.subheader(f"Confidence score: {float(preds[0,0]):.2f}")
    elif (classification_preds[0, 1] == 1) :
        st.header(f"Results for CAD: :yellow[Ischemic]")
        st.subheader(f"Confidence score: {float(preds[0,1]):.2f}")
    elif (classification_preds[0, 2] == 1) :
        st.header(f"Results for CAD: :green[Normal]")
        st.subheader(f"Confidence score: {float(preds[0,2]):.2f}")

def save_uploaded_file(uploaded_file):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return file_path

def sig(x):
    return 1/(1 + np.exp(-x))

def show_random_forest_results(feature_values):
    data = pd.DataFrame([feature_values])
    dataframe = pd.DataFrame(data.values, columns=data.columns)
    # Load the saved Random Forest model
    loaded_model = joblib.load('trained_models/RandomForestClassifier.joblib')
    model_prediction = loaded_model.predict(dataframe)    
    # Display the prediction
    cad_pred_print(model_prediction[0])
    st.session_state['class_result'] = model_prediction[0] #TODO do this for all models

    st.markdown("""Random Forest is expecting Doctor's yield as input. This pretrained model has a reported accuracy of 83.52%, sensitivity of 83.06% and specificity of 85.49%.""")
    st.markdown("""
### Explainability through the SHAP model

The Waterfall SHAP plot provides a detailed explanation of how each feature in the model contributes to the prediction for a single instance.""")
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader(f"Results Explanation")
        # code for SHAP visualization
        tree_xai(loaded_model, dataframe, model_prediction[0])
    with col2:

        st.markdown("""
#### Interpreting the Plot:
- **Bars in Red and Blue**: 
  - **Red bars** indicate features that push the prediction to the right, towards the CAD class.
  - **Blue bars** indicate features that push the prediction to the left, towards the Healthy class.
- **SHAP values**: the numbers that appear inside (or beside) the bars are the SHAP values. These values indicate the impact of each feature on the model's prediction.
- **Order of Features**: Features are sorted in descending order based on their absolute SHAP values, highlighting the most influential features first.
""")
    # st.subheader(f"Results Explanation")
    # # code for SHAP visualization
    # tree_xai(loaded_model, dataframe, model_prediction[0])

def show_catboost_results(feature_values):
    data = pd.DataFrame([feature_values])
    dataframe = pd.DataFrame(data.values, columns=data.columns)
    # Load the saved Random Forest model
    loaded_model = joblib.load('trained_models/CatBoostClassifier.joblib')
    model_prediction = loaded_model.predict(dataframe)    
    # Display the prediction
    cad_pred_print(model_prediction[0])
    st.subheader(f"Results Explanation")
    # code for SHAP visualization
    xai_cat(loaded_model, dataframe)

def show_adaboost_results(feature_values):
    data = pd.DataFrame([feature_values])
    dataframe = pd.DataFrame(data.values, columns=data.columns)
    # Load the saved Random Forest model
    loaded_model = joblib.load('trained_models/AdaBoostClassifier.joblib')
    model_prediction = loaded_model.predict(dataframe)    
    # Display the prediction
    nsclc_pred_print(model_prediction[0])
    st.subheader(f"Results Explanation")
    # code for SHAP visualization
    tree_xai(loaded_model, dataframe, model_prediction[0])

def FCM_based_prediction(clinical_array, best_position, num_dimensions, study):
    if study == 'cad':
        user_bmi = int(clinical_array['BMI'])
        # Min and max values for age
        min_bmi = 15
        max_bmi = 45
        # Perform min-max normalization for the user-provided age
        normalized_bmi = (user_bmi - min_bmi) / (max_bmi - min_bmi)
        clinical_array['BMI'] = normalized_bmi
        
    if study == 'nsclc' or study == 'nsclc_multimodal':
        user_bmi = int(clinical_array['BMI'])
        normalized_bmi = user_bmi / 3
        clinical_array['BMI'] = normalized_bmi
        
        user_diameter = float(clinical_array['Diameter'])
        normalized_diameter = user_diameter / 70
        clinical_array['Diameter'] = normalized_diameter
        
        user_suv = float(clinical_array['SUV'])
        normalized_suv = user_suv / 30
        clinical_array['SUV'] = normalized_suv
 
    # User-provided age
    print(f"clinical_array: {clinical_array}")
    user_age = int(clinical_array['age'])

    # Min and max values for age
    min_age = 1
    max_age = 120

    # Perform min-max normalization for the user-provided age
    normalized_age = (user_age - min_age) / (max_age - min_age)

    # Update the dictionary with the normalized age value
    clinical_array['age'] = normalized_age

    clinical_array_list_values = [int(value) for value in clinical_array.values()]
    clinical_array_list_values = [int(value) for value in clinical_array.values()]

    clinical_array_list_values.append(0.5)

    # Initialize predicted_results with zeros
    predicted_results = [0.0] * num_dimensions
    best_position = best_position[1:]

    for i in range(0, num_dimensions):
        sum_temp = 0

        # Iterate through each dimension
        for j in range(0, num_dimensions):
            if i == j:
                continue
            else:
                sum_temp += best_position[j][i] * clinical_array_list_values[j]

        # Update the predicted result
        predicted_results[i] += sum_temp
        predicted_results[i] = sig(predicted_results[i])
    output_prediction = predicted_results[-1]

    limit = 0.6
    # if study== 'nsclc_multimodal':
    #     limit = 0.8
    
    binary_prediction = output_prediction > limit

    return binary_prediction, output_prediction, predicted_results

def show_fcm_pso_results(feature_values):
    if st.session_state['cad_patient_info']:
        study = "cad"
        df = pd.read_excel("trained_models/fcm_cad_mean_values.xlsx", header=None)
        if feature_values['Doctor: Healthy'] == 1:
            feature_values['Doctor: Healthy'] = 0
        else:
            feature_values['Doctor: Healthy'] = 1
    else:
        feature_values = transform_fcm_nsclc_vals(feature_values)
        study = "nsclc"
        df = pd.read_excel("trained_models/fcm_nsclc_mean_values.xlsx", header=None)
    best_position = df.to_numpy()
    bin_pred, out_pred, pred_results = FCM_based_prediction(feature_values,best_position,num_dimensions=len(feature_values), study=study)
    if study == 'cad':
        cad_pred_print(bin_pred, out_pred)
        print(f"pred_results: {pred_results}") #TODO pass to NLP
        st.text("The concept_values for each concept are the following:")
        for name, val in zip(concept_names_list_cad_fcm, pred_results):
            st.text(f"For {name}: {val:.2f}")
    else:
        nsclc_pred_print(bin_pred, out_pred)
        print(f"pred_results: {pred_results}") #TODO pass to NLP
        st.text("The concept_values for each concept are the following:")
        for name, val in zip(concept_names_list_nsclc_fcm, pred_results):
            st.text(f"For {name}: {val:.2f}")

# Function to preprocess an image object
def preprocess_image(image, size=(640, 640)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Function for Feature Ablation
def feature_ablation(model, image, mask_size=60):
    original_tensor = preprocess_image(image)
    num_rows = (image.height + mask_size - 1) // mask_size
    num_cols = (image.width + mask_size - 1) // mask_size
    heatmap = np.zeros((num_rows, num_cols))

    # Perform inference on the original image
    with torch.no_grad():
        original_pred = model(original_tensor)
    original_confidence = original_pred[0].probs.top1conf.item()    

    for i in range(0, image.width, mask_size):
        for j in range(0, image.height, mask_size):
            ablated_image = image.copy()
            draw = ImageDraw.Draw(ablated_image)
            draw.rectangle([i, j, i + mask_size, j + mask_size], fill="black")
            
            ablated_tensor = preprocess_image(ablated_image)
            with torch.no_grad():
                ablated_pred = model(ablated_tensor)
            ablated_confidence = ablated_pred[0].probs.top1conf.item()

            diff = np.abs(original_confidence - ablated_confidence)
            heatmap[j // mask_size, i // mask_size] = diff

    return heatmap

# Function to overlay the heatmap on the image
def overlay_heatmap(image, heatmap, colormap=plt.cm.autumn):
    heatmap_resized = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 1))
    heatmap_resized = colormap(heatmap_resized)
    heatmap_resized = Image.fromarray((heatmap_resized[:, :, :3] * 255).astype(np.uint8))
    heatmap_resized = heatmap_resized.resize(image.size, Image.LANCZOS)
    overlayed_image = Image.blend(image, heatmap_resized, alpha=0.5)
    return overlayed_image

def yolo_results(image, mode="PET", mask_size=60):
    yolo_model_path_ct = 'trained_models/YOLO_ct.pt'
    yolo_model_path_pet = 'trained_models/YOLO_pet.pt'
    if mode == "CT":
        model = YOLO(yolo_model_path_ct)
    else:
        model = YOLO(yolo_model_path_pet)
    original_image = image.convert('RGB')
    heatmap = feature_ablation(model, original_image, mask_size)
    overlayed_image = overlay_heatmap(original_image, heatmap, colormap=plt.cm.BuPu)

    # Print the XAI result
    # Perform inference
    results = model(image)
    # Display the results
    res_data = results[0].probs
    prediction = res_data.top1
    conf_score = res_data.top1conf
    nsclc_pred_print(prediction, conf_score)
    # some padding
    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        col1.header("Original")
        st.image(image, caption="Uploaded Image", width=200)
    with col3:
        col3.header("XAI Image")
        st.image(overlayed_image, caption="Ablation Heat-map", width=200)
    return

### Anna's models
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        # Ensure both arrays are of type float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if heatmap.dtype != np.float32:
            heatmap = heatmap.astype(np.float32)
            
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
###
def show_cnn_grad_cam_results(image, mode="PET", case='default'):
    original_height, original_width = image.shape[:2]
    pixel_size = 200
    model_layer = ''

    if mode == 'PET':
        cnn_model = keras.models.load_model("trained_models/pet_vgg16.keras")
        pixel_size = 100
        model_layer = 'block5_conv3'

    if mode == 'CT':
        cnn_model = keras.models.load_model("trained_models/ct_vgg16.keras")
        pixel_size = 100
        model_layer = 'block5_conv3'

    if mode == 'Polar Maps':
        cnn_model = keras.models.load_model("trained_models/polar_maps.keras")
        pixel_size = 300
        model_layer = 'conv2d_2'

    if mode == 'SPECT':
        cnn_model = keras.models.load_model("trained_models/spect_model")
        pixel_size = 250
        model_layer = 'conv2d_3'

    res_image = cv2.resize(image, (pixel_size, pixel_size))
    res_image = res_image.astype('float32') / 255
    res_image = np.expand_dims(res_image, axis=0)

    preds = cnn_model.predict(res_image)

    if mode == 'SPECT':
        classification_preds = (cnn_model.predict(res_image) > 0.5).astype(int)

        if classification_preds[0, 0] == 1:
            print("\n The model predicts that this image instance exhibits signs indicative of an infarction with a probability ", preds[0, 0])
        if classification_preds[0, 1] == 1:
            print("\n The model predicts that this image instance exhibits signs indicative of ischemic case with a probability ", preds[0, 1])
        if classification_preds[0, 2] == 1:
            print("\n The model predicts that this image instance exhibits signs indicative of normal case with a probability ", preds[0, 2])
    else:
        binary_preds = (preds > 0.5).astype(int)
        if binary_preds == 0:
            print("\nThe image is predicted as Benign with a probability ", preds)
        else:
            print("\nThe image is predicted as Malignant with a probability ", preds)

    i = np.argmax(preds[0])

    icam = GradCAM(cnn_model, i, model_layer)
    heatmap = icam.compute_heatmap(res_image)
    heatmap = cv2.resize(heatmap, (pixel_size, pixel_size))

    (heatmap, output) = icam.overlay_heatmap(heatmap, res_image[0], alpha=0.5)
    st.session_state['xai_image'] = heatmap #TODO do this for all models


    heatmap = cv2.resize(output, (original_width, original_height))

    # Normalize heatmap to [0, 1]
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()

    if case != 'Multimodal':
        if mode == 'Polar Maps':
            cad_pred_print(binary_preds, preds)
        elif mode == 'SPECT':
            spect_pred_print(classification_preds, preds)
        else:
            nsclc_pred_print(binary_preds, preds)
    else:
        return preds, heatmap

    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        col1.header("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    with col3:
        col3.header("XAI Image")
        st.image(heatmap, caption="GRAD-CAM Image", use_column_width=True)

    return

def show_deep_fcm_results(feature_values, image, mode="PET"):
    if st.session_state['cad_patient_info']:
        study = "cad"
        df = pd.read_excel("trained_models/deepfcm_cad_mean_values.xlsx", header=None)
        best_position = df.to_numpy()
        num_dimensions= len(feature_values) + 2
        cnn_prediction, heatmap = show_cnn_grad_cam_results(image, mode='Polar Maps', case='Multimodal')
        img_type = 'Polar Maps'
    else:
        feature_values = transform_fcm_nsclc_vals(feature_values)
        study = "nsclc"
        if mode == "PET":
            df = pd.read_excel("trained_models/deepfcm_nsclc_pet_mean_values.xlsx", header=None)
            best_position = df.to_numpy()
            num_dimensions= len(feature_values) + 1
            cnn_prediction, heatmap = show_cnn_grad_cam_results(image, mode='PET', case='Multimodal')
            img_type = 'PET'
        else:
            df = pd.read_excel("trained_models/deepfcm_nsclc_ct_mean_values.xlsx", header=None)
            best_position = df.to_numpy()
            num_dimensions= len(feature_values) + 1
            cnn_prediction, heatmap = show_cnn_grad_cam_results(image, mode='CT', case='Multimodal')
            img_type = 'CT'
        
    cnn_prediction = cnn_prediction > 0.5    
    cnn_prediction = int(cnn_prediction[0, 0])
    feature_values['cnn_prediction'] = cnn_prediction
    st.session_state['cnn_prediction'] = cnn_prediction
    
    bin_pred, out_pred, pred_results = FCM_based_prediction(feature_values,best_position,num_dimensions, study=img_type)
    st.session_state['class_result'] = bin_pred #TODO do this for all models
    if study == 'cad':
        cad_pred_print(bin_pred, out_pred)
        print(f"pred_results: {pred_results}") #TODO pass to NLP
        st.text("The concept_values for each concept are the following:")
        temp_name_list = concept_names_list_cad_fcm
        temp_name_list.append("CNN prediction")
        for name, val in zip(temp_name_list, pred_results):
            st.text(f"For {name}: {val:.2f}")
    else:
        nsclc_pred_print(bin_pred, out_pred)
        print(f"pred_results: {pred_results}") #TODO pass to NLP
        st.text("The concept_values for each concept are the following:")
        temp_name_list = concept_names_list_nsclc_fcm
        temp_name_list.append("CNN prediction")
        for name, val in zip(temp_name_list, pred_results):
            st.text(f"For {name}: {val:.2f}")

    # some padding
    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        col1.header("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=200)
    with col3:
        col3.header("XAI Image")
        st.image(heatmap, caption="GRAD-CAM Image", width=200)
    
    return

cad_models_dict = {
    'cad_models_clinical': [
        {
            "name": "Random Forest",
            "function": show_random_forest_results,
            "info": "Random Forest is expecting Doctor's yield as input. This pretrained model has a reported accuracy of 83.52%.",
            "features": ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease', 'ANGINA LIKE', 'RST ECG', 'male', 'u40', 'Doctor: Healthy']
        },
        {
            "name": "CatBoost",
            "function": show_catboost_results,
            "info": "Catboost does not require the Doctor's yield as input. This pretrained model achieves a reported accuracy of 78.82%.",
            "features": ['known CAD', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Angiopathy', 
                        'Chronic Kindey Disease', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Obese', '40b50', '50b60']
        },
        {
            "name": "FCM-PSO",
            "function": show_fcm_pso_results,
            "info": "FCM-PSO requires the Doctor's yield as input. This pretrained model achieves a reported accuracy of 74.98%.",
            "features": ['male', 'age', 'BMI', 'known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension',
			            'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION',
			            'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'Doctor: Healthy']
        },
    ],
    'cad_models_image': [
        {
            "name": "RGB-CNN-Polar-Maps",
            "function": show_cnn_grad_cam_results,
            "info": "RGB-CNN is trained on Polar maps image data. This pretrained model has a reported accuracy of 81.25%.",
            "modes": ["Polar Maps"],
        },
        {
            "name": "RGB-CNN-SPECT-MPI",
            "function": show_cnn_grad_cam_results,
            "info": "RGB-CNN is trained on SPECT-MPI image data. This pretrained model has a reported accuracy of 84.37%.",
            "modes": ["SPECT"],
        },
    ],
    'cad_models_multimodal': [
        {
            "name": "DeepFCM",
            "function": show_deep_fcm_results,
            "info": "DeepFCM is a multimodal approach able to handle both clinical and Polar Maps imaging data. PSO is utilized as a learning technique. This pretrained model has a reported accuracy of 84.21%.",
            "features": ['male', 'age', 'BMI', 'known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Arterial Hypertension',
			            'Dislipidemia', 'Angiopathy', 'Chronic Kindey Disease', 'Family History of CAD', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE', 'DYSPNOEA ON EXERTION',
			            'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'Doctor: Healthy']
        },
    ]
}

nsclc_models_dict = {
    'nsclc_models_clinical': [
        {
            "name": "AdaBoost",
            "function": show_adaboost_results,
            "info": "AdaBoost model architecture. This pretrained model has a reported accuracy of 94.33%.",
            "features": ['Gender','Fdg','Age','BMI','GLU','SUV','Diameter','Location','Type','Limits']
        },
        {
            "name": "FCM-PSO",
            "function": show_fcm_pso_results,
            "info": "FCM-PSO handles clinical data as input concepts. PSO is utilized as a learning technique. This pretrained model has a reported accuracy of 82.61%.",
            "features": ['Gender','Fdg','Age','BMI','GLU','SUV','Diameter','Location','Type','Limits']
        }
    ],
    'nsclc_models_image': [
        {
            "name": "YOLOv8",
            "function": yolo_results,
            "info": "You Only Look Once (YOLO) version 8 can be applied to CT or PET medical scans. This pretrained model has a reported accuracy of 89% for PET images and 92.3% for CT.",
        },
        {
            "name": "VGG-16",
            "function": show_cnn_grad_cam_results,
            "info": "VGG-16 is a pre-trained network, which has been fine-tuned to CT or PET medical scans. This pretrained model has a reported accuracy of 85% for PET images and 87.5% for CT.",
        },
    ],
    'nsclc_models_multimodal': [
        {
            "name": "DeepFCM",
            "function": show_deep_fcm_results,
            "info": "DeepFCM is a multimodal approach able to handle both clinical and PET/CT imaging data. PSO is utilized as a learning technique. This pretrained model has a reported accuracy of 86.45% for PET and 86.96% for CT.",
            "features": ['Gender','Fdg','Age','BMI','GLU','SUV','Diameter','Location','Type','Limits']
        },
    ]
}

# st.session_state[] initializations
session_tate_defaults = {
    'step': 'home',
    'model': '',
    'cad_model_type': '',
    'cad_model_info': '',
    'cad_patient_info': '',
    'nsclc_model_info': '',
    'nsclc_patient_info': '',
    'age': 0,
    'weight': 0.0,
    'warning': [],
    'cv_image': None,
    'xai_image': None,
    'class_result': None,
    'nlp_text': None,
    'entry_id': None,
}
for key, value in session_tate_defaults.items():
    if st.session_state.get(key) is None:
        st.session_state[key] = value

def transform_quantitive_cad(val, feature):
    if feature in age_cats:
        if val < 40:
            return int('u40' == feature)
        elif val >= 40 and val < 50:
            return int('40b50' == feature)
        elif val >= 50 and val < 60:
            return int('50b60' == feature)
        else:
            return int('o60' == feature)
    elif feature in weight_cats:
        if val < 25.0:
            return int('normal_weight' == feature)
        elif val >= 25.0 and val < 30.0:
            return int('Overweight' == feature)
        else:
            return int('Obese' == feature)
        
def transform_nsclc_vals(val, feature):
    if feature == "Location":
        location_mapping = {
        "left-down": 0,
        "lingula": 1,
        "middle": 2,
        "right-down": 3,
        "right-up": 4,
        "left-up": 5,
        }
        return location_mapping.get(val, None)
    elif feature == "Type":
        type_mapping = {
        "cavitary": 0,
        "ground-class": 1,
        "inconclusive": 2,
        "semi-solid": 3,
        "solid": 4,
        "stikto": 5,
        }
        return type_mapping.get(val, None)
    elif feature == "Limits":
        limits_mapping = {
        "ground-class": 0,
        "ill-defined": 1,
        "inconclusive": 2,
        "lobulated": 3,
        "speculated": 4,
        "well-defined": 5,
        }
        return limits_mapping.get(val, None)
    
def transform_fcm_nsclc_vals(features_dict):
    if "Location" in features_dict.keys():
        for name in ['LOCATION_LEFT_UPPER_LOBE', 'LOCATION_RIGHT_UPPER_LOBE', 'LOCATION_LEFT_LOWER_LOBE',
                     'LOCATION_RIGHT_LOWER_LOBE', 'LOCATION_MIDDLE', 'LOCATION_LINGULA']:
            features_dict[name] = 0
        if features_dict["Location"] == 0:
            features_dict["LOCATION_LEFT_LOWER_LOBE"] = 1
        elif features_dict["Location"] == 1:
            features_dict["LOCATION_LINGULA"] = 1
        elif features_dict["Location"] == 2:
            features_dict["LOCATION_MIDDLE"] = 1
        elif features_dict["Location"] == 3:
            features_dict["LOCATION_RIGHT_LOWER_LOBE"] = 1
        elif features_dict["Location"] == 4:
            features_dict["LOCATION_RIGHT_UPPER_LOBE"] = 1
        elif features_dict["Location"] == 5:
            features_dict["LOCATION_LEFT_UPPER_LOBE"] = 1
        del features_dict["Location"]

    if "Type" in features_dict.keys():
        for name in ['TYPE_SOLID', 'TYPE_GROUND_CLASS', 'TYPE_CONSOLIDATED',
                     'TYPE_SPECKLED', 'TYPE_SEMI_SOLID', 'TYPE_calcified', 'TYPE_cavitary']:
            features_dict[name] = 0
        if features_dict["Type"] == 0:
            features_dict["TYPE_cavitary"] = 1
        elif features_dict["Type"] == 1:
            features_dict["TYPE_GROUND_CLASS"] = 1
        elif features_dict["Type"] == 2:
            features_dict["TYPE_calcified"] = 1
        elif features_dict["Type"] == 3:
            features_dict["TYPE_SEMI_SOLID"] = 1
        elif features_dict["Type"] == 4:
            features_dict["TYPE_SOLID"] = 1
        elif features_dict["Type"] == 5:
            features_dict["TYPE_SPECKLED"] = 1
        elif features_dict["Type"] == 6:
            features_dict["TYPE_CONSOLIDATED"] = 1
        del features_dict["Type"]

    if "Limits" in features_dict.keys():
        for name in ['BOUNDARIES_spiculated', 'BOUNDARIES_lobulated', 'BOUNDARIES_well_defined',
                     'BOUNDARIES_ill-defined']:
            features_dict[name] = 0
        if features_dict["Limits"] == 4:
            features_dict["BOUNDARIES_spiculated"] = 1
        elif features_dict["Limits"] == 3:
            features_dict["BOUNDARIES_lobulated"] = 1
        elif features_dict["Limits"] == 5:
            features_dict["BOUNDARIES_well_defined"] = 1
        elif features_dict["Limits"] == 1:
            features_dict["BOUNDARIES_ill-defined"] = 1
        del features_dict["Limits"]
    if "Age" in features_dict.keys():
        features_dict["age"] = features_dict["Age"]
        del features_dict["Age"]
    return features_dict

# function to print SHAP values and plots for tree ML models
def tree_xai(model, X, val):
    # Create a mapping dictionary for descriptive feature names
    feature_name_mapping = {
        'Sex': 'Female/Male',
        'Age': 'Age: Under 40/Over 40',
        'Doctor: Healthy': 'Doctor: Healthy/CAD'
        # Add other feature name mappings as needed
    }

    # # # Rename the columns in the data for SHAP explanation
    # X_renamed = X.rename(columns=feature_name_mapping)
    X_renamed = X.copy()
    # Convert all 1s to True and 0s to False in the renamed DataFrame
    X_renamed = X_renamed.applymap(lambda x: True if x == 1 else False if x == 0 else x)

    print(f"X_renamed: {X_renamed}")

    # Explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_renamed)
    sv = explainer(X_renamed)
    
    # Create the explanation with the renamed features
    exp = shap.Explanation(sv[:, :, 1], sv.base_values[:, 1], X_renamed, feature_names=X_renamed.columns)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Show waterfall plot for the first instance
    fig_cad, _ = plt.subplots()
    shap.waterfall_plot(exp[0])
    st.pyplot(fig_cad)

    # Show force plot for class 0 (Healthy) here input prediction
    fig_force_0, _ = plt.subplots()
    fig_force_0 = shap.force_plot(val, shap_values[0][0], X_renamed.iloc[0, :], matplotlib=True)
    st.pyplot(fig_force_0)

    # st.text(trigger_gpt("CAD", exp, X.columns))

# function to print SHAP values and plots for CatBoost
def xai_cat(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sv = explainer.shap_values(X.loc[[0]])
    exp = shap.Explanation(sv,explainer.expected_value, data=X.loc[[0]].values, feature_names=X.columns)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Show waterfall plot for idx_cad
    fig_cad, _ = plt.subplots()
    shap.waterfall_plot(exp[0])
    st.pyplot(fig_cad)
    # Show force plot for class 0 (Healthy) here input prediction
    fig_force_0, _ = plt.subplots()
    fig_force_0 = shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)
    st.pyplot(fig_force_0)

# App flow
# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "CAD", "NSCLC", "Load results from DB", "Contact Us"],
        icons=["house", "heart", "lungs", "open_file_folder", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

# Body
if selected == "Home":
    st.header("EMERALD Medical Decision Support System")
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    with col2:
        st.image("https://emerald.uth.gr/wp-content/uploads/2022/08/cropped-logo_emerald_official-1-150x150.jpg", width=200)
    with col4:
        st.image("https://emerald.uth.gr/wp-content/uploads/2022/05/UTH-logo-english-300x300.png", width=200)
    st.markdown('''EMERALD pioneers a holistic approach to patient-specific predictive modeling and MDSS development, 
                leveraging advanced ICT technologies like Data Mining, Deep Learning, and Fuzzy Cognitive Tools. The 
                integration of dynamic Fuzzy Cognitive Models (FCMs) and expert knowledge enhances data interpretability 
                and modeling complexity, facilitating personalized treatment and health technology assessment. The 
                introduction of DeepFCMs, an innovative component of XAI-MDSS, enables the fusion of diverse medical 
                data and provides visual and self-explanatory capabilities, fostering optimal physician decision-making 
                and a high-value healthcare ecosystem.''')
    st.video("https://emerald.uth.gr/wp-content/uploads/EMERALD_final.mp4")
    if st.session_state['step'] != 'home':
        st.session_state['step'] = 'home'
        st.rerun()

elif selected == "CAD":
    if st.session_state['step'] != 'cad' and st.session_state['step'] != 'cad_results':
        st.session_state['step'] = 'cad'

elif selected == "NSCLC":
    if st.session_state['step'] != 'nsclc' and st.session_state['step'] != 'nsclc_results':
        st.session_state['step'] = 'nsclc'

# elif selected == "CAD":
    # if st.session_state['step'] == 'home':
    #     st.session_state['step'] = 'cad'

# elif selected == "NSCLC":
#     if st.session_state['step'] == 'home':
#         st.session_state['step'] = 'nsclc'

elif selected == "Load results from DB":
    st.header("Load Results from Database")

    entry_id = st.text_input("Enter Entry ID:")
    table_option = st.selectbox("Select Case", ["cad", "nsclc"])

    if st.button("Load Results"):
        print_func = cad_pred_print
        if table_option == "nsclc":
            print_func = nsclc_pred_print
        
        if entry_id:
            results = load_results_to_db(entry_id, table_option)
            if results:
                st.subheader("Results")
                print_func(results['dss_pred'])
                # st.write(f"Prediction: {results['dss_pred']}")
                if results['nlp_text']:
                    st.write(f"NLP Text: {results['nlp_text']}")
                if results['xai_img_path']:
                    st.subheader("XAI Image")
                    st.image(results['xai_img_path'], caption='XAI Image')
            else:
                st.error("No results found for the given Entry ID.")
        else:
            st.error("Please enter an Entry ID.")

elif selected == "Contact Us":
    st.header("EMERALD Medical Decision Support System")
    st.subheader(f"Homepage: https://emerald.uth.gr/")
    st.subheader(f"Github: https://github.com/emeraldUTH/EMERALD")
    st.subheader(f"Get in touch!")
    if st.session_state['step'] != 'contact':
        st.session_state['step'] = 'contact'
        st.rerun()

# Process CAD Prediction
if st.session_state['step'] == 'cad':
    st.sidebar.write("**Program instructions**")
    st.sidebar.write("Step 1. Input patient info")
    st.sidebar.write("Step 2. Select a desired model configuration")
    st.sidebar.write("Step 3. Select a model from the available selection")
    tab1, tab2, tab3 = st.tabs(["Step 1. Input patient info", "Step 2. Model configuration", "Step 3. Select a model"])
    with tab1:
        st.header("Input patient info")
        with st.form(key='CAD PATIENT INFO'):
            age_asked = False
            weight_asked = False
            for feature in cad_features_dict:
                if feature not in st.session_state:
                    st.session_state[feature] = 0
                if feature in age_cats:
                    if age_asked:
                        st.session_state[feature] = int(transform_quantitive_cad(st.session_state['age'], feature))
                    else:
                        val = st.slider("Age:", min_value=0, max_value=100, step=1, value=st.session_state['age'])
                        age_asked = True
                        if val:
                            st.session_state['age'] = val
                            st.session_state[feature] = int(transform_quantitive_cad(val, feature))
                elif feature == "BMI":
                    val = st.slider("Body Mass Index (BMI):", min_value=0.0, max_value=50.0, step=0.01, value=1.0*st.session_state[feature])
                    if val:
                        st.session_state[feature] = val
                elif feature in weight_cats:
                    if weight_asked:
                        st.session_state[feature] = int(transform_quantitive_cad(st.session_state['weight'], feature))
                    else:
                        val = st.slider("Weight:", min_value=0.0, max_value=150.0, step=0.1, value=st.session_state['weight'])
                        weight_asked = True
                        if val:
                            st.session_state['weight'] = val
                            st.session_state[feature] = int(transform_quantitive_cad(val, feature))
                elif feature == "male":
                    val = st.selectbox(cad_features_dict[feature], ['Yes','No'])
                    if val:
                        if val == 'Yes':
                            st.session_state[feature] = 1
                        else:
                            st.session_state[feature] = 0
                else:
                    val = st.selectbox(cad_features_dict[feature], ['Yes','No','N/A'])
                    if val:
                        if val == 'Yes':
                            st.session_state[feature] = 1
                        elif val == 'No':
                            st.session_state[feature] = 0
                        else:
                            # if N/A is selected use the default values
                            st.session_state[feature] = cad_default_values[feature]
                            st.session_state['warning'].append(f"Using default value {cad_default_values[feature]} for {feature}.")

            col1, col2, col3 = st.columns([2, 6, 1])
            if col1.form_submit_button("Submit"):
                st.session_state['cad_patient_info'] = True
                print(f"state: {st.session_state}")

                if any( st.session_state[feature] == 0 for feature in cad_mandatory_features):
                    for feature in cad_mandatory_features:
                        if st.session_state[feature] == 0:
                            st.warning(f'{feature} cannot be empty', icon="❌")
                else:
                    if len(st.session_state['warning']) > 0:
                        for warning in st.session_state['warning']:
                            st.warning(warning, icon="⚠️")
                        time.sleep(5)
                        st.session_state['warning'].clear()
                    st.rerun()
            if col3.form_submit_button("Reset"):
                for feature in cad_features_dict:
                    if feature in st.session_state:
                        st.session_state[feature] = 0
                st.session_state['age'] = 0
                st.session_state['weight'] = 0.0
                st.session_state['cad_patient_info'] = False
                print(f"state: {st.session_state}")
                st.rerun()
        # Upload image
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff", "bmp"])
        # Select image type (Polar Maps or SPECT)
        st.session_state['image_type'] = st.selectbox("Select image type:", ["Polar Maps", "SPECT"])
        if uploaded_img:
            # Convert the uploaded image to PIL Image
            # pil_image = Image.open(uploaded_img)
            file_path = save_uploaded_file(uploaded_img)
            cv_image = cv2.imread(file_path)
            st.session_state['cv_image'] = cv_image
            #TODO remove below before commit
            st.markdown("")
            st.markdown("")
            col1.header("Original")
            st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

        # Save to DB button
        if st.button("Save to DB"):
            data = {feature: st.session_state[feature] for feature in cad_features_dict}
            data['age'] = st.session_state['age'] # age is not included in dict cad_features_dict
            entry_id = save_to_db(data, "cad")
            st.success(f"Data saved to DB with entry ID: {entry_id}")
            st.session_state['entry_id'] = entry_id

        # Load from DB section
        with st.form(key='Load from DB'):
            entry_id = st.number_input("Enter entry ID to load", min_value=1, step=1)
            if st.form_submit_button("Load"):
                record = load_from_db(entry_id, "cad")
                if record:
                    for feature in cad_features_dict:
                        st.session_state[feature] = record.get(feature, None)
                        if isinstance(st.session_state[feature], decimal.Decimal):
                            st.session_state[feature] = float(st.session_state[feature])
                    st.success(f"Data loaded from DB for entry ID: {entry_id}")
                else:
                    st.error(f"No record found for entry ID: {entry_id}")

        # Save to DB button
        if st.button("Save to DB"):
            data = {feature: st.session_state[feature] for feature in cad_features_dict}
            data['age'] = st.session_state['age'] # age is not included in dict cad_features_dict
            entry_id = save_to_db(data, "cad")
            st.success(f"Data saved to DB with entry ID: {entry_id}")
            st.session_state['entry_id'] = entry_id

        # Load from DB section
        with st.form(key='Load from DB'):
            entry_id = st.number_input("Enter entry ID to load", min_value=1, step=1)
            if st.form_submit_button("Load"):
                record = load_from_db(entry_id, "cad")
                if record:
                    for feature in cad_features_dict:
                        st.session_state[feature] = record.get(feature, None)
                        if isinstance(st.session_state[feature], decimal.Decimal):
                            st.session_state[feature] = float(st.session_state[feature])
                    st.success(f"Data loaded from DB for entry ID: {entry_id}")
                else:
                    st.error(f"No record found for entry ID: {entry_id}")

    with tab2:
        st.header("Input model preferences")
        with st.form(key=f'CAD MODEL INFO'):
            val = st.selectbox("Please select the type of input data for the AI model:", ['Clinical only','Image only','Multimodal'])
            if val:
                if val == 'Clinical only':
                    st.session_state['cad_model_type'] = 'cad_models_clinical'
                elif val == 'Image only':
                    st.session_state['cad_model_type'] = 'cad_models_image'
                elif val == 'Multimodal':
                    st.session_state['cad_model_type'] = 'cad_models_multimodal'

            if st.form_submit_button("Submit"):
                if st.session_state['cad_model_type'] == 'cad_models_multimodal' and any((
                        st.session_state['cv_image'] is None,
                        st.session_state['cad_patient_info'] is not True
                    )):
                        st.warning('Multimodal prediction for CAD requires clinical data and Polar Maps as input image', icon="❌")
                else:
                    st.session_state['cad_model_info'] = True
                    st.rerun()

    with tab3:
        st.header("Select a model from the available list")
        if any((
            st.session_state['cad_model_type'] is None,
            st.session_state['cad_model_info'] is not True
        )):
            st.markdown("Please first complete Steps 1 & 2")
        else:
            selected_model_dict = None
            for model_dict in cad_models_dict:
                if model_dict == st.session_state['cad_model_type']:
                    selected_model_dict = cad_models_dict.get(model_dict)

            # st.header("Clinical data only models")
            for model in selected_model_dict:
                # do not show models that require the expert's yield if we do not have the yield
                if "features" in model.keys() and 'Doctor: Healthy' in model["features"] and st.session_state["Doctor: Healthy"] is None:
                    continue
                if "modes" in model.keys() and st.session_state['image_type'] not in model["modes"]:
                    continue
                with st.container(border=True):
                    # Divide the container into two columns: 1/3 button, 2/3 info
                    col1, col2 = st.columns([1, 2])

                    # Button in the first column (1/3)
                    if col1.button(model["name"]):
                        # if not st.session_state['test']:
                        #     st.warning('This is a warning', icon="⚠️")
                        st.session_state['step'] = 'cad_results'
                        st.session_state['model'] = model
                        # st.session_state['model_type'] = 'clinical'
                        st.rerun()

                    # Info in the second column (2/3)
                    col2.write(model["info"])

# Show CAD Results
if st.session_state['step'] == 'cad_results':
    print(f"state: {st.session_state}")
    spec_model = st.session_state['model']
    st.sidebar.subheader("CAD Results")
    features_vals = {}
    if st.session_state['cad_model_type'] == 'cad_models_clinical':
        for feature in spec_model['features']:
            features_vals[feature] = st.session_state[feature]
        spec_model["function"](features_vals)
    elif st.session_state['cad_model_type'] == 'cad_models_image':
        spec_model["function"](st.session_state['cv_image'], st.session_state['image_type'], "default")
    elif st.session_state['cad_model_type'] == 'cad_models_multimodal':
        for feature in spec_model['features']:
            features_vals[feature] = st.session_state[feature]
        spec_model["function"](features_vals, st.session_state['cv_image'], st.session_state['image_type'])

    if st.session_state['entry_id']:
        if st.button("Save results to DB"):
            # save_results_to_db("dfa", "daf", "sdf")
            save_results_to_db(nlp_text="", xai_img=st.session_state['xai_image'], result=st.session_state['class_result'], table="cad")
    if st.button("Close"):
        selected = "Home"
        st.session_state['step'] = 'Home'
        st.rerun()

# Process NSCLC Prediction
if st.session_state['step'] == 'nsclc':
    st.sidebar.write("**Program instructions**")
    st.sidebar.write("Step 1. Input patient info")
    st.sidebar.write("Step 2. Select a desired model configuration")
    st.sidebar.write("Step 3. Select a model from the available selection")
    tab1, tab2, tab3 = st.tabs(["Step 1. Input patient info", "Step 2. Model configuration", "Step 3. Select a model"])
    with tab1:
        st.header("Input patient info")
        with st.form(key='NSCLC PATIENT INFO'):
            age_asked = False
            weight_asked = False
            for feature in nsclc_features_dict:
                if feature not in st.session_state:
                    st.session_state[feature] = 0
                if feature == "Age":
                    val = st.slider(nsclc_features_dict[feature], min_value=0, max_value=100, step=1, value=st.session_state[feature])
                    if val:
                        st.session_state[feature] = val
                elif feature == "BMI":
                    val = st.slider(nsclc_features_dict[feature], min_value=0.0, max_value=50.0, step=0.01, value=1.0*st.session_state[feature])
                    if val:
                        st.session_state[feature] = val
                elif feature == "Gender":
                    # for clinical_nsclc just the Gender
                    val = st.selectbox(nsclc_features_dict[feature], ['Yes','No'])
                    if val:
                        if val == 'Yes':
                            st.session_state[feature] = 1
                        else:
                            st.session_state[feature] = 0
                elif feature == "Location":
                    val = st.selectbox(feature, ['left-up','left-down','lingula','middle','right-down','right-up'])
                    if val:
                        st.session_state[feature] = transform_nsclc_vals(val, feature)
                elif feature == "Type":
                    val = st.selectbox(feature, ['cavitary','ground-class','semi-solid','solid','stikto','inconclusive'])
                    if val:
                        st.session_state[feature] = transform_nsclc_vals(val, feature)
                elif feature == "Limits":
                    val = st.selectbox(feature, ['ground-class','ill-defined','inconclusive','lobulated','speculated','well-defined'])
                    if val:
                        st.session_state[feature] = transform_nsclc_vals(val, feature)
                else:
                    val = st.number_input(feature, value=None, placeholder="Accepts positive values with single digit precision - e.g. 1.8", min_value=0.1, step=0.1, format="%0.1f")
                    if val:
                        if feature == "GLU":
                            st.session_state[feature] = int(val)
                        else:
                            st.session_state[feature] = val

            col1, col2, col3 = st.columns([2, 6, 1])
            if col1.form_submit_button("Submit"):
                st.session_state['Fdg'] = st.session_state['SUV']
                if st.session_state['BMI'] == 0:
                    st.session_state['BMI'] = 26.23 # set to avg
                    st.session_state['warning'].append("Using default value 26.23 for BMI.")
                if st.session_state['GLU'] == 0:
                    st.session_state['warning'].append("Using default value 111 for GLU.")
                    st.session_state['GLU'] = 111 # set to avg
                st.session_state['nsclc_patient_info'] = True
                print(f"state: {st.session_state}")
                
                if any( st.session_state[feature] == 0 for feature in nsclc_mandatory_features):
                    for feature in nsclc_mandatory_features:
                        if st.session_state[feature] == 0:
                            st.warning(f'{feature} cannot be empty', icon="❌")
                else:
                    if len(st.session_state['warning']) > 0:
                        for warning in st.session_state['warning']:
                            st.warning(warning, icon="⚠️")
                        time.sleep(5)
                        st.session_state['warning'].clear()
                    st.rerun()
            if col3.form_submit_button("Reset"):
                for feature in cad_features_dict:
                    if feature in st.session_state:
                        st.session_state[feature] = 0
                st.session_state['Age'] = 0
                st.session_state['BMI'] = 0.0
                st.session_state['nsclc_patient_info'] = False
                print(f"state: {st.session_state}")
                st.rerun()

        # Upload image
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        # Select image type (PET or CT)
        st.session_state['image_type'] = st.selectbox("Select image type:", ["PET", "CT"])
        if uploaded_img:
            # Convert the uploaded image to PIL Image
            pil_image = Image.open(uploaded_img)
            st.session_state['pil_image'] = pil_image

            file_path = save_uploaded_file(uploaded_img)
            cv_image = cv2.imread(file_path)
            st.session_state['cv_image'] = cv_image

        # Save to DB button
        if st.button("Save to DB"):
            data = {feature: st.session_state[feature] for feature in nsclc_features_dict}
            data['Fdg'] = st.session_state['Fdg'] # Fdg is a duplicate of SUN and as such not included in dict nsclc_features_dict
            entry_id = save_to_db(data, "nsclc")
            st.success(f"Data saved to DB with entry ID: {entry_id}")
            st.session_state['entry_id'] = entry_id

        # Load from DB section
        with st.form(key='Load from DB'):
            entry_id = st.number_input("Enter entry ID to load", min_value=1, step=1)
            if st.form_submit_button("Load"):
                record = load_from_db(entry_id, "nsclc")
                if record:
                    for feature in nsclc_features_dict:
                        st.session_state[feature] = record.get(feature, None)
                        if isinstance(st.session_state[feature], decimal.Decimal):
                            st.session_state[feature] = float(st.session_state[feature])
                    st.session_state['Fdg'] = st.session_state['SUV']
                    st.success(f"Data loaded from DB for entry ID: {entry_id}")
                else:
                    st.error(f"No record found for entry ID: {entry_id}")

    with tab2:
        st.header("Input model preferences")
        with st.form(key=f'NSCLC MODEL INFO'):
            val = st.selectbox("Please select the type of input data for the AI model:", ['Clinical only','Image only','Multimodal'])
            if val:
                if val == 'Clinical only':
                    st.session_state['nsclc_model_type'] = 'nsclc_models_clinical'
                elif val == 'Image only':
                    st.session_state['nsclc_model_type'] = 'nsclc_models_image'
                elif val == 'Multimodal':
                    st.session_state['nsclc_model_type'] = 'nsclc_models_multimodal'

            if st.form_submit_button("Submit"):
                if st.session_state['nsclc_model_type'] == 'nsclc_models_multimodal' and any((
                        st.session_state['cv_image'] is None,
                        st.session_state['nsclc_patient_info'] is not True
                    )):
                        st.warning('Multimodal prediction for NSCLC requires clinical data and PET/CT as input image', icon="❌")
                else:
                    st.session_state['nsclc_model_info'] = True
                    st.rerun()

    with tab3:
        st.header("Select a model from the available list")
        if any((
            st.session_state['nsclc_model_type'] is None,
            st.session_state['nsclc_model_info'] is not True
        )):
            st.markdown("Please first complete Steps 1 & 2")
        else:
            selected_model_dict = None
            for model_dict in nsclc_models_dict:
                if model_dict == st.session_state['nsclc_model_type']:
                    selected_model_dict = nsclc_models_dict.get(model_dict)

            # st.header("Clinical data only models")
            for model in selected_model_dict:
                with st.container(border=True):
                    # Divide the container into two columns: 1/3 button, 2/3 info
                    col1, col2 = st.columns([1, 2])

                    # Button in the first column (1/3)
                    if col1.button(model["name"]):
                        st.session_state['step'] = 'nsclc_results'
                        st.session_state['model'] = model
                        st.session_state['model_type'] = 'clinical' #TODO is this needed?
                        st.rerun()

                    # Info in the second column (2/3)
                    col2.write(model["info"])

# Show NSCLC Results
if st.session_state['step'] == 'nsclc_results':
    print(f"state: {st.session_state}")
    spec_model = st.session_state['model']
    st.sidebar.subheader("NSCLC Results")
    features_vals = {}
    if st.session_state['nsclc_model_type'] == 'nsclc_models_clinical':
        for feature in spec_model['features']:
            features_vals[feature] = st.session_state[feature]
        spec_model["function"](features_vals)
    elif st.session_state['nsclc_model_type'] == 'nsclc_models_image':
        if spec_model["name"] == "YOLOv8": 
            spec_model["function"](st.session_state['pil_image'], st.session_state['image_type'])
        else:
            spec_model["function"](st.session_state['cv_image'], st.session_state['image_type'])
    elif st.session_state['nsclc_model_type'] == 'nsclc_models_multimodal':
        for feature in spec_model['features']:
            features_vals[feature] = st.session_state[feature]
        spec_model["function"](features_vals, st.session_state['cv_image'], st.session_state['image_type'])
    if st.button("Close"):
        selected = "Home"
        st.session_state['step'] = 'Home'
        st.rerun()
