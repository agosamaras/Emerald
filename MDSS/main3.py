import streamlit as st
import pandas as pd
import numpy as np
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
    "Overweight": "",
    "Obese": "",
    "normal_weight":"",
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
    "Doctor: Healthy": None,
}

nsclc_features_dict = {
    "Age": "What is the age of the patient?",
    "BMI": "What is the Body Mass Index of the patient?",
    "GLU": "What is the GLU measurement of the patient?",
    "SUV": "What is the SUVmax measurement of the patient?",
    "Diameter": "What is the diameter of the SPN?",
    "Location": "What is the location of the SPN?",
    "Type": "What is the type of the SPN?",
    "Limits": "How would you define the limits of the SPN?",
    "Gender": "Is the patient male?"
}

nsclc_mandatory_features = ["SUV", "Diameter"]

weight_cats = ["Overweight","Obese","normal_weight"]
age_cats = ["u40","40b50","50b60","o60"]

def cad_pred_print(pred):
    if not pred:
        st.header(f"Results for CAD: :green[Healthy]")
    else:
        st.header(f"Results for CAD: :red[CAD]")

def nsclc_pred_print(pred, score=None):
    if not pred:
        st.header(f"Results for NSCLC: :green[Benign]")
    else:
        st.header(f"Results for NSCLC: :red[Malignant]")
    if score:
        st.subheader(f"Confidence score: {score:.2f}")

def show_random_forest_results(feature_values):
    data = pd.DataFrame([feature_values])
    dataframe = pd.DataFrame(data.values, columns=data.columns)
    # Load the saved Random Forest model
    loaded_model = joblib.load('trained_models/RandomForestClassifier.joblib')
    model_prediction = loaded_model.predict(dataframe)    
    # Display the prediction
    cad_pred_print(model_prediction[0])
    st.subheader(f"Results Explanation")
    # code for SHAP visualization
    tree_xai(loaded_model, dataframe, model_prediction[0])

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
    # print(f"prediction: {prediction}")
    # print(f"confidence_score: {conf_score}")
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
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
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
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

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
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
###

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
            "name": "Support Vector Machine",
            "function": show_catboost_results, #TODO
            "info": "Catboost does not require the Doctor's yield as input. This pretrained model achieves a reported accuracy of 78.82%.",
            "features": ['known CAD', 'previous PCI', 'previous CABG', 'previous STROKE', 'Diabetes', 'Smoking', 'Angiopathy', 
                        'Chronic Kindey Disease', 'ANGINA LIKE', 'INCIDENT OF PRECORDIAL PAIN', 'RST ECG', 'male', 'Obese', '40b50', '50b60']
        },
    ],
    'cad_models_image': [
        {
            "name": "Random Forest",
            "function": show_random_forest_results,
            "info": "Random Forest is expecting Doctor's yield as input. This pretrained model has a reported accuracy of 83.52%.",
            "features": ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease', 'ANGINA LIKE', 'RST ECG', 'male', 'u40', 'Doctor: Healthy']
        },
    ],
    'cad_models_multimodal': [
        {
            "name": "Random Forest",
            "function": show_random_forest_results,
            "info": "Random Forest is expecting Doctor's yield as input. This pretrained model has a reported accuracy of 83.52%.",
            "features": ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease', 'ANGINA LIKE', 'RST ECG', 'male', 'u40', 'Doctor: Healthy']
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
    ],
    'nsclc_models_image': [
        {
            "name": "YOLOv8",
            "function": yolo_results,
            "info": "You Only Look Once (YOLO) version 8 can be applied to CAD or PET medical scans. This pretrained model has a reported accuracy of 89% for PET images and 92.3% for CT.",
        },
    ],
    'nsclc_multimodal': [
    ]
}

# st.session_state[]
if st.session_state.get('step') is None:
    st.session_state['step'] = 'home'
if st.session_state.get('model') is None:
    st.session_state['model'] = ''
if st.session_state.get('cad_model_type') is None:
    st.session_state['cad_model_type'] = ''
if st.session_state.get('cad_model_info') is None:
    st.session_state['cad_model_info'] = ''
if st.session_state.get('cad_patient_info') is None:
    st.session_state['cad_patient_info'] = ''
if st.session_state.get('nsclc_model_info') is None:
    st.session_state['nsclc_model_info'] = ''
if st.session_state.get('nsclc_patient_info') is None:
    st.session_state['nsclc_patient_info'] = ''
if st.session_state.get('age') is None:
    st.session_state['age'] = 0
if st.session_state.get('weight') is None:
    st.session_state['weight'] = 0.0
if st.session_state.get('warning') is None:
    st.session_state['warning'] = []
# if st.session_state.get('test') is None:
#     st.session_state['test'] = False

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

# function to print SHAP values and plots for tree ML models
def tree_xai(model, X, val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sv = explainer(X)
    exp = shap.Explanation(sv[:, :, 1], sv.base_values[:, 1], X, feature_names=X.columns)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Show waterfall plot for idx_cad
    fig_cad, _ = plt.subplots()
    shap.waterfall_plot(exp[0])
    st.pyplot(fig_cad)

    # Show force plot for class 0 (Healthy) here input prediction
    fig_force_0, _ = plt.subplots()
    fig_force_0 = shap.force_plot(val, shap_values[0][0], X.iloc[0, :], matplotlib=True)
    st.pyplot(fig_force_0)

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
        options=["Home", "CAD", "NSCLC", "Contact Us"],
        icons=["house", "heart", "lungs", "envelope"],
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
    st.sidebar.write("Select a CAD model:")
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

            col1, col2, col3 = st.columns([2, 6, 1])
            if col1.form_submit_button("Submit"):
                st.session_state['cad_patient_info'] = True
                print(f"state: {st.session_state}")
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
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        # Select image type (PET or CT)
        st.session_state['image_type'] = st.selectbox("Select image type:", ["PET", "CT"])
        if uploaded_img:
            # Convert the uploaded image to PIL Image
            pil_image = Image.open(uploaded_img)
            st.session_state['image'] = pil_image

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
                st.session_state['cad_model_info'] = True
                st.rerun()

    with tab3:
        st.header("Select a model From the available list")
        if any((
            st.session_state['cad_model_type'] is None,
            st.session_state['cad_model_info'] is not True,
            st.session_state['cad_patient_info'] is not True
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
                if 'Doctor: Healthy' in model["features"] and st.session_state["Doctor: Healthy"] is None:
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
                        st.session_state['model_type'] = 'clinical'
                        st.rerun()

                    # Info in the second column (2/3)
                    col2.write(model["info"])

# Show CAD Results
if st.session_state['step'] == 'cad_results':
    spec_model = st.session_state['model']
    st.sidebar.subheader("CAD Results")
    if st.session_state['model_type'] == 'clinical':
        features_vals = {}
        for feature in spec_model['features']:
            features_vals[feature] = st.session_state[feature]
        spec_model["function"](features_vals)
    elif st.session_state['model_type'] == 'images':
        st.markdown("ZONK!")
    elif st.session_state['model_type'] == 'multimodal':
        st.markdown("ZONK!")
    if st.button("Close"):
        selected = "Home"
        st.session_state['step'] = 'Home'
        st.rerun()

# Process NSCLC Prediction
if st.session_state['step'] == 'nsclc':
    st.sidebar.write("Select a NSCLC model:")
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
                    # val = st.checkbox(feature, value=st.session_state[feature])
                    val = st.number_input(feature, value=None, placeholder="Insert a value...", min_value=0.1, step=0.1, format="%g")
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
                
                print([feature for feature in nsclc_mandatory_features])
                if any( st.session_state[feature] == 0 for feature in nsclc_mandatory_features):
                    for feature in nsclc_mandatory_features:
                        if st.session_state[feature] == 0:
                            st.warning(f'{feature} cannot be empty', icon="❌")
                            print(f"yes!: {feature} | {st.session_state[feature]}")
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
            st.session_state['image'] = pil_image

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
                st.session_state['nsclc_model_info'] = True
                st.rerun()

    with tab3:
        st.header("Select a model From the available list")
        if any((
            st.session_state['nsclc_model_type'] is None,
            st.session_state['nsclc_model_info'] is not True,
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
                        st.session_state['model_type'] = 'clinical'
                        st.rerun()

                    # Info in the second column (2/3)
                    col2.write(model["info"])

# Show NSCLC Results
if st.session_state['step'] == 'nsclc_results':
    spec_model = st.session_state['model']
    st.sidebar.subheader("NSCLC Results")
    if st.session_state['nsclc_model_type'] == 'nsclc_models_clinical':
        features_vals = {}
        for feature in spec_model['features']:
            features_vals[feature] = st.session_state[feature]
        print(f"features_vals: {features_vals}")
        spec_model["function"](features_vals)
    elif st.session_state['nsclc_model_type'] == 'nsclc_models_image':
        spec_model["function"](st.session_state['image'], st.session_state['image_type'])
    elif st.session_state['nsclc_model_type'] == 'nsclc_models_multimodal':
        st.markdown("ZONK!")
    if st.button("Close"):
        selected = "Home"
        st.session_state['step'] = 'Home'
        st.rerun()
