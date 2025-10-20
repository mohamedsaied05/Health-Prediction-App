import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Page configuration
st.set_page_config(page_title="Health Prediction App",
                   layout="wide",
                   page_icon="❤️")

# -----------------------
# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# Load saved models
# Diabetes
diabetes_models = {
    'Logistic Regression': pickle.load(open(f'{working_dir}/saved_modelsD/model_1.pkl', 'rb')),
    'SVC': pickle.load(open(f'{working_dir}/saved_modelsD/model_2.pkl', 'rb')),
    'Random Forest': pickle.load(open(f'{working_dir}/saved_modelsD/model_3.pkl', 'rb')),
    'Gradient Boosting': pickle.load(open(f'{working_dir}/saved_modelsD/model_4.pkl', 'rb'))
}

# Heart Disease
heart_models = {
    'Logistic Regression': pickle.load(open(f'{working_dir}/saved_modelsH/model_1.pkl', 'rb')),
    'XGBoost': pickle.load(open(f'{working_dir}/saved_modelsH/model_2.pkl', 'rb')),
    'Decision Tree': pickle.load(open(f'{working_dir}/saved_modelsH/model_3.pkl', 'rb'))
}

# -----------------------
# Sidebar navigation
with st.sidebar:
    disease_choice = option_menu('Select Prediction Type',
                                 ['Diabetes', 'Heart Disease'],
                                 menu_icon='activity',
                                 icons=['123', 'heart'],
                                 default_index=0)

# -----------------------
st.title("Health Prediction using ML")

# -----------------------
# Function to preprocess heart disease input
def preprocess_heart_input(user_input):
    mapping = {
        'sex': {'M':1, 'F':0},
        'chest_pain': {'ATA':1, 'NAP':2, 'ASY':3},
        'fasting_bs': {'0':0, '1':1},
        'resting_ecg': {'Normal':0, 'ST':1},
        'exercise_angina': {'N':0, 'Y':1},
        'st_slope': {'Up':1, 'Flat':2, 'Down':3}
    }
    user_input[1] = mapping['sex'][user_input[1]]
    user_input[2] = mapping['chest_pain'][user_input[2]]
    user_input[5] = mapping['fasting_bs'][user_input[5]]
    user_input[6] = mapping['resting_ecg'][user_input[6]]
    user_input[8] = mapping['exercise_angina'][user_input[8]]
    user_input[10] = mapping['st_slope'][user_input[10]]
    return [float(x) for x in user_input]

# -----------------------
# Input forms
if disease_choice == "Diabetes":
    st.subheader("Enter Diabetes Patient Data")
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input('Number of Pregnancies')
    with col2: Glucose = st.text_input('Glucose Level')
    with col3: BloodPressure = st.text_input('Blood Pressure')
    with col1: SkinThickness = st.text_input('Skin Thickness')
    with col2: Insulin = st.text_input('Insulin Level')
    with col3: BMI = st.text_input('BMI')
    with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2: Age = st.text_input('Age')
    
    model_choice = st.selectbox("Select Model", list(diabetes_models.keys()))
    
    if st.button("Predict Diabetes"):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) for x in user_input]
            model = diabetes_models[model_choice]
            result = model.predict([user_input])
            
            if result[0]==1:
                st.success("The person is diabetic")
            else:
                st.success("The person is not diabetic")
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                feat_imp = pd.Series(model.feature_importances_, index=[
                    'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                    'BMI','DiabetesPedigreeFunction','Age']).sort_values(ascending=False)
            else:
                feat_imp = pd.Series(np.ones(8), index=[
                    'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                    'BMI','DiabetesPedigreeFunction','Age'])  # Dummy
            st.subheader("Feature Importance")
            st.bar_chart(feat_imp)
            
        except Exception as e:
            st.error(f"Error: {e}")

else:  # Heart Disease
    st.subheader("Enter Heart Disease Patient Data")
    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input('Age')
    with col2: sex = st.selectbox('Sex', ['M','F'])
    with col3: chest_pain = st.selectbox('Chest Pain Type', ['ATA','NAP','ASY'])
    with col1: resting_bp = st.text_input('Resting Blood Pressure')
    with col2: cholesterol = st.text_input('Cholesterol')
    with col3: fasting_bs = st.selectbox('Fasting Blood Sugar', ['0','1'])
    with col1: resting_ecg = st.selectbox('Resting ECG', ['Normal','ST'])
    with col2: max_hr = st.text_input('Max Heart Rate Achieved')
    with col3: exercise_angina = st.selectbox('Exercise Induced Angina', ['N','Y'])
    with col1: oldpeak = st.text_input('Oldpeak')
    with col2: st_slope = st.selectbox('ST Slope', ['Up','Flat','Down'])
    
    model_choice = st.selectbox("Select Model", list(heart_models.keys()))
    
    if st.button("Predict Heart Disease"):
        try:
            user_input = [age, sex, chest_pain, resting_bp, cholesterol,
                          fasting_bs, resting_ecg, max_hr, exercise_angina,
                          oldpeak, st_slope]
            user_input = preprocess_heart_input(user_input)
            model = heart_models[model_choice]
            result = model.predict([user_input])
            
            if result[0]==1:
                st.success("The person is likely to have heart disease")
            else:
                st.success("The person is unlikely to have heart disease")
            
            # Feature Importance
            feature_names = ['age','sex','chest_pain','resting_bp','cholesterol','fasting_bs',
                             'resting_ecg','max_hr','exercise_angina','oldpeak','st_slope']
            if hasattr(model, 'feature_importances_'):
                feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            else:
                feat_imp = pd.Series(np.ones(len(feature_names)), index=feature_names)  # Dummy
            st.subheader("Feature Importance")
            st.bar_chart(feat_imp)
            
        except Exception as e:
            st.error(f"Error: {e}")
