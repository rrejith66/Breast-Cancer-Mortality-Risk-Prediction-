# Breast Cancer 10-Year Mortality Risk Prediction - Streamlit App

import streamlit as st
import pandas as pd
import joblib

# Load models and transformers
logistic_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
selected_columns = joblib.load('selected_columns.pkl')

# Define features based on model training
numerical_features = [
    'Age at Diagnosis', 'Lymph nodes examined positive', 'Tumor Size',
    'Nottingham prognostic index', 'Mutation Count'
]
ordinal_features = ['Tumor Stage', 'Neoplasm Histologic Grade']
nominal_features = [
    'Type of Breast Surgery', 'Cellularity', 'Chemotherapy',
    'Pam50 + Claudin-low subtype', 'ER Status', 'PR Status', 'HER2 Status',
    'Hormone Therapy', 'Radio Therapy', 'Inferred Menopausal State',
    '3-Gene classifier subtype'
]

# Dummy columns used during training (get from Logistic or RF model input)
example_df = pd.DataFrame(columns=selected_columns)
nominal_dummy_columns = [col for col in example_df.columns if col not in numerical_features + ordinal_features]

# Prediction function
def predict_mortality_risk(patient_data, model_name='Logistic Regression'):
    patient_df = pd.DataFrame([patient_data])

    # Scale numerical
    patient_df[numerical_features] = scaler.transform(patient_df[numerical_features])

    # Encode ordinal
    for col in ordinal_features:
        le = label_encoders[col]
        val = str(patient_df[col].values[0])
        if val in le.classes_:
            patient_df[col] = le.transform([val])
        else:
        # Assign a default known value (e.g., mode or most frequent label)
            default_val = le.classes_[0]  # Or choose based on domain knowledge
            patient_df[col] = le.transform([default_val])


    # Encode nominal
    patient_encoded = pd.get_dummies(patient_df[nominal_features], drop_first=True)
    for col in nominal_dummy_columns:
        if col not in patient_encoded:
            patient_encoded[col] = 0

    # Final input
    final_input = pd.concat([
        patient_df[numerical_features + ordinal_features], patient_encoded
    ], axis=1)
    final_input = final_input.reindex(columns=selected_columns, fill_value=0)

    # Model prediction
    model = logistic_model if model_name == 'Logistic Regression' else rf_model
    prob = model.predict_proba(final_input)[0][1]
    risk = 'Low' if prob < 0.3 else 'Medium' if prob < 0.7 else 'High'

    return prob, risk

# Streamlit App UI
st.title("Breast Cancer 10-Year Mortality Risk Prediction")
st.write("Provide patient information to estimate the risk of 10-year mortality.")

with st.form("patient_form"):
    age = st.slider("Age at Diagnosis", 20, 100, 50)
    lymph_nodes = st.number_input("Lymph Nodes Examined Positive", 0, 30, 1)
    tumor_size = st.number_input("Tumor Size (mm)", 1, 200, 25)
    npi = st.number_input("Nottingham Prognostic Index", 0.0, 10.0, 4.5)
    mutation_count = st.number_input("Mutation Count", 0, 100, 20)

    tumor_stage = st.selectbox("Tumor Stage", ['1', '2', '3', '4'])
    grade = st.selectbox("Neoplasm Histologic Grade", ['1', '2', '3'])

    surgery = st.selectbox("Type of Breast Surgery", ['BREAST CONSERVING', 'MASTECTOMY'])
    chemo = st.selectbox("Chemotherapy", ['YES', 'NO'])
    cellularity = st.selectbox("Cellularity", ['Low', 'Moderate', 'High'])
    er_status = st.selectbox("ER Status", ['Positive', 'Negative'])
    pr_status = st.selectbox("PR Status", ['Positive', 'Negative'])
    her2 = st.selectbox("HER2 Status", ['Positive', 'Negative'])
    hormone = st.selectbox("Hormone Therapy", ['YES', 'NO'])
    menopause = st.selectbox("Inferred Menopausal State", ['Pre', 'Post'])
    subtype = st.selectbox("Pam50 + Claudin-low subtype", ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal'])
    radio = st.selectbox("Radio Therapy", ['YES', 'NO'])
    gene3 = st.selectbox("3-Gene classifier subtype", ['ER-/HER2-', 'ER+/HER2-', 'HER2+'])

    model_choice = st.radio("Select Prediction Model", ['Logistic Regression', 'Random Forest'])
    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'Age at Diagnosis': age,
        'Lymph nodes examined positive': lymph_nodes,
        'Tumor Size': tumor_size,
        'Nottingham prognostic index': npi,
        'Mutation Count': mutation_count,
        'Tumor Stage': tumor_stage,
        'Neoplasm Histologic Grade': grade,
        'Type of Breast Surgery': surgery,
        'Chemotherapy': chemo,
        'Cellularity': cellularity,
        'ER Status': er_status,
        'PR Status': pr_status,
        'HER2 Status': her2,
        'Hormone Therapy': hormone,
        'Radio Therapy': radio,
        'Inferred Menopausal State': menopause,
        'Pam50 + Claudin-low subtype': subtype,
        '3-Gene classifier subtype': gene3
    }

    prob, risk = predict_mortality_risk(input_data, model_choice)
    st.markdown(f"### Predicted Risk: **{risk}**")
    st.markdown(f"Probability of 10-Year Mortality: **{prob:.1%}**")
