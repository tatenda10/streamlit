import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and preprocessor
model = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Define the input fields for user input
def user_input_features():
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', ['Male', 'Female'])  # Male: 1, Female: 0
    sex = 1 if sex == 'Male' else 0
    chest_pain_type = st.selectbox('Chest Pain Type', [1, 2, 3, 4])
    resting_bp_s = st.number_input('Resting Blood Pressure (systolic)', min_value=50, max_value=250, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])  # False: 0, True: 1
    fasting_blood_sugar = 1 if fasting_blood_sugar == 'True' else 0
    resting_ecg = st.selectbox('Resting ECG', [0, 1, 2])
    max_heart_rate = st.number_input('Max Heart Rate', min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])  # No: 0, Yes: 1
    exercise_angina = 1 if exercise_angina == 'Yes' else 0
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox('ST Slope', [1, 2, 3])
    
    data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_bp_s': resting_bp_s,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'resting_ecg': resting_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_angina': exercise_angina,
        'oldpeak': oldpeak,
        'st_slope': st_slope
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app
def main():
    st.title("Heart Disease Prediction Using Random Forest")
    
    # User input features
    input_df = user_input_features()
    
    # Display user inputs
    st.write("### User Input Features")
    st.write(input_df)
    
    # Add a Predict button
    if st.button('Predict'):
        # Preprocess the input data
        input_preprocessed = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_preprocessed)
        prediction_proba = model.predict_proba(input_preprocessed)
        
        # Display prediction
        st.write("### Prediction")
        if prediction[0] == 0:
            st.write("The model predicts that there is no heart disease.")
        else:
            st.write("The model predicts that there is heart disease.")
        
        # Display prediction probability
        st.write("### Prediction Probability")
        st.write(f"Probability of no heart disease: {prediction_proba[0][0]:.2f}")
        st.write(f"Probability of heart disease: {prediction_proba[0][1]:.2f}")

if __name__ == "__main__":
    main()
