import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Title and description
st.title("Lung Cancer Prediction App")
st.write("This app predicts the likelihood of lung cancer based on user-provided data.")

# Sidebar inputs for user data
st.sidebar.header("User Input Features")
def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 50)
    smoking = st.sidebar.selectbox('Smoking (1 = Yes, 0 = No)', [1, 0])
    yellow_fingers = st.sidebar.selectbox('Yellow Fingers (1 = Yes, 0 = No)', [1, 0])
    anxiety = st.sidebar.selectbox('Anxiety (1 = Yes, 0 = No)', [1, 0])
    peer_pressure = st.sidebar.selectbox('Peer Pressure (1 = Yes, 0 = No)', [1, 0])
    chronic_disease = st.sidebar.selectbox('Chronic Disease (1 = Yes, 0 = No)', [1, 0])
    fatigue = st.sidebar.selectbox('Fatigue (1 = Yes, 0 = No)', [1, 0])
    allergy = st.sidebar.selectbox('Allergy (1 = Yes, 0 = No)', [1, 0])
    wheezing = st.sidebar.selectbox('Wheezing (1 = Yes, 0 = No)', [1, 0])
    alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption (1 = Yes, 0 = No)', [1, 0])
    coughing = st.sidebar.selectbox('Coughing (1 = Yes, 0 = No)', [1, 0])
    shortness_of_breath = st.sidebar.selectbox('Shortness of Breath (1 = Yes, 0 = No)', [1, 0])
    swallowing_difficulty = st.sidebar.selectbox('Swallowing Difficulty (1 = Yes, 0 = No)', [1, 0])
    chest_pain = st.sidebar.selectbox('Chest Pain (1 = Yes, 0 = No)', [1, 0])

    data = {
        'Age': age,
        'Smoking': smoking,
        'Yellow_Fingers': yellow_fingers,
        'Anxiety': anxiety,
        'Peer_Pressure': peer_pressure,
        'Chronic_Disease': chronic_disease,
        'Fatigue': fatigue,
        'Allergy': allergy,
        'Wheezing': wheezing,
        'Alcohol_Consumption': alcohol_consumption,
        'Coughing': coughing,
        'Shortness_of_Breath': shortness_of_breath,
        'Swallowing_Difficulty': swallowing_difficulty,
        'Chest_Pain': chest_pain
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.write("### User Input Features")
st.write(input_df)

# Load dataset (Replace 'lung_cancer_data.csv' with actual data file)
@st.cache
def load_data():
    # Simulated data loading for demonstration
    # Replace with actual data loading code
    data = pd.DataFrame({
        'Age': np.random.randint(20, 100, size=100),
        'Smoking': np.random.randint(0, 2, size=100),
        'Yellow_Fingers': np.random.randint(0, 2, size=100),
        'Anxiety': np.random.randint(0, 2, size=100),
        'Peer_Pressure': np.random.randint(0, 2, size=100),
        'Chronic_Disease': np.random.randint(0, 2, size=100),
        'Fatigue': np.random.randint(0, 2, size=100),
        'Allergy': np.random.randint(0, 2, size=100),
        'Wheezing': np.random.randint(0, 2, size=100),
        'Alcohol_Consumption': np.random.randint(0, 2, size=100),
        'Coughing': np.random.randint(0, 2, size=100),
        'Shortness_of_Breath': np.random.randint(0, 2, size=100),
        'Swallowing_Difficulty': np.random.randint(0, 2, size=100),
        'Chest_Pain': np.random.randint(0, 2, size=100),
        'Lung_Cancer': np.random.randint(0, 2, size=100)
    })
    return data

data = load_data()

# Splitting the data
X = data.drop('Lung_Cancer', axis=1)
y = data['Lung_Cancer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display the prediction
st.write("### Prediction")
lung_cancer_risk = 'High' if prediction[0] == 1 else 'Low'
st.write(f"Predicted Risk: {lung_cancer_risk}")

# Display prediction probabilities
st.write("### Prediction Probabilities")
st.write(f"Low Risk: {prediction_proba[0][0]:.2f}, High Risk: {prediction_proba[0][1]:.2f}")

# Model accuracy
st.write("### Model Performance")
y_pred = clf.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
#st.write("Classification Report:")
#st.text(classification_report(y_test, y_pred))
