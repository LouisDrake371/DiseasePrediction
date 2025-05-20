import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("disease_model_svm.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

# Medication dictionary
medication_dict = {
    'flu': ['Oseltamivir', 'Paracetamol', 'Rest', 'Fluids'],
    'cold': ['Antihistamines', 'Decongestants', 'Vitamin C'],
    'dengue': ['Acetaminophen', 'Hydration', 'Rest'],
    'measles': ['Ibuprofen', 'Vitamin A', 'Rest'],
    'pneumonia': ['Antibiotics', 'Oxygen therapy', 'Hospitalization if severe'],
    'asthma': ['Inhaler', 'Bronchodilators', 'Steroids'],
    'covid-19': ['Paracetamol', 'Isolation', 'Fluids', 'Rest'],
    'food_poisoning': ['Oral rehydration', 'Activated charcoal', 'Antibiotics (if bacterial)']
}


# App title
st.title("🩺 Disease Diagnosis and Medication Suggestion")
st.markdown("Enter three symptoms below to get a predicted disease and medication suggestions.")

# Dropdowns for symptoms
symptoms = list(symptom_encoder.classes_)

symptom1 = st.selectbox("Symptom 1", symptoms)
symptom2 = st.selectbox("Symptom 2", symptoms)
symptom3 = st.selectbox("Symptom 3", symptoms)

if st.button("Predict Disease"):
    # Encode the symptoms
    encoded_s1 = symptom_encoder.transform([symptom1])[0]
    encoded_s2 = symptom_encoder.transform([symptom2])[0]
    encoded_s3 = symptom_encoder.transform([symptom3])[0]

    # Predict
    prediction = model.predict([[encoded_s1, encoded_s2, encoded_s3]])
    predicted_disease = disease_encoder.inverse_transform(prediction)[0]

    # Get medications
    medications = medication_dict.get(predicted_disease.lower(), ["Consult a doctor"])

    # Display results
    st.success(f"Predicted Disease: **{predicted_disease.upper()}**")
    st.markdown("### 💊 Suggested Medications:")
    for med in medications:
        st.markdown(f"- {med}")
