import joblib

# Load model and encoders
model = joblib.load("disease_model.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

# Get valid symptom list from encoder
valid_symptoms = list(symptom_encoder.classes_)

print("🤖 Disease Predictor")
print("Enter 3 symptoms (e.g., fever, cough, fatigue)")
print("Available symptoms:", valid_symptoms)

# Get input
s1 = input("Symptom 1: ").strip().lower()
s2 = input("Symptom 2: ").strip().lower()
s3 = input("Symptom 3: ").strip().lower()

# Check for validity
for s in [s1, s2, s3]:
    if s not in valid_symptoms:
        print(f"❌ Invalid symptom: {s}")
        exit()

# Encode symptoms
encoded_s1 = symptom_encoder.transform([s1])[0]
encoded_s2 = symptom_encoder.transform([s2])[0]
encoded_s3 = symptom_encoder.transform([s3])[0]

# Predict
prediction = model.predict([[encoded_s1, encoded_s2, encoded_s3]])
predicted_disease = disease_encoder.inverse_transform(prediction)[0]

# Medication suggestion dictionary
medication_dict = {
    'flu': ['Oseltamivir', 'Paracetamol', 'Rest', 'Fluids'],
    'cold': ['Antihistamines', 'Decongestants', 'Vitamin C'],
    'dengue': ['Acetaminophen', 'Hydration', 'Rest'],
    'measles': ['Ibuprofen', 'Vitamin A', 'Rest']
}

# Get medications
medications = medication_dict.get(predicted_disease.lower(), ["Consult a doctor"])

# Display results
print(f"\n✅ You may have: **{predicted_disease.upper()}**")
print("💊 Suggested medications or care:")
for med in medications:
    print(f" - {med}")

