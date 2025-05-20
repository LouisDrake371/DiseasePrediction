import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import joblib

# Sample data
data = {
    'Symptom1': ['fever', 'headache', 'fever', 'sore_throat', 'fatigue', 'chest_pain', 'fever', 'nausea'],
    'Symptom2': ['cough', 'nausea', 'rash', 'fever', 'fatigue', 'breathing_difficulty', 'sore_throat', 'vomiting'],
    'Symptom3': ['fatigue', 'vomiting', 'itching', 'runny_nose', 'shortness_of_breath', 'dizziness', 'chills', 'diarrhea'],
    'Disease': ['flu', 'dengue', 'measles', 'cold', 'pneumonia', 'asthma', 'covid-19', 'food_poisoning']
}


df = pd.DataFrame(data)

# Create a symptom encoder
symptom_encoder = LabelEncoder()

# Fit on all symptom values combined
all_symptoms = pd.concat([df['Symptom1'], df['Symptom2'], df['Symptom3']])
symptom_encoder.fit(all_symptoms)


# Encode symptoms
df['Symptom1'] = symptom_encoder.transform(df['Symptom1'])
df['Symptom2'] = symptom_encoder.transform(df['Symptom2'])
df['Symptom3'] = symptom_encoder.transform(df['Symptom3'])

# Encode diseases separately
disease_encoder = LabelEncoder()
df['Disease'] = disease_encoder.fit_transform(df['Disease'])

# Split features and labels
X = df[['Symptom1', 'Symptom2', 'Symptom3']]
y = df['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save everything
joblib.dump(model, "disease_model_svm.pkl")
joblib.dump(symptom_encoder, "symptom_encoder.pkl")
joblib.dump(disease_encoder, "disease_encoder.pkl")

print("✅ Model and encoders saved!")
