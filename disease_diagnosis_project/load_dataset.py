import pandas as pd

# Simulated dataset
data = {
    'Symptom1': ['fever', 'headache', 'fever', 'sore_throat'],
    'Symptom2': ['cough', 'nausea', 'rash', 'fever'],
    'Symptom3': ['fatigue', 'vomiting', 'itching', 'runny_nose'],
    'Disease': ['flu', 'dengue', 'measles', 'cold']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Show dataset
print("Sample Dataset:\n")
print(df)
