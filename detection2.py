import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np

# Load your data
data = pd.read_csv('Dataframe_RiskOn_joined.csv')

# Drop columns that are not useful for modeling
# Here, we'll drop identifiers and columns like Date, which needs special handling
data = data.drop(columns=['Client ID', 'RM ID', 'Date'])

# Handling missing or incorrect data (fixing any date formats if needed)
# For this example, assuming "45272" and "45541" are erroneous and converted to actual dates or filled with median date

# Encode categorical columns (Sex_Client, Country_Client, etc.)
data = pd.get_dummies(data, drop_first=True)

# Feature scaling (standardizing)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(X_scaled)

# Map results (1 for normal, -1 for anomaly)
data['Anomaly'] = data['Anomaly'].map({1: 'Normal', -1: 'Fraud'})

# Filter and print anomalies (transactions flagged as 'Fraud')
anomalies = data[data['Anomaly'] == 'Fraud']
print("Anomalies (Flagged as Fraud):")
print(anomalies)

# Optional: Save anomalies to a CSV file
anomalies.to_csv('fraud_detection_anomalies.csv', index=False)
