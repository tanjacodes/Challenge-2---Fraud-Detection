import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load your data (replace 'data.csv' with the actual file path)
data = pd.read_csv('Dataframe_RiskOn_joined.csv')
data = data.drop(columns=['Client ID', 'RM ID', 'Date'])

# Handle missing or incorrect data
# One-hot encode categorical variables (Sex_Client, Country_Client, etc.)
data = pd.get_dummies(data, drop_first=True)

# Feature scaling (standardizing)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(X_scaled)

# Map results (1 for normal, -1 for anomaly)
data['Anomaly'] = data['Anomaly'].map({1: 'Normal', -1: 'Fraud'})

# Filter anomalies (transactions flagged as 'Fraud')
anomalies = data[data['Anomaly'] == 'Fraud']

# Calculate the percentage of fraud
total_rows = len(data)
fraud_count = len(anomalies)
fraud_percentage = (fraud_count / total_rows) * 100

print(f"Total Transactions: {total_rows}")
print(f"Fraudulent Transactions: {fraud_count}")
print(f"Percentage of Fraudulent Transactions: {fraud_percentage:.2f}%")

# Optional: Save anomalies to a CSV file
anomalies.to_csv('fraud_detection_anomalies.csv', index=False)
