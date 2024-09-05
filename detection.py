import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'your_data.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Preprocess data
# Convert 'Date' to datetime and extract useful features
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Encode categorical features
le_country = LabelEncoder()
le_sex = LabelEncoder()
data['Country_Client'] = le_country.fit_transform(data['Country_Client'])
data['Sex_Client'] = le_sex.fit_transform(data['Sex_Client'])
data['Sex_RM'] = le_sex.fit_transform(data['Sex_RM'])
data['Nationality_RM'] = le_country.fit_transform(data['Nationality_RM'])

# Drop irrelevant columns for anomaly detection
features = data.drop(['Client ID', 'RM ID', 'Date'], axis=1)

# Scale numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(features_scaled)

# Predict anomalies
data['Anomaly'] = model.predict(features_scaled)

# -1 for anomalies, 1 for normal points
data['Anomaly'] = data['Anomaly'].map({-1: 'Anomaly', 1: 'Normal'})
