import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 2))  # Normal transactions
fraud_data = np.random.normal(loc=5, scale=1, size=(50, 2))   # Fraudulent transactions
data = np.vstack([normal_data, fraud_data])
labels = np.array([1]*1000 + [-1]*50)  # 1 for normal, -1 for fraud

# Create a DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
df['Label'] = labels

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[['Feature1', 'Feature2']])

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies
df['Predicted'] = model.predict(X)
df['Predicted'] = df['Predicted'].map({1: 'Normal', -1: 'Fraud'})

# Plotting results
plt.figure(figsize=(10, 6))
plt.scatter(df['Feature1'], df['Feature2'], c=df['Predicted'].apply(lambda x: 1 if x == 'Fraud' else 0), cmap='coolwarm', label='Anomaly')
plt.title('Fraud Detection using Isolation Forest')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend(['Fraud', 'Normal'])
plt.colorbar(label='Anomaly')
plt.show()

# Print some results
print(df.head(10))
