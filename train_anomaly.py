import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
data = pd.read_csv("industrial_ai_training_data.csv")

print("Dataset loaded successfully")

# Select features
X = data[
    [
        "temperature",
        "gas",
        "vibration",
        "load",
        "distance"
    ]
]

# Create model
model = IsolationForest(
    contamination=0.2,
    random_state=42
)

# Train model
model.fit(X)

print("Model trained successfully")

# Save model
joblib.dump(
    model,
    "anomaly_model.pkl"
)

print("Model saved as anomaly_model.pkl")