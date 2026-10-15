import joblib
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =====================================
# LOAD AI MODELS
# =====================================

anomaly_model = joblib.load(
    "anomaly_model.pkl"
)

lstm_model = load_model(
    "lstm_model.keras"
)

# =====================================
# CURRENT SENSOR VALUES
# (FROM MQTT/ESP32 LATER)
# =====================================

temperature = 85
gas = 20
vibration = 25
load = 95
distance = 20

# =====================================
# ANOMALY DETECTION
# =====================================

sample = pd.DataFrame(
    [[
        temperature,
        gas,
        vibration,
        load,
        distance
    ]],
    columns=[
        "temperature",
        "gas",
        "vibration",
        "load",
        "distance"
    ]
)

result = anomaly_model.predict(sample)

print()
print("====================================")
print("      INDUSTRIAL AI SYSTEM")
print("====================================")

print("\nCURRENT ANOMALIES:\n")

if result[0] == -1:

    if temperature > 60:
        print("✓ Temperature anomaly")

    if gas > 60:
        print("✓ Gas anomaly")

    if vibration > 20:
        print("✓ Vibration anomaly")

    if load > 90:
        print("✓ Load anomaly")

    if distance < 50:
        print("✓ Worker safety risk")

else:
    print("No anomaly detected")

# =====================================
# LSTM PREDICTION
# =====================================

data = pd.read_csv(
    "lstm_data.csv"
)

dataset = data[
    [
        "temperature",
        "vibration",
        "load"
    ]
].values

scaler = MinMaxScaler()

dataset = scaler.fit_transform(
    dataset
)

last_values = dataset[-10:]

X = np.array([last_values])

prediction = lstm_model.predict(
    X,
    verbose=0
)

prediction = scaler.inverse_transform(
    prediction
)

pred_temp = prediction[0][0]
pred_vib = prediction[0][1]
pred_load = prediction[0][2]

print("\n====================================")
print("     LSTM FAILURE PREDICTION")
print("====================================")

print(
    f"\nPredicted Temperature : "
    f"{pred_temp:.2f}°C"
)

print(
    f"Predicted Vibration  : "
    f"{pred_vib:.2f}"
)

print(
    f"Predicted Load       : "
    f"{pred_load:.2f}%"
)

# =====================================
# FAILURE PROBABILITY
# =====================================

probability = 0

if pred_temp > 70:
    probability += 40

if pred_vib > 20:
    probability += 35

if pred_load > 90:
    probability += 25

probability = min(
    probability,
    95
)

health = 100 - probability

time_to_failure = max(
    30 - probability//3,
    5
)

print(
    f"\nMachine Health Score : "
    f"{health}/100"
)

print(
    f"\nFailure Probability : "
    f"{probability}%"
)

print(
    f"\nEstimated Failure Time : "
    f"{time_to_failure} minutes"
)

# =====================================
# RISK LEVEL
# =====================================

if probability > 80:
    risk = "CRITICAL"

elif probability > 50:
    risk = "HIGH"

elif probability > 20:
    risk = "MEDIUM"

else:
    risk = "LOW"

print(
    f"\nRisk Level : {risk}"
)

# =====================================
# RECOMMENDATIONS
# =====================================

print("\nRecommended Actions:")

if pred_temp > 70:
    print(
        "• Reduce machine temperature"
    )

if pred_vib > 20:
    print(
        "• Inspect motor vibration"
    )

if pred_load > 90:
    print(
        "• Reduce machine load"
    )

if distance < 50:
    print(
        "• Move worker to safe distance"
    )

if probability > 80:
    print(
        "• Immediate maintenance required"
    )

print("\nFinal Status:")

if probability > 50:
    print(
        "⚠ MACHINE FAILURE LIKELY"
    )
else:
    print(
        "✓ MACHINE HEALTHY"
    )

print("\n====================================")