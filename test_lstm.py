import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==================================
# LOAD DATA
# ==================================

data = pd.read_csv("lstm_data.csv")

dataset = data[
    [
        "temperature",
        "vibration",
        "load"
    ]
].values

# ==================================
# NORMALIZE
# ==================================

scaler = MinMaxScaler()

dataset = scaler.fit_transform(
    dataset
)

# ==================================
# LOAD LSTM
# ==================================

model = load_model(
    "lstm_model.keras"
)

# ==================================
# LAST 10 READINGS
# ==================================

last_values = dataset[-10:]

X = np.array([last_values])

# ==================================
# PREDICT
# ==================================

prediction = model.predict(X)

prediction = scaler.inverse_transform(
    prediction
)

temp = prediction[0][0]
vib = prediction[0][1]
load = prediction[0][2]

print("\n========== LSTM PREDICTION ==========\n")

print(
    "Predicted Temperature:",
    round(temp,2),
    "°C"
)

print(
    "Predicted Vibration:",
    round(vib,2)
)

print(
    "Predicted Load:",
    round(load,2),
    "%"
)

# ==================================
# FAILURE PROBABILITY
# ==================================

probability = 0

if temp > 70:
    probability += 40

if vib > 20:
    probability += 35

if load > 90:
    probability += 25

probability = min(
    probability,
    95
)

time_to_failure = max(
    30 - probability//3,
    5
)

print()

print(
    "Failure Probability:",
    probability,
    "%"
)

print(
    "Estimated Failure Time:",
    time_to_failure,
    "minutes"
)

print("\n===================================")