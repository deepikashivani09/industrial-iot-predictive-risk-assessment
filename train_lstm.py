import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

data = pd.read_csv("lstm_data.csv")

print(data.head())

# ==========================================
# SELECT FEATURES
# ==========================================

dataset = data[
    [
        "temperature",
        "vibration",
        "load"
    ]
].values

# ==========================================
# NORMALIZE DATA
# ==========================================

print("\nNormalizing data...")

scaler = MinMaxScaler()

dataset = scaler.fit_transform(dataset)

# Save scaler
joblib.dump(scaler, "lstm_scaler.pkl")

print("Scaler saved as lstm_scaler.pkl")

# ==========================================
# CREATE SEQUENCES
# ==========================================

sequence_length = 10

X = []
Y = []

for i in range(sequence_length, len(dataset)):

    X.append(dataset[i-sequence_length:i])

    Y.append(dataset[i])

X = np.array(X)
Y = np.array(Y)

print("\nTraining samples :", len(X))
print("Input shape      :", X.shape)
print("Output shape     :", Y.shape)

# ==========================================
# BUILD LSTM MODEL
# ==========================================

print("\nBuilding LSTM model...")

model = Sequential()

model.add(Input(shape=(sequence_length, 3)))

model.add(
    LSTM(
        units=64,
        activation="tanh",
        return_sequences=False
    )
)

model.add(Dense(32, activation="relu"))

model.add(Dense(3))

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining started...\n")

history = model.fit(
    X,
    Y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save("lstm_model.keras")

print("\n======================================")
print("LSTM MODEL TRAINED SUCCESSFULLY")
print("======================================")

print("Model saved  : lstm_model.keras")
print("Scaler saved : lstm_scaler.pkl")