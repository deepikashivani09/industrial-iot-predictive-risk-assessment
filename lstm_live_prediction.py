import pandas as pd
import numpy as np
import joblib

from influxdb_client import InfluxDBClient
from tensorflow.keras.models import load_model

# ======================================
# InfluxDB Configuration
# ======================================

url = "http://localhost:8086"

token = "6nhTDXteHfWnE_4k7Kx1XQ5PYiwza293nyYfqMw_qQdW3DBtl_GV0txHWf_KYsSM0juKjJWs8cQDeIHgxzBd_g=="

org = "ABC"

bucket = "industrial_data"

# ======================================
# Connect to InfluxDB
# ======================================

client = InfluxDBClient(
    url=url,
    token=token,
    org=org
)

query_api = client.query_api()

# ======================================
# Read Last 10 Readings
# ======================================

query = f'''
from(bucket: "{bucket}")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "industry")
  |> filter(fn: (r) =>
      r._field == "temperature" or
      r._field == "vibration" or
      r._field == "load")
  |> pivot(
      rowKey:["_time"],
      columnKey:["_field"],
      valueColumn:"_value"
  )
  |> sort(columns:["_time"])
  |> tail(n:10)
'''

df = query_api.query_data_frame(query)

client.close()

# If multiple tables are returned
if isinstance(df, list):
    df = pd.concat(df, ignore_index=True)

df = df[["temperature", "vibration", "load"]]

print("\n========== LAST 10 SENSOR READINGS ==========\n")
print(df)

# ======================================
# Load Scaler
# ======================================

scaler = joblib.load("lstm_scaler.pkl")

scaled = scaler.transform(df)

X = np.array([scaled])

# ======================================
# Load LSTM Model
# ======================================

model = load_model("lstm_model.keras")

prediction = model.predict(X, verbose=0)

prediction = scaler.inverse_transform(prediction)

pred_temp = prediction[0][0]
pred_vibration = prediction[0][1]
pred_load = prediction[0][2]



# ======================================
# Failure Probability
# ======================================

# =====================================
# MACHINE HEALTH CALCULATION
# =====================================

temp_score = min((pred_temp / 100) * 100, 100)
vib_score = min((pred_vibration / 25) * 100, 100)
load_score = min((pred_load / 100) * 100, 100)

failure_probability = (
    temp_score * 0.4 +
    vib_score * 0.35 +
    load_score * 0.25
)

failure_probability = int(min(failure_probability, 95))

health_score = 100 - failure_probability

if failure_probability < 30:
    failure_time = "30+ minutes"

elif failure_probability < 60:
    failure_time = "15-30 minutes"

elif failure_probability < 80:
    failure_time = "5-15 minutes"

else:
    failure_time = "<5 minutes"

print("\n======================================")
print("     LSTM FAILURE PREDICTION")
print("======================================")

print(f"\nPredicted Temperature : {pred_temp:.2f} °C")
print(f"Predicted Vibration  : {pred_vibration:.2f}")
print(f"Predicted Load       : {pred_load:.2f} %")

print(f"\nMachine Health Score : {health_score}/100")
print(f"Failure Probability  : {failure_probability}%")
print(f"Estimated Failure Time : {failure_time}")

if failure_probability >= 80:
    risk = "CRITICAL"
elif failure_probability >= 60:
    risk = "HIGH"
elif failure_probability >= 30:
    risk = "MEDIUM"
else:
    risk = "LOW"

print(f"Risk Level : {risk}")