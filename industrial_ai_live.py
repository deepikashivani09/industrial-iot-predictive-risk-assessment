import joblib
import pandas as pd
import numpy as np

from influxdb_client import InfluxDBClient
from tensorflow.keras.models import load_model

# ==========================================
# INFLUXDB CONFIGURATION
# ==========================================

url = "http://localhost:8086"

token = "INFLUXDB_TOKEN"

org = "organisation name"

bucket = "industrial_data"

# ==========================================
# CONNECT TO INFLUXDB
# ==========================================

client = InfluxDBClient(
    url=url,
    token=token,
    org=org
)

query_api = client.query_api()

# ==========================================
# READ LATEST SENSOR VALUES
# ==========================================

query = f'''
from(bucket: "{bucket}")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "industry")
  |> last()
'''

tables = query_api.query(query)

sensor = {}

for table in tables:
    for record in table.records:
        sensor[record.get_field()] = record.get_value()

if len(sensor) == 0:
    print("No live sensor data found.")
    exit()

temperature = sensor["temperature"]
humidity = sensor["humidity"]
gas = sensor["gas"]
vibration = sensor["vibration"]
load = sensor["load"]
distance = sensor["distance"]

print("\n=========================================")
print("          INDUSTRIAL AI SYSTEM")
print("=========================================")

print("\nLIVE SENSOR DATA\n")

print(f"Temperature : {temperature:.2f} °C")
print(f"Humidity    : {humidity:.2f} %")
print(f"Gas         : {gas:.2f}")
print(f"Vibration   : {vibration:.2f}")
print(f"Load        : {load:.2f} %")
print(f"Distance    : {distance:.2f} cm")

# ==========================================
# ISOLATION FOREST
# ==========================================

anomaly_model = joblib.load("anomaly_model.pkl")

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

prediction = anomaly_model.predict(sample)

print("\n=========================================")
print("CURRENT ANOMALY DETECTION")
print("=========================================\n")

if prediction[0] == -1:

    print("⚠ ANOMALY DETECTED\n")

    anomaly_found = False

    if temperature > 60:
        print(f"• Temperature anomaly ({temperature:.1f} °C)")
        anomaly_found = True

    if gas > 70:
        print(f"• Gas anomaly ({gas:.1f})")
        anomaly_found = True

    if vibration > 15:
        print(f"• Vibration anomaly ({vibration:.2f})")
        anomaly_found = True

    if load > 80:
        print(f"• Load anomaly ({load:.1f} %)")
        anomaly_found = True

    if distance < 50:
        print(f"• Worker safety risk ({distance:.1f} cm)")
        anomaly_found = True

    if not anomaly_found:
        print("• Abnormal operating pattern detected.")

else:

    print("✓ Machine operating normally")

# ==========================================
# READ LAST 10 READINGS FOR LSTM
# ==========================================

query = f'''
from(bucket: "{bucket}")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "industry")
  |> filter(fn:(r)=>
      r._field=="temperature" or
      r._field=="vibration" or
      r._field=="load")
  |> pivot(
      rowKey:["_time"],
      columnKey:["_field"],
      valueColumn:"_value")
  |> sort(columns:["_time"])
  |> tail(n:10)
'''

df = query_api.query_data_frame(query)

client.close()

if isinstance(df, list):
    df = pd.concat(df, ignore_index=True)

df = df[["temperature", "vibration", "load"]]

# ==========================================
# LOAD SCALER
# ==========================================

scaler = joblib.load("lstm_scaler.pkl")

scaled = scaler.transform(df)

X = np.array([scaled])

# ==========================================
# LOAD LSTM MODEL
# ==========================================

lstm_model = load_model("lstm_model.keras")

prediction_lstm = lstm_model.predict(X, verbose=0)

prediction_lstm = scaler.inverse_transform(prediction_lstm)

pred_temp = prediction_lstm[0][0]
pred_vibration = prediction_lstm[0][1]
pred_load = prediction_lstm[0][2]

print("\n=========================================")
print("LSTM FAILURE PREDICTION")
print("=========================================\n")

print(f"Predicted Temperature : {pred_temp:.2f} °C")
print(f"Predicted Vibration  : {pred_vibration:.2f}")
print(f"Predicted Load       : {pred_load:.2f} %")
# ==========================================
# TREND ANALYSIS
# ==========================================

# ==========================================
# TREND ANALYSIS USING LAST 10 READINGS
# ==========================================

print("\n=========================================")
print("TREND ANALYSIS")
print("=========================================\n")

recommendations = []

# Get last 10 readings
temp_values = df["temperature"].values
vib_values = df["vibration"].values
load_values = df["load"].values

# Calculate trend (slope)
temp_slope = np.polyfit(range(len(temp_values)), temp_values, 1)[0]
vib_slope = np.polyfit(range(len(vib_values)), vib_values, 1)[0]
load_slope = np.polyfit(range(len(load_values)), load_values, 1)[0]

# ==========================================
# TEMPERATURE
# ==========================================

if temp_slope > 0.2 and pred_temp > temperature:

    print("Temperature is increasing continuously.")
    print("Prediction : Possible overheating.\n")

    recommendations.append("Reduce machine temperature")

# ==========================================
# VIBRATION
# ==========================================

if vib_slope > 0.10 and pred_vibration > vibration:

    print("Vibration is increasing continuously.")
    print("Prediction : Possible bearing or motor failure.\n")

    recommendations.append("Inspect motor bearings")

# ==========================================
# LOAD
# ==========================================

if load_slope > 0.5 and pred_load > load:

    print("Load is increasing continuously.")
    print("Prediction : Possible overload condition.\n")

    recommendations.append("Reduce machine load")

# ==========================================
# GAS
# ==========================================

if gas > 70:

    print("Gas concentration is increasing.")
    print("Prediction : Possible gas leakage.\n")

    recommendations.append("Check gas leakage")

# ==========================================
# WORKER SAFETY
# ==========================================

if distance < 50:

    print("Worker is too close to the machine.")
    print("Prediction : Worker safety risk.\n")

    recommendations.append("Move worker to a safe distance")
# ==========================================
# MACHINE HEALTH
# ==========================================

temp_score = min((pred_temp / 100) * 100, 100)
vib_score = min((pred_vibration / 25) * 100, 100)
load_score = min((pred_load / 100) * 100, 100)

failure_probability = int(
    temp_score * 0.40 +
    vib_score * 0.35 +
    load_score * 0.25
)

failure_probability = min(failure_probability, 95)

health_score = 100 - failure_probability

# ==========================================
# FAILURE TIME
# ==========================================

if failure_probability < 30:
    failure_time = "30+ minutes"

elif failure_probability < 60:
    failure_time = "15-30 minutes"

elif failure_probability < 80:
    failure_time = "5-15 minutes"

else:
    failure_time = "<5 minutes"

# ==========================================
# RISK LEVEL
# ==========================================

if failure_probability >= 80:
    risk = "CRITICAL"

elif failure_probability >= 60:
    risk = "HIGH"

elif failure_probability >= 30:
    risk = "MEDIUM"

else:
    risk = "LOW"

# ==========================================
# RECOMMENDATIONS
# ==========================================

if failure_probability >= 80:
    recommendations.append("Immediate maintenance required")

elif failure_probability >= 60:
    recommendations.append("Schedule maintenance immediately")

elif failure_probability >= 30:
    recommendations.append("Continue monitoring machine")

# Remove duplicate recommendations
recommendations = list(dict.fromkeys(recommendations))

# ==========================================
# DISPLAY AI RESULT
# ==========================================

print("\n=========================================")
print("AI HEALTH REPORT")
print("=========================================\n")

print(f"Predicted Temperature : {pred_temp:.2f} °C")
print(f"Predicted Vibration  : {pred_vibration:.2f}")
print(f"Predicted Load       : {pred_load:.2f} %")

print(f"\nMachine Health Score : {health_score}/100")
print(f"Failure Probability  : {failure_probability}%")
print(f"Estimated Failure Time : {failure_time}")
print(f"Risk Level : {risk}")

print("\nRecommended Actions:")

if len(recommendations) == 0:
    print("• Machine operating normally")

else:
    for item in recommendations:
        print(f"• {item}")

# ==========================================
# FINAL STATUS
# ==========================================

print("\n=========================================")
print("FINAL STATUS")
print("=========================================\n")

if prediction[0] == -1 and risk in ["HIGH", "CRITICAL"]:
    final_status = "⚠ MACHINE FAILURE LIKELY"

elif prediction[0] == -1:
    final_status = "⚠ ANOMALY DETECTED - CHECK MACHINE"

elif risk in ["HIGH", "CRITICAL"]:
    final_status = "⚠ FAILURE POSSIBLE SOON"

elif risk == "MEDIUM":
    final_status = "⚠ MACHINE NEEDS MONITORING"

else:
    final_status = "✓ MACHINE OPERATING NORMALLY"

print(final_status)

# ==========================================
# SAVE AI RESULTS TO INFLUXDB
# ==========================================

from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(
    url=url,
    token=token,
    org=org
)

write_api = client.write_api(write_options=SYNCHRONOUS)

point = (
    Point("ai_results")
    .field("predicted_temperature", float(pred_temp))
    .field("predicted_vibration", float(pred_vibration))
    .field("predicted_load", float(pred_load))
    .field("health_score", int(health_score))
    .field("failure_probability", int(failure_probability))
    .field("risk_level", risk)
    .field("failure_time", failure_time)
    .field("current_temperature", float(temperature))
    .field("current_vibration", float(vibration))
    .field("current_load", float(load))
    .field("current_gas", float(gas))
    .field("worker_distance", float(distance))
    .field("status", final_status)
)

write_api.write(
    bucket=bucket,
    org=org,
    record=point
)

client.close()

print("\n=========================================")
print("AI RESULTS SAVED TO INFLUXDB")
print("=========================================")