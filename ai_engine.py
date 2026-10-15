# =========================
# IMPORTS
# =========================
import time
import numpy as np
import joblib
from collections import deque

from influxdb_client import InfluxDBClient
from tensorflow.keras.models import load_model

# =========================
# LOAD MODELS
# =========================
anomaly_model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")
lstm_model = load_model("lstm_model.keras")

# =========================
# INFLUXDB CONFIG
# =========================
url = "http://localhost:8086"

token = "INFLUXDB_TOKEN"

org = "organisation name"

bucket = "industrial_data"
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# =========================
# LSTM WINDOW
# =========================
window_size = 10
data_window = deque(maxlen=window_size)

# =========================
# FETCH DATA FROM INFLUXDB
# =========================
def fetch_latest_data():

    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -1m)
      |> filter(fn: (r) => r._measurement == "industry")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> last()
    '''

    result = query_api.query_data_frame(query)

    if result.empty:
        return None

    row = result.iloc[0]

    return np.array([
        row["temperature"],
        row["gas"],
        row["vibration"],
        row["load"],
        row["distance"]
    ])

# =========================
# PREPROCESS
# =========================
def preprocess(data):
    data = np.array(data).reshape(1, -1)
    return scaler.transform(data)[0]

# =========================
# ANOMALY DETECTION
# =========================
def detect_anomaly(data):
    return 1 if anomaly_model.predict([data])[0] == -1 else 0

# =========================
# LSTM PREDICTION
# =========================
def predict_lstm(window):
    X = np.array(window).reshape(1, window_size, 5)
    return lstm_model.predict(X)[0]

# =========================
# RISK ENGINE
# =========================
def calculate_risk(anomaly, prediction):

    temp, gas, vib, load, dist = prediction
    risk = 0

    if anomaly == 1:
        risk += 50
    if temp > 70:
        risk += 20
    if vib > 20:
        risk += 20
    if load > 90:
        risk += 10
    if dist < 50:
        risk += 10

    if risk < 30:
        return "LOW"
    elif risk < 70:
        return "MEDIUM"
    else:
        return "HIGH"

# =========================
# MAIN LOOP
# =========================
print("AI ENGINE STARTED...")

while True:

    data = fetch_latest_data()

    if data is None:
        print("Waiting for InfluxDB data...")
        time.sleep(2)
        continue

    scaled = preprocess(data)
    data_window.append(scaled)

    if len(data_window) < window_size:
        print("Collecting data window...")
        time.sleep(2)
        continue

    anomaly = detect_anomaly(scaled)
    prediction = predict_lstm(data_window)
    risk = calculate_risk(anomaly, prediction)

    print("\n========================")
    print("DATA:", data)
    print("ANOMALY:", anomaly)
    print("PREDICTION:", prediction)
    print("RISK:", risk)
    print("========================\n")

    time.sleep(2)