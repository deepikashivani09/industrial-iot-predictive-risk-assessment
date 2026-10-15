import joblib
import pandas as pd

# ===================================
# LOAD TRAINED MODEL
# ===================================

model = joblib.load("anomaly_model.pkl")

# ===================================
# TEST SENSOR VALUES
# CHANGE THESE VALUES TO TEST
# ===================================

temperature = 85      # °C
gas = 20              # %
vibration = 25        # vibration value
load = 95             # %
distance = 20         # cm

# ===================================
# CREATE DATAFRAME
# ===================================

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

# ===================================
# PREDICT ANOMALY
# ===================================

result = model.predict(sample)

# ===================================
# DISPLAY RESULT
# ===================================

if result[0] == -1:

    print("\n================================")
    print("        AI ALERT SYSTEM")
    print("================================")

    anomaly_found = False

    # Temperature
    if temperature > 60:
        print(
            f"Temperature anomaly detected "
            f"(Current value = {temperature}°C)"
        )
        anomaly_found = True

    # Gas
    if gas > 60:
        print(
            f"Gas anomaly detected "
            f"(Current value = {gas})"
        )
        anomaly_found = True

    # Vibration
    if vibration > 20:
        print(
            f"Vibration anomaly detected "
            f"(Current value = {vibration})"
        )
        anomaly_found = True

    # Load
    if load > 90:
        print(
            f"Load anomaly detected "
            f"(Current value = {load}%)"
        )
        anomaly_found = True

    # Worker Safety
    if distance < 50:
        print(
            f"Worker safety risk detected "
            f"(Distance = {distance} cm)"
        )
        anomaly_found = True

    if anomaly_found:
        print("\nSTATUS : MAINTENANCE REQUIRED")

    print("================================")

else:

    print("\n================================")
    print("      NORMAL CONDITION")
    print("================================")

    print(f"Temperature : {temperature}°C")
    print(f"Gas         : {gas}")
    print(f"Vibration   : {vibration}")
    print(f"Load        : {load}%")
    print(f"Distance    : {distance} cm")

    print("\nSTATUS : SAFE")
    print("================================")