import numpy as np

# ====================================
# SENSOR HISTORY
# ====================================

temperature = [35,36,35,37,38,40,39,42]

vibration = [2,3,4,5,7,9,12,15]

load = [50,55,58,60,68,75,82,90]

print("\n========== PREDICTIVE AI SYSTEM ==========\n")

# ====================================
# TEMPERATURE
# ====================================

x = np.arange(len(temperature))
temp_slope = np.polyfit(x, temperature, 1)[0]

if temp_slope > 0.5:

    probability = min(int(temp_slope*40),95)
    time_to_failure = max(30-int(temp_slope*15),5)

    print("Temperature trend increasing")
    print("Prediction : Possible overheating")
    print(f"Failure probability : {probability}%")
    print(f"Estimated failure time : {time_to_failure} minutes\n")

# ====================================
# VIBRATION
# ====================================

x = np.arange(len(vibration))
vib_slope = np.polyfit(x, vibration, 1)[0]

if vib_slope > 0.5:

    probability = min(int(vib_slope*35),95)
    time_to_failure = max(30-int(vib_slope*10),5)

    print("Vibration trend increasing")
    print("Prediction : Possible mechanical failure")
    print(f"Failure probability : {probability}%")
    print(f"Estimated failure time : {time_to_failure} minutes\n")

# ====================================
# LOAD
# ====================================

x = np.arange(len(load))
load_slope = np.polyfit(x, load, 1)[0]

if load_slope > 0.5:

    probability = min(int(load_slope*5),95)
    time_to_failure = max(30-int(load_slope),5)

    print("Load trend increasing")
    print("Prediction : Possible overload condition")
    print(f"Failure probability : {probability}%")
    print(f"Estimated failure time : {time_to_failure} minutes\n")

print("==========================================")