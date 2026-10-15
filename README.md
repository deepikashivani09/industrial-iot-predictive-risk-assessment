# AI-Driven Predictive Risk Assessment System for Industrial Machines Using IoT and Machine Learning

## Project Overview
This project develops an IoT-based industrial monitoring system that uses machine learning to analyze sensor data, detect abnormal operating conditions, and assess potential machine risks.

## Problem Statement
Industrial machines may develop abnormal operating conditions that can lead to equipment damage, downtime, and safety risks. This project explores real-time sensor monitoring and machine learning techniques to identify unusual patterns and support early risk assessment.

## System Architecture
ESP32 and Sensors → MQTT (HiveMQ) → Node-RED → InfluxDB → Python ML Engine → Grafana Dashboard

## Technologies Used
- **Programming:** Python
- **IoT Platform:** ESP32 and sensor simulation
- **Communication:** MQTT, HiveMQ
- **Data Processing:** Node-RED, InfluxDB
- **Machine Learning:** Scikit-learn, Isolation Forest
- **Time-Series Prediction:** LSTM, TensorFlow/Keras
- **Visualization:** Grafana

## Key Features
- Sensor data collection and real-time transmission
- Anomaly detection using machine learning
- Time-series trend prediction using LSTM
- Machine risk assessment based on analyzed sensor readings
- Time-series data storage and dashboard visualization

## How It Works
1. Sensors collect operating-condition readings from the simulated industrial machine.
2. Sensor data is transmitted through MQTT.
3. Node-RED processes and routes the incoming data.
4. InfluxDB stores the sensor measurements.
5. The Python ML engine analyzes the data for abnormal patterns and generates risk-related outputs.
6. Grafana displays monitoring information and trends.

## Machine Learning Approach
- **Isolation Forest:** Identifies unusual sensor patterns.
- **LSTM:** Analyzes sequential data to estimate future trends.

Model performance depends on the training data, preprocessing, and evaluation. Predictions should be validated before real-world industrial use.

## Repository Contents
The repository contains Python source files for data retrieval, model training, anomaly detection, trend prediction, and live analysis, along with the required model artifacts and dependencies where included.



