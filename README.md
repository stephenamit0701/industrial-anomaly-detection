# Predictive Maintenance using NASA C-MAPSS Turbofan Dataset

## Overview
This project focuses on predictive maintenance using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The goal is to analyze multivariate time-series sensor data from turbofan engines to detect degradation patterns and estimate Remaining Useful Life (RUL).

The system leverages machine learning techniques to identify early signs of failure, enabling proactive maintenance and improved operational reliability.

---

## Problem Statement
Aircraft engine failures can lead to significant operational and financial risks. Traditional maintenance strategies are often reactive or scheduled, which may not accurately reflect the actual health of the system.

This project aims to:
- Analyze engine degradation using sensor data
- Detect abnormal behavior in time-series data
- Estimate Remaining Useful Life (RUL)
- Support predictive maintenance decisions

---

## Dataset

The project uses the NASA C-MAPSS FD001 dataset, which contains simulated run-to-failure data for multiple turbofan engines.

### Dataset Characteristics:
- Multiple engines with different operational cycles
- 21 sensor measurements per cycle
- 3 operational settings
- Run-to-failure trajectories for each engine

Each engine starts in a healthy state and degrades over time until failure.

---

## Approach

### Data Preprocessing
- Removal of irrelevant or constant features
- Normalization of sensor data
- Grouping data by engine ID
- Creation of time-series sequences

### Feature Engineering
- Rolling mean and standard deviation
- Trend analysis over time
- Degradation pattern extraction

### Modeling

#### 1. Anomaly Detection
- Isolation Forest for identifying abnormal patterns in sensor behavior

#### 2. Predictive Modeling (Planned/Extended)
- LSTM-based models for time-series forecasting
- Remaining Useful Life (RUL) prediction

### Evaluation
- Visualization of degradation trends
- Detection of anomalous behavior prior to failure
- Model performance analysis

---

## Project Structure


---

## Results

The system analyzes sensor data across engine cycles and identifies degradation patterns.

Typical outputs include:
- Time-series plots of sensor behavior
- Detection of abnormal operating conditions
- Early indicators of engine failure

(Add your plots here, e.g. anomaly detection graphs or degradation curves)

---

## Demo

To run the project locally:


---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras (for future LSTM implementation)
- Streamlit
- Docker

---

## Future Work

- Implement LSTM/GRU models for accurate RUL prediction
- Add sequence-based deep learning models
- Deploy model using FastAPI
- Integrate real-time data streaming
- Improve evaluation using RMSE for RUL prediction

---

## Applications

- Aircraft engine predictive maintenance
- Industrial equipment monitoring
- Condition-based maintenance systems
- Fault detection in complex systems

---

## Conclusion

This project demonstrates how machine learning can be applied to multivariate time-series data for predictive maintenance. By leveraging the NASA C-MAPSS dataset, it provides a foundation for building scalable and reliable health monitoring systems for industrial applications.

---

## Author

Amit Stephen