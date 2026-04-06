import matplotlib.pyplot as plt

def plot_anomalies(errors, threshold):
    plt.figure()
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    
    plt.legend()
    plt.title("Anomaly Detection")
    plt.xlabel("Time")
    plt.ylabel("Error")
    
    plt.show()