from evaluate import plot_anomalies
import numpy as np

def get_reconstruction_error(model, sequences):
    reconstructed = model.predict(sequences)
    
    error = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
    
    return error
from sklearn.ensemble import IsolationForest
from preprocessing import load_data, normalize_data


def train_isolation_forest(X):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    return model


def get_anomaly_scores(model, X):
    scores = model.decision_function(X)
    return scores

from preprocessing import load_data, normalize_data, create_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def build_lstm_autoencoder(timesteps, features):
    inputs = Input(shape=(timesteps, features))

    
    encoded = LSTM(64, activation='relu')(inputs)

  
    decoded = RepeatVector(timesteps)(encoded)

    
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)

   
    outputs = TimeDistributed(Dense(features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    return model

if __name__ == "__main__":
    print("Loading data...")

    df = load_data("data/raw/train_FD001.txt")

    X_scaled, scaler = normalize_data(df)

    sequences = create_sequences(X_scaled, seq_length=30)

    print("Sequence shape:", sequences.shape)

    timesteps = sequences.shape[1]
    features = sequences.shape[2]

    model = build_lstm_autoencoder(timesteps, features)

    print("Training LSTM Autoencoder...")

    model.fit(
        sequences,
        sequences,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    print("Training complete.")

    
    print("Calculating reconstruction error...")

    errors = get_reconstruction_error(model, sequences)

    print("Sample errors:")
    print(errors[:10])

    
    threshold = np.percentile(errors, 95)

    print("Threshold:", threshold)

    anomalies = errors > threshold

    print("Number of anomalies detected:", np.sum(anomalies))

plot_anomalies(errors, threshold)

