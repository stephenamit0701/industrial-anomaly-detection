import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Correct column names
columns = (
    ["unit", "time"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    df.columns = columns
    return df


def normalize_data(df):
    scaler = MinMaxScaler()
    
    # Drop non-feature columns
    features = df.drop(["unit", "time"], axis=1)
    
    scaled = scaler.fit_transform(features)
    
    return scaled, scaler


def create_sequences(data, seq_length=30):
    sequences = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    
    return np.array(sequences)


if __name__ == "__main__":
    df = load_data("data/raw/train_FD001.txt")
    
    print("Original Shape:", df.shape)

    scaled_data, scaler = normalize_data(df)
    
    print("Scaled Shape:", scaled_data.shape)

    sequences = create_sequences(scaled_data, seq_length=30)
    
    print("Sequences Shape:", sequences.shape)