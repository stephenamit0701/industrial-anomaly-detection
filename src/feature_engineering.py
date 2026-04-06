import pandas as pd
import numpy as np


def add_rolling_features(df, window=5):
    df_new = df.copy()
    
    sensor_cols = [col for col in df.columns if "sensor" in col]
    
    for col in sensor_cols:
        df_new[f"{col}_roll_mean"] = df[col].rolling(window).mean()
        df_new[f"{col}_roll_std"] = df[col].rolling(window).std()
    
    return df_new


def add_fft_features(df):
    df_new = df.copy()
    
    sensor_cols = [col for col in df.columns if "sensor" in col]
    
    for col in sensor_cols:
        fft_vals = np.fft.fft(df[col].fillna(0))
        
        df_new[f"{col}_fft_real"] = np.real(fft_vals)
        df_new[f"{col}_fft_imag"] = np.imag(fft_vals)
    
    return df_new


if __name__ == "__main__":
    from preprocessing import load_data
    
    df = load_data("data/raw/train_FD001.txt")
    
    df = df.fillna(0)
    
    print("Original:", df.shape)
    
    df = add_rolling_features(df)
    print("After Rolling:", df.shape)
    
    df = add_fft_features(df)
    print("After FFT:", df.shape)
    
    print(df.head())