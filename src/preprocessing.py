import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length - 1])
    return np.array(xs), np.array(ys)

def run_preprocessing_pipeline(df, target_col, seq_length, test_size):
    # Label encoding
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_size, shuffle=False
    )
    
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    
    # Normalization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    smote = SMOTE()
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    
    # Sequences
    X_train_seq, y_train_seq = create_sequences(X_train_bal, y_train_bal, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler, le
