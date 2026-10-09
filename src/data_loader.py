import pandas as pd
from pathlib import Path

def load_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported format")
    return df

def clean_dataset(df, target_col):
    df = df.dropna(subset=[target_col])
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Drop identifiers
    drop_candidates = ['id', 'sample_id', 'timestamp', 'station_id']
    for col in drop_candidates:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df

def separate_features(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} missing")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
