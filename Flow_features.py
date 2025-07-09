import os
import pandas as pd
import numpy as np
from difflib import get_close_matches

cic_dir = '/Users/shishirkumarvallapuneni/Desktop/IDS/data/CICIDS2017Dataset/MachineLearningCVE'
darknet_path = '/Users/shishirkumarvallapuneni/Desktop/IDS/data/Darknet.csv'

target_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Label'
]

def normalize_columns(df):
    df.columns = [col.strip() for col in df.columns]
    return df

def map_columns(df, target_cols):
    col_map = {}
    existing_cols = df.columns.tolist()
    for col in target_cols:
        matches = get_close_matches(col, existing_cols, n=1, cutoff=0.7)
        if matches:
            col_map[col] = matches[0]
    return col_map

def load_and_clean_csv(path, target_cols):
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    col_map = map_columns(df, target_cols)
    aligned_df = df[[col_map[col] for col in col_map]]
    aligned_df.columns = list(col_map.keys())
    return aligned_df

def load_cic_data(cic_dir):
    all_files = [f for f in os.listdir(cic_dir) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        full_path = os.path.join(cic_dir, file)
        df = load_and_clean_csv(full_path, target_features)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def load_darknet_data(darknet_path):
    return load_and_clean_csv(darknet_path, target_features)

def main():
    cic_df = load_cic_data(cic_dir)
    darknet_df = load_darknet_data(darknet_path)
    os.makedirs('outputs/processed', exist_ok=True)
    cic_df.to_csv('outputs/processed/cic_flow_data.csv', index=False)
    darknet_df.to_csv('outputs/processed/darknet_flow_data.csv', index=False)

if __name__ == '__main__':
    main()