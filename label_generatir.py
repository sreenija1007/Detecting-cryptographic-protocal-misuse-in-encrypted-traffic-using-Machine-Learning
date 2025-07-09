import pandas as pd
import os

cic_path = 'outputs/processed/cic_flow_data.csv'
darknet_path = 'outputs/processed/darknet_flow_data.csv'

def map_cic_labels(label):
    label = label.lower()
    return {
        'is_mitm': int('infiltration' in label or 'man in the middle' in label or 'ftp-patator' in label or 'ssh-patator' in label),
        'is_ssl_stripping': int('heartbleed' in label or 'sql injection' in label or 'xss' in label),
        'is_weak_cert': int('bot' in label or 'web attack' in label or 'dos' in label or 'ddos' in label),
        'is_encrypted_malware': int('malware' in label or 'tor' in label or 'vpn' in label or 'darknet' in label)
    }

def map_darknet_labels(label):
    label = label.lower()
    return {
        'is_mitm': 0,
        'is_ssl_stripping': 0,
        'is_weak_cert': 0,
        'is_encrypted_malware': int('malware' in label or 'tor' in label or 'vpn' in label or 'darknet' in label)
    }

def process_and_merge():
    cic_df = pd.read_csv(cic_path)
    darknet_df = pd.read_csv(darknet_path)

    cic_df['Label'] = cic_df['Label'].astype(str)
    darknet_df['Label'] = darknet_df['Label'].astype(str)

    cic_labels = cic_df['Label'].apply(map_cic_labels).apply(pd.Series)
    darknet_labels = darknet_df['Label'].apply(map_darknet_labels).apply(pd.Series)

    cic_df = pd.concat([cic_df.drop(columns=['Label']), cic_labels], axis=1)
    darknet_df = pd.concat([darknet_df.drop(columns=['Label']), darknet_labels], axis=1)

    malware_df = darknet_df[darknet_df['is_encrypted_malware'] == 1]
    malware_sampled = malware_df.sample(n=3000, random_state=42)

    mitm_df = cic_df[cic_df['is_mitm'] == 1]
    sslstrip_df = cic_df[cic_df['is_ssl_stripping'] == 1]
    weakcert_df = cic_df[cic_df['is_weak_cert'] == 1]
    benign_df = cic_df[(cic_df['is_mitm'] == 0) & (cic_df['is_ssl_stripping'] == 0) & (cic_df['is_weak_cert'] == 0)]

    mitm_sampled = mitm_df.sample(n=min(len(mitm_df), 1000), random_state=42)
    sslstrip_sampled = sslstrip_df.sample(n=min(len(sslstrip_df), 1000), random_state=42)
    weakcert_sampled = weakcert_df.sample(n=min(len(weakcert_df), 1000), random_state=42)
    benign_sampled = benign_df.sample(n=1000, random_state=42)

    final_df = pd.concat([
        mitm_sampled,
        sslstrip_sampled,
        weakcert_sampled,
        malware_sampled,
        benign_sampled
    ], ignore_index=True)

    os.makedirs('outputs/processed', exist_ok=True)
    final_df.to_csv('outputs/processed/flow_features_with_labels.csv', index=False)

    label_columns = ['is_mitm', 'is_ssl_stripping', 'is_weak_cert', 'is_encrypted_malware']
    label_counts = final_df[label_columns].sum().reset_index()
    label_counts.columns = ['Label', 'Count']
    label_counts.to_csv('outputs/processed/label_distribution_counts.csv', index=False)

    combination_counts = final_df[label_columns].value_counts().reset_index()
    combination_counts.columns = label_columns + ['Count']
    combination_counts.to_csv('outputs/processed/label_combination_counts.csv', index=False)

    print("Balanced label counts:\n")
    print(label_counts)
    print("\nLabel combination distribution:\n")
    print(combination_counts)

if __name__ == '__main__':
    process_and_merge()