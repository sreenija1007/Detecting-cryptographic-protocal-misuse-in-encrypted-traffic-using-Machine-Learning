import os
import pyshark
import pandas as pd
from tqdm import tqdm
def extract_tls_from_pcap(pcap_path, label='unknown', max_packets=10000):
    cap = pyshark.FileCapture(pcap_path, display_filter="tls.handshake", keep_packets=False)
    records = []
    count = 0

    for pkt in cap:
        try:
            if 'IP' not in pkt or 'TLS' not in pkt:
                continue

            proto = pkt.transport_layer
            record = {
                'pcap_file': os.path.basename(pcap_path),
                'label': label,
                'src_ip': pkt.ip.src,
                'dst_ip': pkt.ip.dst,
                'src_port': getattr(pkt[proto], 'srcport', 'NA'),
                'dst_port': getattr(pkt[proto], 'dstport', 'NA'),
                'tls_version': pkt.tls.get_field_value('handshake_version') or "NA",
                'cipher_suite': pkt.tls.get_field_value('handshake_ciphersuite') or "NA",
                'cert_issuer': pkt.tls.get_field_value('handshake_certificate_issuer') or "NA",
                'cert_subject': pkt.tls.get_field_value('handshake_certificate_subject') or "NA",
                'cert_expiry': pkt.tls.get_field_value('handshake_certificate_validity_not_after') or "NA",
                'handshake_time': float(pkt.frame_info.time_delta) if hasattr(pkt.frame_info, 'time_delta') else 0.0
            }
            records.append(record)
            count += 1
            if max_packets and count >= max_packets:
                break
        except Exception:
            continue

    cap.close()
    df = pd.DataFrame(records)
    return df

def extract_all_from_directory(base_dir, max_packets=500):
    all_data = []
    pcaps = [f for f in os.listdir(base_dir) if f.endswith('.pcap')]
    for pcap in tqdm(pcaps, desc="Processing CIC IDS 2017 PCAPs"):
        full_path = os.path.join(base_dir, pcap)
        df = extract_tls_from_pcap(full_path, max_packets=max_packets)
        if not df.empty:
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def main():
    base_dir = '/Users/shishirkumarvallapuneni/Desktop/IDS/data/CICIDS2017Dataset/'
    os.makedirs('outputs/tls', exist_ok=True)
    df = extract_all_from_directory(base_dir, max_packets=30000)
    df.to_csv('outputs/tls/cic_tls_features.csv', index=False)
    print("CIC TLS feature extraction completed and saved to outputs/tls/cic_tls_features.csv")

if __name__ == '__main__':
    main()