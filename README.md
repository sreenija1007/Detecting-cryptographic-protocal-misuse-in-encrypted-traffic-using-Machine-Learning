

# 🔐 IDS for Cryptographic Protocol Misuses

This project implements a machine learning–based intrusion detection system (IDS) to identify misuses of TLS/SSL cryptographic protocols — such as downgrade attacks, invalid ciphers, and insecure certificate configurations — using only encrypted traffic metadata.

## 🚀 Project Overview

- **Goal:** Detect cryptographic protocol misuses without decrypting packet payloads.
- **Scope:** Focused on TLS handshake features and traffic flow characteristics.
- **Privacy-preserving:** Detection is based entirely on metadata, ensuring no content-level inspection.

## 🧠 What We Built

- A supervised learning pipeline to classify malicious/misused TLS sessions.
- Feature extraction from **CIC-IDS2017** and **CIC-Darknet2020** datasets.
- Model training with:
  - **Random Forest**
  - **XGBoost**
  - **Deep MLP (Multi-layer Perceptron)**

## 🔍 Features Used

Extracted from encrypted TLS flows:
- TLS Version & Cipher Suite
- Packet Flow Count
- Byte Entropy
- Inter-arrival Time
- Server Name Indication (SNI) presence
- Certificate Validity
- And more...

## 📊 Results

| Model         | Accuracy | Micro-F1 | Macro-F1 |
|---------------|----------|----------|----------|
| Random Forest | 96.1%    | 95.4%    | 94.7%    |
| XGBoost       | 97.2%    | 96.8%    | 95.9%    |
| Deep MLP      | 94.3%    | 93.6%    | 92.4%    |

✅ XGBoost performed best in terms of generalization and consistency.

## 🔐 Why This Matters

TLS is widely used but often misconfigured. Detecting misuses without decrypting payloads is:
- Scalable
- Non-invasive
- Deployment-friendly in privacy-sensitive systems

This research has potential applications in:
- **Cloud-based IDS tools**
- **Enterprise SOCs**
- **AI-powered security automation**


## ⚙️ Requirements

- Python 3.8+
- Scikit-learn
- XGBoost
- TensorFlow or PyTorch (for MLP)
- Pandas, NumPy, Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt

📈 Usage

To train and evaluate all models:

python train_and_evaluate.py

To test on new encrypted traffic data:

python predict.py --input new_data.csv

📚 References
	•	CIC-IDS2017 Dataset
	•	CIC-Darknet2020 Dataset
	•	XGBoost Documentation

👥 Contributors
	•	Sreenija Kanugonda
	•	Shishir Kumar Vallapuneni

---
