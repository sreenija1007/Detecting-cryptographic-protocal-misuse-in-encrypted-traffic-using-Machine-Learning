import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import GlorotUniform
import logging
import joblib
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

LABEL_COLUMNS = ['is_mitm', 'is_ssl_stripping', 'is_weak_cert', 'is_encrypted_malware']
def load_data():
    data_path = "/Users/shishirkumarvallapuneni/Desktop/IDS/outputs/processed/flow_features_with_labels.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=LABEL_COLUMNS)
    y = df[LABEL_COLUMNS].astype(int)
    X = df.drop(columns=LABEL_COLUMNS)
    X = X.select_dtypes(include=[np.number])
    return X, y
def evaluate_model(y_true, y_pred, model_name="Model"):
    hamming = hamming_loss(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    subset_acc = accuracy_score(y_true, y_pred)

    logging.info(f" {model_name} Evaluation:")
    print("Hamming Loss     :", hamming)
    print("Micro F1 Score   :", micro_f1)
    print("Macro F1 Score   :", macro_f1)
    print("Subset Accuracy  :", subset_acc)

    plot_prediction_errors(y_true, y_pred, model_name)

    return {
        'model': model_name,
        'Hamming Loss': hamming,
        'Micro F1': micro_f1,
        'Macro F1': macro_f1,
        'Subset Accuracy': subset_acc
    }

def train_random_forest(X_train, y_train, X_test, y_test,model_path):
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=25, max_depth=7, random_state=42))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = evaluate_model(y_test, preds, "Random Forest")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Random Forest model saved to {model_path}")

    return score

def train_lightgbm(X_train, y_train, X_test, y_test,model_path):
    model = MultiOutputClassifier(LGBMClassifier(n_estimators=6, max_depth=4, learning_rate=0.05))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = evaluate_model(y_test, preds, "LightGBM")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"LightGBM model saved to {model_path}")

    return score

def train_xgboost(X_train, y_train, X_test, y_test,model_path):
    model = MultiOutputClassifier(XGBClassifier(n_estimators=13, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss'))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = evaluate_model(y_test, preds, "XGBoost")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"XGBoost model saved to {model_path}")

    return score

def train_deep_model(X_train, y_train, X_test, y_test,model_path):
    model = Sequential()
    model.add(Dense(512, kernel_initializer=GlorotUniform(), input_dim=X_train.shape[1]))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(256, kernel_initializer=GlorotUniform()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(128, kernel_initializer=GlorotUniform()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1], activation='sigmoid'))  

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    preds = model.predict(X_test)
    preds_binary = (preds > 0.5).astype(int)
    score = evaluate_model(y_test, preds_binary, " MLP")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(model, model_path)
    logging.info(f"Deep model saved to {model_path}")
    plot_training_history(history)

    return score

def plot_training_history(history, save_path="plots/deep_training_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Deep Learning Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f" Saved training loss plot to {save_path}")
    plt.close()

def plot_model_comparison(scores, save_path="plots/model_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(scores)
    df_melt = df.melt(id_vars='model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='model')
    plt.title("Model Evaluation Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f" Saved model comparison plot to {save_path}")
    plt.close()

def plot_prediction_errors(y_true, y_pred, model_name, save_dir="plots/errors"):
    os.makedirs(save_dir, exist_ok=True)
    cm_all = multilabel_confusion_matrix(y_true, y_pred)

    for i, cm in enumerate(cm_all):
        label = LABEL_COLUMNS[i]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Not {label}", label])
        disp.plot(cmap='Reds', values_format='d')
        plt.title(f"{model_name} - {label}")
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{model_name}_{label}_cm.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f" Saved confusion matrix for '{label}' to {plot_path}")

def main():
    logging.info(" Loading and preparing data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    all_scores = []

    logging.info(" Training Random Forest...")
    rf_score = train_random_forest(X_train, y_train, X_test, y_test, model_path="models/random_forest.pkl")
    rf_score['model'] = "Random Forest"
    all_scores.append(rf_score)

    logging.info(" Training LightGBM...")
    lgbm_score = train_lightgbm(X_train, y_train, X_test, y_test, model_path="models/lightgbm.pkl")
    lgbm_score['model'] = "LightGBM"
    all_scores.append(lgbm_score)

    logging.info(" Training XGBoost...")
    xgb_score = train_xgboost(X_train, y_train, X_test, y_test, model_path="models/xgboost.pkl")
    xgb_score['model'] = "XGBoost"
    all_scores.append(xgb_score)

    logging.info("Training Deep Learning MLP...")
    mlp_score = train_deep_model(X_train, y_train, X_test, y_test, model_path="models/MLP.h5")
    mlp_score['model'] = "Deep MLP"
    all_scores.append(mlp_score)

    plot_model_comparison(all_scores)


if __name__ == "__main__":
    main()