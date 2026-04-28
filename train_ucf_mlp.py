"""
Treino do MLP para classificação de Crime vs Normal.
Usa features extraídas pela CNN (ResNet50).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import h5py
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração
FEATURES_TRAIN = Path("models/ucf_cnn_features_train.h5")
FEATURES_TEST = Path("models/ucf_cnn_features_test.h5")
MODEL_OUTPUT = Path("models/ucf_crime_mlp.pkl")
SCALER_OUTPUT = Path("models/ucf_scaler.pkl")
REPORT_OUTPUT = Path("models/ucf_mlp_report.txt")

# Hiperparâmetros do MLP
MLP_PARAMS = {
    "hidden_layer_sizes": (512, 256, 128),
    "activation": "relu",
    "solver": "adam",
    "learning_rate": "adaptive",
    "learning_rate_init": 0.001,
    "max_iter": 200,
    "batch_size": 32,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "random_state": 42,
    "verbose": True
}


def load_features():
    """Carrega features extraídas de HDF5."""
    print("[INFO] Carregando features...")
    
    with h5py.File(FEATURES_TRAIN, "r") as f:
        X_train = f["features"][:]
        y_train = f["labels"][:]
    
    with h5py.File(FEATURES_TEST, "r") as f:
        X_test = f["features"][:]
        y_test = f["labels"][:]
    
    print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def train_mlp():
    """Treina o MLP."""
    
    print("\n" + "="*60)
    print("Treino do MLP - Classificação de Crime")
    print("="*60)
    
    # Carregar features
    X_train, y_train, X_test, y_test = load_features()
    
    # Normalizar features
    print("\n[INFO] Normalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
    
    # Criar e treinar modelo
    print("\n[INFO] Treinando MLP...")
    print(f"   Arquitetura: {MLP_PARAMS['hidden_layer_sizes']}")
    print(f"   Iterações máx: {MLP_PARAMS['max_iter']}")
    
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train_scaled, y_train)
    
    print(f"[OK] Treino concluído em {mlp.n_iter_} iterações")
    
    # Avaliar
    print("\n[INFO] Avaliação...")
    
    y_train_pred = mlp.predict(X_train_scaled)
    y_test_pred = mlp.predict(X_test_scaled)
    
    y_train_proba = mlp.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")
    print(f"   Train AUC:      {train_auc:.4f}")
    print(f"   Test AUC:       {test_auc:.4f}")
    
    # Relatório detalhado
    print("\n" + "-"*60)
    print("TRAIN SET")
    print("-"*60)
    print(classification_report(y_train, y_train_pred, 
                              target_names=["Normal", "Crime"]))
    
    print("\n" + "-"*60)
    print("TEST SET")
    print("-"*60)
    print(classification_report(y_test, y_test_pred, 
                              target_names=["Normal", "Crime"]))
    
    # Salvar relatório
    with open(REPORT_OUTPUT, "w") as f:
        f.write("="*60 + "\n")
        f.write("UCF CRIME MLP - RELATÓRIO DE TREINAMENTO\n")
        f.write("="*60 + "\n\n")
        
        f.write("HIPERPARÂMETROS:\n")
        for k, v in MLP_PARAMS.items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\nRESULTADOS:\n")
        f.write(f"  Train Accuracy: {train_acc:.4f}\n")
        f.write(f"  Test Accuracy:  {test_acc:.4f}\n")
        f.write(f"  Train AUC:      {train_auc:.4f}\n")
        f.write(f"  Test AUC:       {test_auc:.4f}\n")
        
        f.write("\nTRAIN CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_train, y_train_pred, 
                                     target_names=["Normal", "Crime"]))
        
        f.write("\nTEST CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_test_pred, 
                                     target_names=["Normal", "Crime"]))
    
    # Salvar modelos
    Path("models").mkdir(exist_ok=True)
    joblib.dump(mlp, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    
    print(f"\n[OK] Modelo salvo em: {MODEL_OUTPUT}")
    print(f"[OK] Scaler salvo em: {SCALER_OUTPUT}")
    print(f"[OK] Relatório salvo em: {REPORT_OUTPUT}")
    
    # Plotar
    plot_results(y_train, y_train_proba, y_test, y_test_proba, 
                y_train_pred, y_test_pred)


def plot_results(y_train, y_train_proba, y_test, y_test_proba,
                y_train_pred, y_test_pred):
    """Cria gráficos de avaliação."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    axes[0, 0].plot(fpr_train, tpr_train, label="Train", linewidth=2)
    axes[0, 0].plot(fpr_test, tpr_test, label="Test", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Confusion Matrix - Train
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0, 1],
                xticklabels=["Normal", "Crime"],
                yticklabels=["Normal", "Crime"])
    axes[0, 1].set_title("Train Confusion Matrix")
    axes[0, 1].set_ylabel("True")
    axes[0, 1].set_xlabel("Predicted")
    
    # Confusion Matrix - Test
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["Normal", "Crime"],
                yticklabels=["Normal", "Crime"])
    axes[1, 0].set_title("Test Confusion Matrix")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_xlabel("Predicted")
    
    # Probability Distribution
    axes[1, 1].hist(y_train_proba[y_train == 0], bins=50, alpha=0.5, label="Normal")
    axes[1, 1].hist(y_train_proba[y_train == 1], bins=50, alpha=0.5, label="Crime")
    axes[1, 1].set_xlabel("Crime Probability")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Train - Probability Distribution")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("models/ucf_mlp_evaluation.png", dpi=150, bbox_inches="tight")
    print("[OK] Gráficos salvos em: models/ucf_mlp_evaluation.png")
    plt.close()


if __name__ == "__main__":
    train_mlp()
