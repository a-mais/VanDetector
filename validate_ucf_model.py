"""
Validação rápida do modelo UCF Crime em frames de amostra.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import random
from tqdm import tqdm

# Configuração
DATASET_CSV = Path("ucf_crime_dataset.csv")
MODEL_PATH = Path("models/ucf_crime_mlp.pkl")
DATASET_ROOT = Path("datasets/ucfcrimedataset")

from src.ucf_features import frame_to_features
from PIL import Image
import cv2


def validate_random_samples(n_samples=100):
    """Valida modelo em N amostras aleatórias."""
    
    print("\n" + "="*60)
    print("🔍 VALIDAÇÃO RÁPIDA DO MODELO UCF CRIME")
    print("="*60)
    
    # Carregar dados
    if not MODEL_PATH.exists():
        print(f"❌ Modelo não encontrado: {MODEL_PATH}")
        print("Execute: python train_ucf_mlp.py")
        return
    
    if not DATASET_CSV.exists():
        print(f"❌ Dataset CSV não encontrado: {DATASET_CSV}")
        print("Execute: python prepare_ucf_dataset.py")
        return
    
    df = pd.read_csv(DATASET_CSV)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load("models/ucf_scaler.pkl")
    
    print(f"\n✅ Dataset carregado: {len(df)} frames")
    print(f"✅ Modelo carregado: {MODEL_PATH}")
    
    # Amostragem
    print(f"\n📊 Amostrando {n_samples} frames aleatórios...")
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    correct = 0
    incorrect = 0
    errors = []
    
    predictions_by_category = {}
    
    for idx, row in tqdm(sample.iterrows(), total=len(sample)):
        try:
            # Carregar frame
            frame_path = DATASET_ROOT / row["path"]
            
            if not frame_path.exists():
                errors.append(f"Frame não encontrado: {frame_path}")
                continue
            
            # Carregar e processar
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(frame_rgb)
            
            # Extrair features (usar função do app)
            from src.ucf_features import frame_to_features
            features_dict = frame_to_features(img)
            
            # Converter para array
            feature_names = sorted(features_dict.keys())
            X = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
            
            # Prever
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][int(pred)]
            
            pred_label = "crime" if pred == 1 else "normal"
            true_label = row["label"]
            
            # Registrar resultado
            category = row["category"]
            if category not in predictions_by_category:
                predictions_by_category[category] = {"correct": 0, "total": 0}
            
            predictions_by_category[category]["total"] += 1
            
            if pred_label == true_label:
                correct += 1
                predictions_by_category[category]["correct"] += 1
            else:
                incorrect += 1
                if len(errors) < 10:  # Guardar primeiros 10 erros
                    errors.append(
                        f"❌ {category}: esperado={true_label}, "
                        f"predito={pred_label} (conf={proba:.4f})"
                    )
        
        except Exception as e:
            errors.append(f"Erro ao processar: {e}")
    
    # Resultados
    total = correct + incorrect
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"RESULTADOS ({n_samples} amostras)")
    print(f"="*60)
    print(f"✅ Corretos:   {correct:5d}")
    print(f"❌ Incorretos: {incorrect:5d}")
    print(f"📊 Acurácia:   {accuracy:6.2f}%")
    
    # Por categoria
    if predictions_by_category:
        print(f"\n{'Categoria':<20} {'Acertos':<10} {'Taxa':<10}")
        print("-"*40)
        for cat in sorted(predictions_by_category.keys()):
            stats = predictions_by_category[cat]
            rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"{cat:<20} {stats['correct']:>2}/{stats['total']:<2} {rate:>6.1f}%")
    
    # Erros
    if errors:
        print(f"\n⚠️  Primeiros erros encontrados:")
        for err in errors[:5]:
            print(f"  {err}")
    
    print(f"\n{'='*60}\n")


def test_single_frame(frame_path):
    """Testa modelo em um frame específico."""
    
    print(f"\n📸 Testando frame: {frame_path}")
    
    model = joblib.load(MODEL_PATH)
    
    frame = cv2.imread(str(frame_path))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(frame_rgb)
    
    from src.ucf_features import frame_to_features
    features_dict = frame_to_features(img)
    
    feature_names = sorted(features_dict.keys())
    X = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][int(pred)]
    
    pred_label = "🔴 CRIME" if pred == 1 else "🟢 NORMAL"
    
    print(f"Predição: {pred_label}")
    print(f"Confiança: {proba:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Teste de frame específico
        test_single_frame(sys.argv[1])
    else:
        # Validação geral
        validate_random_samples(n_samples=100)
