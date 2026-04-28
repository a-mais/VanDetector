"""
Extração de features usando CNN pré-treinada (ResNet50).
Processa frames do dataset UCF Crime e extrai features para treino do MLP.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm

# Configuração
DATASET_ROOT = Path("datasets/ucfcrimedataset")
DATASET_CSV = Path("ucf_crime_dataset.csv")
FEATURES_OUTPUT = Path("models/ucf_cnn_features_train.h5")
FEATURES_TEST_OUTPUT = Path("models/ucf_cnn_features_test.h5")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model info
MODEL_NAME = "resnet50"
FEATURE_DIM = 2048

print(f"🔧 Usando device: {DEVICE}")


class FrameDataset(Dataset):
    """Dataset para carregar frames e extrair features."""
    
    def __init__(self, csv_path, split="Train", root_dir=DATASET_ROOT):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        
        # Transform para ResNet50
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["path"]
        label = 1 if row["label"] == "crime" else 0
        
        # Carregar imagem
        from PIL import Image
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label, idx
        except Exception as e:
            print(f"Erro ao carregar {img_path}: {e}")
            # Retornar imagem preta em caso de erro
            return torch.zeros(3, 224, 224), label, idx


def create_feature_extractor():
    """Cria modelo de extração de features."""
    model = models.resnet50(pretrained=True)
    # Remove última camada (classificação)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(DEVICE)
    model.eval()
    return model


def extract_features(split="Train"):
    """Extrai features de um split do dataset."""
    
    print(f"\n📊 Extraindo features do {split} set...")
    
    # Carregar modelo
    model = create_feature_extractor()
    
    # Criar dataset e dataloader
    dataset = FrameDataset(DATASET_CSV, split=split)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Total de amostras: {len(dataset)}")
    
    # Extrair features
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for images, labels, indices in tqdm(dataloader, desc=f"Extraindo {split}"):
            images = images.to(DEVICE)
            
            # Forward pass
            features = model(images)  # [batch, 2048, 1, 1]
            features = features.view(features.size(0), -1)  # [batch, 2048]
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())
    
    # Consolidar
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"✅ Features extraídas: {X.shape}")
    print(f"   Dimensões por amostra: {X.shape[1]}")
    print(f"   Labels: {np.bincount(y)}")
    
    return X, y, np.array(all_indices)


def main():
    """Executa extração para Train e Test."""
    
    print("="*60)
    print("🚀 Extração de Features - UCF Crime Dataset")
    print("="*60)
    
    # Train set
    X_train, y_train, idx_train = extract_features("Train")
    
    # Test set
    X_test, y_test, idx_test = extract_features("Test")
    
    # Salvar em HDF5 (mais eficiente em espaço)
    Path("models").mkdir(exist_ok=True)
    
    print("\n💾 Salvando Train features em HDF5...")
    with h5py.File(FEATURES_OUTPUT, "w") as f:
        f.create_dataset("features", data=X_train, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=y_train, compression="gzip")
        f.create_dataset("indices", data=idx_train, compression="gzip")
    print(f"✅ Train features salvo em: {FEATURES_OUTPUT}")
    
    print("💾 Salvando Test features em HDF5...")
    with h5py.File(FEATURES_TEST_OUTPUT, "w") as f:
        f.create_dataset("features", data=X_test, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=y_test, compression="gzip")
        f.create_dataset("indices", data=idx_test, compression="gzip")
    print(f"✅ Test features salvo em: {FEATURES_TEST_OUTPUT}")
    
    print("\n✨ Extração concluída!")
    print(f"Train: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
    
    # Verificar tamanho dos arquivos
    import os
    train_size = os.path.getsize(FEATURES_OUTPUT) / 1024**3
    test_size = os.path.getsize(FEATURES_TEST_OUTPUT) / 1024**3
    print(f"\n📊 Tamanho em disco:")
    print(f"   Train: {train_size:.2f} GB")
    print(f"   Test:  {test_size:.2f} GB")


if __name__ == "__main__":
    main()
