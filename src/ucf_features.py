"""
Extração de features de frames usando CNN para o app.py.
Interface para extrair features de frames em tempo real.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import joblib
from pathlib import Path

# Modelo global (carregado uma vez)
_cnn_model = None
_scaler = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuração
CNN_MODEL_PATH = Path("models/resnet50_feature_extractor.pt")
SCALER_PATH = Path("models/ucf_scaler.pkl")


def load_cnn_model():
    """Carrega modelo CNN para extração de features."""
    global _cnn_model
    
    if _cnn_model is not None:
        return _cnn_model
    
    print(f"📦 Carregando CNN (device: {_device})...")
    
    # Criar modelo ResNet50
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(_device)
    model.eval()
    
    _cnn_model = model
    return _cnn_model


def load_scaler():
    """Carrega scaler para normalização."""
    global _scaler
    
    if _scaler is not None:
        return _scaler
    
    if SCALER_PATH.exists():
        _scaler = joblib.load(SCALER_PATH)
    else:
        print(f"⚠️  Scaler não encontrado em {SCALER_PATH}")
        _scaler = None
    
    return _scaler


def frame_to_features(frame) -> dict[str, float]:
    """
    Extrai features de um frame (numpy array ou PIL Image).
    
    Args:
        frame: numpy array (H, W, 3) ou PIL Image
    
    Returns:
        dict com features: {"feature_0": valor, "feature_1": valor, ...}
    """
    
    # Converter para PIL Image se necessário
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame).convert("RGB")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Preparar frame
    img_tensor = transform(frame).unsqueeze(0).to(_device)
    
    # Extrair features
    model = load_cnn_model()
    with torch.no_grad():
        features = model(img_tensor)
        features = features.view(features.size(0), -1).cpu().numpy()[0]
    
    # Normalizar se scaler disponível
    scaler = load_scaler()
    if scaler is not None:
        features = scaler.transform([features])[0]
    
    # Converter para dict
    return {f"feature_{i}": float(features[i]) for i in range(len(features))}


def cv2_frame_to_features(frame) -> dict[str, float]:
    """
    Versão otimizada para frames lidos com OpenCV (BGR).
    
    Args:
        frame: BGR array do OpenCV
    
    Returns:
        dict com features
    """
    # Converter BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_to_features(frame_rgb)


def batch_extract_features(frames: list) -> np.ndarray:
    """
    Extrai features de múltiplos frames em batch (mais eficiente).
    
    Args:
        frames: lista de numpy arrays ou PIL Images
    
    Returns:
        array [num_frames, 2048] com features
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Converter frames para tensores
    img_tensors = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame).convert("RGB")
        img_tensors.append(transform(frame))
    
    # Stack e processar
    batch = torch.stack(img_tensors).to(_device)
    
    model = load_cnn_model()
    with torch.no_grad():
        features = model(batch)
        features = features.view(batch.size(0), -1).cpu().numpy()
    
    # Normalizar se scaler disponível
    scaler = load_scaler()
    if scaler is not None:
        features = scaler.transform(features)
    
    return features


def frame_rgb_to_feature_vector(frame) -> np.ndarray:
    """Extrai o vetor de features bruto de um frame RGB."""

    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(frame).unsqueeze(0).to(_device)
    model = load_cnn_model()

    with torch.no_grad():
        features = model(img_tensor)
        features = features.view(features.size(0), -1).cpu().numpy()[0]

    scaler = load_scaler()
    if scaler is not None:
        features = scaler.transform([features])[0]

    return features


def occlusion_suspicion_map(
    frame_rgb: np.ndarray,
    model,
    grid_size: int = 6,
    occlusion_ratio: float = 0.35,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, float]:
    """
    Gera um mapa de suspeição por oclusão.

    A saída é uma aproximação visual do tipo YOLO: áreas que mais reduzem
    a probabilidade de crime quando occluídas recebem maior destaque.
    """

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb deve ter shape HxWx3")

    height, width = frame_rgb.shape[:2]
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    baseline_features = frame_rgb_to_feature_vector(frame_rgb)
    baseline_proba = float(model.predict_proba([baseline_features])[0][1])

    cell_w = max(1, width // grid_size)
    cell_h = max(1, height // grid_size)
    occlusion_color = tuple(int(v) for v in frame_rgb.mean(axis=(0, 1)))

    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = width if col == grid_size - 1 else (col + 1) * cell_w
            y2 = height if row == grid_size - 1 else (row + 1) * cell_h

            occluded = frame_rgb.copy()
            occluded[y1:y2, x1:x2] = occlusion_color

            occluded_features = frame_rgb_to_feature_vector(occluded)
            occluded_proba = float(model.predict_proba([occluded_features])[0][1])

            heatmap[row, col] = max(0.0, baseline_proba - occluded_proba)

    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value

    threshold = 0.65
    mask = heatmap >= threshold
    bbox = None

    if np.any(mask):
        rows, cols = np.where(mask)
        x1 = int(cols.min() * cell_w)
        y1 = int(rows.min() * cell_h)
        x2 = int(min(width, (cols.max() + 1) * cell_w))
        y2 = int(min(height, (rows.max() + 1) * cell_h))
        bbox = (x1, y1, x2, y2)

    return heatmap, bbox, baseline_proba


def draw_suspicion_overlay(frame_bgr: np.ndarray, heatmap: np.ndarray, bbox=None) -> np.ndarray:
    """Desenha um overlay tipo YOLO com mapa de calor e caixa destacando a área suspeita."""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (frame_rgb.shape[1], frame_rgb.shape[0]))
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized * 255, 0, 255))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_rgb, 0.68, colored, 0.32, 0)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            overlay,
            "CRIME",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return overlay
