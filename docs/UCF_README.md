# VanDetector - Abordagem UCF Crime

Nova abordagem do VanDetector usando o **UCF Crime Dataset** com arquitetura **CNN + MLP**.

## 🎯 Mudanças Principais

| Aspecto | Anterior (YOLO) | Novo (UCF) |
|--------|-----------------|-----------|
| **Modelo Visual** | YOLO Object Detection | ResNet50 (CNN) |
| **Extração Features** | Bounding boxes → features | ResNet50 penúltima camada |
| **Classificação** | MLP em features YOLO | MLP em CNN features |
| **Dataset** | Roboflow (com anotações) | UCF Crime (classificação de frames) |
| **Foco** | Detecção de objetos + crime | Classificação direta de frames |

## 📊 Dataset UCF Crime

- **11 categorias**: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, NormalVideos
- **~1.2M frames de treino** + test set
- **Formato**: PNG (frames extraídos de vídeos)
- **Split**: Train/ e Test/ pré-organizados

## 🚀 Pipeline de Treinamento

### 1. Preparar Dataset

```powershell
python prepare_ucf_dataset.py
```

Cria `ucf_crime_dataset.csv` com mapeamento de todos os frames.

**Output:**
- `ucf_crime_dataset.csv` - paths, categorias e labels (crime/normal)

### 2. Extrair Features (CNN)

```powershell
python extract_ucf_features.py
```

Processa **todos os frames** com ResNet50 e extrai features (2048-dim).

**Output:**
- `models/ucf_cnn_features_train.pkl` - features [1.2M, 2048]
- `models/ucf_cnn_features_test.pkl` - features [test_size, 2048]

**⏱️ Estimativa:** 4-6 horas em GPU, ~24h em CPU

### 3. Treinar MLP

```powershell
python train_ucf_mlp.py
```

Treina MLP para classificar frames como "crime" ou "normal".

**Output:**
- `models/ucf_crime_mlp.pkl` - Modelo treinado
- `models/ucf_scaler.pkl` - StandardScaler para normalização
- `models/ucf_mlp_report.txt` - Relatório de resultados
- `models/ucf_mlp_evaluation.png` - Gráficos (ROC, confusion matrix)

## 💻 Usar em Produção

### Interface Streamlit

```powershell
streamlit run app_ucf.py
```

**Recursos:**
- 📹 Processar vídeos locais
- 📸 Classificar imagens
- 🎬 Webcam em tempo real (em desenvolvimento)
- ⚙️ Ajustar confiança e filtros temporais

### API (Code)

```python
from src.ucf_features import cv2_frame_to_features, batch_extract_features
import joblib

# Carregar modelo
model = joblib.load("models/ucf_crime_mlp.pkl")

# Classificar um frame
import cv2
frame = cv2.imread("frame.jpg")

features = cv2_frame_to_features(frame)  # dict com 2048 features
prediction = model.predict(features_array)  # 0=normal, 1=crime
probability = model.predict_proba(features_array)[:, 1]
```

## 📈 Arquitetura

```
Frame (H, W, 3)
    ↓
ResNet50 (pré-treinada)
    ↓
Features [2048] (penúltima camada)
    ↓
StandardScaler (normalização)
    ↓
MLP [2048 → 512 → 256 → 128 → 1]
    ↓
Saída: P(crime) ∈ [0, 1]
```

### Hiperparâmetros MLP

```python
{
    "hidden_layer_sizes": (512, 256, 128),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "early_stopping": True,
    "batch_size": 32,
    "max_iter": 200
}
```

## 🛠️ Filtros de Detecção

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `confidence_threshold` | Probabilidade mín. para considerar "crime" | 0.60 |
| `temporal_window` | Frames em análise temporal | 10 |
| `min_crime_frames` | Mínimo de frames "crime" na janela | 7 |

**Exemplo:** Com window=10 e min_frames=7, detecta crime se 70%+ dos últimos 10 frames forem classificados como crime.

## 📝 Logs

Eventos de crime são salvos em `outputs/logs/ucf_crime_events.csv`:

```
timestamp,source,mode,frame,confidence,prediction
2026-04-27T15:30:45.123456,video.mp4,video,1542,0.8543,crime
```

## 📋 Checklist de Setup

- [ ] Dataset UCF Crime extraído em `datasets/ucfcrimedataset/`
- [ ] `pip install -r requirements.txt` (com torch/torchvision)
- [ ] `python prepare_ucf_dataset.py` ✅
- [ ] `python extract_ucf_features.py` ⏳ (longo!)
- [ ] `python train_ucf_mlp.py` ✅
- [ ] `streamlit run app_ucf.py` 🚀

## 🔍 Comparação com Abordagem Anterior

### ✅ Vantagens UCF
- Dataset **público e bem documentado**
- Não precisa de YOLO (mais simples e rápido)
- **11 categorias de crime** vs genérico "vandalism"
- CNN pré-treinada = menos overfitting
- Pipeline mais limpo: Frame → CNN → MLP

### ⚠️ Desvantagens UCF
- Features CNN são genéricas (não específicas de crime)
- Não captura **ações localizadas** (bounding boxes)
- Requer **GPU para inferência rápida**
- Treino é longo (extraction de features)

## 🎓 Próximos Passos

1. **Fine-tuning**: Treinar CNN também (não apenas MLP)
2. **Ensemble**: Combinar com YOLO para melhor acurácia
3. **Temporal**: Usar LSTM/Transformer para sequências de frames
4. **Attention**: Adicionar attention maps para interpretabilidade
5. **Data Augmentation**: Aumentar dataset com transformações

## 📚 Referências

- [UCF Crime Dataset](https://www.crcv.ucf.edu/datasets/crime-dataset/)
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## 👥 Contribuidores

- Dataset: UCF CRCV
- Framework: PyTorch, scikit-learn, OpenCV, Streamlit
