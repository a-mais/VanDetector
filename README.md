# VanDetector - UCF Crime + MLP

Aplicacao Streamlit para classificar frames de video como `crime` ou `normal` usando a abordagem UCF Crime Dataset com ResNet50 + MLP.

## O que esta nesta versao

Esta e a abordagem atual do projeto:

```text
Frame de video
    -> ResNet50 pre-treinada
    -> 2048 features visuais
    -> StandardScaler
    -> MLP treinado
    -> P(crime)
```

O app principal e:

```text
app_ucf.py
```

A documentacao tecnica completa esta em:

```text
UCF_MLP_ABORDAGEM.md
```

Os arquivos da abordagem anterior Roboflow/YOLO foram movidos para:

```text
.old/
```

## Arquivos necessarios para rodar no Streamlit

Para executar a aplicacao em outro computador, estes arquivos precisam estar no repositorio:

```text
app_ucf.py
src/ucf_features.py
src/__init__.py
requirements.txt
models/ucf_crime_mlp.pkl
models/ucf_scaler.pkl
UCF_MLP_ABORDAGEM.md
```

Arquivos opcionais, mas uteis para consulta:

```text
models/ucf_mlp_report.txt
models/ucf_mlp_evaluation.png
UCF_README.md
quickstart_ucf.py
validate_ucf_model.py
```

Os arquivos `models/ucf_cnn_features_train.h5` e `models/ucf_cnn_features_test.h5` nao sao necessarios para rodar o Streamlit. Eles sao muito grandes e servem apenas para retreinar o MLP sem extrair features novamente.

## Requisitos

- Python 3.10 ou superior recomendado.
- Internet no primeiro uso, caso o PyTorch precise baixar os pesos pre-treinados da ResNet50.
- CPU suficiente para inferencia. GPU ajuda na extracao de features pela ResNet50 se CUDA estiver disponivel, mas o MLP roda em CPU.

## Como rodar em outro computador

Clone ou baixe o projeto:

```powershell
git clone <URL_DO_REPOSITORIO>
cd VanDetector
```

Crie o ambiente virtual:

```powershell
python -m venv venv
```

Ative o ambiente:

```powershell
.\venv\Scripts\activate
```

Instale as dependencias:

```powershell
pip install -r requirements.txt
```

Rode o Streamlit:

```powershell
streamlit run app_ucf.py
```

Ou diretamente pelo Python do ambiente:

```powershell
.\venv\Scripts\python.exe -m streamlit run app_ucf.py
```

Depois abra no navegador:

```text
http://localhost:8501
```

## Como usar

1. Abra o app Streamlit.
2. Selecione o modo `Video Local` ou `Imagem`.
3. Envie um arquivo de video ou imagem.
4. Ajuste a confianca minima para crime se necessario.
5. Execute a analise.
6. Ao final, use a navegacao de frames registrados como crime para revisar os pontos detectados.

## Observacao sobre treino e GPU

O arquivo `train_ucf_mlp.py` usa `sklearn.neural_network.MLPClassifier`, portanto o treino do MLP e feito em CPU.

A GPU pode ser usada na etapa de extracao de features com ResNet50, feita por `extract_ucf_features.py`, porque essa etapa usa PyTorch:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Para apenas rodar o Streamlit, nao e necessario ter o dataset UCF completo nem os arquivos `.h5` de features.

## Estrutura atual

```text
VanDetector/
    app_ucf.py
    train_ucf_mlp.py
    extract_ucf_features.py
    prepare_ucf_dataset.py
    validate_ucf_model.py
    requirements.txt
    UCF_MLP_ABORDAGEM.md
    UCF_README.md
    src/
        __init__.py
        ucf_features.py
    models/
        ucf_crime_mlp.pkl
        ucf_scaler.pkl
        ucf_mlp_report.txt
        ucf_mlp_evaluation.png
    .old/
        arquivos da abordagem anterior
```

## Arquivos grandes ignorados

O `.gitignore` mantem fora do GitHub:

- `venv/`
- `__pycache__/`
- `datasets/`
- `runs/`
- arquivos `.pt`
- arquivos `.h5` grandes dentro de `models/`
- videos ou uploads gerados em `outputs/`

Mas permite subir os artefatos pequenos necessarios para rodar o Streamlit:

- `models/ucf_crime_mlp.pkl`
- `models/ucf_scaler.pkl`
- `models/ucf_mlp_report.txt`
- `models/ucf_mlp_evaluation.png`
