# Abordagem UCF Crime com CNN + MLP no VanDetector

Este documento descreve a abordagem do VanDetector baseada no UCF Crime Dataset, explicando a origem dos dados, o fluxo de processamento, a extracao de features com ResNet50, o treinamento do MLP, a funcao de ativacao utilizada, os neuronios da rede e a justificativa para escolher MLP em vez de Perceptron ou Adaline.

## 1. Objetivo da abordagem

O objetivo desta abordagem e classificar frames de video como:

- `normal`: frame sem indicio de crime.
- `crime`: frame associado a uma categoria de evento criminal.

No codigo atual, a abordagem nao detecta objetos diretamente como um YOLO faria. Ela classifica o frame inteiro. Para isso, o sistema usa duas etapas:

1. Uma CNN pre-treinada, ResNet50, transforma cada frame em um vetor numerico de 2048 caracteristicas.
2. Um MLP recebe esse vetor de 2048 valores e decide se o frame representa `crime` ou `normal`.

O fluxo geral e:

```text
Frame de video
    -> ResNet50 pre-treinada
    -> Vetor de features com 2048 dimensoes
    -> StandardScaler
    -> MLPClassifier
    -> Probabilidade P(crime)
    -> Decisao: crime ou normal
```

## 2. Origem dos dados

A abordagem utiliza o UCF Crime Dataset, um dataset publico criado para pesquisa em reconhecimento de anomalias e eventos criminais em videos de vigilancia.

No projeto, os dados ficam organizados em:

```text
datasets/ucfcrimedataset/
    Train/
    Test/
```

O script responsavel por preparar o mapeamento dos dados e:

```text
prepare_ucf_dataset.py
```

Esse script percorre os diretorios `Train` e `Test`, identifica a categoria de cada frame e gera o arquivo:

```text
ucf_crime_dataset.csv
```

O CSV final contem, para cada frame:

- `path`: caminho relativo do frame dentro do dataset.
- `category`: categoria original do UCF.
- `label`: rotulo binario usado pelo VanDetector, `crime` ou `normal`.
- `split`: divisao original, `Train` ou `Test`.

## 3. Categorias utilizadas

No arquivo `prepare_ucf_dataset.py`, as categorias sao agrupadas em duas classes.

Categorias consideradas `crime`:

```python
{
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Robbery",
    "RoadAccidents",
    "Shooting",
}
```

Categoria considerada `normal`:

```python
{
    "NormalVideos",
}
```

Portanto, o modelo final nao aprende uma classificacao multiclasse do tipo `Assault`, `Robbery`, `Fighting`, etc. Ele aprende uma classificacao binaria:

```text
crime vs normal
```

## 4. Quantidade de dados utilizada

Com base no relatorio salvo em `models/ucf_mlp_report.txt`, o treinamento atual usou:

| Split | Normal | Crime | Total |
| --- | ---: | ---: | ---: |
| Treino | 947.768 | 235.314 | 1.183.082 |
| Teste | 64.952 | 35.638 | 100.590 |

Essa distribuicao mostra que o conjunto de treino e desbalanceado: existem muito mais frames normais do que frames de crime. Isso deve ser considerado na interpretacao das metricas.

## 5. Extracao de features com ResNet50

O arquivo responsavel por extrair features e:

```text
extract_ucf_features.py
```

Ele usa uma ResNet50 pre-treinada da biblioteca `torchvision`.

No codigo:

```python
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
```

A ultima camada da ResNet50, que originalmente faria classificacao ImageNet, e removida. Assim, a rede passa a funcionar como extratora de caracteristicas visuais.

Cada frame passa por:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

Esses valores de media e desvio padrao sao os valores padrao usados para modelos treinados no ImageNet.

Depois da passagem pela ResNet50, cada frame vira um vetor:

```text
[2048]
```

Ou seja, cada imagem passa a ser representada por 2048 numeros.

Os arquivos gerados pela extracao sao:

```text
models/ucf_cnn_features_train.h5
models/ucf_cnn_features_test.h5
```

Cada arquivo HDF5 armazena:

- `features`: matriz com os vetores de 2048 caracteristicas.
- `labels`: rotulos numericos, onde `0 = normal` e `1 = crime`.
- `indices`: indices originais dos frames.

## 6. Por que usar ResNet50 antes do MLP

O MLP sozinho nao recebe pixels crus da imagem. Isso seria inviavel e pouco eficiente, porque uma imagem 224x224 com 3 canais possui:

```text
224 * 224 * 3 = 150.528 valores
```

Treinar um MLP diretamente nesses pixels exigiria muitos parametros e tenderia a generalizar mal.

A ResNet50 resolve esse problema porque ja aprendeu representacoes visuais gerais em um grande dataset. Ela extrai padroes como:

- bordas;
- formas;
- textura;
- objetos;
- pessoas;
- cenas;
- composicoes visuais.

Assim, o MLP recebe uma representacao mais compacta e semanticamente mais rica: 2048 features em vez de pixels crus.

## 7. Treinamento do MLP

O treinamento acontece no arquivo:

```text
train_ucf_mlp.py
```

O script carrega as features ja extraidas:

```python
FEATURES_TRAIN = Path("models/ucf_cnn_features_train.h5")
FEATURES_TEST = Path("models/ucf_cnn_features_test.h5")
```

Depois carrega os arrays:

```python
X_train = f["features"][:]
y_train = f["labels"][:]
X_test = f["features"][:]
y_test = f["labels"][:]
```

Onde:

- `X_train`: matriz de treino com shape aproximado `[1183082, 2048]`.
- `y_train`: rotulos de treino.
- `X_test`: matriz de teste com shape aproximado `[100590, 2048]`.
- `y_test`: rotulos de teste.

## 8. Normalizacao com StandardScaler

Antes de treinar o MLP, o codigo aplica:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

O `StandardScaler` transforma cada feature para ter, aproximadamente:

```text
media = 0
desvio padrao = 1
```

Isso e importante porque redes neurais treinam melhor quando as entradas estao em escalas parecidas.

O scaler treinado e salvo em:

```text
models/ucf_scaler.pkl
```

Durante a inferencia, as features de novos frames tambem precisam passar pelo mesmo scaler. Isso e feito no modulo `src/ucf_features.py`, que carrega `models/ucf_scaler.pkl`.

## 9. Arquitetura do MLP

No arquivo `train_ucf_mlp.py`, os hiperparametros estao definidos em:

```python
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
```

A arquitetura pode ser representada assim:

```text
Entrada: 2048 features
    -> Camada oculta 1: 512 neuronios
    -> Camada oculta 2: 256 neuronios
    -> Camada oculta 3: 128 neuronios
    -> Saida binaria: P(crime)
```

### Quantidade de neuronios

O MLP possui:

| Parte da rede | Neuronios |
| --- | ---: |
| Entrada | 2048 |
| Camada oculta 1 | 512 |
| Camada oculta 2 | 256 |
| Camada oculta 3 | 128 |
| Saida | 1 |

Total de neuronios ocultos:

```text
512 + 256 + 128 = 896 neuronios ocultos
```

Considerando entrada, camadas ocultas e saida:

```text
2048 + 512 + 256 + 128 + 1 = 2945 unidades
```

### Quantidade aproximada de parametros treinaveis

Os parametros treinaveis sao pesos e vieses.

Entre a entrada e a primeira camada oculta:

```text
2048 * 512 + 512 = 1.049.088 parametros
```

Entre a primeira e a segunda camada oculta:

```text
512 * 256 + 256 = 131.328 parametros
```

Entre a segunda e a terceira camada oculta:

```text
256 * 128 + 128 = 32.896 parametros
```

Entre a terceira camada oculta e a saida:

```text
128 * 1 + 1 = 129 parametros
```

Total aproximado:

```text
1.049.088 + 131.328 + 32.896 + 129 = 1.213.441 parametros treinaveis
```

## 10. Onde esta a funcao de ativacao

A funcao de ativacao das camadas ocultas e definida em `train_ucf_mlp.py`:

```python
"activation": "relu"
```

Esse parametro e passado para:

```python
mlp = MLPClassifier(**MLP_PARAMS)
```

Portanto, quem implementa a ativacao e o `MLPClassifier` do scikit-learn.

### Funcao ReLU

`relu` significa Rectified Linear Unit. Sua formula e:

```text
ReLU(x) = max(0, x)
```

Isso significa:

- se `x` for negativo, a saida e `0`;
- se `x` for positivo, a saida e o proprio `x`.

Exemplos:

```text
ReLU(-3) = 0
ReLU(0) = 0
ReLU(2.5) = 2.5
```

### Por que ReLU e usada

A ReLU e usada porque:

- adiciona nao linearidade ao modelo;
- ajuda o MLP a aprender relacoes mais complexas;
- costuma treinar mais rapido do que funcoes como sigmoid/tanh nas camadas ocultas;
- reduz o problema de gradientes muito pequenos em redes com varias camadas.

### Ativacao da saida

Embora o codigo defina explicitamente apenas:

```python
"activation": "relu"
```

isso vale para as camadas ocultas. A camada de saida e tratada internamente pelo `MLPClassifier`.

Como o problema e binario (`normal` ou `crime`), o scikit-learn usa uma saida probabilistica equivalente a uma ativacao logistica/sigmoide para produzir probabilidades.

A interface usa:

```python
model.predict_proba(X)[0][1]
```

Esse valor representa:

```text
P(crime)
```

## 11. Otimizador Adam

O parametro:

```python
"solver": "adam"
```

define o algoritmo de otimizacao usado no treinamento.

Adam ajusta os pesos da rede usando gradientes, mas adapta a taxa de aprendizado para cada parametro. Na pratica, ele costuma ser uma boa escolha para redes neurais porque:

- converge mais rapido em muitos problemas;
- e menos sensivel a escala dos gradientes;
- funciona bem com mini-batches;
- reduz a necessidade de ajuste manual fino em comparacao com gradiente descendente simples.

A taxa inicial de aprendizado e:

```python
"learning_rate_init": 0.001
```

E a estrategia:

```python
"learning_rate": "adaptive"
```

permite adaptar a taxa caso o treinamento pare de melhorar.

## 12. Early stopping

O treinamento usa:

```python
"early_stopping": True
"validation_fraction": 0.1
"n_iter_no_change": 5
```

Isso significa que 10% dos dados de treino sao separados internamente para validacao. Se o modelo nao melhora por 5 iteracoes consecutivas, o treinamento pode parar antes de `max_iter = 200`.

Esse mecanismo ajuda a reduzir overfitting.

## 13. Por que usar MLP

O MLP foi escolhido porque o problema nao e apenas linear.

As features da ResNet50 representam informacoes visuais complexas. A separacao entre `crime` e `normal` pode depender de combinacoes nao triviais dessas features.

Exemplos:

- presenca de pessoas;
- postura corporal;
- contexto da cena;
- objetos visiveis;
- padroes de movimento implicitos no frame;
- aparencia de fogo, explosao, tumulto ou acidente;
- composicao geral da imagem.

Um MLP com camadas ocultas consegue aprender combinacoes nao lineares entre as 2048 features.

## 14. Por que nao usar Perceptron

O Perceptron classico possui uma limitacao importante: ele aprende apenas uma fronteira de decisao linear.

Em termos simples, ele tenta separar as classes com uma unica reta, plano ou hiperplano:

```text
w1*x1 + w2*x2 + ... + wn*xn + b
```

Para problemas simples e linearmente separaveis, isso pode funcionar. Mas a classificacao de cenas de crime em frames reais nao tende a ser linearmente separavel.

Limites do Perceptron neste projeto:

- nao possui camadas ocultas;
- nao aprende combinacoes nao lineares profundas;
- nao produz uma probabilidade bem calibrada como `P(crime)`;
- e menos adequado para features visuais complexas;
- pode falhar quando crime e normal compartilham muitos padroes visuais parecidos.

No VanDetector, dois frames podem ter pessoas, mesas, carros ou ruas, mas apenas um deles representar crime. Essa diferenca pode depender de uma combinacao complexa de sinais visuais.

## 15. Por que nao usar Adaline

Adaline, Adaptive Linear Neuron, tambem e um modelo linear. Ele usa uma funcao de ativacao linear durante o treinamento e otimiza erro quadratico medio.

A ideia geral e:

```text
saida = w1*x1 + w2*x2 + ... + wn*xn + b
```

Depois, a decisao pode ser feita por um limiar.

Adaline e util para estudar fundamentos de aprendizado supervisionado, gradiente descendente e ajuste de pesos. Porem, para este problema, ele tem limitacoes parecidas com o Perceptron:

- fronteira de decisao linear;
- pouca capacidade de modelar relacoes complexas;
- menor flexibilidade para dados visuais;
- desempenho limitado quando as classes se misturam no espaco de features.

O MLP pode ser visto como uma extensao mais poderosa: varias camadas, funcoes de ativacao nao lineares e capacidade de aprender representacoes intermediarias.

## 16. Comparacao resumida

| Modelo | Tipo | Fronteira de decisao | Camadas ocultas | Probabilidade | Adequado para este projeto |
| --- | --- | --- | --- | --- | --- |
| Perceptron | Linear | Linear | Nao | Nao ideal | Baixo |
| Adaline | Linear | Linear | Nao | Nao ideal | Baixo |
| MLP | Nao linear | Nao linear | Sim | Sim, via `predict_proba` | Alto |

## 17. Como o codigo classifica um frame no app

Na interface `app_ucf.py`, o processo principal acontece em:

```python
features = cv2_frame_to_features(frame)
prediction, confidence = classify_ucf_frame(model, features)
```

A funcao `cv2_frame_to_features` esta em:

```text
src/ucf_features.py
```

Ela:

1. converte o frame OpenCV de BGR para RGB;
2. aplica o mesmo preprocessamento da ResNet50;
3. extrai o vetor de 2048 features;
4. aplica o scaler salvo em `models/ucf_scaler.pkl`;
5. retorna as features em formato de dicionario.

A funcao `classify_ucf_frame` converte esse dicionario em vetor ordenado:

```python
feature_names = sorted(
    frame_features.keys(),
    key=lambda name: int(name.rsplit("_", 1)[-1])
)
```

Essa ordenacao numerica e importante porque evita erro como:

```text
feature_10 antes de feature_2
```

Depois o modelo calcula:

```python
crime_probability = float(model.predict_proba(X)[0][1])
```

E decide:

```python
label = "crime" if crime_probability >= 0.5 else "normal"
```

Na interface, ainda existe um limiar configuravel:

```python
DEFAULT_CRIME_CONFIDENCE = 0.6
```

Ou seja, mesmo que a rede considere `crime` a partir de 0.5, o app pode exigir uma confianca maior para registrar o evento.

## 18. Filtro temporal

O app nao registra crime com base em um unico frame isolado. Ele usa uma janela temporal:

```python
DEFAULT_TEMPORAL_WINDOW = 10
DEFAULT_MIN_CRIME_FRAMES = 7
```

Isso significa:

- o app observa os ultimos 10 frames analisados;
- se pelo menos 7 forem classificados como crime acima do limiar, registra um evento.

Esse filtro reduz falsos positivos pontuais.

Exemplo:

```text
Janela de 10 frames:
normal, crime, crime, crime, crime, normal, crime, crime, crime, crime
```

Nesse caso existem 8 frames de crime na janela. Como `8 >= 7`, o evento e registrado.

## 19. Logs e navegacao de frames

Quando um evento e detectado, o app registra em:

```text
outputs/logs/ucf_crime_events.csv
```

As colunas sao:

```text
timestamp, source, mode, frame, confidence, prediction
```

O app tambem guarda frames registrados como crime em memoria, durante a execucao, para permitir navegacao visual apos a analise.

A navegacao exibe:

- indice do frame coletado;
- numero do frame no video;
- tempo aproximado em segundos;
- confianca;
- quantidade de frames positivos na janela temporal;
- imagem anotada do frame.

## 20. Resultados atuais do treinamento

Segundo `models/ucf_mlp_report.txt`, os resultados atuais sao:

| Metrica | Treino | Teste |
| --- | ---: | ---: |
| Accuracy | 0.9975 | 0.8026 |
| AUC | 0.9999 | 0.8656 |

Relatorio no conjunto de teste:

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Normal | 0.83 | 0.88 | 0.85 | 64.952 |
| Crime | 0.75 | 0.67 | 0.71 | 35.638 |

Interpretacao:

- O resultado de treino e muito alto.
- O resultado de teste e menor, indicando diferenca entre treino e teste e possivel overfitting.
- O recall de `Crime` no teste e 0.67, ou seja, aproximadamente 33% dos frames de crime do teste nao foram identificados como crime.
- Para uma aplicacao de vigilancia, esse recall pode ser insuficiente se o objetivo for minimizar crimes nao detectados.

## 21. Limitacoes da abordagem atual

A principal limitacao e que o modelo classifica frames individualmente.

Eventos como:

- arremessar objetos;
- quebrar mesa de vidro;
- chutar portas;
- destruir objetos;
- iniciar uma briga;

sao acoes temporais. Muitas vezes, um unico frame nao mostra a acao completa. O movimento acontece ao longo de varios frames.

Por isso, um MLP baseado em frame isolado pode falhar em cenas onde:

- o objeto arremessado aparece pequeno;
- o impacto acontece rapido;
- o vidro quebrado e visualmente sutil;
- a cena parece normal antes ou depois da acao;
- o comportamento suspeito depende mais do movimento do que da aparencia estatica.

## 22. Como melhorar a confiabilidade

Para aumentar a confiabilidade da deteccao de vandalismo e crime, especialmente em casos de arremesso e quebra de objetos, as melhorias recomendadas sao:

1. Coletar falsos negativos usando a navegacao de frames do app.
2. Criar um conjunto proprio com exemplos de vandalismo, arremesso, impacto e quebra de vidro.
3. Balancear melhor o dataset, reduzindo excesso de frames normais ou aumentando exemplos positivos.
4. Ajustar o limiar de `P(crime)` para favorecer recall quando for mais importante detectar crime do que evitar falso positivo.
5. Treinar com sequencias de frames, nao apenas frames isolados.
6. Testar modelos temporais como LSTM, GRU, 3D CNN, SlowFast, I3D ou VideoMAE.
7. Combinar a abordagem UCF com YOLO para localizar objetos, pessoas e regioes relevantes.
8. Usar validacao com videos parecidos com o ambiente real da aplicacao.

## 23. Arquivos principais

| Arquivo | Funcao |
| --- | --- |
| `prepare_ucf_dataset.py` | Gera o CSV com paths, categorias, labels e split. |
| `extract_ucf_features.py` | Usa ResNet50 para extrair 2048 features por frame. |
| `train_ucf_mlp.py` | Treina o MLP com as features extraidas. |
| `src/ucf_features.py` | Extrai features em tempo real para uso no app. |
| `app_ucf.py` | Interface Streamlit para analisar videos e imagens. |
| `models/ucf_crime_mlp.pkl` | Modelo MLP treinado. |
| `models/ucf_scaler.pkl` | Normalizador usado antes do MLP. |
| `models/ucf_mlp_report.txt` | Relatorio de metricas do treinamento. |
| `models/ucf_mlp_evaluation.png` | Graficos de avaliacao. |

## 24. Comandos principais

Preparar o CSV:

```powershell
python prepare_ucf_dataset.py
```

Extrair features:

```powershell
python extract_ucf_features.py
```

Treinar o MLP:

```powershell
python train_ucf_mlp.py
```

Executar a interface:

```powershell
streamlit run app_ucf.py
```

Ou, usando o ambiente virtual do projeto:

```powershell
.\venv\Scripts\python.exe -m streamlit run app_ucf.py
```

## 25. Configuracao do computador usado

O modelo MLP registrado em `models/ucf_crime_mlp.pkl` foi treinado no ambiente local do projeto, com a seguinte configuracao identificada em 28/04/2026:

| Componente | Configuracao |
| --- | --- |
| Sistema operacional | Microsoft Windows 11 Pro 64 bits |
| Processador | AMD Ryzen 7 5700X 8-Core Processor |
| Nucleos fisicos | 8 |
| Processadores logicos / threads | 16 |
| Clock maximo informado | 3401 MHz |
| Memoria RAM | 32 GB aproximados |
| GPU disponivel | NVIDIA GeForce RTX 4060 |
| Memoria da GPU | 8188 MiB, conforme `nvidia-smi` |
| Driver NVIDIA | 595.79 |
| CUDA reportado pelo driver | 13.2 |
| Python do ambiente | Python 3.14.3 |
| scikit-learn | 1.8.0 |
| NumPy | 2.4.4 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |

### Uso de CPU e GPU

O treino do arquivo `train_ucf_mlp.py` usa `sklearn.neural_network.MLPClassifier`. Esse classificador do scikit-learn treina em CPU, nao em GPU.

Portanto, mesmo existindo uma RTX 4060 disponivel na maquina, o treinamento do MLP em si foi executado no processador.

A GPU pode ser usada na etapa anterior, de extracao de features, porque `extract_ucf_features.py` usa PyTorch:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Assim, a divisao correta e:

| Etapa | Arquivo | Biblioteca principal | Dispositivo |
| --- | --- | --- | --- |
| Extracao de features ResNet50 | `extract_ucf_features.py` | PyTorch / torchvision | GPU se CUDA estiver disponivel; caso contrario CPU |
| Treino do MLP | `train_ucf_mlp.py` | scikit-learn | CPU |
| Inferencia no app | `app_ucf.py` e `src/ucf_features.py` | PyTorch + scikit-learn | ResNet50 pode usar GPU; MLP usa CPU |

O `MLPClassifier` tambem nao define explicitamente uma quantidade de nucleos via `n_jobs`. Operacoes internas de algebra linear podem usar threads das bibliotecas numericas instaladas, mas o codigo do treino nao fixa esse numero. A maquina possui ate 16 threads logicas disponiveis.

## 26. Conclusao

A abordagem UCF + ResNet50 + MLP e uma solucao pratica para transformar videos em uma classificacao binaria de crime ou normal.

O uso da ResNet50 reduz drasticamente a complexidade do problema, convertendo imagens em vetores de 2048 features. O MLP, por sua vez, e mais adequado que Perceptron ou Adaline porque consegue aprender relacoes nao lineares entre essas features.

Apesar disso, a abordagem atual ainda tem uma limitacao importante: ela opera principalmente por frame. Para detectar melhor vandalismo dinamico, como arremesso de objetos e quebra de vidro, o proximo passo tecnicamente mais forte e adicionar modelagem temporal com sequencias de frames ou combinar o classificador com deteccao/localizacao via YOLO.
