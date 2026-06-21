# VanDetector: Detecção de Vandalismo em Vídeos por YOLO e MLP

## Resumo

O VanDetector é um sistema de visão computacional para identificar cenários de vandalismo e possíveis atos de destruição em vídeos. A solução combina detecção de objetos por YOLO com um Perceptron Multicamadas (MLP). O YOLO interpreta cada frame e gera caixas delimitadoras, classes e coordenadas; o MLP recebe uma representação numérica dessas detecções e estima a probabilidade de vandalismo. A arquitetura separa a percepção visual da decisão final, permitindo usar um classificador supervisionado leve sobre sinais espaciais e quantitativos.

O projeto mantém ainda uma linha experimental UCF-Crime, baseada em ResNet50 e MLP para classificar frames como crime ou normal. Como o UCF-Crime não contém as classes específicas de vandalismo, quebra de porta e destruição, o fluxo YOLO + MLP é a referência para os assets de vandalismo.

**Palavras-chave:** visão computacional; vandalismo; YOLO; rede neural artificial; MLP; análise de vídeo.

## 1. Objetivo

O objetivo do sistema é automatizar a triagem de vídeos de monitoramento, destacando trechos que possam conter vandalismo, dano patrimonial ou comportamento destrutivo. A saída não substitui a revisão humana: ela prioriza os vídeos e instantes que devem ser analisados.

~~~text
vídeo -> frames amostrados -> YOLO -> atributos numéricos -> MLP
      -> P(vandalismo) -> alerta ou normal
~~~

## 2. Tecnologias utilizadas

| Tecnologia | Papel no projeto |
| --- | --- |
| Python | Linguagem principal. |
| Ultralytics YOLO | Detecção visual em cada frame. |
| OpenCV | Leitura dos vídeos, FPS, contagem e captura de frames. |
| scikit-learn | MLPClassifier, StandardScaler e métricas. |
| NumPy e Pandas | Operações numéricas e organização tabular dos atributos. |
| Joblib | Persistência do modelo treinado. |
| Streamlit | Interface web para análise de imagens e vídeos. |
| PyTorch e Torchvision | Infraestrutura de deep learning e ResNet50 da linha UCF. |
| h5py | Armazenamento de características em HDF5. |
| Matplotlib e Seaborn | Gráficos e matrizes de confusão. |
| PyYAML | Leitura de configurações de datasets YOLO. |

As dependências e versões mínimas estão registradas em requirements.txt.

## 3. Arquitetura especializada em vandalismo

O fluxo validado para os assets utiliza os pesos YOLO em runs/detect/vandalism2/weights/best.pt e o classificador models/vandalism_mlp.pkl.

~~~text
Arquivo de vídeo
      |
      v
OpenCV: amostragem de frames
      |
      v
YOLO: classes, caixas, confiança e coordenadas normalizadas
      |
      v
Extração de atributos geométricos e contagens por classe
      |
      v
StandardScaler + MLP
      |
      v
P(vandalismo) e decisão por limiar
~~~

O YOLO é responsável pela percepção visual. O MLP não recebe a imagem diretamente; ele combina os atributos derivados das detecções e retorna uma decisão binária. Essa separação torna possível inspecionar as variáveis usadas pela etapa de classificação.

## 4. Extração de atributos

O módulo src/roboflow_features.py converte as caixas YOLO no vetor de entrada do MLP. Para cada frame, são extraídos:

| Grupo | Atributos |
| --- | --- |
| Volume | total_detections |
| Área | max_area, sum_area, mean_area |
| Dimensão | max_width, max_height |
| Posição | mean_center_x, mean_center_y |
| Semântica | count_nome-da-classe para cada classe do detector |

Para uma caixa de largura normalizada w e altura normalizada h, a área é calculada como a = w × h. Os valores são agregados para todas as detecções do frame. Na ausência de caixas, as contagens e medidas geométricas assumem valor zero, evitando dados ausentes.

No detector usado para a validação dos assets, as classes são normal e vandalism-B5md. Na base combinada experimental, breaking_the_door e destroying são remapeadas para a classe positiva vandalism; normal, person e car são utilizadas como contexto não positivo.

## 5. Dados e rotulagem

Foram considerados dois conjuntos no formato YOLO:

1. **Vandalism2:** classes normal e vandalism-B5md.
2. **Vandalism-nighttime:** classes breaking_the_door, car, destroying e person.

Os arquivos de rótulo YOLO contêm a classe e a geometria dos objetos anotados. Esses rótulos são transformados em uma tabela com atributos, rótulo binário, split e identificação da imagem.

No experimento combinado, foram geradas 3.305 imagens: 1.692 negativas e 1.613 positivas. O holdout usado no relatório possui 514 amostras: 253 normais e 261 positivas.

Além das imagens rotuladas, o projeto possui três vídeos para validação funcional, definidos em asset_cases.json:

| Caso | Rótulo esperado |
| --- | --- |
| assets/vandalismbrasil1.mp4 | vandalism |
| assets/vandalismbrasil2.mp4 | vandalism |
| Vídeo de multidão em Times Square | normal |

## 6. Treinamento do MLP

O script train_vandalism_mlp.py implementa o experimento de vandalismo e destruição.

~~~text
atributos tabulares -> StandardScaler -> MLPClassifier -> classe binária
~~~

A normalização é necessária porque os atributos possuem escalas diferentes: contagens são inteiros, coordenadas são normalizadas entre zero e um e áreas representam proporções do frame.

| Hiperparâmetro | Configuração |
| --- | --- |
| Camadas ocultas | (32, 16) |
| Ativação | ReLU |
| Otimizador | Adam |
| Máximo de iterações | 500 |
| Parada antecipada | Habilitada |
| Semente aleatória | 42 |

Simplificadamente, cada camada oculta calcula h = ReLU(Wx + b). A camada final fornece a probabilidade da classe positiva. O Adam ajusta os pesos para reduzir o erro de classificação durante o treinamento.

A avaliação utiliza precisão, recall, F1-score, acurácia e matriz de confusão. Os exemplos de treino são usados no ajuste, enquanto valid e test são usados para avaliação.

## 7. Resultados do experimento combinado

O relatório models/vandalism_destruction_mlp.report.txt registrou os seguintes resultados no holdout:

| Classe | Precisão | Recall | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| Normal | 1,00 | 1,00 | 1,00 | 253 |
| Vandalismo ou destruição | 1,00 | 1,00 | 1,00 | 261 |
| **Acurácia global** |  |  | **1,00** | **514** |

A matriz de confusão foi [[253, 0], [0, 261]].

Esse resultado deve ser interpretado no escopo do holdout disponível. Antes de qualquer uso em campo, é necessário testar o modelo com vídeos independentes, coletados em condições diferentes das imagens de treinamento.

## 8. Validação de integração nos vídeos

O script validate_vandalism_assets.py é um teste de regressão de ponta a ponta. Ele percorre os vídeos definidos em asset_cases.json, amostra frames a cada 3 segundos, executa YOLO, cria o vetor de atributos e calcula a probabilidade pelo MLP.

Um vídeo é classificado como vandalismo quando pelo menos um frame atinge o limiar de 0,75. Caso contrário, é classificado como normal. O script gera um CSV e retorna erro se qualquer caso falhar.

Resultados registrados em outputs/validation/vandalism_assets.csv:

| Caso | Predição | Maior probabilidade | Primeiro alerta | Resultado |
| --- | --- | ---: | ---: | --- |
| vandalismbrasil1.mp4 | vandalism | 0,9123 | 18,0 s | aprovado |
| vandalismbrasil2.mp4 | vandalism | 0,9078 | 48,05 s | aprovado |
| Times Square | normal | 0,5603 | — | aprovado |

Portanto, os três assets rotulados foram classificados corretamente. O percentual de 100% se refere exclusivamente a esses três vídeos e não representa garantia de generalização para vídeos externos.

## 9. Linha complementar UCF-Crime

A linha UCF é mantida como experimento de classificação genérica de crime. Nela, uma ResNet50 pré-treinada remove a última camada de classificação e transforma cada frame em um vetor de 2.048 atributos.

~~~text
frame -> ResNet50 -> vetor de 2.048 atributos -> StandardScaler
      -> MLP (512, 256, 128) -> P(crime)
~~~

Os frames são redimensionados para 224 × 224 pixels e normalizados com parâmetros do ImageNet. As categorias Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery e Shooting foram agrupadas como crime; NormalVideos representa a classe negativa.

No relatório atual, essa linha obteve acurácia de teste de 0,7205 e AUC de 0,8025. Para a classe crime, foram obtidos precisão de 0,59, recall de 0,66 e F1-score de 0,63. Por não possuir classes específicas de vandalismo, esse modelo não é a referência para a validação dos assets.

## 10. Interface e saídas

A interface Streamlit em app_ucf.py permite enviar imagens e vídeos, configurar limiares e visualizar eventos detectados. A linha UCF oferece ainda filtro temporal por janela de frames e um mapa de suspeição por oclusão, usado para destacar regiões cuja ocultação reduz a probabilidade de crime.

No fluxo especializado, o relatório de validação inclui:

- vídeo analisado;
- classe esperada e classe predita;
- situação de aprovação;
- maior probabilidade estimada;
- tempo do primeiro alerta;
- número de frames amostrados.

## 11. Reprodutibilidade

### Instalação

~~~powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
~~~

### Validação dos assets

~~~powershell
.\venv\Scripts\python.exe validate_vandalism_assets.py
~~~

O comando usa, por padrão, asset_cases.json, os pesos YOLO em runs/detect/vandalism2/weights/best.pt e o modelo models/vandalism_mlp.pkl.

### Treinamento experimental da base combinada

~~~powershell
.\venv\Scripts\python.exe train_vandalism_mlp.py --dataset vandalism_mlp_dataset.csv --output models/vandalism_destruction_mlp.pkl
~~~

Um novo modelo só deve substituir o modelo de produção após passar pela validação de integração e por análise de falsos positivos e falsos negativos.

## 12. Limitações e trabalhos futuros

1. A validação funcional contém três vídeos; ela comprova o comportamento nesses assets, não em todos os cenários reais.
2. O MLP depende da qualidade das detecções YOLO. Uma falha de detecção pode impedir a decisão correta.
3. Vandalismo é um evento temporal: impactos, arremessos e quebras podem ocorrer em poucos frames.
4. Iluminação, câmera, resolução, multidões e objetos novos podem reduzir a generalização.
5. Os resultados devem ser complementados por validação separada por vídeo de origem, para evitar que frames semelhantes de uma mesma gravação apareçam em treino e teste.

Como continuidade, recomenda-se ampliar a base com vídeos do domínio de uso, anotar intervalos temporais de ação, criar splits por vídeo e comparar a solução com redes temporais como LSTM, GRU, 3D CNN, SlowFast ou VideoMAE.

## 13. Conclusão

O VanDetector demonstra uma arquitetura híbrida para análise de vandalismo em vídeos. O YOLO realiza a percepção visual e o MLP integra atributos geométricos e semânticos para a decisão final. A solução possui validação automatizada para os assets do projeto e classificou corretamente os três casos rotulados disponíveis. A evolução do trabalho deve concentrar-se em ampliar os dados e medir a robustez em vídeos independentes e temporalmente variados.
