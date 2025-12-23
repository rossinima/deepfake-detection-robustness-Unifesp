# Análise de Robustez de Arquiteturas CNN na Detecção de Deepfakes sob Compressão

Este repositório contém o código-fonte e os experimentos realizados para avaliar o impacto da compressão JPEG (comumente aplicada por redes sociais) na eficácia de diferentes modelos de Deep Learning na detecção de vídeos manipulados. O projeto foca em quantificar o fenômeno do **"Compression Crash"** e validar a necessidade de treinamentos robustos.

## 📂 Estrutura do Projeto

Abaixo estão detalhadas as funções de cada pasta e arquivo visíveis na estrutura do diretório:

### 📁 Diretórios

- **`data/`**: Contém as sequências de vídeo originais do dataset, divididas em `videos_real` (originais) e `videos_fake` (manipulados).
- **`frames/`**: Armazena os rostos extraídos dos vídeos, organizados por níveis de qualidade: `hq` (alta qualidade), `q60`, `q30` e `q10` (baixa qualidade/alta compressão).
- **`frames_split/`**: Contém os dados organizados para a validação final, divididos em conjuntos de `train` (treino) e `test` (teste) seguindo uma separação rigorosa de 65/35 por IDs de vídeo para evitar vazamento de dados (_data leakage_).
- **`models/`**: Pasta destinada aos arquivos de pesos dos modelos treinados (`.keras` e `.h5`) e definições de arquitetura.
- **`scripts/`**: Contém todos os códigos em Python responsáveis pelo processamento, treinamento e avaliação do projeto.

---

### 📜 Descrição dos Scripts (`scripts/`)

Os scripts devem ser seguidos conforme a numeração para reproduzir os experimentos:

1.  **`01_extract_faces.py`**: Realiza a detecção facial e a extração sistemática de frames dos vídeos brutos.
2.  **`02_create_lq_images.py`**: Gera as versões comprimidas das imagens originais (HQ) nos níveis q60, q30 e q10 para simular a degradação do canal de transmissão.
3.  **`03_run_mesonet.py`** e **`04_run_mesonet_F2F.py`**: Executam as predições utilizando a arquitetura MesoNet como linha de base (_baseline_) para diferentes métodos de manipulação.
4.  **`05_train_xception.py`**: Código para o treinamento (via _Transfer Learning_) do modelo Xception.
5.  **`06_train_mobilenet.py`**: Código para o treinamento do modelo MobileNetV2, focado em eficiência computacional.
6.  **`07_train_efficientnet.py`**: Realiza o treinamento do EfficientNetB0, explorando seus blocos de atenção para maior resiliência.
7.  **`08_stress_evaluation.py`**: Executa o teste de estresse cruzado, avaliando modelos treinados em HQ contra todos os níveis de compressão.
8.  **`09_split_data.py`**: Realiza a divisão automática dos dados por IDs de vídeo para garantir uma validação de generalização justa.
9.  **`10_robust_validation.py`**: Script de validação final que compara o desempenho de modelos padrão versus modelos treinados com simulação de ruído e compressão.

---

### 📊 Arquivos de Análise e Resultados

- **`analise.ipynb`**: Notebook utilizado para a visualização dos dados, geração de gráficos de acurácia e curvas AUC.
- **`MatrizDeConfunsao(q10).png`**: Representação visual do erro sistemático induzido pela compressão severa, evidenciando o aumento de falsos positivos.
- **`VALIDACAO_ROBUSTEZ_FINAL.csv`**: Arquivo com as métricas consolidadas do teste de generalização cruzada.
- **`results_*.csv`**: Arquivos contendo as predições e métricas brutas de cada arquitetura testada.

---

## 💾 Dataset Utilizado

Os experimentos foram conduzidos utilizando o **SDFVD (Self-Deepfake Video Dataset)**.

- **Link para acesso e download**: [Mendeley Data - SDFVD Dataset](https://data.mendeley.com/datasets/bcmkfgct2s/1).

---

## 👥 Autoras

- Marcela Rossini
- Bruna Surur Bergara

_Projeto desenvolvido para a disciplina de Segurança da Informação - ICT/UNIFESP._
