import os
import sys
import cv2
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mesonet_model import Meso4

IMAGE_ROOT = "frames"
SCENARIOS = ["hq", "q60", "q30", "q10"]
FINAL_RESULTS_FILE = "results_ESTRESSE_COMPLETO.csv"

# funções de pré-processamento específicas de cada modelo
from tensorflow.keras.applications.xception import preprocess_input as prep_xception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as prep_efficient

MODELS_INFO = [
    {"name": "Xception", "path": "models/xception_model.keras", "size": 299, "prep": prep_xception},
    {"name": "MobileNetV2", "path": "models/mobilenet_model.keras", "size": 224, "prep": prep_mobilenet},
    {"name": "EfficientNetB0", "path": "models/efficientnet_model.keras", "size": 224, "prep": prep_efficient}
]

results_list = []

print("--- INICIANDO AVALIAÇÃO DE ESTRESSE RECURSIVA ---")

# 1. AVALIAÇÃO DOS MODELOS TREINADOS
for m_info in MODELS_INFO:
    print(f"\n>>> Avaliando Modelo: {m_info['name']}")
    if not os.path.exists(m_info['path']):
        print(f"Erro: Arquivo {m_info['path']} não encontrado!")
        continue
    
    model = load_model(m_info['path'])
    
    for scenario in SCENARIOS:
        
        search_pattern = os.path.join(IMAGE_ROOT, scenario, "**", "*.jpg")
        image_paths = glob.glob(search_pattern, recursive=True)
        
        if not image_paths:
            print(f"Aviso: Nenhuma imagem encontrada em {scenario}")
            continue

        print(f"   Processando {scenario}: {len(image_paths)} imagens")
        
        for img_path in tqdm(image_paths, desc=f"      {scenario}", leave=False):
            # Identifica se é real ou fake pelo caminho da pasta
            label = 1 if "videos_fake" in img_path else 0
            
            img = cv2.imread(img_path)
            img = cv2.resize(img, (m_info['size'], m_info['size']))
            img_array = np.expand_dims(img.astype(np.float32), axis=0)
            img_array = m_info['prep'](img_array) # Usa o pré-processamento correto
            
            score = model.predict(img_array, verbose=0)[0][0]
            results_list.append({"model": m_info['name'], "scenario": scenario, "label": label, "score": score})

# 2. AVALIAÇÃO DO MESONET
print("\n>>> Avaliando Modelo: MesoNet (Pré-treinado)")
model_meso = Meso4(input_shape=(256, 256, 3))
weights_path = "models/Meso4_DF.h5"

if os.path.exists(weights_path):
    model_meso.load_weights(weights_path)
    for scenario in SCENARIOS:
        search_pattern = os.path.join(IMAGE_ROOT, scenario, "**", "*.jpg")
        image_paths = glob.glob(search_pattern, recursive=True)
        
        for img_path in tqdm(image_paths, desc=f"      {scenario}", leave=False):
            label = 1 if "videos_fake" in img_path else 0
            img = cv2.resize(cv2.imread(img_path), (256, 256))
            img = np.expand_dims(img / 255.0, axis=0) # MesoNet normaliza 0-1
            score = model_meso.predict(img, verbose=0)[0][0]
            results_list.append({"model": "MesoNet (Incompatível)", "scenario": scenario, "label": label, "score": score})
else:
    print("Pesos do MesoNet não encontrados.")

# 3. SALVAR 
df = pd.DataFrame(results_list)
df.to_csv(FINAL_RESULTS_FILE, index=False)
print(f"\n--- SUCESSO! Resultados salvos em: {FINAL_RESULTS_FILE} ---")