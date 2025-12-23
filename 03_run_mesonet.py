import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import glob
from tensorflow.keras.preprocessing.image import img_to_array

# Importa o "corpo" do nosso modelo, do arquivo que já criamos
from models.mesonet_model import Meso4

# --- Configurações ---
MODEL_WEIGHTS_PATH = "models/Meso4_DF.h5" # O "cérebro" que baixamos
IMAGE_ROOT_DIR = "frames"                 # A pasta que contém 'hq' e 'lq'
RESULTS_FILE = "results.csv"              # O arquivo final onde salvaremos as notas

# --- 1. Carregar o Modelo ---
print("Carregando o modelo MesoNet...")

# Cria o "corpo" do modelo
try:
    model = Meso4(input_shape=(256, 256, 3))
    # Coloca o "cérebro" (pesos) dentro do corpo
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("Modelo e pesos carregados com sucesso.")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    print("Verifique se os arquivos 'models/mesonet_model.py' e 'models/Meso4_DF.h5' estão corretos.")
    exit()

# --- 2. Preparar Coleta de Resultados ---
results_list = [] # Uma lista para guardar os dicionários de resultados

# --- 3. Processar as Imagens ---
start_time = time.time()
print("Iniciando detecção... Isso pode levar alguns minutos.")

# Vamos rodar nos dois cenários: 'hq' e 'lq'
for scenario in ["hq", "lq"]:
    
    current_image_dir = os.path.join(IMAGE_ROOT_DIR, scenario)
    
    # Encontra todas as imagens .jpg em todas as sub-pastas
    search_path = os.path.join(current_image_dir, "**", "*.jpg")
    image_files = glob.glob(search_path, recursive=True)
    
    if not image_files:
        print(f"\nAviso: Nenhuma imagem encontrada em '{current_image_dir}'. Pulando.")
        continue
        
    print(f"\nProcessando {len(image_files)} imagens do cenário: '{scenario}'")
    
    # Usa tqdm para a barra de progresso
    for img_path in tqdm(image_files, desc=f"Cenário {scenario}", unit="imagem"):
        try:
            # Descobre o rótulo (label) e o nome do vídeo original
            # O caminho é algo como: frames\hq\videos_fake\v1\frame_0.jpg
            parts = img_path.split(os.path.sep)
            label_str = parts[-3] # 'videos_fake' ou 'videos_real'
            video_name = parts[-2] # 'v1'
            frame_name = parts[-1] # 'frame_0.jpg'
            
            # Converte o rótulo para 1 (fake) ou 0 (real)
            label_int = 1 if label_str == "videos_fake" else 0
            
            # Carrega a imagem e prepara para o modelo
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256)) # O MesoNet espera 256x256
            img_array = img_to_array(img)
            img_array = img_array / 255.0     # Normaliza (de 0-255 para 0.0-1.0)
            img_array = np.expand_dims(img_array, axis=0) # Cria um "batch" de 1 imagem
            
            # === A MÁGICA ACONTECE AQUI ===
            # O modelo dá a "nota" (score)
            prediction_score = model.predict(img_array, verbose=0)[0][0]
            
            # Salva o resultado
            results_list.append({
                "video": video_name,
                "frame": frame_name,
                "label": label_int,
                "label_str": label_str,
                "scenario": scenario,
                "score": prediction_score
            })
            
        except Exception as e:
            print(f"Erro ao processar imagem {img_path}: {e}")

# --- 4. Salvar os Resultados ---
print("\nProcessamento de detecção concluído.")

if results_list:
    # Converte a lista de resultados em um DataFrame do Pandas
    df = pd.DataFrame(results_list)
    
    # Salva o DataFrame em um arquivo CSV
    df.to_csv(RESULTS_FILE, index=False, encoding='utf-8')
    
    end_time = time.time()
    print(f"\n--- SUCESSO ---")
    print(f"Tempo total de detecção: {((end_time - start_time) / 60):.2f} minutos")
    print(f"Resultados (scores) salvos em: {os.path.abspath(RESULTS_FILE)}")
else:
    print("Nenhum resultado foi gerado. Verifique as pastas de frames.")