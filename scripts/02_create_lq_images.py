import os
import cv2
from tqdm import tqdm
import time
import glob

# --- Configurações ---
HQ_FRAME_ROOT = "frames/hq"      # Pasta de ONDE VAMOS LER (os originais)
BASE_OUTPUT_ROOT = "frames"      # Pasta base para as novas compressões
QUALITIES = [60, 30, 10]         # Níveis de qualidade JPEG que vamos testar

#  Execução Principal 
start_time = time.time()
print(f"Iniciando criação de múltiplos níveis de compressão a partir de '{HQ_FRAME_ROOT}'...")

# 1. Encontrar TODAS as imagens HQ
search_path = os.path.join(HQ_FRAME_ROOT, "**", "*.jpg")
hq_image_files = glob.glob(search_path, recursive=True)

if not hq_image_files:
    print(f"ERRO: Nenhuma imagem .jpg encontrada em '{HQ_FRAME_ROOT}'. Verifique a pasta.")
    exit()

print(f"Encontradas {len(hq_image_files)} imagens HQ para processar.\n")

# 2. Processar para cada nível de qualidade
for quality in QUALITIES:
    current_lq_root = os.path.join(BASE_OUTPUT_ROOT, f"q{quality}")
    print(f"Gerando imagens para Qualidade {quality}% em '{current_lq_root}'...")
    
    total_saved = 0
    
    for hq_image_path in tqdm(hq_image_files, desc=f"Qualidade {quality}", unit="imagem"):
        try:
            # Carrega a imagem HQ
            img = cv2.imread(hq_image_path)
            
            if img is None:
                continue

            # Cria o novo caminho de saída 
            relative_path = os.path.relpath(hq_image_path, HQ_FRAME_ROOT)
            lq_image_path = os.path.join(current_lq_root, relative_path)
            
            # Cria a pasta de destino
            os.makedirs(os.path.dirname(lq_image_path), exist_ok=True)

            # Define os parâmetros de compressão
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
            # Salva a imagem
            cv2.imwrite(lq_image_path, img, compression_params)
            total_saved += 1
            
        except Exception as e:
            print(f"Erro ao processar imagem {hq_image_path}: {e}")
    
    print(f"Finalizado Qualidade {quality}: {total_saved} imagens salvas.\n")

end_time = time.time()
print("--- Processamento de Todos os Níveis Concluído ---")
print(f"Tempo total: {((end_time - start_time)):.2f} segundos")