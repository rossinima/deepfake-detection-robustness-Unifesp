import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm
import time

#  Configurações 

VIDEO_ROOT = "data"       
FRAME_ROOT = "frames/hq"  # Pasta de saída para os rostos (HQ = High Quality)
SAMPLE_RATE = 15          # Salvar 1 frame a cada 15 frames 
IMG_SIZE = 256            # Tamanho final da imagem do rosto (256x256 pixels)
PAD = 20                  # Um preenchimento (padding) para pegar um pouco do contexto do rosto

# Inicia
try:
    detector = MTCNN()
    print("Carregando detector de rostos MTCNN...")
except Exception as e:
    print(f"Erro ao carregar MTCNN: {e}")
    print("Verifique se o TensorFlow está instalado corretamente.")
    exit()

def process_video(video_path, output_dir):
    """
    Abre um vídeo, detecta rostos, recorta e salva os frames.
    """
    
    video_name = os.path.basename(video_path).split('.')[0]
    video_output_dir = os.path.join(output_dir, video_name)
    

    if os.path.exists(video_output_dir): 
        return 0 
        
    os.makedirs(video_output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return 0

    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        
        if frame_count % SAMPLE_RATE == 0:
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(f"Erro ao converter frame (vídeo: {video_name}, frame: {frame_count}): {e}")
                continue 
            
            # Detecta rostos
            results = detector.detect_faces(frame_rgb)
            
            if results:
                # Pega o primeiro rosto (geralmente o maior)
                result = results[0]
                x, y, w, h = result['box']
                
                # Adiciona padding e garante que não saia dos limites da imagem
                x1 = max(0, x - PAD)
                y1 = max(0, y - PAD)
                x2 = min(frame.shape[1], x + w + PAD)
                y2 = min(frame.shape[0], y + h + PAD)
                
                try:
                    # Recorta o rosto do frame original (BGR)
                    face = frame[y1:y2, x1:x2]
                    # Redimensiona para o tamanho padrão
                    resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    
                    # Salva a imagem
                    save_path = os.path.join(video_output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(save_path, resized_face)
                    saved_count += 1
                except Exception as e:
                    
                    pass

        frame_count += 1
        
    cap.release()
    return saved_count


start_time = time.time()
print("Iniciando extração de rostos... Isso pode demorar.")

total_saved = 0
videos_processados = 0

for label_folder in os.listdir(VIDEO_ROOT):
    label_path = os.path.join(VIDEO_ROOT, label_folder)
    
    if os.path.isdir(label_path):
        output_label_path = os.path.join(FRAME_ROOT, label_folder)
        
        video_files = [f for f in os.listdir(label_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print(f"Nenhum vídeo (.mp4, .avi, .mov) encontrado em {label_path}")
            continue
            
        print(f"\nProcessando pasta: {label_folder}")
        
        
        for video_file in tqdm(video_files, desc=f"Pasta {label_folder}", unit="vídeo"):
            video_path = os.path.join(label_path, video_file)
            saved = process_video(video_path, output_label_path)
            total_saved += saved
            if saved > 0:
                videos_processados += 1

end_time = time.time()
print("\n--- Processamento Concluído ---")
print(f"Tempo total: {((end_time - start_time) / 60):.2f} minutos")
print(f"Vídeos processados: {videos_processados}")
print(f"Total de frames de rosto salvos: {total_saved}")
print(f"Frames salvos em: {os.path.abspath(FRAME_ROOT)}")