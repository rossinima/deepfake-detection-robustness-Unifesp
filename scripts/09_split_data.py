import os
import shutil
import random

BASE_DIR = "frames"
SCENARIOS = ["hq", "q60", "q30", "q10"]
TRAIN_SPLIT = 0.65
OUTPUT_DIR = "frames_split"

# Limpar execução anterior 
if os.path.exists(OUTPUT_DIR):
    print(f"Limpando pasta antiga {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)

# Processar cada classe (Real e Fake) 
for label in ["videos_real", "videos_fake"]:
    sample_path = os.path.join(BASE_DIR, "hq", label)
    
    if not os.path.exists(sample_path):
        print(f"Aviso: Pasta {sample_path} não encontrada. Pulando...")
        continue

    
    video_ids = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]
    random.shuffle(video_ids)

    split_idx = int(len(video_ids) * TRAIN_SPLIT)
    train_ids = video_ids[:split_idx]
    
    print(f"Classe {label}: Total {len(video_ids)} | Treino {len(train_ids)} | Teste {len(video_ids)-len(train_ids)}")

    for scenario in SCENARIOS:
        for vid in video_ids:
            src = os.path.join(BASE_DIR, scenario, label, vid)
            if not os.path.exists(src): continue
            
            category = "train" if vid in train_ids else "test"
            dest = os.path.join(OUTPUT_DIR, category, scenario, label, vid)
            
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)

print("\n--- SUCESSO! Verifique se 'frames_split' agora tem as duas pastas em cada cenário ---")