import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import pandas as pd
import os

IMG_SIZE = 224
TRAIN_DIR = "frames_split/train/hq"
TEST_ROOT = "frames_split/test"

def build_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(base.input, x)

# 1. Carregar Treino 
print("\n>>> Carregando dados de treino...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, 
    image_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=32,
    label_mode='binary' 
)

# 2. Treino Padrão
print("\n>>> Treinando Modelo Padrão (Sem aumento de dados)...")
model_std = build_model()
model_std.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_std.fit(train_ds, epochs=3)

# 3. Treino Robusto 
print("\n>>> Treinando Modelo Robusto (Com Simulação de Compressão)...")
augmentation = models.Sequential([
    layers.RandomContrast(0.2),
    layers.GaussianNoise(0.1) 
])

model_robust = build_model()
model_robust.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_ds_robust = train_ds.map(lambda x, y: (augmentation(x), y))
model_robust.fit(train_ds_robust, epochs=3)

# 4. Avaliação Cruzada
results = []
for scenario in ["hq", "q60", "q30", "q10"]:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(TEST_ROOT, scenario),
        image_size=(IMG_SIZE, IMG_SIZE),
        label_mode='binary'
    )
    acc_std = model_std.evaluate(test_ds, verbose=0)[1]
    acc_robust = model_robust.evaluate(test_ds, verbose=0)[1]
    results.append({"Cenário": scenario, "Acurácia_Padrão": acc_std, "Acurácia_Robusta": acc_robust})

df = pd.DataFrame(results)
df.to_csv("VALIDACAO_ROBUSTEZ_FINAL.csv", index=False)
print("\n--- RESULTADO FINAL ---")
print(df)