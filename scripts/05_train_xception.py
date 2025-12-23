import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import glob
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

HQ_FRAME_ROOT = "frames/hq"
IMAGE_SIZE = 299 
RESULTS_FILE = "results_xception_TRAINADO.csv"
MODEL_SAVE_PATH = "models/xception_model.keras" 


search_path = os.path.join(HQ_FRAME_ROOT, "**", "*.jpg")
hq_image_files = glob.glob(search_path, recursive=True)
data = [{"path": f, "label": (1 if f.split(os.path.sep)[-3] == "videos_fake" else 0)} for f in hq_image_files]
df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])


base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=Dense(1, activation='sigmoid')(x))
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

def preprocess_input(img): return (img / 127.5) - 1.0
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=10, horizontal_flip=True)
train_gen = train_datagen.flow_from_dataframe(train_df, x_col='path', y_col='label', target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=16, class_mode='raw')

print("\nIniciando Treinamento Xception...")
model.fit(train_gen, epochs=5)


os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")