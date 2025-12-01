# --- IMPORTS ---
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.optimizers import Adam
import gc


from resnet_34 import ResidualUnit, BuildRESNET34



tf.keras.utils.get_custom_objects()['ResidualUnit'] = ResidualUnit

# --- PARAMÈTRES ---
MIXED_IMAGE_DIR = "../data/processed/mixed_spectrograms_large_3channel_npy"
TARGET_INSTRUMENTS = ["Violin", "Piano", "Cymbals", "flute", "vibraphone"]
INPUT_SHAPE = (257, 520, 3)
BATCH_SIZE = 16
EPOCHS = 50
MODEL_OUTPUT_DIR = "../models"
OUTPUT_MODEL_FILENAME = "best_multilabel_classifier.keras"
SHUFFLE_BUFFER_SIZE = 1000

# --- CONFIGURATION GPU ---

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"-> Activation de la 'Memory Growth' pour le GPU : {gpus[0].name}")
  except RuntimeError as e:
    print(e)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("-> Politique de Précision Mixte (mixed_float16) activée.")

# --- 1. PRÉPARATION DES DONNÉES MULTI-LABEL ---

def get_label_from_filename(filename, target_instruments):
    # ... (code identique) ...
    label = np.zeros(len(target_instruments), dtype=np.float32)
    base_name = os.path.basename(filename)
    if '_mix_' in base_name:
        present_instruments = base_name.split('_mix_')[0].split('_')
        for i, instrument in enumerate(target_instruments):
            if instrument in present_instruments: label[i] = 1.0
    return label
    
all_mixed_files = tf.io.gfile.glob(os.path.join(MIXED_IMAGE_DIR, '*.npy'))
if not all_mixed_files: exit("Error: No NPY files found.")
random.shuffle(all_mixed_files)
labels = np.array([get_label_from_filename(f, TARGET_INSTRUMENTS) for f in all_mixed_files])

def load_and_process_npy(path):
    path_str = path.numpy().decode('utf-8')
    data = np.load(path_str)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    data.set_shape(INPUT_SHAPE)
    return data

def map_fn(path, label):
    spectrogram = tf.py_function(func=load_and_process_npy, inp=[path], Tout=tf.float32)
    spectrogram.set_shape(INPUT_SHAPE)
    return spectrogram, label

# --- CORRECTION DE L'ORDRE D'APPEL ---
# 1. Créer les datasets de base (chemins et labels)
image_ds = tf.data.Dataset.from_tensor_slices(all_mixed_files)
label_ds = tf.data.Dataset.from_tensor_slices(labels)

# 2. Zipper les chemins et les labels EN PREMIER
dataset = tf.data.Dataset.zip((image_ds, label_ds))

# 3. Appliquer la fonction .map MAINTENANT (sur le dataset zippé)
dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
# --- FIN DE LA CORRECTION ---

dataset_size = len(all_mixed_files); train_size = int(0.8 * dataset_size); val_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
print(f"Dataset multi-label (NPY 3 canaux) créé: {dataset_size} images...")


# --- 2. CRÉER LE MODÈLE MULTI-LABEL ---

print("Création du modèle multi-label (ResNet-34) depuis zéro...")
resnet_base = BuildRESNET34(npdInputShape=INPUT_SHAPE, iNbClasses=len(TARGET_INSTRUMENTS))
multilabel_model_scratch = models.Sequential([
    layers.Input(shape=INPUT_SHAPE, name="input_spectrogram"), # Pas de Rescaling
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
    models.Sequential(resnet_base.layers[:-2], name="resnet_trunk_scratch"),
    layers.Dropout(0.5, name="new_dropout"),
    layers.Dense(len(TARGET_INSTRUMENTS), activation='sigmoid', name='instrument_multi_output')
], name="multilabel_classifier_scratch_3ch")
print("Architecture du nouveau modèle multi-label (from scratch, 3ch) :")
multilabel_model_scratch.summary()

# --- 3. COMPILER LE MODÈLE MULTI-LABEL ---

multilabel_model_scratch.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# --- 4. CALLBACKS POUR L'ENTRAÎNEMENT ---

checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_OUTPUT_DIR + f"/{OUTPUT_MODEL_FILENAME}",
    save_weights_only=False, monitor='val_binary_accuracy', mode='max', save_best_only=True
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=20,
    verbose=1, restore_best_weights=True
)

# --- 5. ENTRAÎNER LE MODÈLE ---

print("Début de l'entraînement (from scratch) sur le grand dataset 3 canaux...")
history = multilabel_model_scratch.fit(
    train_dataset, validation_data=validation_dataset, epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# --- 6. INFORMATION ET GRAPHIQUE ---

print(f"Le meilleur modèle multi-label (from scratch, 3ch) a été sauvegardé sous '{MODEL_OUTPUT_DIR}/{OUTPUT_MODEL_FILENAME}'.")

plt.plot(history.history['binary_accuracy'], label='binary_accuracy'); plt.plot(history.history['val_binary_accuracy'], label = 'val_binary_accuracy');
plt.xlabel('Epoch'); plt.ylabel('Binary Accuracy'); plt.ylim([0, 1]); plt.legend(loc='lower right'); plt.show()
plt.figure(); plt.plot(history.history['loss'], label='loss'); plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (Binary Crossentropy)'); plt.legend(loc='upper right'); plt.show()
