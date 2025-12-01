# --- IMPORTS ---
import tensorflow as tf
from matplotlib import pyplot as plt
from resnet_34 import BuildRESNET34
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import RandomFlip, RandomZoom

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam # Pour régler le learning rate

# --- CONFIGURATION GPU  ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"-> Activation de la 'Memory Growth' pour le GPU : {gpus[0].name}")
  except RuntimeError as e:
    print(e)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("-> Politique de Précision Mixte (mixed_float16) activée.")

# --- PARAMÈTRES OPTIMISÉS ---
IMAGE_DIR = ".././data/processed/no_normed"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 26
EPOCHS = 20
MODEL_OUTPUT_DIR = ".././models"

# --- 1. CHARGER LES DONNÉES ---
train_dataset = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print("Classes détectées :", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. CONSTRUIRE LE MODÈLE  ---
model_base = BuildRESNET34(npdInputShape=[IMG_SIZE[0], IMG_SIZE[1], 3], iNbClasses=NUM_CLASSES)

model_full = tf.keras.Sequential([
    Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    RandomFlip("horizontal"),
    RandomZoom(0.1),
    model_base
])

print("Architecture du modèle :")
model_full.summary()

# --- 3. COMPILER LE MODÈLE ---
# --- MODIF : OPTIMIZER AVEC LEARNING RATE  ---
model_full.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- CRÉATION DES CALLBACKS ---
# 1. Pour sauvegarder le meilleur modèle basé sur val_accuracy
checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_OUTPUT_DIR + "/best_instrument_classifier.keras",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
# --- MODIF : AJOUT DE EARLY STOPPING ---
# 2. Pour arrêter l'entraînement si val_loss ne s'améliore plus
early_stopping_callback = EarlyStopping(
    monitor='val_loss',     # On surveille l'erreur de validation
    patience=5,             # Nb d'époques sans amélioration avant d'arrêter
    verbose=1,              # Affiche un message quand ça s'arrête
    restore_best_weights=True # Revient aux poids du meilleur moment
)

# --- 4. ENTRAÎNER LE MODÈLE ---
print("Début de l'entraînement...")
history = model_full.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,

    callbacks=[checkpoint_callback, early_stopping_callback]
)

# --- 5. INFORMATION SUR LA SAUVEGARDE ---
print(f"Le meilleur modèle (basé sur val_accuracy) a été sauvegardé automatiquement sous '{MODEL_OUTPUT_DIR}/best_instrument_classifier.keras' pendant l'entraînement.")
if early_stopping_callback.stopped_epoch > 0:
    print(f"L'entraînement s'est arrêté prématurément à l'époque {early_stopping_callback.stopped_epoch + 1} car la val_loss ne s'améliorait plus.")
    print("Les poids restaurés sont ceux de la meilleure époque (basée sur val_loss).")

# --- 6. AFFICHER LE GRAPHIQUE ---
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
