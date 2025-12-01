import tensorflow as tf
import numpy as np
from scipy.io import wavfile # Gardé car create_spectrogram_image l'utilise
# Retiré : ShortTimeFFT, hamming car c'est dans preprocess maintenant
import os
import glob
import matplotlib.pyplot as plt

# --- IMPORTS DE TES AUTRES FICHIERS ---
from spectrogram import lstINSTRUMENTS_CLASSES, BuildFilePath2Wave, szDATASET_DIR
from resnet_34 import ResidualUnit
# --- MODIF : IMPORTER LA FONCTION DE PREPROCESS ---
from preprocess import create_spectrogram_image # <--- ON IMPORTE LA FONCTION ORIGINALE

# --- AJOUT : ENREGISTREMENT MANUEL DE LA CLASSE ---
tf.keras.utils.get_custom_objects()['ResidualUnit'] = ResidualUnit

# --- PARAMÈTRES ---
IMG_SIZE = (128, 128) # Important : Doit correspondre à l'entraînement
MODEL_PATH = "../models/best_instrument_classifier.keras"
NUM_SAMPLES_PER_INSTRUMENT = 4
INSTRUMENTS_TO_TEST = ["Piano", "Violin", "Trumpet", "Acoustic_Guitar", "Flute"]

# --- FONCTION DE PRÉTRAITEMENT (Supprimée, on utilise celle importée) ---
# def create_spectrogram_for_prediction(...):
#     ...

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":

    print(f"Chargement du modèle depuis : {MODEL_PATH}")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        exit()

    # --- Collecte des fichiers ---
    files_to_test = []
    true_labels = []
    print("\nCollecte des fichiers de test...")
    # ... (code de collecte identique) ...
    for instrument_name in INSTRUMENTS_TO_TEST:
        instrument_folder = os.path.join(szDATASET_DIR, instrument_name)
        wav_files = glob.glob(os.path.join(instrument_folder, '*.wav'))[:NUM_SAMPLES_PER_INSTRUMENT]
        files_to_test.extend(wav_files)
        true_labels.extend([instrument_name] * len(wav_files))
        print(f"- Ajout de {len(wav_files)} fichiers pour {instrument_name}")

    # --- Boucle de prédiction (avec adaptation) ---
    results = []
    print("\nLancement des prédictions...")
    for i, audio_file in enumerate(files_to_test):
        print(f"Test {i+1}/{len(files_to_test)}: {os.path.basename(audio_file)} ({true_labels[i]})")

        # --- MODIF : UTILISER LA FONCTION IMPORTÉE ---
        # 1. Appeler la fonction de preprocess.py
        #    Attention: Elle attend IMG_SIZE en paramètre global dans preprocess.py
        #    On s'assure que IMG_SIZE dans preprocess.py est aussi (128, 128)
        #    ou on passe la taille en argument si on modifie la fonction.
        #    Pour l'instant, on suppose qu'elle utilise (128, 128) ou qu'on l'a modifiée.
        #    Alternativement, on peut redéfinir IMG_SIZE dans preprocess.py avant d'appeler.
        #    La fonction renvoie un uint8 tensor (0-255).
        img_uint8 = create_spectrogram_image(audio_file) # Utilise la fonction importée

        if img_uint8 is not None:
            # 2. Convertir en float32 (attendu par le modèle/rescaling)
            img_float32 = tf.cast(img_uint8, tf.float32)

            # 3. Ajouter la dimension Batch
            img_batch = tf.expand_dims(img_float32, axis=0)
            # --- FIN MODIF ---

            predictions = model.predict(img_batch, verbose=0) # On passe img_batch
            predicted_index = np.argmax(predictions[0])
            predicted_instrument = lstINSTRUMENTS_CLASSES[predicted_index]
            confidence = np.max(predictions[0]) * 100
            results.append((true_labels[i], predicted_instrument, confidence))
            print(f"  -> Prédit: {predicted_instrument} ({confidence:.1f}%) - {'CORRECT' if true_labels[i] == predicted_instrument else 'INCORRECT'}")
        else:
            print(f"  -> Échec du prétraitement.")
            results.append((true_labels[i], "Erreur", 0))

    # --- Affichage graphique (ne change pas) ---
    # ... (code matplotlib) ...
    print("\nAffichage des résultats...")
    labels = [f"{os.path.basename(f)}\n({r[0]})" for f, r in zip(files_to_test, results)]
    confidences = [r[2] for r in results]
    colors = ['green' if r[0] == r[1] else 'red' for r in results]
    plt.figure(figsize=(15, 7))
    bars = plt.bar(range(len(results)), confidences, color=colors)
    plt.ylabel('Confiance (%)')
    plt.title(f'Résultats de prédiction sur {len(results)} fichiers')
    plt.xticks(range(len(results)), labels, rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 105)
    plt.tight_layout()
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', va='bottom', ha='center', fontsize=7)
    plt.show()
