# --- IMPORTS ---
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import os
import sys # Pour lire les arguments de la ligne de commande
import matplotlib.pyplot as plt # Pour afficher optionnellement les notes


from spectrogram import lstINSTRUMENTS_CLASSES # Liste des noms
from resnet_34 import ResidualUnit # Pour charger le modèle
from preprocess import create_spectrogram_image # Pour l'image du classifieur

from detect_note import frequency_to_note, find_fundamental_frequency, Wav_Reader

tf.keras.utils.get_custom_objects()['ResidualUnit'] = ResidualUnit

# --- PARAMÈTRES ---
MODEL_PATH = "../models/best_instrument_classifier.keras" # Chemin vers le meilleur modèle sauvegardé
IMG_SIZE_CLASSIFIER = (128, 128) # Taille d'image utilisée pour l'entraînement du classifieur

# Paramètres pour la détection de note
NOTE_TIME_STEP_SEC = 0.25 # Analyse tous les 1/4s 
NOTE_WINDOW_SIZE = 4096 # Fenêtre d'analyse FFT

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":

    # 1. Vérifier l'argument (chemin du fichier .wav)
    if len(sys.argv) != 2:
        print("Usage: python analyze_single_instrument.py <chemin_vers_fichier.wav>")
        sys.exit(1)
    audio_file_path = sys.argv[1]

    if not os.path.exists(audio_file_path):
        print(f"Erreur: Le fichier '{audio_file_path}' n'existe pas.")
        sys.exit(1)

    print("--- Analyse de l'instrument ---")
    # 2. Charger le modèle de classification
    print(f"Chargement du modèle depuis : {MODEL_PATH}")
    # Activer la précision mixte si le modèle a été entraîné avec
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    try:
        instrument_classifier_model = tf.keras.models.load_model(MODEL_PATH)
        print("Modèle classifieur chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # 3. Prétraiter le fichier audio pour la classification
    print(f"Prétraitement de {audio_file_path} pour classification...")
    img_uint8 = create_spectrogram_image(audio_file_path)

    predicted_instrument = "Inconnu"
    confidence = 0
    if img_uint8 is not None:
        # Convertir en float32 et ajouter dimension Batch
        img_float32 = tf.cast(img_uint8, tf.float32)
        img_batch = tf.expand_dims(img_float32, axis=0)

        # 4. Prédire l'instrument
        predictions = instrument_classifier_model.predict(img_batch, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_instrument = lstINSTRUMENTS_CLASSES[predicted_index]
        confidence = np.max(predictions[0]) * 100
        print(f"Instrument prédit : {predicted_instrument} (Confiance: {confidence:.1f}%)")
    else:
        print("Échec du prétraitement pour la classification.")

    print("\n--- Analyse des notes ---")
    # 5. Charger l'audio pour la détection de notes
    try:
        sampling_rate, sound_data = Wav_Reader(audio_file_path) # Utilise la fonction de detect_note
        duration_sec = len(sound_data) / sampling_rate
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier audio pour l'analyse des notes: {e}")
        sys.exit(1)

    # 6. Boucle d'analyse des notes
    notes_sequence = []
    timestamps = []
    print(f"Analyse des notes tous les {NOTE_TIME_STEP_SEC} secondes...")

    # Calculer le nombre total d'étapes pour l'estimation du temps
    total_steps = int(duration_sec / NOTE_TIME_STEP_SEC)
    print(f"Durée totale: {duration_sec:.2f}s, Nombre d'étapes estimé: {total_steps}")
    
    step_counter = 0
    for current_time in np.arange(0, duration_sec - (NOTE_WINDOW_SIZE / sampling_rate), NOTE_TIME_STEP_SEC):
        step_counter += 1
        # Afficher la progression toutes les N étapes pour éviter de spammer
        if step_counter % 10 == 0 or step_counter == total_steps:
             print(f"  Progression: Étape {step_counter}/{total_steps} ({current_time:.2f}s / {duration_sec:.2f}s)")
             
        start_sample = int(current_time * sampling_rate)
        end_sample = start_sample + NOTE_WINDOW_SIZE

        if end_sample > len(sound_data):
            break

        sound_segment = sound_data[start_sample:end_sample]

        # Trouver la fréquence fondamentale et la convertir en note
        # find_fundamental_frequency renvoie (freq, freqs_fft, magnitude_fft)
        try:
             fundamental_freq, _, _ = find_fundamental_frequency(sound_segment, sampling_rate)
             note = frequency_to_note(fundamental_freq)
             notes_sequence.append(note)
             timestamps.append(current_time)
        except Exception as e:
             # Gérer les erreurs possibles dans find_fundamental_frequency si le segment est trop silencieux etc.
             print(f"  Avertissement: Erreur à t={current_time:.2f}s - {e}")
             notes_sequence.append("Erreur")
             timestamps.append(current_time)


    # 7. Afficher les résultats
    print("\n--- RÉSULTAT FINAL ---")
    print(f"Fichier analysé : {os.path.basename(audio_file_path)}")
    print(f"Instrument détecté : {predicted_instrument}")
    print("\nSéquence de notes (toutes les 1/4s) :")
    output_string = ""
    for t, note in zip(timestamps, notes_sequence):
        output_string += f"{note.split(' ')[0]} " # Affiche juste "La4", "Do5", etc.
    print(output_string)

   #Afficher un graphique des notes détectées
    plt.figure(figsize=(15, 5))
    plt.step(timestamps, notes_sequence, where='post')
    plt.xticks(np.arange(0, duration_sec, 1.0)) 
    plt.xlabel("Temps (s)")
    plt.ylabel("Note Détectée")
    plt.title(f"Partition Détectée pour {predicted_instrument}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
