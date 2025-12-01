# --- IMPORTS ---
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import os
import glob
import matplotlib.pyplot as plt
import random

# Correction du bug Wayland/Qt (pour afficher le graphique final)
os.environ['QT_QPA_PLATFORM'] = 'xcb'


from resnet_34 import ResidualUnit
from preprocess_mixed import build_3channel_spectrogram_from_wav 


tf.keras.utils.get_custom_objects()['ResidualUnit'] = ResidualUnit

# --- PARAMÈTRES ---
MODEL_PATH = "../models/best_multilabel_classifier.keras"
INPUT_SHAPE = (257, 520, 3) 

TARGET_INSTRUMENTS = ["violin", "piano", "cymbals", "flute", "vibraphone"] 
MIXED_WAV_DIR = "../data/raw/music_dataset_mixed_large_normed"
NUM_FILES_TO_TEST = 15
PREDICTION_THRESHOLD = 0.5

# --- FONCTION POUR CALCULER LE SCORE---
def calculate_prediction_score(true_set, pred_set, all_instruments_list):
    """Calcule le score de réussite (Vrais Positifs + Vrais Négatifs)."""
    score = 0
    for instrument in all_instruments_list: # Utilise la liste TARGET_INSTRUMENTS (minuscule)
        is_present_true = instrument in true_set # true_set est maintenant en minuscule
        is_present_pred = instrument in pred_set # pred_set est (normalement) en minuscule
        
        if is_present_true and is_present_pred: # Vrai Positif
            score += 1
        elif not is_present_true and not is_present_pred: # Vrai Négatif
            score += 1
    
    success_ratio = score / len(all_instruments_list)
    score_text = f"{score}/{len(all_instruments_list)} Corrects"
    return success_ratio, score_text

# --- FONCTION POUR LA COULEUR ---
def get_color_from_ratio(ratio):
    if ratio == 1.0: return 'green'
    elif ratio >= 0.8: return 'limegreen'
    elif ratio >= 0.6: return 'orange'
    else: return 'red'

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":
    print(f"Chargement du modèle multi-label depuis : {MODEL_PATH}")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modèle multi-label chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        exit()

    # --- Collecte des fichiers .wav mixés à tester ---
    mixed_wav_files = glob.glob(os.path.join(MIXED_WAV_DIR, '*.wav'))
    if not mixed_wav_files: exit(f"ERREUR: Aucun fichier .wav trouvé dans {MIXED_WAV_DIR}.")
    num_to_sample = min(NUM_FILES_TO_TEST, len(mixed_wav_files))
    if num_to_sample > 0: files_to_test = random.sample(mixed_wav_files, num_to_sample)
    else: files_to_test = []
    print(f"\nSélection de {len(files_to_test)} fichiers mixés pour le test...")

    # --- Boucle de prédiction ---
    print("\nLancement des prédictions multi-label...")
    test_results = []
    
    for i, audio_file in enumerate(files_to_test):
        print(f"\n--- Test {i+1}/{len(files_to_test)}: {os.path.basename(audio_file)} ---")
        
        # --- MODIF : Convertir les noms parsés en minuscule ---
        true_set_raw = set()
        base_name = os.path.basename(audio_file)
        if '_mix_' in base_name:
            # S'assurer que le nom est en minuscule AVANT de le splitter
            true_set_raw = set(base_name.lower().split('_mix_')[0].split('_'))
        # Filtrer pour ne garder que les instruments cibles
        true_set = {inst for inst in true_set_raw if inst in TARGET_INSTRUMENTS} # Ex: {"flute", "vibraphone"}
        # --- FIN MODIF ---
        
        print(f"  (Instruments réellement présents: {', '.join(sorted(true_set)) or 'Aucun'})")

        spectro_3ch_01 = build_3channel_spectrogram_from_wav(audio_file)
        
        if spectro_3ch_01 is not None and spectro_3ch_01.shape == INPUT_SHAPE:
            img_batch = tf.expand_dims(tf.cast(spectro_3ch_01, tf.float32), axis=0)
            predictions = model.predict(img_batch, verbose=0)[0]
            
            pred_set = set()
            print("  Probabilités prédites:")
            for j, instrument_name in enumerate(TARGET_INSTRUMENTS):
                prob = predictions[j]
                print(f"  - {instrument_name}: {prob*100:.1f}%")
                if prob >= PREDICTION_THRESHOLD:
                    pred_set.add(instrument_name)
            

            success_ratio, score_text = calculate_prediction_score(true_set, pred_set, TARGET_INSTRUMENTS)
            print(f"  --> Score: {score_text}")

            test_label = base_name.split('_mix_')[0].replace('_', '\n')
            true_str = f"Vrai: {', '.join(sorted(true_set)) or 'Rien'}"
            pred_str = f"Prédit: {', '.join(sorted(pred_set)) or 'Rien'}"
            test_results.append((test_label, success_ratio, score_text, true_str, pred_str))
        else:
            print("  -> Échec du prétraitement de l'image.")
            test_results.append((f"Erreur_{i}", 0.0, "0/5 Corrects", "N/A", "Erreur Preproc"))

    # --- GRAPHIQUE VISUEL AMÉLIORÉ (inchangé) ---
    print("\nAffichage du résumé visuel des tests...")
    if not test_results: exit("Aucun test n'a pu être effectué.")
        
    labels = [r[0] for r in test_results] 
    ratios_percent = [r[1] * 100 for r in test_results]
    colors = [get_color_from_ratio(r[1]) for r in test_results]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(test_results)), ratios_percent, color=colors, tick_label=labels)
    plt.ylabel('Taux de Réussite de la Prédiction (%)')
    plt.title(f'Résumé des {len(test_results)} Tests (Score = (Vrais Positifs + Vrais Négatifs) / 5)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylim(0, 105)
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        score_text = test_results[i][2]; true_str = test_results[i][3]; pred_str = test_results[i][4]
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, score_text, va='bottom', ha='center', color='black', weight='bold', fontsize=8)
        if yval > 30: 
            plt.text(bar.get_x() + bar.get_width()/2.0, yval/2 + 5, pred_str, va='center', ha='center', color='white', weight='bold', fontsize=7, rotation=90)
            plt.text(bar.get_x() + bar.get_width()/2.0, yval/2 - 5, f"({true_str})", va='top', ha='center', color='white', style='italic', fontsize=6, rotation=90)
        
    plt.tight_layout()
    plt.show()
