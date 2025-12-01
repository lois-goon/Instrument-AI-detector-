# --- IMPORTS ---
import numpy as np
from scipy.io import wavfile
import os
import glob
import random
import itertools


from spectrogram import szDATASET_DIR, lstINSTRUMENTS_CLASSES

# --- PARAMÈTRES ---
TARGET_INSTRUMENTS = ["Violin", "Piano", "Cymbals", "Flute", "Vibraphone"]
OUTPUT_MIXED_DIR = "../data/raw/music_dataset_mixed_large_normed"
NUM_MIXES_PER_COMBINATION_TYPE = 500
TARGET_DURATION_SEC = 3.0

# --- FONCTIONS UTILES  ---
def read_wav_safe(filepath):
    try:
        sample_rate, data = wavfile.read(filepath)
        if data.ndim > 1: data = data.mean(axis=1)
        if data.dtype == np.int16: data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32: data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8: data = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype == np.float32: pass
        else: return sample_rate, None
        if np.max(np.abs(data)) < 1e-4: return sample_rate, None
        return sample_rate, data
    except Exception as e: return None, None

def adjust_duration(sample_rate, data, target_sec):
    target_samples = int(sample_rate * target_sec)
    current_samples = len(data)
    if current_samples > target_samples: return data[:target_samples]
    elif current_samples < target_samples:
        padding = np.zeros(target_samples - current_samples, dtype=data.dtype)
        return np.concatenate((data, padding))
    else: return data

def mix_n_audios_normalized(sr, data_list):
    if not data_list: return np.array([])
    normalized_tracks = []
    for data in data_list:
        max_val = np.max(np.abs(data))
        if max_val > 1e-5: normalized_tracks.append(data / max_val)
        else: normalized_tracks.append(data)
    if not normalized_tracks: return np.array([])
    mixed_data = np.sum(normalized_tracks, axis=0)
    num_tracks = len(normalized_tracks)
    if num_tracks > 1:
        scale_factor = np.sqrt(num_tracks)
        mixed_data = mixed_data / scale_factor
    mixed_data = np.clip(mixed_data, -1.0, 1.0)
    return mixed_data

def save_wav_safe(filepath, sample_rate, data):
     if data.size == 0 or np.max(np.abs(data)) < 1e-5: return False # Vérifie silence ici aussi par sécurité
     try:
        data = np.clip(data, -1.0, 1.0)
        data_int16 = (data * 32767.0).astype(np.int16)
        wavfile.write(filepath, sample_rate, data_int16)
        return True
     except Exception as e:
        print(f"  Erreur lors de la sauvegarde de {filepath} - {e}") # filepath ne devrait plus être None
        return False

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":

    print(f"Création du grand dataset mixé normalisé dans '{OUTPUT_MIXED_DIR}'...")
    os.makedirs(OUTPUT_MIXED_DIR, exist_ok=True)

    # 1. Lister les fichiers sources (Identique)
    source_files = {}
    print("Recherche des fichiers sources...")
    all_instruments_found = True
    for instrument in TARGET_INSTRUMENTS:
        folder_path = os.path.join(szDATASET_DIR, instrument)
        wav_list_initial = glob.glob(os.path.join(folder_path, '*.wav'))
        wav_list = [f for f in wav_list_initial if os.path.getsize(f) > 100]
        if not wav_list:
             print(f"  ERREUR: Aucun fichier .wav valide trouvé pour {instrument} dans '{folder_path}'.")
             all_instruments_found = False
        source_files[instrument] = wav_list
        print(f"- {len(source_files[instrument])} fichiers valides trouvés pour {instrument}")
    if not all_instruments_found: exit("\nArrêt: Instruments cibles manquants.")

    mix_counter_total = 0
    # Boucle sur le nombre d'instruments à mixer (de 2 à 5)
    for num_instruments_in_mix in range(2, len(TARGET_INSTRUMENTS) + 1):
        print(f"\n--- Génération des mixages à {num_instruments_in_mix} instruments ({NUM_MIXES_PER_COMBINATION_TYPE} par combinaison) ---")
        instrument_combinations = list(itertools.combinations(TARGET_INSTRUMENTS, num_instruments_in_mix))
        print(f"Nombre de combinaisons possibles: {len(instrument_combinations)}")

        # Boucle sur chaque combinaison
        for combination in instrument_combinations:
            if not all(source_files.get(inst) for inst in combination):
                print(f"    -> Skip combinaison {'_'.join(combination)}: manque de fichiers source valides.")
                continue

            combination_str = "_".join(sorted(combination))
            print(f"  Traitement de la combinaison : {combination_str}")
            mix_counter_combination = 0
            attempts = 0
            MAX_ATTEMPTS = NUM_MIXES_PER_COMBINATION_TYPE * 10

            # Créer N mixages pour cette combinaison
            while mix_counter_combination < NUM_MIXES_PER_COMBINATION_TYPE and attempts < MAX_ATTEMPTS:
                attempts += 1
                audio_data_list = []
                target_sr = None
                valid_selection = True

                # Lire fichiers
                for instrument in combination:
                    if not source_files[instrument]: valid_selection = False; break
                    file_path = random.choice(source_files[instrument])
                    sr, data = read_wav_safe(file_path)
                    if data is None: valid_selection = False; break
                    if target_sr is None: target_sr = sr
                    elif sr != target_sr: valid_selection = False; break
                    data_adj = adjust_duration(sr, data, TARGET_DURATION_SEC)
                    audio_data_list.append(data_adj)

                if not valid_selection or len(audio_data_list) != num_instruments_in_mix:
                    continue

                # Mixer
                mixed_audio = mix_n_audios_normalized(target_sr, audio_data_list)

                # --- CORRECTION DE LA VÉRIFICATION ET SAUVEGARDE ---
                # 1. Vérifier si le mix est audible AVANT de sauvegarder
                if np.max(np.abs(mixed_audio)) > 1e-5:
                    mix_counter_combination += 1
                    output_filename = f"{combination_str}_mix_{mix_counter_combination:03d}.wav"
                    output_filepath = os.path.join(OUTPUT_MIXED_DIR, output_filename)

                    # 2. Appeler save_wav_safe UNIQUEMENT avec le vrai chemin
                    if save_wav_safe(output_filepath, target_sr, mixed_audio):
                        mix_counter_total += 1
                    else:
                        # Si la sauvegarde échoue pour une autre raison
                        mix_counter_combination -= 1
                # else: print("Mix silencieux, skip")
                # --- FIN CORRECTION ---

            if attempts >= MAX_ATTEMPTS:
                 print(f"    Avertissement: Nombre maximum d'essais ({MAX_ATTEMPTS}) atteint pour {combination_str}.")
            print(f"    -> {mix_counter_combination} mixages créés.")

    print(f"\nTerminé ! {mix_counter_total} fichiers mixés (normalisés) ont été créés dans '{OUTPUT_MIXED_DIR}'.")
