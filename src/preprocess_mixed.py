import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
import tensorflow as tf
import glob
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


try:
    from spectrogram import szDATASET_DIR
except ImportError:
    # Fallback au cas où
    szDATASET_DIR = "../data/raw/music_dataset"


# --- PARAMÈTRES ---
SOURCE_MIXED_DIR = "../data/raw/music_dataset_mixed_large_normed"
OUTPUT_DIR = "../data/processed/mixed_spectrograms_large_3channel_npy"

SPECTRUM_PARAMS = [
    {"WIDTH": 128, "STRIDE": 32},  # Fenêtre étroite
    {"WIDTH": 512, "STRIDE": 128}, # Fenêtre moyenne
    {"WIDTH": 1024, "STRIDE": 256} # Fenêtre large
]
TARGET_SHAPE = (257, 520) # Taille cible des spectrogrammes

# --- FONCTIONS DE TRAITEMENT ---

def build_single_spectrogram(fe, wvData, width, stride):
    """Crée UN SEUL spectrogramme normalisé 0-1 et redimensionné."""
    try:
        w = hamming(width, sym=True)
        SFT = ShortTimeFFT(w, hop=stride, fs=fe, mfft=width, scale_to="magnitude")
        wv_mono = wvData
        if wvData.ndim > 1:
            wv_mono = wvData.mean(axis=1)
        Sx = SFT.stft(wv_mono)
        Sx_abs = np.abs(Sx)
        Sx_log = np.log1p(Sx_abs) # log(1+x)

        min_val = np.min(Sx_log)
        max_val = np.max(Sx_log)
        if max_val > min_val:
            Sx_norm = (Sx_log - min_val) / (max_val - min_val)
        else:
            Sx_norm = np.zeros_like(Sx_log)

        Sx_tensor = tf.convert_to_tensor(Sx_norm, dtype=tf.float32)[..., tf.newaxis]
        Sx_resized = tf.image.resize(Sx_tensor, TARGET_SHAPE, method='bilinear')
        return tf.squeeze(Sx_resized).numpy()
    except Exception as e:
        # print(f"    Erreur build_single_spectrogram: {e}")
        return None

def build_3channel_spectrogram_from_wav(file_path):
    """Lit un WAV et construit le spectrogramme 3 canaux."""
    try:
        fe, wvData = wavfile.read(file_path)
        
        # Convertir en float32
        if wvData.dtype == np.int16:
            wvData = wvData.astype(np.float32) / 32768.0
        elif wvData.dtype == np.int32:
            wvData = wvData.astype(np.float32) / 2147483648.0
        elif wvData.dtype == np.uint8:
             wvData = (wvData.astype(np.float32) - 128.0) / 128.0
        
        if np.max(np.abs(wvData)) < 1e-4: return None # Fichier silencieux

        spectros = []
        valid = True
        for params in SPECTRUM_PARAMS:
            spectro = build_single_spectrogram(fe, wvData, params["WIDTH"], params["STRIDE"])
            if spectro is None: valid = False; break
            spectros.append(spectro)

        if not valid or len(spectros) != 3: return None
        stacked_spectro = np.stack(spectros, axis=-1)
        return stacked_spectro.astype(np.float32)

    except Exception as e:
        print(f"  Erreur lors du traitement complet de {os.path.basename(file_path)}: {e}")
        return None

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":
    print(f"Début du pré-traitement (3 canaux NPY) depuis '{SOURCE_MIXED_DIR}'...")
    if not os.path.isdir(SOURCE_MIXED_DIR):
        print(f"ERREUR: Dossier source '{SOURCE_MIXED_DIR}' introuvable.")
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Les spectrogrammes NPY seront sauvegardés dans '{OUTPUT_DIR}'...")

    wav_files = glob.glob(os.path.join(SOURCE_MIXED_DIR, '*.wav'))
    if not wav_files:
        print(f"ERREUR: Aucun fichier .wav trouvé dans '{SOURCE_MIXED_DIR}'.")
        exit()

    print(f"Traitement de {len(wav_files)} fichiers .wav mixés...")
    file_counter = 0; error_counter = 0; total_files = len(wav_files)

    for index, wav_path in enumerate(wav_files):
        spectro_3ch = build_3channel_spectrogram_from_wav(wav_path)
        if spectro_3ch is not None and spectro_3ch.shape == (TARGET_SHAPE[0], TARGET_SHAPE[1], 3):
            base_name = os.path.basename(wav_path)
            file_name_npy = os.path.splitext(base_name)[0] + ".npy"
            output_path = os.path.join(OUTPUT_DIR, file_name_npy)
            try:
                np.save(output_path, spectro_3ch)
                file_counter += 1
            except Exception as e:
                 print(f"  Erreur lors de la sauvegarde de {file_name_npy}: {e}")
                 error_counter += 1
        else:
            if spectro_3ch is not None:
                print(f"  Avertissement: Forme inattendue {spectro_3ch.shape} pour {os.path.basename(wav_path)}, ignoré.")
            error_counter += 1

        processed_count = index + 1
        if processed_count % 100 == 0 or processed_count == total_files:
             print(f"  Progression: {processed_count}/{total_files} fichiers traités...")

    print(f"\nPré-traitement (3 canaux NPY) terminé !")
    print(f"-> {file_counter} spectrogrammes .npy créés dans '{OUTPUT_DIR}'.")
    if error_counter > 0:
        print(f"-> {error_counter} fichiers n'ont pas pu être traités ou sauvegardés.")
