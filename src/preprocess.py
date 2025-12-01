import os
import numpy as np
import tensorflow as tf
from spectrogram import BuildFilePath2Wave, lstINSTRUMENTS_CLASSES, szDATASET_DIR
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
import glob # Pour trouver les fichiers

# --- Paramètres ---
# Le dossier où sont les .wav
SOURCE_DIR = szDATASET_DIR 

# Le dossier où on va sauvegarder les images PNG
OUTPUT_DIR = "../data/processed/no_normed" 

# Taille d'image pour le ResNet-34
IMG_SIZE = (128,128) 

# --- Fonction pour traiter UN SEUL fichier ---
# On adapte BuildSpectrogram pour ne pas afficher et ne pas jouer le son

def create_spectrogram_image(szFileName, width=512, stride=128):
    try:
        freq, wvData = wavfile.read(szFileName)
        
        # Si le son est stéréo, on le moyenne en mono
        if wvData.ndim > 1:
            wvData = wvData.mean(axis=1)
            
        N = wvData.shape[0]
        if N == 0:
            print(f"Fichier vide (0 sample): {szFileName}")
            return None

        w = hamming(width, sym=True)
        SFT = ShortTimeFFT(w, hop=stride, fs=freq, mfft=width, scale_to='magnitude')
        Sx = SFT.stft(wvData)  # perform the STFT

        # 1. Prendre la magnitude (valeurs absolues)
        Sx_mag = np.abs(Sx)

        # 2. Convertir en Log-Spectrogramme

        Sx_log = np.log1p(Sx_mag) 

        # 3. Redimensionner l'image
        # tf.image.resize attend un format (batch, height, width, channels)
        Sx_resized = tf.image.resize(Sx_log[..., np.newaxis], IMG_SIZE)
        
        # 4. Normaliser l'image (entre 0 et 1)
        img_normalized = (Sx_resized - tf.reduce_min(Sx_resized)) / (tf.reduce_max(Sx_resized) - tf.reduce_min(Sx_resized))
        
        # 5. Dupliquer sur 3 canaux (pour ResNet)
        img_3_channels = tf.image.grayscale_to_rgb(img_normalized)
        
        # 6. Convertir en format image 8-bit (0-255)
        img_uint8 = tf.image.convert_image_dtype(img_3_channels, dtype=tf.uint8)
        
        return img_uint8

    except Exception as e:
        print(f"Erreur lors du traitement de {szFileName}: {e}")
        return None

# --- Point d'entrée principal ---
if __name__ == "__main__":
    print("Début du pré-traitement...")
    
    # Créer le dossier de sortie principal
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Boucler sur chaque instrument
    for instrument_name in lstINSTRUMENTS_CLASSES:
        print(f"Traitement de : {instrument_name}")
        
        # Créer le sous-dossier pour cet instrument
        instrument_output_dir = os.path.join(OUTPUT_DIR, instrument_name)
        os.makedirs(instrument_output_dir, exist_ok=True)
        
        # Trouver tous les fichiers .wav pour cet instrument
        instrument_source_dir = os.path.join(SOURCE_DIR, instrument_name)
        # glob.glob trouve tous les fichiers qui correspondent au pattern
        wav_files = glob.glob(os.path.join(instrument_source_dir, '*.wav'))
        
        file_counter = 0
        for wav_path in wav_files:
            # 1. Créer l'image du spectrogramme
            img_data = create_spectrogram_image(wav_path)
            
            if img_data is not None:
                # 2. Encoder en PNG
                png_data = tf.io.encode_png(img_data)
                
                # 3. Sauvegarder le fichier PNG
                # On garde le nom original
                base_name = os.path.basename(wav_path)
                file_name_png = os.path.splitext(base_name)[0] + ".png"
                output_path = os.path.join(instrument_output_dir, file_name_png)
                
                tf.io.write_file(output_path, png_data)
                file_counter += 1
        
        print(f"-> {file_counter} images créées pour {instrument_name}.")

    print("Pré-traitement terminé !")
