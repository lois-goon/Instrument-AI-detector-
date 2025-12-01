import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.signal.windows import hamming
from scipy.io import wavfile

from spectrogram import BuildFilePath2Wave, usage, lstINSTRUMENTS_CLASSES, szDATASET_DIR, szINSTRUMENT
import simpleaudio as sa
import time
import os


NOTE_NAMES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

def frequency_to_note(frequency):
    """
    Convertit une fréquence (en Hz) en la note de musique la plus proche.
    """
    if frequency <= 0:
        return "Silence"
        
    # Formule pour trouver le nombre de demi-tons (n) par rapport au La 440Hz
    # f = 440 * 2^(n/12)  ->  n = 12 * log2(f / 440)
    n = 12 * np.log2(frequency / 440.0)
    
    # Arrondir au demi-ton le plus proche
    n = round(n)
    
    # Trouver le nom de la note (0=La, 1=La#, 2=Si, 3=Do, ...)
    # On utilise 9 car "La" est la 9ème note en partant de "Do" (index 0)
    note_index = (n + 9) % 12 
    note = NOTE_NAMES[note_index]
    
    # Trouver l'octave
    # (n + 57) vient de (n + 9 (décalage 'La') + 48 (décalage octave 0))
    # 4 correspond à l'octave du La 440
    octave = (n + 57) // 12 + 3 # On ajuste pour que 440Hz soit La4
    
    return f"{note}{octave} ({frequency:.2f} Hz)"

def find_fundamental_frequency(sound_data, sampling_rate):
    """
    Analyse un segment de son et trouve la fréquence fondamentale (le pic).
    """
    # Appliquer une fenêtre (Hamming) pour lisser les bords
    window = np.hamming(len(sound_data))
    sound_data_windowed = sound_data * window
    
    # Calculer la FFT
    # np.fft.rfft est optimisé pour les signaux réels (non-complexes)
    fft_spectrum = np.fft.rfft(sound_data_windowed)
    
    # Obtenir les fréquences correspondantes à chaque "bac" de la FFT
    fft_freqs = np.fft.rfftfreq(len(sound_data), 1.0 / sampling_rate)
    
    # Trouver la magnitude (l'énergie) de chaque fréquence
    fft_magnitude = np.abs(fft_spectrum)
    
    # Trouver l'index de la magnitude maximale
    # On ignore les très basses fréquences (ex: < 20 Hz) qui sont souvent du bruit
    min_freq_index = 0
    if np.any(fft_freqs > 20):
        min_freq_index = np.where(fft_freqs > 20)[0][0]
        
    peak_index = np.argmax(fft_magnitude[min_freq_index:]) + min_freq_index
    
    # La fréquence fondamentale est la fréquence à cet index
    fundamental_frequency = fft_freqs[peak_index]
    
    # On retourne aussi les données FFT pour l'affichage
    return fundamental_frequency, fft_freqs, fft_magnitude

# --- Ton ancienne fonction pour lire le .wav ---
# (Elle ne change pas)
def Wav_Reader(szFileName):
   freq, sound_data =  wavfile.read(szFileName)
   
   # Si stéréo, on moyenne en mono
   if sound_data.ndim > 1:
       sound_data = sound_data.mean(axis=1)
       
   print(f"Fichier lu. Fréquence d'échantillonnage: {freq} Hz")
   print(f"Durée: {sound_data.shape[0] / freq:.2f} secondes")
   return freq, sound_data


# --- Voici comment tout utiliser ---
if __name__ == "__main__":
    
    # 1. Choisir un instrument et un échantillon pour le test
    INSTRUMENT_INDEX = 25  # Index pour "Violin"
    SAMPLE_NUMBER = 69     # Le fichier 69.wav (celui de l'image du PDF)

    try:
        # 2. Obtenir le chemin complet
        path = BuildFilePath2Wave(INSTRUMENT_INDEX, SAMPLE_NUMBER)
        print(f"Analyse du fichier : {path}")

        # 3. Lire le fichier
        sampling_rate, sound_data = Wav_Reader(path)

        # 4. Analyser la note à différents instants
        duration_sec = len(sound_data) / sampling_rate
        time_step_sec = 0.25 
        window_size = 4096 

        print("\n--- Analyse des notes (par 0.25s) ---")
        
        # --- CONFIGURATION DU GRAPHIQUE STYLÉ --- # <--- NOUVEAU
        plt.ion() # Mode interactif ON (très important)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Analyse en temps réel de: {os.path.basename(path)}', fontsize=16)
        # ---------------------------------------- # <--- NOUVEAU

        for current_time in np.arange(0, duration_sec - (window_size/sampling_rate), time_step_sec):
            
            start_sample = int(current_time * sampling_rate)
            end_sample = start_sample + window_size
            
            if end_sample > len(sound_data):
                break 
                
            sound_segment = sound_data[start_sample:end_sample]
            
            # Récupérer les données FFT pour l'affichage
            freq, freqs, magnitude = find_fundamental_frequency(sound_segment, sampling_rate)
            
            note = frequency_to_note(freq)
            
            print(f"Temps {current_time:.2f}s : {note}")
            
            # --- MISE À JOUR DU GRAPHIQUE --- # <--- NOUVEAU
            
            # 1. Nettoyer les anciens graphiques
            ax1.cla()
            ax2.cla()
            
            # 2. Graphique du HAUT : Onde sonore (Domaine Temporel)
            ax1.plot(np.linspace(current_time, current_time + (window_size/sampling_rate), window_size), sound_segment, color='dodgerblue')
            ax1.set_title(f"Segment à {current_time:.2f}s")
            ax1.set_ylabel("Amplitude")
            ax1.set_ylim([np.min(sound_data), np.max(sound_data)]) # Garder la même échelle Y
            ax1.set_xlabel("Temps (s)")

            # 3. Graphique du BAS : Spectre (Domaine Fréquentiel)
            ax2.plot(freqs, magnitude, color='crimson')
            ax2.set_title(f"Spectre de Fréquence - Note détectée : {note}")
            ax2.set_ylabel("Magnitude (Énergie)")
            ax2.set_xlabel("Fréquence (Hz)")
            ax2.set_xlim([0, 5000]) # On se concentre sur les fréquences audibles (0-5kHz)
            
            # On trace une ligne sur le pic détecté
            ax2.axvline(freq, color='black', linestyle='--', label=f'Pic: {freq:.2f} Hz')
            ax2.legend(loc='upper right')

            # 4. Rafraîchir la fenêtre
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster la mise en page
            plt.pause(0.1) # Pause très courte pour rafraîchir (0.1s)
            
            # --------------------------------- # <--- NOUVEAU

    except FileNotFoundError:
        print("\n--- ERREUR ---")
        print("Le fichier n'a pas été trouvé à l'emplacement :")
        print(BuildFilePath2Wave(INSTRUMENT_INDEX, SAMPLE_NUMBER))
    except Exception as e:
        print(f"Une autre erreur est survenue : {e}")

    finally:
        # <--- NOUVEAU
        print("Analyse terminée. Fermeture du graphique.")
        plt.ioff() # Mode interactif OFF
        plt.show() # Garder la dernière image affichée
        # <--- NOUVEAU
