
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.signal.windows import hamming
from scipy.io import wavfile
import simpleaudio as sa
import time
import os

lstINSTRUMENTS_CLASSES = [
    "Acoustic_Guitar",
    "Banjo",
    "Bass_Guitar",
    "Clarinet",
    "Cowbell",
    "Cymbals",
    "Dobro",
    "Drum_set",
    "Electro_Guitar",
    "Floor_Tom",
    "Flute",
    "Harmonica",
    "Harmonium",
    "Hi_Hats",
    "Horn",
    "Keyboard",
    "Mandolin",
    "Organ",
    "Piano",
    "Saxophone",
    "Shakers",
    "Tambourine",
    "Trumpet",
    "Ukulele",
    "Vibraphone",
    "Violin"
]
# the following string is the location of the dataset : 
szDATASET_DIR = '.././data/raw/music_dataset'
szINSTRUMENT = 'Violin'
#&&&&&&&
# help : 
#&&&&&&&
def usage( szPgmName):
    print(szPgmName + "<instrument number> <sample number> <outfile>")
    print("where <instrument number> is the integer corresponding to a given instrument")
    print("and   <sample number> is the number of the sequence to be processed (in brief : wave file name")
    print("without the .wav extension")
    print("the correspondancy between intruments and instruments number is the following:")
    for i in range(len(lstINSTRUMENTS_CLASSES)):
        print(str(i)+ "--->" + lstINSTRUMENTS_CLASSES[i])
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# build the complete file access path to the wave 
# file given the instrument and the sample number
# IN : 
#       iInstrument : instrument number
#       iSample     : sample number
# OUT :
#       szFileName
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def BuildFilePath2Wave( iInstrument, iSample):
    szFileName = szDATASET_DIR   +  '/' + lstINSTRUMENTS_CLASSES[iInstrument] + '/' + str(iSample) + '.wav'
    return szFileName
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# process a wave file to generate a npy array corresponding to a correlogram
# IN : 
#      szFileName           : path to the WAVE file
#      width = 512          : size of the hamming window
#      stride = 16          : value of the hamming shift in number of samples
#      bDisplay = False     : display the spectrogram if True
#      bPlaySound = False   : play the wave file if True
# OUT :
#      npaSpgm    : spectrogram
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def BuildSpectrogram( szFileName, width = 512, stride = 128, bDisplay = False, bPlaySound = True  ):
    freq, wvData = wavfile.read( szFileName)
    N = wvData.shape[0]
    print("Sampling period = " + str(freq)  + "Hz - number of samples = " + str(N) )
    w = hamming(width, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=stride, fs=freq, mfft=width, scale_to='magnitude')
    Sx = SFT.stft(wvData)  # perform the STFT
    #____________________________________________________________________
    # display if required
    if bDisplay:
        plt.imshow(abs(Sx), origin='lower', aspect='auto',
                extent=SFT.extent(N), cmap='viridis')
        plt.xlabel('time ($s$)')
        plt.ylabel('frequency ($Hz$)')
        plt.title('spectrum evolution over time \n File : ' + szFileName)
        plt.show()
    #____________________________________________________________________
    #___________________________
    # play sound if required
    if bPlaySound:
        wave_obj = sa.WaveObject.from_wave_file(szFileName)
        play_obj = wave_obj.play()
        time.sleep(N/freq + 0.1)
    #___________________________
    return Sx

##################################################
# entry point : 
if __name__ == "__main__":
    argc = len(os.sys.argv)
    if argc != 4:
        usage( os.sys.argv[0])
    else:
        iInstrument   = eval(os.sys.argv[1])
        iSample       = eval(os.sys.argv[2])
        szFile        = BuildFilePath2Wave( iInstrument, iSample)
        npaSpectro    = BuildSpectrogram(szFile, bDisplay = True, bPlaySound= True)
        szOutFileName = os.sys.argv[3]+ '.npy'
        print('saving spectrogram to ' + szOutFileName + ' ....')
        np.save(szOutFileName, npaSpectro)
        print('done.')
        
