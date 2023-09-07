import librosa
import numpy as np

def FFT(y1,fs,DB=True):

    n = len(y1)
    k = np.arange(n)
    T = n/fs
    freq = k/T 
    freq = freq[range(int(n/2))] 
    Y1 = np.fft.fft(y1)/n 
    Y1 = Y1[range(int(n/2))]

    if DB==True:
        Y1=librosa.amplitude_to_db(abs(Y1))
        
        
    return Y1