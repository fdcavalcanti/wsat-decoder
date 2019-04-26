import scipy.io.wavfile as sio
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

#Importa raw 11025 Hz wav file
(Fs, audioRaw) = sio.read('N18_4827.wav')
lineLen = 2080
lineTotal = int(np.floor(len(audioRaw)/lineLen))
audioRaw = audioRaw[0:lineTotal*lineLen]

#Filter audio with Hilbert transform and get absolute value
audioFiltered = sig.hilbert(audioRaw)
signal = np.abs(audioFiltered)

#Generate syncA signal
T = 1/4160
t = np.linspace(0,28*T,11025,endpoint=True)
sqr = sig.square(2*np.pi*1040*t-4*T)
corr = np.correlate(signal,sqr)

plt.plot(corr[1:np.floor(len(corr)/2)])
plt.show()
