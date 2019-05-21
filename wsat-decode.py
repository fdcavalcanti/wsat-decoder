import scipy.io.wavfile as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Import raw 11025 Hz wav file
(Fs, audioRaw) = sio.read('DataAq/Resampled/N18_4827.wav')
audioRaw =  audioRaw/(2.**15)
lineLen = int(np.floor(Fs/2))
lineTotal = int(np.floor(len(audioRaw)/lineLen))
audioRaw = audioRaw[0:lineTotal*lineLen]
medFiltOn = 0
 
#Filter audio with Hilbert transform and get absolute value
audioFiltered = sig.hilbert(audioRaw)
signal = np.abs(audioFiltered)
#Optional
if medFiltOn == 1:
    signal = sig.medfilt(signal,7)

#Generate syncA signal
T = 1/4160
t = np.linspace(0,28*T,77,endpoint=True)
sqr = 2*sig.square(2*np.pi*1040*t-4*T)
corr = np.correlate(signal,sqr,'full')
syncA = corr[len(sqr):len(corr)]

#Map to grayscale
maxVal = np.amax(signal)
Ydig = np.floor(255*signal/maxVal)
for i in range(len(Ydig)):
    if Ydig[i] < 0:
        Ydig[i] = 0

#Get max correlation points
range1 = int(1); range2 = lineLen; matpos = []
for i in np.arange(lineLen,len(syncA),lineLen):
    M = np.amax(syncA[range1:range2])
    pos = np.amax(np.where(syncA[range1:range2] == M))
    matpos.append(pos+range1)
    range1 = 1+i-lineLen; range2 = i

#Generate image by initiating new line every maximum correlation point
fig1 = np.empty((1,lineLen))
matpos = np.array(matpos,dtype=int)
for pos in matpos:
    newData = Ydig[pos:pos+lineLen]
    fig1 = np.vstack((fig1,newData))

fig1 = np.array(fig1,dtype=int) #plt.imshow(fig1,cmap='gray')
fig2 = Image.fromarray(fig1) #fig2.show()
