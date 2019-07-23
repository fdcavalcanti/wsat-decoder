import scipy.io.wavfile as sio
import scipy.signal as sig
import scipy.ndimage as ndimg
import numpy as np
import cv2

#Import raw 11025 Hz wav file
(Fs, audioRaw) = sio.read('DataAq/Resampled/20190523_195020Z_137100000Hz_AF.wav')
audioRaw =  audioRaw/(2.**15)
lineLen = int(np.floor(Fs/2))
lineTotal = int(np.floor(len(audioRaw)/lineLen))
audioRaw = audioRaw[0:lineTotal*lineLen]
medFiltOn = 1

#Filter audio with Hilbert transform and get absolute value
audioFiltered = sig.hilbert(audioRaw)
signal = np.abs(audioFiltered)
#Optional
if medFiltOn == 1:
    #signal = sig.medfilt(signal,5)
    signal = ndimg.median_filter(signal,7)

#Generate syncA signal
T = 1/4160
t = np.linspace(0,28*T,77,endpoint=True)#77 -> 29*2.65
sqr = 2*sig.square(2*np.pi*1040*t-4*T)
corr = np.correlate(signal,sqr,'full')
syncA = corr[len(sqr):len(corr)]

#Map to grayscale
maxVal = np.amax(signal)
Ydig = np.floor(255*signal/maxVal)
Ydig[Ydig<0] = 0

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

#fig1 = sig.medfilt2d(fig1,3) #3x3 mean

fig1 = np.uint8(fig1) #plt.imshow(fig1,cmap='gray')
fig2 = cv2.resize(np.float32(fig1),(2080,lineTotal)) #Fig1 com dimensão corrigida
fig2 = np.uint8(fig2)
#Plot
#cv2.namedWindow('APT Fig', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('APT Fig', 1010, 600)
cv2.imwrite("ITAescura.jpg",fig2)
cv2.imshow("APT Fig", fig2)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################################
print('Sample frequency: ' + str(Fs) + ' Hz')
print('Original resolution: ' + str(np.shape(fig1)))