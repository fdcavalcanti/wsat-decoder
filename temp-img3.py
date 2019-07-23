import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.stats as reg
import pandas as pd

SyncA = 0; SpaceA = 39; VideoA = 86; TelA = 995
SyncB = 1039; SpaceB = 1079; VideoB = 1125; TelB = 2035;
res_prop = Fs/4160

def mean_mat(telemetryData):
    #Pega a média das linhas da coluna de telemetria
    telemetry_Mean_Mat = []
    for ii in range(len(telemetryData)):
        telemetry_Mean = np.mean(telemetryData[ii,:])
        telemetry_Mean_Mat = np.append(telemetry_Mean_Mat,telemetry_Mean)
    telemetry_Mean_Mat = np.uint8(telemetry_Mean_Mat)
    
    return telemetry_Mean_Mat


def get_Frame(xFig,mode,plot):
    if mode == 1:
        telemetry = np.uint8(xFig[:,TelA:SyncB-3])
    else:
        telemetry = np.uint8(xFig[:,TelB:2080-3])

    telemetry_Mean_Mat = mean_mat(telemetry)
    corrPosData = plot_telemetry_analysis(telemetry_Mean_Mat,plot)
    frame = telemetry[corrPosData-63:corrPosData+64, :]
    if plot == 1:
        cv2.imshow("Telemetry Data", telemetry)
        cv2.imshow("Frame Data", frame)
        cv2.waitKey(0)        
        cv2.destroyAllWindows()
        cv2.imwrite("frame.jpg",frame)

    return frame, corrPosData

def get_SpaceScan(xFig,linePos):
    #Space Count
    spaceData = xFig[linePos:linePos+34,SpaceB:VideoB]
    '''cv2.imshow("Space B", spaceData)
    cv2.waitKey(0)        
    cv2.destroyAllWindows()'''

    Cs = np.mean(spaceData)
    return Cs

def get_BBTemp(frame):
    #Wedges 10-13: Black Body Temperatures 1-4
    #Step 1 (Dados NOAA19)
    show = np.vstack((frame[9*8:10*8],frame[10*8:11*8]))
    show = np.vstack((show,frame[11*8:12*8]))
    show = np.vstack((show,frame[12*8:13*8]))
    
    #cv2.imshow("BB Data", show)
    #cv2.waitKey(0)        
    #cv2.destroyAllWindows()

    Cprt1 = np.mean(frame[9*8:10*8])
    Cprt2 = np.mean(frame[10*8:11*8])
    Cprt3 = np.mean(frame[11*8:12*8])
    Cprt4 = np.mean(frame[12*8:13*8])
    Cbb = (Cprt1 + Cprt2 + Cprt3 + Cprt4)/4
    #print(Cprt1);print(Cprt2);print(Cprt3);print(Cprt4)
    #Tprt 1
    d0 = 276.6067; d1 = 0.051111; d2 = 1.405783e-6;
    Tprt1 = d0 + d1*Cprt1 + d2*Cprt1**2
    #Tprt 2
    d0 = 276.6119; d1 = 0.051090; d2 = 1.496037e-6;
    Tprt2 = d0 + d1*Cprt2 + d2*Cprt2**2
    #Tprt 3
    d0 = 276.6311; d1 = 0.051033; d2 = 1.496990e-6;
    Tprt3 = d0 + d1*Cprt3 + d2*Cprt3**2
    #Tprt 4
    d0 = 276.6268; d1 = 0.051058; d2 = 1.493110e-6;
    Tprt4 = d0 + d1*Cprt4 + d2*Cprt4**2

    #print(Tprt1);print(Tprt2);print(Tprt3);print(Tprt4)
    Tbb = (Tprt1+Tprt2+Tprt3+Tprt4)/4
    
    return Tbb, Cbb

def get_PatchTemp(frame):
    #Wedge 14
    temp = np.mean(frame[13*8:14*8])
    return temp

def get_BackScan(frame):
    #Wedge 15
    BackScan_Mean = np.mean(frame[14*8:15*8])
    return BackScan_Mean

def get_ChannelInfo(frame):
    #Wedge 16: Channel Identification
    x = np.array([31, 63, 95, 127, 159, 191, 223, 255])
    channel_ID = frame[-8:,:]
    channel_ID = np.mean(channel_ID)
    index = (np.abs(x-channel_ID)).argmin() + 1
    return index

def get_Radiance(Tbb):
    #Step 2
    #Calculate black body radiance
    #Channel 4 NOAA 19
    A = 0.53959; B = 0.998534; uc = 928.9
    c1 = 1.1910427e-5; c2 = 1.4387752

    Tbb2 = A + B*Tbb
    Nbb = (c1*uc**3) / (np.exp(c2*uc/Tbb2)-1.0) #Radiance
    
    return Nbb

def get_earthRadiance(Nbb, Cbb, Cs, Ce, Regression10Bit):
    #Step 3
    #Table D.6-2.
    Ns = -5.49
    b0 = 5.7; b1 = -0.11187; b2 = 0.00054668
    
    #Valores originais do manual
    raw = pd.read_excel('NlinLookUp.xlsx')
    NLinLookUp = raw.values
    NLinLookUp = NLinLookUp.flatten()
    
    #Gera o mesmo range do manual para o obtido aqui (55-134 8bit)
    testNlin = Ns + (Nbb - Ns)*(Cs-Regression10Bit)/(Cs-Cbb)
    testNlin = testNlin.flatten()
    regData = reg.linregress(testNlin,NLinLookUp)
    print("Slope: " + str(regData[0]))
    print("Temp Int: " + str(regData[1]))
    ''' 
    plt.figure(5)
    plt.plot(testNlin)
    plt.plot(NLinLookUp)
    plt.plot(testNlin*regData[0]+regData[1])
    '''
    Nlin = Ns + (Nbb - Ns)*(Cs-Ce)/(Cs-Cbb)
    Nlin = Nlin*regData[0]+regData[1]
    Nlin[Nlin<=0] = 0
    Ncor = b0 + b1*Nlin + b2*Nlin**2
    Ne = Nlin + Ncor
    
    return Ne

def rad2bb(Ne):
    A = 0.53959; B = 0.998534;
    c1 = 1.1910427e-5; c2 = 1.4387752; uc = 928.9
    Te2 = c2*uc / np.log(1 + c1*(uc**3)/Ne)
    Te = (Te2-A)/B
    
    return Te, Te2
    
def get_CalibrationWedges(frame):
    #Wedges 1-8: Calibration Values
    calibrationMean = []
    calFrame = frame[0:64]     
    for ii in np.arange(0,len(calFrame),8):
        mean = np.mean(calFrame[ii:ii+8])
        calibrationMean = np.append(calibrationMean,mean)
        
    return calibrationMean    

def get_Coeffs(calibrationMean):
    #Coefficients for linear regression analysis
    X = np.array([31, 63, 95, 127, 159, 191, 223, 255])
    Y = calibrationMean
    sumXY = np.dot(X,Y)
    B = (8*sumXY - sum(X)*sum(Y)) / (8*sum(X**2)-(sum(X)**2))
    A = (sum(Y)-B*sum(X)) / 8
    #print('A: ' + str(A) + ' B: ' + str(B))
            
    return A,B

def normalize_APT(fig,A,B,show):
    #Uses the linear regression to correct image values
    figNorm = (fig-A)/B
    figNorm[figNorm>=255] = 254
    figNorm[figNorm<=0] = 0
    figNorm = np.uint8(figNorm)
    if show == 1:
        cv2.namedWindow('APT Fig', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('APT Fig', 1010, 600)
        cv2.imshow("APT Fig", figNorm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return figNorm

def plot_telemetry_analysis(telemetryData,okplot):    
    wedgeArray = np.array([31]*8 + [63]*8 + [95]*8 + [127]*8 + [159]*8 + [191]*8\
                      + [223]*8 + [255]*8)

    #Correlação da coluna de telemetria com a referência
    corrWedges = np.correlate(telemetryData,wedgeArray,'full')
    maxCorr = np.amax(corrWedges)                #Define maior valor do vetor
    maxCorrPos = np.where(corrWedges == maxCorr) #Encontra posição de maior valor
    maxCorrPos = int(maxCorrPos[0])
    #Para visualizar posição de maior correlação
    wedgeShift = np.array([0]*(maxCorrPos-len(wedgeArray)))
    #Desloca wedge referência para da posição de maior correlação
    wedgeArrayShift = np.concatenate((wedgeShift, wedgeArray)) 
    frameVector = telemetryData[maxCorrPos-63:maxCorrPos+64] #Todo o frame
    
    if okplot == 1:
        plot1 = plt.figure()
        ax1 = plot1.add_subplot(311); ax2 = plot1.add_subplot(312, sharex = ax1)
        ax3 = plot1.add_subplot(313)
        ticks = np.arange(0,127,8); ax3.set_xticks(ticks); ax3.grid()
        ax1.plot(telemetryData, linewidth=1, color = 'b')
        ax1.plot(wedgeArrayShift, linewidth=1, color = 'g')
        ax2.plot(corrWedges, linewidth=1, color = 'r')
        ax3.plot(frameVector, color = 'b')
        plot1.suptitle('Visualização da Telemetria',fontsize=20)
        ax1.set_title('Telemetria',fontsize=16)
        ax2.set_title('Correlação',fontsize=16)
        ax3.set_title('Frame',fontsize=16)
        ax3.set_xlabel('Posição',fontsize=15)
        ax1.set_ylabel('Amplitude',fontsize=16)
        ax2.set_ylabel('Amplitude',fontsize=16)
        ax3.set_ylabel('Amplitude',fontsize=16)
    return maxCorrPos

def get_VideoA(xFig):
    videoA = xFig[:,VideoA:TelA-5]
    return videoA

def get_VideoB(xFig):
    videoB = xFig[:,VideoB:TelB-5]
    return videoB

def plotNormalized(xFigNorm,flip):
    #xFigNorm = cv2.resize(np.float32(xFigNorm),(2080,lineTotal)) #Fig1 com dimensão corrigida
    #xFigNorm = np.uint8(xFigNorm)
    if flip == 1:
        xFigNorm = np.flipud(xFigNorm)
        xFigNorm = np.fliplr(xFigNorm)
    
    cv2.namedWindow('APT Fig',cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
    cv2.imshow("APT Fig", xFigNorm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("flp.jpg",xFigNorm)


def viewEarthTemp(te2plot):
    #te2plot = cv2.resize(np.float32(te2plot),(909,lineTotal)) #Fig1 com dimensão corrigida
    plotTe = plt.axes()
    img = plotTe.imshow(te2plot,cmap='gist_rainbow_r')
    plt.colorbar(img)
    plt.suptitle('Mapa de Temperatura (ºC)',fontsize=20)
    

def compareNlin(Coeffs):
    orig8Bit = np.arange(55,135,1)
    orig8BitCount = (orig8Bit-Coeffs[0])/Coeffs[1]
    Corr10Bit = orig8BitCount*4
    return Corr10Bit
   
def generateImage(OKFig,tempFig):
    newImg = OKFig[:,:VideoB]
    #newImg = np.hstack((newImg,tempFig))
    print(np.shape(newImg))
    cv2.namedWindow('Nova Imagem',cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Nova Imagem", newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Visible image correction
frame, corrPosData    = get_Frame(fig2,0,1) #(fig, A:1/B:0, plot)
calibration_Mean      = get_CalibrationWedges(frame)
Coeffs                = get_Coeffs(calibration_Mean)
OKFig                 = normalize_APT(fig2,Coeffs[0],Coeffs[1],1) #(fig,coeff1,coeff2,plot)
plotNormalized(OKFig,1) #2nd: flip

newFrame, corrPosData = get_Frame(OKFig,0,0)#(fig, A:1/B:0, plot)
channel               = get_ChannelInfo(newFrame)

x4Frame               = 4*np.int16(newFrame)
FigX4                 = 4*np.int16(OKFig)

Cs                    = get_SpaceScan(FigX4,corrPosData)
Tbb, Cbb              = get_BBTemp(x4Frame)
PatchTemp             = get_PatchTemp(newFrame)
backScan              = get_BackScan(x4Frame)

videoA                = get_VideoA(FigX4)
videoB                = get_VideoB(FigX4)
Nbb                   = get_Radiance(Tbb)
Regression10Bit       = compareNlin(Coeffs)
Ne                    = get_earthRadiance(Nbb, Cbb, Cs, videoB, Regression10Bit)
Te, Te2               = rad2bb(Ne)
#generateImage(OKFig,Te)
Te                    = np.fliplr(np.flipud(Te))
viewEarthTemp(Te-273)

print('Calibration on channel ' + str(channel))
print('Deep space scan (Cs): ' + str(Cs))
print('Mean black body temp (Tbb): ' + str(Tbb) + ' K')
print('Radiance (Nbb): ' + str(Nbb))
print('Patch temp (105 K ideal): ' + str(.124*PatchTemp+90.113) + ' K')
