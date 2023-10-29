#Versione del 2023-08-24 n.1

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch, stft
from scipy.fft import fft
import hfda
from scipy.stats import skew, kurtosis
#import eeglib
import threading
import time

from statsmodels.tsa.ar_model import AutoReg #autoregressive model
from scipy.linalg import norm                   #calcolo norma
import matplotlib.pyplot as plt                 #per histogramma

from tqdm import tqdm #for update progress bar
import warnings
warnings.filterwarnings('ignore') #stop showing messages 

#from editable_inputs import *


CHANNELS = 128 #number of channels
fs = 2048 #sampling frequency





def feature(df, i, fl):
  #print(f'Starting the task {i}...')
  
  value = df.iloc[i].to_numpy()

  N = value.size # Number of samples 
  #print(N)
  
  #MAVS = 0 #Mean absolute value slope (MAVS) Sara un campo riempito dopo
  mean = np.mean(value)
  median = np.median(value)  
  MAV = np.mean(np.absolute(value))  # Mean absolute value
  STD = np.std(value)  # Standard deviation
  Var = np.var(value)  # Variance
  WL = np.sum(np.absolute(value[1:] - value[:-1]))  # Waveform length
  ZC = np.sum((-value[:-1] * value[1:]) > 0)  # Zero crossing
  RMS = np.sqrt(np.mean(value ** 2))  # Root mean square
  peaks = find_peaks(value, height=RMS)
  NP = len(peaks[1]['peak_heights']) # Number of peaks

  # MPV = len(np.array(peaks[1]['peak_heights']))/(NP)  # Mean of peak values
  # MFD = (np.sum(np.absolute(
  #           np.array(peaks[1]['peak_heights'])[1:] - 
  #           np.array(peaks[1]['peak_heights'])[:-1])))/(NP) 
  #       #Differendo da Khang, sarebbe meglio dividere per NP-1 secondo me
        # Mean firing difference

  MFV = NP / (N/(fs/1000)) #((fs/1000)*N) # Mean firing velocity 
  
  DAMV = ((np.sum(np.absolute(value[1:] - value[:-1])))/(N))  # Difference absolute mean value
  IAV = np.sum(np.absolute(value))  # Integrated absolute value
  HFD = hfda.measure(value, 5)  # Higuchi’s fractal dimension
  SKEW = skew(value)  # Skewness
  
  HMob = eeglib.features.hjorthMobility(value)  # Hjorth mobility parameter
  HCom = eeglib.features.hjorthComplexity(value)  # Hjorth complexity parameter
  K = kurtosis(value)  # Kurtosis
  PER = np.percentile(value, 75)  # 75° percentile
  f = lambda x: (x >= 0).astype(float)
  SSC = sum(f(np.diff(value, prepend=1)[1:-1] * -np.diff(value)[1:])) # Slope sign changes
  
  #new features added
  MFL = np.log10(np.sqrt(np.sum((value[1:] - value[:-1])** 2))) #maximum fractal length
  DASDV = np.sqrt((1/(value.size-1)) * (np.sum((value[1:] - value[:-1])** 2))) #Difference absolute standard deviation value 
  #LOG = #np.exp(np.mean(np.log10(np.absolute(value)))) #Logarithm
  SSI = np.sum(value**2) #  Simple Square Integral (SSI)

  SogliaWAM = 2 #Valore desunto dai paper
  f = lambda x: (x > SogliaWAM).astype(float)
  WAM = np.sum(f(np.absolute(value[:-1] - value[1:])))  #Willison amplitude

  SogliaMYOP = 0.0066 #Valore desunto dai paper (da ricontrollare) in mV
  f = lambda x: (x > SogliaMYOP).astype(float)
  #MYOP = (np.sum(f(np.absolute(value[:-1] - value[1:])))/N) #Myopulse Value
  MYOP = (np.sum(f(np.absolute(value))))/N #Myopulse Percentage Rate

  AAC = ((np.sum(np.absolute(value[1:] - value[:-1])))/N) #Average amplitude change

  
  value_00_to_25 = value[0:int(N*0.25)] #il primo 25% dei valori
  value_25_to_75 = value[int(N*0.25):int((N*0.75))] #dal 25esimo al 75esimo valore (ragionando su 100)
  value_75_to_100 = value[int(N*0.75):] #dal 75esimo valore in poi

  MAV1 = np.mean([
     *(np.absolute(value_00_to_25) * 0.5), 
     *(np.absolute(value_25_to_75) ), 
     *(np.absolute(value_75_to_100) * 0.5)])
  
  MAV2 = np.mean([
    *(np.absolute(value_00_to_25)  * ((4 * np.array(range(1, int((N)*0.25)+1)))/N)), #coefficiente = 4i/N
    *(np.absolute(value_25_to_75)), #coefficiente = 1
    *(np.absolute(value_75_to_100) * ( 4 * (np.array(range(int(N*0.75), int(N))) - N)) / N) #coefficiente = (4(i-N))/N
    ]) 


  TM3 = np.absolute(np.mean(value**3)) #Third central moment
  TM4 = (np.mean(value**4)) #Fourth central moment
  TM5 = np.absolute(np.mean(value**5)) #Fifth central moment

  # #L2 = norm(value, 2) #L2 norm
  VOrderVal = 2 #valore trvato dai paper (che lo rende uguale a VAR)
  VOrder = (np.mean((value**VOrderVal)))**VOrderVal #V-order
  
  #AROrder = 4 #da qualche parte ho letto che 4 è il valore migliore
  #AR = AutoReg(pd.DataFrame({'': value.ravel()}), lags=AROrder).fit().params.values #AR (Autoreressive) model
  #restituisce una lista di coefficienti, che sono i parametri del modello
  
  bins = 9 #Numero di "cestini" Scelto 9 dai paper
  HIST = np.histogram(value, bins=bins)[0] #Histogram
  

  #Frequency domain features FDF
  ##Per il momento vengono lasciate da stare visto che creano problemi nel csv con i nomi delle colonne
  
  #power spectral density
  #f contains the frequency components
  #S is the PSD
  #f1, Pxx = welch(value, fs) ### Feature dominio della frequenza

  #short time fourier transform
  #f2, t, Zxx = stft(value, fs, nperseg=len(value)) ###

  #discrete Fourier Transform
  ##dft = fft(value) ###
  

  #Crea array
  
  features = np.array([
    mean, median, MAV,  
    STD, Var, WL, 
    ZC, RMS, NP, 
    MFV,
    DAMV, IAV, HFD, 
    SKEW, HMob, HCom, 
    K, PER, SSC, 
    MFL, DASDV, SSI, 
    WAM, MYOP, AAC, 
    MAV1, MAV2, TM3, 
    TM4, TM5, VOrder
    
    ])
  #features = np.concatenate((features, AR), axis=None)
  features = np.concatenate((features, HIST), axis=None)  
  fl[i] = features


#if __name__ == '__main__':
def mainFE(filename, savefile, soggetto):

    start_time = time.time()


    df = pd.read_csv(filename, header=None)#('EMGdataset.sw_STD.csv', header=None)
    print("\nDataset loaded")
    [l,r] = df.shape

    df2 = pd.read_csv('EMGdataset_'+str(soggetto)+'.subrepetitions_idx.csv', header=None, index_col=None)
    df3 = pd.read_csv('EMGdataset_'+str(soggetto)+'.subrepetitions_classes.csv', header=None, index_col=None)
    print("Classes loaded")
    print("Extracting features...")

    fl = [None] * l #vettore temporaneo per contenere le features
    count = 0

    #initiliaze progress bar
    pbar = tqdm(range(l-1)) #set progress bar

    # We used threads to speed up the process
    # We extract the features from all the sw and we put it in fl
    for k in range(0, l):
      feature(df, k, fl)
                
      #update progress bar
      pbar.n = k
      pbar.refresh()

    print("\nFeatures extracted")

    #re-initialize progress bar
    print("Merging feature vectors...")
    pbar.n = 0
    pbar.refresh()

    #feature for GNN
    #pd.DataFrame(fl[:]).to_csv('EMGdataset.node_attrs.csv', mode='a', header=False, index=False)

    num_subrepetitions = int(l/CHANNELS)
    fv = [None] * num_subrepetitions

    # We create feature vectors and we put it in fv with the class and the id of the repetition
    count = 0
    for j in range(0, l, CHANNELS):
        #add the feature
        fvj = np.concatenate((fl[j:j+CHANNELS]), axis=None)
        #add the class column
        classe = list(df3.iloc[count])
        fvj = np.append(fvj, [classe]) #this add the class of the repetition
        #add the index column
        idx = list(df2.iloc[count])
        fvj = np.append(fvj, [idx]) #this add the index of the repetition
        
        #add the feature vector to the list
        fv[count] = fvj
        count = count + 1

        #update progress bar
        pbar.n = j
        pbar.refresh()

    print("\nFeature estratte in %s seconds" % (time.time() - start_time))
    
    #pd.DataFrame(fv).to_csv('EMGdataset.feature_vectors.csv', header = False, index = False)


    pd.DataFrame(fv).to_csv(
        savefile, #'EMGdataset.feature_vectors_STD_5.csv', 

        #le colonne hanno nome di numero progressivo * 128 canali + 2 colonne aggiuntive per classe e indice
        header = [str(y) for y in range(0,(len(fl[0])*CHANNELS))] + ['class','id'], 

        index = False)
    print("Script terminato in %s seconds" % (time.time() - start_time))

    #salvare un dataframe con le feature estratte e il tempo impiegato

    
    print("Inserisci la dimensione della finestra (es. 150): ")
    sw = editable_input("150")
    print("Inseriscile feature usate: ")
    feature_usate = editable_input('MEAN MEDIAN MAV STD VAR WL ZC RMS NP MPV MFD MFV DAMV IAV HFD SKEW HMob HCom K PER SSC MFL DASDV SSI WAM MYOP AAC MAV1 MAV2 TM3 TM4 TM5 VOrder hist1 hist2 hist3 hist4 hist5 hist6 hist7 hist8 hist9')
    print("Inserisci i canali esclusi: ")
    canali_esclusi = editable_input('(nessuno)')
    print("Inserisci l'overlap: ")
    overlap = editable_input("0")
    print("Inserisci le operazioni di pre-processing (es. STD, RAW, NORM): ")
    op_prepr = editable_input('STD')
    print("Inserisci nome del file di salvataggio delle metriche: ")
    savefile_metriche = editable_input("Results_extraction_ID_" + str(soggetto)+".csv")


  
    
    df_info = pd.DataFrame(
        {'feature_extraction_time': [float(time.time() - start_time)],
         'features names': [feature_usate],
         'subject': [soggetto],
         'sw': [sw],
         'canali_esclusi': [canali_esclusi],
         'overlap': [overlap],
         'op_prepr': [op_prepr],
        })

    pd.DataFrame(df_info).to_csv( savefile_metriche,   header=True, index = False)
