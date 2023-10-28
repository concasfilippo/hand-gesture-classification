import math
import threading
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import hfda
from scipy.stats import skew, kurtosis
import eeglib
import time
import h5py
from sklearn.preprocessing import StandardScaler
from utils import *

from tqdm import tqdm #for update progress bar
import warnings
warnings.filterwarnings('ignore') #stop showing messages 

import os
import re

CHANNELS = 128 #number of channels
fs = 2048 #sampling frequency
N = 307
WS = 307
SHIFT = 307


def extraction(lista_nomi_features, id_soggetto):
    PATH_FILE_RAW = 'Subjects/s' + str(id_soggetto) + '.mat'
    #carico file raw
    print("Caricamento file raw in corso...")
    matF = h5py.File(PATH_FILE_RAW,'r')
    print("Caricamento file raw completato")

    E = np.array(matF['emg_extensors'])
    F = np.array(matF['emg_flexors'])

    # Reshape the files so that the channels are aligned and no longer in a square matrix
    [l,r,c] = E.shape
    E = E.reshape(l*r, c)
    F = F.reshape(l*r, c)

    C = np.array(matF['adjusted_class'])

    #standardizzazione dei dati
    timer = time.time()
    print("Standardizzazione dei dati in corso...")
    scaler = StandardScaler()
    scaler.fit(E)
    E = scaler.transform(E)
    scaler.fit(F)
    F = scaler.transform(F)
    std_timer = time.time() - timer
    print("Standardizzazione dei dati completata in " + str(std_timer) + " secondi")

    # We find the index of the final sample of each repetition
    end_reps_ind = end_ind(C[0])

    # We divide the recorded samples into lists. Each list contains the samples of a single repetition.
    l1 = create_sub_list(E, end_reps_ind)
    l2 = create_sub_list(F, end_reps_ind)

    # We create sliding windows for each repetition. 
    # We do not consider the case where no action takes place.
    fl = [None] * math.floor((len(end_reps_ind)-1)/2)
    count = 0

    pbar = tqdm(range(int(len(end_reps_ind)))) #set progress bar
    print("Suddivisione in sliding windows in corso...")

    # We used threads to speed up the process
    for k in range(1, len(end_reps_ind), 20):
        threads = list()
        for i in range(0,19,2):
            if k + i < len(end_reps_ind):
                t = threading.Thread(target=sliding_windows_attrs, args=(WS, SHIFT, l1[k + i], l2[k + i], count, fl))
                threads.append(t)
                t.start()
                count = count + 1
        for index, thread in enumerate(threads):
            thread.join()

        #update progress bar
        pbar.n = k + 20
        pbar.refresh()


    # We create a vector containing the number of sliding windows obtained from each repetition and 
    # concatenate the sliding windows of all the repetitions.
    sw_per_repetition = []
    df_dati = pd.DataFrame()

    #creazione dataset dati grezzi
    for j in range(0, len(fl)):
        sw_per_repetition.append(fl[j].shape[0])
        df_dati = pd.concat([df_dati, fl[j]])

    # We create the subrepetitions_classes and the subrepetitions_idx files
    subrepetitions_classes = pd.DataFrame(columns = ['label'])
    subrepetitions_idx = pd.DataFrame(columns = ['idx'])
    idx = 1
    idx_class = 1

    for j in range(0, len(sw_per_repetition)):
        num_subrepetitions = math.floor(sw_per_repetition[j] / CHANNELS)
        # For each subrepetition we have the label
        subrepetitions_class = C[:, end_reps_ind[idx_class]]
        idx_class = idx_class+2
        i = pd.DataFrame(subrepetitions_class * np.ones(num_subrepetitions), dtype=int, columns=['label'])
        subrepetitions_classes = pd.concat([subrepetitions_classes, i])
        # For each subrepetition we have an id
        p = pd.DataFrame(idx * np.ones(num_subrepetitions), dtype=int, columns=['idx'])
        if idx == 5:
            idx =1
        else:
            idx = idx + 1
        subrepetitions_idx = pd.concat([subrepetitions_idx, p])

    #estrazione features
    [l,r] = df_dati.shape
    

    pbar = tqdm(range(int(l/CHANNELS))) #set progress bar

    nomi_colonne = lista_nomi_features.iloc[1,:]
    nomi_colonne = nomi_colonne.append(pd.Series(['class', 'idx']))
    df_feature_vectors = pd.DataFrame(columns=nomi_colonne)

    count = 0
    for i in range(0,l,CHANNELS):
        df_sw = df_dati.iloc[i:i+CHANNELS].to_numpy()

        riga_feature_estratte = []

        for j in range(lista_nomi_features.shape[1]):
            feature_canale_da_estrarre = str(lista_nomi_features.iloc[:,j])
            nome_feature_da_estrarre = re.search(r'1\s+(.*)', re.search(r'(\d+)\s+(.*?)\s+Ch\.', str(feature_canale_da_estrarre)).group(2)).group(1)
            canale_da_estrarre = int(re.search(r'(\d+)\n', str(feature_canale_da_estrarre.split(".")[1])).group(1))
           

            value = df_sw[canale_da_estrarre-1]

            match nome_feature_da_estrarre:
                case "MEAN":
                    riga_feature_estratte.append(np.mean(value))
                case "MEDIAN":
                    riga_feature_estratte.append(np.median(value))
                case "MAV":
                    riga_feature_estratte.append(np.mean(np.absolute(value)))
                case "STD":
                    riga_feature_estratte.append(np.std(value))
                case "VAR":
                    riga_feature_estratte.append(np.var(value))
                case "WL":
                    riga_feature_estratte.append(np.sum(np.absolute(value[1:] - value[:-1])))
                case "ZC":
                    riga_feature_estratte.append(np.sum((-value[:-1] * value[1:]) > 0))
                case "RMS":
                    riga_feature_estratte.append(np.sqrt(np.mean(value ** 2)))
                case "NP":
                    RMS = np.sqrt(np.mean(value ** 2))
                    peaks = find_peaks(value, height=RMS)
                    riga_feature_estratte.append(len(peaks[1]['peak_heights']))
                case "MFV":
                    RMS = np.sqrt(np.mean(value ** 2))  # Root mean square
                    peaks = find_peaks(value, height=RMS)
                    NP = len(peaks[1]['peak_heights']) # Number of peaks
                    riga_feature_estratte.append(NP / (N/(fs/1000)))
                case "DAMV":
                    riga_feature_estratte.append(((np.sum(np.absolute(value[1:] - value[:-1])))/(N)))
                case "IAV":
                    riga_feature_estratte.append(np.sum(np.absolute(value)))
                case "HFD":
                    riga_feature_estratte.append(hfda.measure(value, 5))
                case "SKEW":
                    riga_feature_estratte.append(skew(value))
                case "HMOB":
                    riga_feature_estratte.append(eeglib.features.hjorthMobility(value))
                case "HCOM":
                    riga_feature_estratte.append(eeglib.features.hjorthComplexity(value))
                case "K":
                    riga_feature_estratte.append(kurtosis(value))
                case "PER":
                    riga_feature_estratte.append(np.percentile(value, 75))
                case "SSC":
                    f = lambda x: (x >= 0).astype(float)
                    riga_feature_estratte.append(sum(f(np.diff(value, prepend=1)[1:-1] * -np.diff(value)[1:])))
                case "MFL":
                    riga_feature_estratte.append(np.log10(np.sqrt(np.sum((value[1:] - value[:-1])** 2))))
                case "DASDV":
                    riga_feature_estratte.append(np.sqrt((1/(value.size-1)) * (np.sum((value[1:] - value[:-1])** 2))))
                case "SSI":
                    riga_feature_estratte.append(np.sum(value**2))
                case "WAM":
                    SogliaWAM = 2 #Valore desunto dai paper
                    f = lambda x: (x > SogliaWAM).astype(float)
                    riga_feature_estratte.append(np.sum(f(np.absolute(value[:-1] - value[1:]))))
                case "MYOP":
                    SogliaMYOP = 0.0066 #Valore desunto dai paper (da ricontrollare) in mV
                    f = lambda x: (x > SogliaMYOP).astype(float)
                    riga_feature_estratte.append((np.sum(f(np.absolute(value))))/N)
                case "AAC":
                    riga_feature_estratte.append((np.sum(np.absolute(value[1:] - value[:-1])))/N)
                case "MAV1":
                    value_00_to_25 = value[0:int(N*0.25)] #il primo 25% dei valori
                    value_25_to_75 = value[int(N*0.25):int((N*0.75))] #dal 25esimo al 75esimo valore (ragionando su 100)
                    value_75_to_100 = value[int(N*0.75):] #dal 75esimo valore in poi
                    riga_feature_estratte.append(np.mean([
                                *(np.absolute(value_00_to_25) * 0.5), 
                                *(np.absolute(value_25_to_75) ), 
                                *(np.absolute(value_75_to_100) * 0.5)]))
                case "MAV2":
                    value_00_to_25 = value[0:int(N*0.25)] #il primo 25% dei valori
                    value_25_to_75 = value[int(N*0.25):int((N*0.75))] #dal 25esimo al 75esimo valore (ragionando su 100)
                    value_75_to_100 = value[int(N*0.75):] #dal 75esimo valore in poi
                    riga_feature_estratte.append(np.mean([
                                *(np.absolute(value_00_to_25)  * ((4 * np.array(range(1, int((N)*0.25)+1)))/N)), #coefficiente = 4i/N
                                *(np.absolute(value_25_to_75)), #coefficiente = 1
                                *(np.absolute(value_75_to_100) * ( 4 * (np.array(range(int(N*0.75), int(N))) - N)) / N) #coefficiente = (4(i-N))/N
                                ]))
                case "TM3":
                    riga_feature_estratte.append(np.absolute(np.mean(value**3)))
                case "TM4":
                    riga_feature_estratte.append(np.mean(value**4))
                case "TM5":
                    riga_feature_estratte.append(np.absolute(np.mean(value**5)))
                case "VORDER":
                    VOrderVal = 2 #valore trvato dai paper (che lo rende uguale a VAR)
                    riga_feature_estratte.append((np.mean((value**VOrderVal)))**VOrderVal)
                case "HIST1":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram 
                    riga_feature_estratte.append(HIST[0])
                case "HIST2":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[1])
                case "HIST3":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[2])
                case "HIST4":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[3])
                case "HIST5":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[4])
                case "HIST6":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[5])
                case "HIST7":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[6])
                case "HIST8":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[7])
                case "HIST9":
                    bins = 9 #Numero di "cestini" Scelto 9 dai paper
                    HIST = np.histogram(value, bins=bins)[0] #Histogram
                    riga_feature_estratte.append(HIST[8])

        #add classe
        classe = list(subrepetitions_classes.iloc[count])[0]
        riga_feature_estratte.append(classe)
        #add idx
        idx = list(subrepetitions_idx.iloc[count])[0]
        riga_feature_estratte.append(idx)
        
    
        df_feature_vectors = df_feature_vectors.append(pd.Series(riga_feature_estratte, index=df_feature_vectors.columns[:len(riga_feature_estratte)]), ignore_index=True)

        count = count + 1

        #update progress bar
        pbar.n = count
        pbar.refresh()

    # print(df_feature_vectors.shape)
    # print(df_feature_vectors)
    return df_feature_vectors, std_timer