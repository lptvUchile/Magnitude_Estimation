# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:28:08 2022

@author: Marc
"""
import numpy.matlib
import os
import numpy as np
from obspy import read, read_inventory
import math
import time
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from obspy.signal.trigger import classic_sta_lta,trigger_onset
import sys
from keras.models import  Model, load_model
from keras.layers import Dense, LSTM, Bidirectional,Concatenate,Input,concatenate
from keras.utils import Sequence
import utils_magnitude
from tensorflow.python.keras import backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import json

def function_features_extration(sac_conc): 
    sac_k = sac_conc.copy()
    
    ##Los siguientes parametros no deberian modificarse, ya que fueron los con que se entreno la red
    
    nro_canales = 3  #Number of channel to use, 1 or 3, if is set to 1, it will take the Z channel.
    umbral_corte = 0.03 # Threshold, based on the decay of the signal #0.1 is 10% # if is setted in 1 preserves the entire trace
    frame_length = 4 #size in seconds of the frame length
    frame_shift = 2  #frame shift
    Energy = 'E3' # There are different types of energy used 
    Vel = True #True: the traces are converted to velocity, False the traces are kept in count
    escala_features_fft = 'logaritmica' # 'lineal' o 'logaritmica' 
    escala_features_energia = 'logaritmica'
    escala_signal = 1e+10 #Scale to amplify the signal, especially useful when the signal is in velocity
    version = 'original'  #'original', 'fourier_completa' o 'raw'

    id_estacion = {'PB09':1,'PB06':2,'AC02':3,'CO02':4,'PB14':5,'CO01':6,'GO01':7,'GO03':8, 'MT16':9, 'PB18':10,
                   'CO10':11}
    features_canales_temporal,features_canales_global = [], []        
    feat_por_evento_temporal, feat_por_evento_global = [], []
    for ch in range(len(sac_k)):
        sac_k[ch].data = sac_k[ch].data*escala_signal
        estoy_en_Z = 0
        
        sta = sac_k[ch].stats.channel
        
        
        fs = int(sac_k[ch].stats.sampling_rate)
        fs_real = int(sac_k[ch].stats.sampling_rate)
        if fs ==100:
            sac_k[ch].data = sp.signal.resample(sac_k[ch].data,int(len(sac_k[ch].data)*40/100))
            sac_k[ch].stats.sampling_rate = 40
            sac_k[ch] = sac_k[ch].slice(sac_k[ch].stats.starttime+1, sac_k[ch].stats.endtime-1)
            fs = 40
        elif fs==40:
            sac_k[ch] = sac_k[ch].slice(sac_k[ch].stats.starttime+1, sac_k[ch].stats.endtime-1)
        elif fs!=40:
            print('Hay sampling rate distinto a 40, revisar!!')
        
        frame_len = frame_length*fs
        frame_shi = frame_shift*fs                
        nfft = pow(2,math.ceil(np.log2(frame_len)))
      
        if Vel == True: # Signal are converted to velocity
                
            sta = sac_k[ch].stats.station
            cha = sac_k[ch].stats.channel

            inv = read_inventory('../../data/xml/'+sta+'.xml')
            sac_k[ch].remove_response(inventory=inv, output="VEL")
    
        data_k = utils_magnitude.butter_highpass_lfilter(sac_k[ch].data, cutoff=1, fs=fs, order=3)   
    
    
        if sac_k[ch].stats.channel[-1]=='Z':
           
    
    
            if Energy == 'E1':
                Energia_Z_ref = utils_magnitude.E1(data_k, frame_len, frame_shi, nfft,escala = 'lineal')
            elif Energy == 'E2':
                Energia_Z_ref = utils_magnitude.E2(data_k, frame_len, frame_shi,escala = 'lineal')
            elif Energy == 'E3':
                Energia_Z_ref = utils_magnitude.E3(data_k, frame_len,frame_shi,escala = 'lineal')
        
    
            arg_amp_maxima = np.argmax(Energia_Z_ref) #Assumption: The maximun energy is in S-wave
            arg_amp_minima = np.argmin(Energia_Z_ref[:arg_amp_maxima]) # take the minimum energy between the start of the signal and the S-wave
            delta_energia = Energia_Z_ref[arg_amp_maxima]-Energia_Z_ref[arg_amp_minima] 
            energia_umbral_corte = delta_energia*umbral_corte+Energia_Z_ref[arg_amp_minima] #energy threshold
    
            arg_fin_nueva_coda = arg_amp_maxima + np.argmin(np.abs(Energia_Z_ref[arg_amp_maxima:]-energia_umbral_corte))                      
            muestra_corte_coda = int(fs*frame_len*arg_fin_nueva_coda/frame_shi)      
            data_k_or = data_k
    
        data_k = data_k[:muestra_corte_coda] #The signal is cut       
    
        feat_k = utils_magnitude.parametrizador(data_k, frame_len, frame_shi,nfft, escala = escala_features_fft)
    
        #The signal is windowed
        feat_t = utils_magnitude.get_frames(data_k,frame_len, frame_shi)      
        #FFT
        feat_fourier_completa = np.fft.fft(feat_t, nfft, axis=1)  
        #Real and imaginary part are concatenated
        feat_fourier_completa = np.hstack((feat_fourier_completa.real[:,:feat_fourier_completa.shape[1]//2 +1],
                   feat_fourier_completa.imag[:,:feat_fourier_completa.shape[1]//2 +1]))   
        feat_fourier_completa = np.delete(feat_fourier_completa,[129,257],1)
        #Energy
        if Energy == 'E1':
            feat_Energy = utils_magnitude.E1(data_k, frame_len, frame_shi, nfft,escala = escala_features_energia)
            feat_k = np.hstack((feat_k, np.array([feat_Energy]).T ))
        elif Energy == 'E2':
            feat_Energy = utils_magnitude.E2(data_k, frame_len, frame_shi,escala = escala_features_energia)
            feat_k = np.hstack((feat_k, np.array([feat_Energy]).T ))
        elif Energy == 'E3':
            feat_Energy = utils_magnitude.E3(data_k, frame_len,frame_shi,escala = escala_features_energia)
            feat_k = np.hstack((feat_k, np.array([feat_Energy]).T))         
            
    
        if version == 'original':
            features_canales_temporal.append(feat_k)
        elif version == 'fourier_completa':
            features_canales_temporal.append(feat_fourier_completa)
        elif version == 'raw':
            features_canales_temporal.append(feat_t)
     
        feat_canales_temporal = np.hstack(features_canales_temporal)   
      
        if sac_k[ch].stats.channel[-1]=='Z':
            ##Global features selection
            #feat_distancia_evento = [lat,lon,depth,dist]
            #features_canales_global.append(np.hstack(([Coordenadas_estaciones[estaciones[i]], id_estacion[estaciones[i]],time_sp, features_distancia_evento])))
            features_canales_global.append(np.hstack(([id_estacion[sta]])))
            
            feat_canales_global = np.hstack(features_canales_global)



        feat_por_evento_temporal.append(feat_canales_temporal)
                
    #print('Shape temporal features: {:d}x{:d}'.format(feat_canales_temporal.shape[0],feat_canales_temporal.shape[1]))  
    #print('Shape global features: {:d}'.format(feat_canales_global.shape[0]))  

    if True:
        feat_out = '../../data/features/' #Path where the features will be generated
       
        if not os.path.isdir(feat_out):
            print('The directory is created ', feat_out)
            os.mkdir(feat_out)
        
        print('The features were saved')
        np.save(feat_out+'feat_temporal_1trace_magnitude.npy', feat_canales_temporal)
        np.save(feat_out+'feat_global_1trace_magnitude.npy', feat_canales_global)    
    
def DNN_magnitude():

    
    tipo_de_escalamiento = 'MinMax' #'Standard', 'MinMax', 'MVN', 'None' #Normalization, must be the same than in training
    #Path to features and labels
    path_root =  os.getcwd()
    path_feat =  "../../data/features/"
    path_feat_in_test_temporal = path_feat + 'feat_temporal_1trace_magnitude.npy'
    path_feat_in_test_global = path_feat + 'feat_global_1trace_magnitude.npy'
    
    #Path where the model was saved
    #path_model = '../../models/DNN_magnitude.h5'
    #path_normalization = '../../models/normalization_parameters_magnitude.json'
    path_model = '../../models/DNN_magnitude_original.h5'
    path_normalization = '../../models/normalization_parameters_magnitude_original.json'

    class MyBatchGenerator(Sequence):
        'Generates data for Keras'
        def __init__(self, X, x, y, batch_size=1, shuffle=False):
            'Initialization'
            self.X = X
            self.x = x
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()
    
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.y)/self.batch_size))
    
        def __getitem__(self, index):
            return self.__data_generation(index)
    
        def on_epoch_end(self):
            'Shuffles indexes after each epoch'
            self.indexes = np.arange(len(self.y))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
    
        def __data_generation(self, index):
            Xb = np.empty((self.batch_size, *self.X[index].shape))
            xb = np.empty((self.batch_size, *self.x[index].shape))
            yb = np.empty((self.batch_size, *self.y[index].shape))
            # naively use the same sample over and over again
            for s in range(0, self.batch_size):
                Xb[s] = self.X[index]
                xb[s] = self.x[index]
                yb[s] = self.y[index]
    
            return [Xb, xb], yb
    
    
    
    
    
    feat_in_test_temporal = np.load(path_feat_in_test_temporal, allow_pickle=True)
    feat_in_test_global = np.load(path_feat_in_test_global, allow_pickle=True)
    
    #mag_real_test = np.load(path_mag_real_test)
    #id_test = mag_real_test[:,0]
    #mag_real_test = np.array([mag_real_test[i,1] for i in range(len(mag_real_test))],dtype=float)
            
    
    dictParameters = open(path_normalization, "r", encoding='utf-8')
    dictParameters = json.load(dictParameters)
    
    
    # =============================================================================
    # Se Normaliza los features: 'MVN' normaliza los features de acuerdo a los features del evento en cuestion
    #'Standard' y 'MinMax' normaliza los features de acuerdo a promedios y desv estandar de features de acuerdo a la base de entrenamiento
    # =============================================================================
    if tipo_de_escalamiento == 'Standard': #The features are normalized using Z-Score over all the data base
        print('Se normalizan los features utilizando', tipo_de_escalamiento)
        mean_over_feat_train_temporal = np.array(dictParameters['mean_temporal'])
        std_over_feat_train_temporal = np.array(dictParameters['std_temporal'])
        mean_over_feat_train_global = np.array(dictParameters['mean_global'])
        std_over_feat_train_global = np.array(dictParameters['std_global'])
    
        feat_norm_test_temporal = np.array([ (feat_in_test_temporal[i]-mean_over_feat_train_temporal)/std_over_feat_train_temporal 
                                    for i in range(len(feat_in_test_temporal))],dtype=object)
        
        feat_norm_test_global = np.array([ (feat_in_test_global[i]-mean_over_feat_train_global)/std_over_feat_train_global 
                                    for i in range(len(feat_in_test_global))],dtype=object)
        
        
    
    elif tipo_de_escalamiento == 'MinMax': #Features are translated into a range between 0 and 1
        print('Se normalizan los features utilizando', tipo_de_escalamiento)
     
        min_f_train_temporal = np.array(dictParameters['min_temporal'])
        max_f_train_temporal = np.array(dictParameters['max_temporal'])
        min_f_train_global = np.array(dictParameters['min_global'])
        max_f_train_global = np.array(dictParameters['max_global'])
        
        
        feat_norm_test_temporal = np.array([(feat_in_test_temporal[i]-min_f_train_temporal)/(max_f_train_temporal-min_f_train_temporal)
                                    for i in range(len(feat_in_test_temporal))],dtype=object)
    
        feat_norm_test_global = np.array([(feat_in_test_global[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                                    for i in range(len(feat_in_test_global))],dtype=object)
    
    elif tipo_de_escalamiento == 'MVN':
        print('Se normalizan los features utilizando', tipo_de_escalamiento)
        feat_norm_test_temporal = np.array([utils_magnitude.cmvn(feat_in_test_temporal[i]) for i in range(len(feat_in_test_temporal))],dtype=object)
    
        feat_norm_test_global = feat_in_test_global
    
        
    elif tipo_de_escalamiento == 'None':
        print('No se normalizan los features')
        feat_norm_test_temporal = np.array([feat_in_test_temporal[i] for i in range(len(feat_in_test_temporal))],dtype=object)
        feat_norm_test_global = np.array([feat_in_test_global[i] for i in range(len(feat_in_test_global))],dtype=object)
    

    
    X_test_temporal, y_test = feat_norm_test_temporal, np.zeros(len(feat_norm_test_temporal))
    X_test_global = feat_norm_test_global
    x_test = MyBatchGenerator( [X_test_temporal],X_test_global, np.zeros(1), batch_size=1, shuffle=False)
    model = load_model(path_model)
    y_estimada_test = np.hstack(model.predict(x_test,verbose=0))               
          
    return y_estimada_test[0]  
          
def magnitude_estimation(sac_k):
    
    function_features_extration(sac_k)
    
    magnitude = DNN_magnitude()
    
    return magnitude



path_1sac = '../../data/sac/2014_01_04 00_11_50_M5_7/PB09/'


sac_conc = read(path_1sac+'PB09_HHZ.SAC')
sac_conc += read(path_1sac+'PB09_HHN.SAC')
sac_conc += read(path_1sac+'PB09_HHE.SAC')


magnitude_estimated = magnitude_estimation(sac_conc)

print('Magnitude: ', magnitude_estimated)