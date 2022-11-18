# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:40:22 2022

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
import utils_magnitude


def function_features_extration(subset):    
    start_time = time.time()
    path_root =  os.getcwd()
    
    conjuntos = subset #sets to use
    path_carpeta_sac = '../../data/sac/' #Path to the database (.sac)    
    feat_out = '../../data/features/' #Path where the features will be generated
    for conjunto in conjuntos:
        
        print('Reading the set of', conjunto)
        nro_canales = 3  #Number of channel to use, 1 or 3, if is set to 1, it will take the Z channel.
        umbral_corte = 0.03 # Threshold, based on the decay of the signal #0.1 is 10% # if is setted in 1 preserves the entire trace
        frame_length = 4 #size in seconds of the frame length
        frame_shift = 2  #frame shift
        Energy = 'E3' # There are different types of energy used 
        Vel = True #True: the traces are converted to velocity, False the traces are kept in count
        escala_features_fft = 'logaritmica' # 'lineal' o 'logaritmica' 
        escala_features_energia = 'logaritmica'
        escala_signal = 1e+10 #Scale to amplify the signal, especially useful when the signal is in velocity
        Estacion_distancia = 'Cercana'  #'Cercana', 'Lejana'/ Select if you want to use the close or the far away earthquakes
        Base = 'Toda' #'Menor_4', 'Mayor_4', 'Toda' # Data base to use; eartquakes<4: 'Menor_4', earthquakes>=4 :'Mayor_4', or whole range of magnitudes: 'Toda'
        
        ##################
        version = 'original'  #'original', 'fourier_completa' o 'raw'
        ##################
        
        if Estacion_distancia == 'Cercana':
            if Base == 'Mayor_4': path_carpeta_conjunto = os.path.join('../../data/set',conjunto +'_250eventos_magnitud_magnitud_mayor4_cerca.csv').replace('\\', '/');       
            elif Base == 'Menor_4': path_carpeta_conjunto = os.path.join('../../data/set',conjunto +'_250eventos_magnitud_magnitud_menor4_cerca.csv').replace('\\', '/');       
            elif Base == 'Toda': path_carpeta_conjunto = os.path.join('../../data/set',conjunto +'_500eventos_magnitud_magnitud_todo_cerca.csv').replace('\\', '/');       
        elif Estacion_distancia == 'Lejana':
            if Base == 'Mayor_4': path_carpeta_conjunto = os.path.join('../../data/set',conjunto +'_250eventos_magnitud_magnitud_mayor4_lejos.csv').replace('\\', '/');       
            elif Base == 'Menor_4': path_carpeta_conjunto = os.path.join('../../data/set',conjunto +'_250eventos_magnitud_magnitud_menor4_lejos.csv').replace('\\', '/');       
            elif Base == 'Toda': path_carpeta_conjunto = os.path.join('../../data/set',conjunto +'_500eventos_magnitud_magnitud_todo_lejos.csv').replace('\\', '/');            
        
        fs = 40
        lista_conjunto = pd.read_csv(path_carpeta_conjunto,delimiter=",", index_col=False)
        lista_eventos = lista_conjunto['Evento'].values
        
        
        feat_por_evento_temporal, feat_por_evento_global = [], []
        magnitud_por_evento, locacion_por_evento = [], {'Evento':[],'Latitud':[],'Longitud':[],'Profundidad':[],'Distancia':[]}
        estaciones = lista_conjunto['Estacion'].values
        if nro_canales ==3:
            canales = {'PB09':['HHZ','HHE','HHN'],'PB06':['HHZ','HHE','HHN'],'AC02':['HHZ','HHE','HHN'],'PB18':['HHZ','HHE','HHN'],
                       'CO02':['HHZ','HHE','HHN'],'PB14':['HHZ','HHE','HHN'],'CO01':['HHZ','HHE','HHN'],
                       'GO01':['BHZ','BHE','BHN'],'GO03':['BHZ','BHE','BHN'],'MT16':['HHZ','HHE','HHN'] } #siempre debe ir el canal Z, ya que de ahi se saca magnitud y distancia
        elif nro_canales==1:
            canales = {'PB09':['HHZ'],'PB06':['HHZ'],'AC02':['HHZ'],'MT16':['HHZ'],
                       'CO02':['HHZ'],'PB14':['HHZ'],'CO01':['HHZ'],'PB18':['HHZ'],
                       'GO01':['BHZ'],'GO03':['BHZ']} 
        
        #Coordenates: Lat, Lon, Height
        Coordenadas_estaciones ={'PB09':[-21.796, -69.242, 1.530],'PB06':[-22.706, -69.572, 1.440],'AC02':[-26.836,-69.129,3.980],
                       'CO02':[-31.204, -71.000, 1.190],'PB14':[-24.626, -70.404, 0.2630],'CO01':[-29.977,-70.094,2.157],
                       'GO01':[-19.669,-69.194,3.809],'GO03':[-27.594,-70.235,0.730],'PB18':[-17.590,-69.480, 4.155],
                       'MT16':[-33.428,-70.523,0.780]} 
        id_estacion = {'PB09':1,'PB06':2,'AC02':3,'CO02':4,'PB14':5,'CO01':6,'GO01':7,'GO03':8, 'MT16':9, 'PB18':10}
            
        
        contador_inexistencia_estacion, contador_tamano_canales = 0,0
        ex_e=0
        ex_ne = []
        tiempo_traza = []
        for i in range(len(lista_conjunto)): 
            
            print('Reading earthquake: ', lista_eventos[i])  
            path_carpeta_evento = path_carpeta_sac + lista_eventos[i]
            feat_estaciones = []        
            path_evento = path_carpeta_evento+'/'+estaciones[i]+'/'
            features_canales_temporal,features_canales_global = [], []        
            for k in range(len(canales[estaciones[i]])): #for sobre los canales
    
                sac_k = read(path_evento+estaciones[i]+'_'+ canales[estaciones[i]][k]+'.sac')
                sac_k[0].data = sac_k[0].data*escala_signal
                estoy_en_Z = 0

    
    
    
                if sac_k[0].stats.channel[-1] =='Z':
                    
                    if conjunto == 'test': 
                        estoy_en_Z = 1                        
                    elif conjunto == 'train' or conjunto == 'val':
                        magnitud_por_evento.append([lista_eventos[i],float(sac_k[0].stats.sac.mag)])                    
                        lat,lon, depth,dist = sac_k[0].stats.sac.evla,sac_k[0].stats.sac.evlo,sac_k[0].stats.sac.evdp, sac_k[0].stats.sac.dist              
                        locacion_por_evento['Latitud'].append(lat)
                        locacion_por_evento['Longitud'].append(lon)
                        locacion_por_evento['Profundidad'].append(depth) 
                        locacion_por_evento['Distancia'].append(dist) 
                        locacion_por_evento['Evento'].append(lista_eventos[i]) 
                        estoy_en_Z = 1
    
               
                    
                fs = int(sac_k[0].stats.sampling_rate)
                fs_real = int(sac_k[0].stats.sampling_rate)
                if fs ==100:
                    sac_k[0].data = sp.signal.resample(sac_k[0].data,int(len(sac_k[0].data)*40/100))
                    sac_k[0].stats.sampling_rate = 40
                    sac_k = sac_k.slice(sac_k[0].stats.starttime+1, sac_k[0].stats.endtime-1)
                    fs = 40
                elif fs==40:
                    sac_k = sac_k.slice(sac_k[0].stats.starttime+1, sac_k[0].stats.endtime-1)
                elif fs!=40:
                    print('Hay sampling rate distinto a 40, revisar!!')
                tiempo_traza.append(sac_k[0].stats.endtime-sac_k[0].stats.starttime)
                frame_len = frame_length*fs
                frame_shi = frame_shift*fs                
                nfft = pow(2,math.ceil(np.log2(frame_len)))
              
                if Vel == True: # Signal are converted to velocity
                    sta = sac_k[0].stats.station
                    cha = sac_k[0].stats.channel
                    inv = read_inventory('../../data/xml/'+sta+'.xml')
                    sac_k.remove_response(inventory=inv, output="VEL")
    
    
    
                data_k = utils_magnitude.butter_highpass_lfilter(sac_k[0].data, cutoff=1, fs=fs, order=3)   
        
        
                if estoy_en_Z:
                    sac_filt_s= utils_magnitude.butter_bandpass_lfilter(sac_k[0].data, lowcut=1, highcut= 2, fs=fs, order=3)
    
    
                    cft = classic_sta_lta(sac_filt_s, int(9 * fs), int(32 * fs))
                    try:
                        frame_p =trigger_onset(cft, 1.8, 0.5)[0][0]
                    except:
                        #print('fallo sta/lta y por lo que se fijo que la P se ubica al inicio de la traza')
                        frame_p = 0
    
                    frame_s = np.argmax(sac_filt_s)
    
                    time_sp = (frame_s-frame_p)/fs
    
    
                    if Energy == 'E1':
                        Energia_Z_ref = utils_magnitude.E1(data_k, frame_len, frame_shi, nfft,escala = 'lineal')
                    elif Energy == 'E2':
                        Energia_Z_ref = utils_magnitude.E2(data_k, frame_len, frame_shi,escala = 'lineal')
                    elif Energy == 'E3':
                        Energia_Z_ref = utils_magnitude.E3(data_k, frame_len,frame_shi,escala = 'lineal')
                
                        
                    if umbral_corte ==1:
                        muestra_corte_coda = len(data_k)
                        energia_umbral_corte = np.max(Energia_Z_ref)
                        data_k_or = data_k
                    else:
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
      
                if sac_k[0].stats.channel[-1] =='Z':

                    #Global features selection
                    #feat_distancia_evento = [lat,lon,depth,dist]
                    #features_canales_global.append(np.hstack(([Coordenadas_estaciones[estaciones[i]], id_estacion[estaciones[i]],time_sp, features_distancia_evento])))
                    features_canales_global.append(np.hstack(([id_estacion[estaciones[i]]])))
                    
                    feat_canales_global = np.hstack(features_canales_global)
                
                        
                        
    
            
            
        
            
            
            
    
            feat_por_evento_temporal.append(feat_canales_temporal)
            feat_por_evento_global.append(feat_canales_global)
                    
                
            print('Shape temporal features: {:d}x{:d}'.format(feat_canales_temporal.shape[0],feat_canales_temporal.shape[1]))  
            print('Shape global features: {:d}'.format(feat_canales_global.shape[0]))  
            print('-----------------------')
            
         
            
        print('***********************')
        print('Set:',conjunto)
        print('{} out of a total of {} traces were read'.format(len(feat_por_evento_temporal),len(lista_conjunto)))    
            
        ind_random = np.random.permutation(np.arange(0,len(feat_por_evento_temporal)))
        feat_por_evento_temporal = np.array(feat_por_evento_temporal, dtype=object)[ind_random]
        feat_por_evento_global = np.array(feat_por_evento_global, dtype=float)[ind_random]
        if conjunto == 'train' or conjunto =='val':
            magnitud_por_evento = np.array(magnitud_por_evento)[ind_random]
            locacion_por_evento['Distancia']= list(np.array(locacion_por_evento['Distancia'])[ind_random])
            locacion_por_evento['Longitud']= list(np.array(locacion_por_evento['Longitud'])[ind_random])
            locacion_por_evento['Latitud']= list(np.array(locacion_por_evento['Latitud'])[ind_random])
            locacion_por_evento['Profundidad']= list(np.array(locacion_por_evento['Profundidad'])[ind_random])
            locacion_por_evento['Evento']= list(np.array(locacion_por_evento['Evento'])[ind_random])
            
        
                
        if not os.path.isdir(feat_out):
            print('The directory is created ', feat_out)
            os.mkdir(feat_out)
        
        if True:
    
            print('The features were saved')
            np.save(feat_out+'feat_temporal_'+conjunto+ '_magnitude.npy', feat_por_evento_temporal)
            np.save(feat_out+'feat_global_'+conjunto+ '_magnitude.npy', feat_por_evento_global)
            
            if conjunto == 'train' or conjunto =='val':
                print('The target were saved')
                np.save(feat_out+'magnitude_'+conjunto +'.npy', magnitud_por_evento)                   
        else:
            print('The features were not saved')
        
    print("--- %s seconds ---" % int((time.time() - start_time)))

