# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:16:29 2022

@author: Marc
"""

import matplotlib.pyplot as plt
from keras.models import  Model, load_model
from keras.layers import Dense, LSTM, Bidirectional,Concatenate,Input,concatenate
import numpy as np
from keras.utils import Sequence
import time
import utils_magnitude
from tensorflow.python.keras import backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping
import os
from keras.optimizers import Adam
from features_extraction import function_features_extration
import json

subset = ['test']

function_features_extration(subset)

start_time = time.time()

tipo_de_escalamiento = 'MinMax' #'Standard', 'MinMax', 'MVN', 'None' #Normalization, must be the same than in training

#Path to features and labels
path_root =  os.getcwd()
path_feat =  "../../data/features/"
path_feat_in_test_temporal = path_feat + 'feat_temporal_test_magnitude.npy'
path_feat_in_test_global = path_feat + 'feat_global_test_magnitude.npy'

#Path where the model was saved
path_salida_modelo = '../../models/'

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
        

dictParameters = open(path_salida_modelo+'normalization_parameters_magnitude_original.json', "r", encoding='utf-8')
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



tam_feat_temporal = np.shape(feat_norm_test_temporal[0])[1]
tam_feat_global = np.shape(feat_norm_test_global)[1]

X_test_temporal, y_test = feat_norm_test_temporal, np.zeros(len(feat_norm_test_temporal))
X_test_global = feat_norm_test_global
x_test = MyBatchGenerator( X_test_temporal,X_test_global, np.zeros(len(y_test)), batch_size=1, shuffle=False)

model = load_model(path_salida_modelo +'/DNN_magnitude_original.h5')


y_estimada_test = np.hstack(model.predict(x_test))                  
    


