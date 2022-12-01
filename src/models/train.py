# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:51:20 2021

@author: Marc
"""

import matplotlib.pyplot as plt
from keras.models import  Model
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

subset = ['train', 'val']

function_features_extration(subset)

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)



start_time = time.time()
tipo_de_escalamiento = 'MinMax' #'Standard', 'MinMax', 'MVN', 'None' #Normalization

#Path to features and labels
path_feat =  "../../data/features/"
path_feat_in_train_temporal = path_feat + 'feat_temporal_train_magnitude.npy'
path_feat_in_train_global = path_feat + 'feat_global_train_magnitude.npy'
path_mag_real_train = path_feat + 'magnitude_train.npy'
path_feat_in_val_temporal = path_feat + 'feat_temporal_val_magnitude.npy'
path_feat_in_val_global = path_feat + 'feat_global_val_magnitude.npy'
path_mag_real_val =  path_feat + 'magnitude_val.npy'

#Path where the model is saved
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




feat_in_train_temporal = np.load(path_feat_in_train_temporal, allow_pickle=True)
feat_in_train_global = np.load(path_feat_in_train_global, allow_pickle=True)
mag_real_train = np.load(path_mag_real_train)
id_train = mag_real_train[:,0]
mag_real_train = np.array([mag_real_train[i,1] for i in range(len(mag_real_train))],dtype=float)

feat_in_val_temporal = np.load(path_feat_in_val_temporal, allow_pickle=True)
feat_in_val_global = np.load(path_feat_in_val_global, allow_pickle=True)
mag_real_val = np.load(path_mag_real_val)
id_val = mag_real_val[:,0]
mag_real_val = np.array([mag_real_val[i,1] for i in range(len(mag_real_val))],dtype=float)


normalization_parameters = {}
# =============================================================================
# Se Normaliza los features: 'MVN' normaliza los features de acuerdo a los features del evento en cuestion
#'Standard' y 'MinMax' normaliza los features de acuerdo a promedios y desv estandar de features de acuerdo a la base de entrenamiento
# =============================================================================
if tipo_de_escalamiento == 'Standard': #The features are normalized using Z-Score over all the data base
    print('Se normalizan los features utilizando', tipo_de_escalamiento)

    pesos = [x.shape[0] for x in feat_in_train_temporal]
    mean_over_feat_train_temporal = np.average([np.mean(x,axis=0) for x in feat_in_train_temporal],axis=0, weights=pesos)
    std_over_feat_train_temporal = np.sqrt(np.average([np.square(np.mean(x,axis=0)-mean_over_feat_train_temporal) for x in feat_in_train_temporal],axis=0,weights=pesos))
    
    feat_norm_train_temporal = np.array([ (feat_in_train_temporal[i]-mean_over_feat_train_temporal)/std_over_feat_train_temporal 
                                for i in range(len(feat_in_train_temporal))],dtype=object)
    feat_norm_val_temporal = np.array([ (feat_in_val_temporal[i]-mean_over_feat_train_temporal)/std_over_feat_train_temporal 
                                for i in range(len(feat_in_val_temporal))],dtype=object)

    
    mean_over_feat_train_global = np.mean(feat_in_train_global,0) 
    std_over_feat_train_global = np.std(feat_in_train_global,0) 
    
    feat_norm_train_global = np.array([ (feat_in_train_global[i]-mean_over_feat_train_global)/std_over_feat_train_global 
                                for i in range(len(feat_in_train_global))],dtype=object)
    feat_norm_val_global = np.array([ (feat_in_val_global[i]-mean_over_feat_train_global)/std_over_feat_train_global 
                                for i in range(len(feat_in_val_global))],dtype=object)

    normalization_parameters['mean_temporal'] = list(mean_over_feat_train_temporal)
    normalization_parameters['std_temporal'] = list(std_over_feat_train_temporal)
    normalization_parameters['mean_global'] = list(mean_over_feat_train_global)
    normalization_parameters['std_global'] = list(std_over_feat_train_global)
    
elif tipo_de_escalamiento == 'MinMax': #Features are translated into a range between 0 and 1
    print('Se normalizan los features utilizando', tipo_de_escalamiento)
    min_f_train_temporal = np.min([np.min(x,0) for x in feat_in_train_temporal],0)
    max_f_train_temporal =  np.max([np.max(x,0) for x in feat_in_train_temporal],0)
    feat_norm_train_temporal = np.array([(feat_in_train_temporal[i]-min_f_train_temporal)/(max_f_train_temporal-min_f_train_temporal)
                                for i in range(len(feat_in_train_temporal))],dtype=object)
    feat_norm_val_temporal = np.array([(feat_in_val_temporal[i]-min_f_train_temporal)/(max_f_train_temporal-min_f_train_temporal)
                                for i in range(len(feat_in_val_temporal))],dtype=object)


    min_f_train_global = np.min(feat_in_train_global,0)
    max_f_train_global =  np.max(feat_in_train_global,0)
    
    feat_norm_train_global = np.array([(feat_in_train_global[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                                for i in range(len(feat_in_train_global))],dtype=object)
    feat_norm_val_global = np.array([(feat_in_val_global[i]-min_f_train_global)/(max_f_train_global-min_f_train_global)
                                for i in range(len(feat_in_val_global))],dtype=object)

    normalization_parameters['min_temporal'] = list(min_f_train_temporal)
    normalization_parameters['max_temporal'] = list(max_f_train_temporal)
    normalization_parameters['min_global'] = list(min_f_train_global)
    normalization_parameters['max_global'] = list(max_f_train_global)


elif tipo_de_escalamiento == 'MVN':
    print('Se normalizan los features utilizando', tipo_de_escalamiento)
    feat_norm_train_temporal = np.array([utils_magnitude.cmvn(feat_in_train_temporal[i]) for i in range(len(feat_in_train_temporal))],dtype=object)
    feat_norm_val_temporal = np.array([utils_magnitude.cmvn(feat_in_val_temporal[i]) for i in range(len(feat_in_val_temporal))],dtype=object)

    feat_norm_train_global = feat_in_train_global
    feat_norm_val_global = feat_in_val_global
    

    
elif tipo_de_escalamiento == 'None':
    print('No se normalizan los features')
    feat_norm_train_temporal = np.array([feat_in_train_temporal[i] for i in range(len(feat_in_train_temporal))],dtype=object)
    feat_norm_val_temporal = np.array([feat_in_val_temporal[i] for i in range(len(feat_in_val_temporal))],dtype=object)

    feat_norm_train_global = np.array([feat_in_train_global[i] for i in range(len(feat_in_train_global))],dtype=object)
    feat_norm_val_global = np.array([feat_in_val_global[i] for i in range(len(feat_in_val_global))],dtype=object)



tam_feat_temporal = np.shape(feat_norm_train_temporal[0])[1]
tam_feat_global = np.shape(feat_norm_train_global)[1]

X_train_temporal, y_train = feat_norm_train_temporal, mag_real_train  
X_train_global = feat_norm_train_global




def fit_model(): 
    #input global features
    input_global = Input(shape=(tam_feat_global,))
    #input temporal features
    temporal_input = Input(shape=(None, tam_feat_temporal))
    hidden1 = Bidirectional(LSTM(10 ,kernel_regularizer=tf.keras.regularizers.l2(0.01)),merge_mode = 'concat')(temporal_input)
    concat = Concatenate()([hidden1, input_global]) #Concatenate the output of the LSTM with the global features
    mlp_hidden = Dense(30, activation= 'relu')(concat)
    mlp_out = Dense(1)(mlp_hidden) #output, magnitude
    
    model = Model(inputs=[temporal_input, input_global],outputs=mlp_out) 
    
    opt = Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=opt) 
    
    X_train_temporal, X_train_global, y_train = feat_norm_train_temporal,feat_norm_train_global, mag_real_train  
    X_val_temporal, X_val_global, y_val = feat_norm_val_temporal,feat_norm_val_global, mag_real_val  
    train_feat_target= MyBatchGenerator( X_train_temporal,X_train_global, y_train, batch_size=1, shuffle=True)
    val_feat_target= MyBatchGenerator( X_val_temporal,X_val_global, y_val, batch_size=1, shuffle=True)
    model.summary()
    
    
    
    
    es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=10,
                           restore_best_weights=True) 
    hist = model.fit(train_feat_target, validation_data=val_feat_target, epochs=200, callbacks=[es])
    
         	
    loss_train=hist.history['loss']
    loss_val=hist.history['val_loss']
       
    if True: #Save the model
        model.save(os.path.join(path_salida_modelo +'/DNN_magnitude.h5').replace('\\', '/'))
        #np.save(path_salida_modelo+'normalization_parameters_magnitude.npu', [min_f_train_temporal,max_f_train_temporal,min_f_train_global,max_f_train_global])
        with open(path_salida_modelo+'normalization_parameters_magnitude.json', 'w') as fp:
            json.dump(normalization_parameters, fp,indent=4)    
            
    
    #Calculo metricas MSE y error relativo en conjunto de entrenamiento
    x_train = MyBatchGenerator( X_train_temporal,X_train_global, np.zeros(len(y_train)), batch_size=1, shuffle=False)        
    y_estimada_train = np.hstack(model.predict(x_train))                  
    error_mse_train = np.mean(np.square(np.abs(y_train - y_estimada_train)))
    error_rel_train = np.mean(100*np.abs(y_train - y_estimada_train)/y_train)
 
    ind_menor_4 = np.where(y_train<4)[0]
    ind_mayor_4 = np.where(y_train>=4)[0]
    error_rel_train_menor_4  = np.mean(100*np.abs(y_train[ind_menor_4] - y_estimada_train[ind_menor_4])/y_train[ind_menor_4])
    error_rel_train_mayor_4  = np.mean(100*np.abs(y_train[ind_mayor_4] - y_estimada_train[ind_mayor_4])/y_train[ind_mayor_4])
    
    
    #Calculo metricas MSE y error relativo en conjunto de validacion
    x_val = MyBatchGenerator( X_val_temporal,X_val_global, np.zeros(len(y_val)), batch_size=1, shuffle=False)
    y_estimada_val = np.hstack(model.predict(x_val))                  
    error_mse_val = np.mean(np.square(np.abs(y_val - y_estimada_val)))
    error_rel_val = np.mean(100*np.abs(y_val - y_estimada_val)/y_val)

    ind_menor_4 = np.where(y_val<4)[0]
    ind_mayor_4 = np.where(y_val>=4)[0]
    error_rel_val_menor_4  = np.mean(100*np.abs(y_val[ind_menor_4] - y_estimada_val[ind_menor_4])/y_val[ind_menor_4])
    error_rel_val_mayor_4  = np.mean(100*np.abs(y_val[ind_mayor_4] - y_estimada_val[ind_mayor_4])/y_val[ind_mayor_4])

     
    #Calculo metricas MSE y error relativo en conjunto de prueba
  
    error = {'Error_mse_train': error_mse_train, 'Error_rel_train':error_rel_train,
             'Error_mse_val': error_mse_val, 'Error_rel_val':error_rel_val,
             'Error_rel_train_<4':error_rel_train_menor_4,'Error_rel_train_>=4':error_rel_train_mayor_4,
             'Error_rel_val_<4':error_rel_val_menor_4,'Error_rel_val_>=4':error_rel_val_mayor_4}
    
    
    
    loss = {'loss_train': loss_train, 'loss_val':loss_val}
    y_estimacion = {'Estimacion_train':y_estimada_train,'Estimacion_val':y_estimada_val}
    

    return y_estimacion, error, loss

y_estimada_repeat_train, error_repeat_train_mse,error_repeat_train_rel, loss_repeat_train= [], [], [], [] 
y_estimada_repeat_val, error_repeat_val_mse, error_repeat_val_rel, loss_repeat_val= [], [], [], [] 
error_repeat_train_rel_menor4, error_repeat_val_rel_menor4 = [], []
error_repeat_train_rel_mayor4, error_repeat_val_rel_mayor4= [], []


y_estimation, error_metrics, loss_final  = fit_model()
   
print('MSE train: {}'.format(np.round(error_metrics['Error_mse_train'],2)))
print('MSE val: {}'.format(np.round(error_metrics['Error_mse_val'],2)))

print('%Error train: {}'.format(np.round(error_metrics['Error_rel_train'],2)))
print('%Error val: {}'.format(np.round(error_metrics['Error_rel_val'],2)))





if True: #Se guardan los resultados
    import pandas as pd
    df_train = pd.DataFrame(data= np.transpose([id_train,mag_real_train,y_estimation['Estimacion_train']]),
                            columns=['id_event','Real','Estimation'])
    df_val = pd.DataFrame(data= np.transpose([id_val, mag_real_val,y_estimation['Estimacion_val']]),
                            columns=['id_event','Real','Estimation'])


    df_train.to_csv('results/train_results_magnitude.csv',index=False)
    df_val.to_csv('results/val_results_magnitude.csv',index=False)

    df_loss_train = pd.DataFrame(data=loss_final['loss_train'], columns = ['loss_train'])
    df_loss_val = pd.DataFrame(data=loss_final['loss_val'], columns = ['loss_val'])

    df_loss_train.to_csv('results/train_loss_magnitude.csv',index=False)
    df_loss_val.to_csv('results/val_loss_magnitude.csv',index=False)


print("--- %s seconds ---" % int((time.time() - start_time)))
