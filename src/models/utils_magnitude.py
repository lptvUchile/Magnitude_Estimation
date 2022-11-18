# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:09:46 2021

@author: Marc
"""
import numpy.matlib
import numpy as np
import re
from scipy import signal
import tensorflow as tf

def get_frames(signal, frame_length, frame_shift, window=None):
    if window is None:
        window=np.hamming(frame_length) 

    L = len(signal)
    N = int(np.fix((L-frame_length)/frame_shift + 1)) #number of frames

    Index = (np.matlib.repmat(np.arange(frame_length),N,1)+np.matlib.repmat(np.expand_dims(np.arange(N)*frame_shift,1),1,frame_length)).T
    hw=np.matlib.repmat(np.expand_dims(window,1),1,N)
    Seg=signal[Index]*hw

    return Seg.T

def parametrizador(senial, frame_length, frame_shift, nfft, window=None, escala='logaritmica'):
    #pdb.set_trace()
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y


def E1(senial, frame_length, frame_shift, nfft,escala='logaritmica'):
    y = get_frames(senial,frame_length,frame_shift)        
    Y = np.fft.fft(y, nfft, axis=1)
    if escala =='logaritmica':
        Y = np.sum(np.log10(np.abs(Y)**2),1) 
    elif escala == 'lineal':
        Y = np.sum(np.abs(Y)**2,1) 
    return Y

def E2(senial, frame_length,frame_shift,escala='logaritmica'):
    y = get_frames(senial,frame_length,frame_shift)        

    if escala =='logaritmica':
        Y = np.log10(np.sum(y**2,1))
    elif escala == 'lineal':
        Y = np.sum(y**2,1)
    return Y

def E3(senial, frame_length,frame_shift,escala='logaritmica'):
    Edos = E2(senial, frame_length,frame_shift,escala)
    if escala =='logaritmica':
        salida = Edos/np.max(Edos) #Edos-np.max(Edos)
    elif escala == 'lineal':
        salida = Edos-np.max(Edos)
    
    return salida


#Filtrar la señal
def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

#filtro no causal. 
def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data) #el orden final del filtro es del doble del filtro original
    return y

#filtro causal
def butter_highpass_lfilter(data, cutoff, fs, order=6):
    b, a = butter_highpass(cutoff, fs, order=order)
    #hago padding hacia la izquierda
    len_padding = 20 * order    #numero de muestras que se agregan
    data_ = np.pad(data, (len_padding, 0), 'symmetric', reflect_type='odd') 
    y = signal.lfilter(b, a, data_)
    y = y[len_padding:]   #elimino el padding
    return y



def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_lfilter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    len_padding = 20 * order    #numero de muestras que se agregan
    data_ = np.pad(data, (len_padding, 0), 'symmetric', reflect_type='odd') 
    y = signal.lfilter(b, a, data_)
    y = y[len_padding:]   #elimino el padding
    return y

#Filtrar la señal
def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#filtro no causal. 
def butter_lowpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data) #el orden final del filtro es del doble del filtro original
    return y

#filtro causal
def butter_lowpass_lfilter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #hago padding hacia la izquierda
    len_padding = 20 * order    #numero de muestras que se agregan
    data_ = np.pad(data, (len_padding, 0), 'symmetric', reflect_type='odd') 
    y = signal.lfilter(b, a, data_)
    y = y[len_padding:]   #elimino el padding
    return y

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = 2**-30
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output


def MinMax_utt(vec):
    
    minimo = np.min(vec,axis=0)
    maximo = np.max(vec,axis=0)
    
    
    output = (vec - minimo)/(maximo-minimo)
    return output
 
def get_angles(pos, i, d_model): # pos: (seq_length, 1) i: (1, d_model)
    angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
    return pos * angles # (seq_length, d_model)

def PositionalEncoding(inputs_layer):
    # input shape batch_size, seq_length, d_model
    seq_length = inputs_layer.shape.as_list()[-2]
    d_model = inputs_layer.shape.as_list()[-1]
    # Calculate the angles given the input
    angles = get_angles(np.arange(seq_length)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)
    # Calculate the positional encodings
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    # Expand the encodings with a new dimension
    pos_encoding = angles[np.newaxis, ...]
    #plt.pcolormesh(pos_encoding[0], cmap='viridis')
    return inputs_layer + tf.cast(pos_encoding, tf.float32)



