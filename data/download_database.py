# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:10:36 2022

@author: Marc
"""

import geopy.distance
import os
from obspy import read
import pandas as pd
from obspy.core import UTCDateTime
from obspy.clients.fdsn  import Client

path_xlsx =  'set/database.xlsx'

path_output = "sac/"

df = pd.read_excel(path_xlsx, index_col=0,)



for i in range(232,len(df)):
    
    event = df.iloc[i]
    print('Downloading: ', event['name'])
    
    name = event['name']
    tin = UTCDateTime(event['starttime'])
    tfin = UTCDateTime(event['endtime'])
    net =  event['network']
    sta = event['station']
    ch = event['channel'][0:2]
    
    path_folder = os.path.join(path_output,name)
    isExistFolder = os.path.exists(path_folder)
    if not isExistFolder:
        os.makedirs(path_folder)

    path_folder_sta = os.path.join(path_output,name,sta)
    isExistFolderSta = os.path.exists(path_folder_sta)
    if not isExistFolderSta:
        os.makedirs(path_folder_sta)
        
    



    
    if net in ['C1','C']:
        client = Client('IRIS')
    elif net == 'CX':
        client = Client('GFZ')
    client.status_delay= 40  

    stz = client.get_waveforms(net, sta, "*", ch+"Z", tin, tfin, attach_response=True)
    ste = client.get_waveforms(net, sta, "*", ch+"E", tin, tfin, attach_response=True)
    stn = client.get_waveforms(net, sta, "*", ch+"N", tin, tfin, attach_response=True)
    

    stz.write(path_folder_sta+ "/" + str(sta)+'_'+ ch+'Z'+ '.SAC',format='SAC')
    ste.write(path_folder_sta+ "/" + str(sta)+'_' +ch+'E' + '.SAC',format='SAC')
    stn.write(path_folder_sta+ "/" + str(sta)+'_' +ch+'N' + '.SAC',format='SAC')

    

    stz = read(path_folder_sta+ "/" + str(sta)+'_'+ ch+'Z'+ '.SAC',format='SAC')
    stz[0].stats.sac.mag = event['magnitude']           
    stz[0].stats.sac.evla = event['latitude']
    stz[0].stats.sac.evlo = event['longitude']    
    stz[0].stats.sac.evdp = event['depth']    
    stz[0].stats.sac.dist = event['distance']                
    stz.write(path_folder_sta+ "/" + str(sta)+'_'+ ch+'Z'+ '.SAC',format='SAC')
                
            
            