<img src="http://www.lptv.cl/wp-content/uploads/2017/08/LOGO_2017_740x150.png" width="1020">

# End-to-End LSTM-Based Earthquake Magnitude Estimation with a Single Station

This repository contains the codes of: “End-to-End LSTM-Based Earthquake Magnitude Estimation with a Single Station” Cofré, A., Marín, M. Pino, O.V., Galleguillos, N., Riquelme, S., Barrientos, S., and Yoma, N.B., 2022., IEEE Geoscience and Remote Sensing Letters, vol. 19, p. 3006905.
*Paper*:   [https://ieeexplore.ieee.org/document/9774421](https://ieeexplore.ieee.org/document/9774421) 

--------------
## Description
<p align="justify"> 
The model proposed is an end-to-end DNN based on long short-term memory (LSTM) is presented to address the problem of earthquake magnitude estimation for earthquake early warning (EEW) and tsunami early warning (TW) purposes using a seismic station.
  
 

--------------
## How to install 
  
conda env create -f env_magnitude.yml

--------------
## Tutorial
* Download the database, download_database.py, which is located in the 'data' folder. It is also available at the [https://drive.google.com/file/d/1dheCE8lLGiZ-aJZRl0wBHN9kaCEYYxPy/view?usp=share_link](https://drive.google.com/file/d/1dheCE8lLGiZ-aJZRl0wBHN9kaCEYYxPy/view?usp=share_link)
* To train the algorithm it is necessary to run train.py
* To test a database, you will need to run test.py
    ** In case you want to test only one trace, it's possible to use test_magnitude_estimation.py
  
  
  
  
  



