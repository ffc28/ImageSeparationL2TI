#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:27:51 2020

@author: fangchenfeng
"""

# This is a script for some tries with the three projections
from __future__ import division
import numpy as np
import museval.metrics as mmetrics
import matplotlib.pyplot as plt
import three_projection_method
from sklearn.decomposition import FastICA
import utils
from scipy.stats import kurtosis

# control the random sequence
np.random.seed(1)

N_sources = 5

Kur_it = np.zeros((1, N_sources))
SDR_soft1_imp = np.zeros((1, N_sources))
SDR_soft2_imp = np.zeros((1, N_sources))
SDR_cube_imp = np.zeros((1, N_sources))
SDR_ica_imp = np.zeros((1, N_sources))
SDR_sp_imp = np.zeros((1, N_sources))

for pic_set in np.arange(N_sources):
    # load images and convert them
    print(pic_set)
    #mixing_matrix = np.array([[0.36, 0.66], [0.03, 0.95]])
    mixing_matrix = np.array([[1, 0.7], [0.02, 1]])
    # mixing_matrix = np.array([[1, 0.3], [0.5, 1]])
    # mixing_matrix = np.array([[0.8488177, 0.17889592], [0.05436321, 0.36153845]])
    X, source = three_projection_method.import_image(pic_set, mixing_matrix)
    # calculate kurtosis, if it's Gaussian distribution, then it's close to zero
    k1 = kurtosis(np.squeeze(source[0,:]))
    k2 = kurtosis(np.squeeze(source[1,:]))
    Kur_it[0, pic_set] = np.abs(k1) + np.abs(k2)
    
    # whitening processing. It's important
    X_nonwhiten = np.copy(X)
    X = three_projection_method.whiten_projection(np.asarray(X))
    
    (sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X))
    print(sdr_ref)
    
    #####################################
    # Using the three projection method with different non linearity
    print('Here is the first algorithm')
    separation_method = 'wavelet'
    sigma = 1e-3
    sigma_final = 1e-5
    max_it = 800
    separation_matrix = three_projection_method.three_projection_demix(X, max_it = max_it, method = separation_method, threshold_value = sigma, threshold_final = sigma_final)   
    # get the estimated sources with the separation matrix
    Se = np.dot(separation_matrix, X)   
    # print('Kurtosis of the estimation 1 with TV is: ', kurtosis(Se[0,:]))
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se)    
    SDR_soft1_imp[0, pic_set] = np.mean(sdr) - np.mean(sdr_ref)
  
    
    ######################################
    # ICA does whitening pre-processing automatiquely
    print('Here is the second algorithm')
    ica = FastICA(n_components = 2, fun = 'cube', max_iter = 400)
    Se_ica = np.asarray(ica.fit_transform(X.T).T)
    # print('Kurtosis of the estimation 1 with ICA is: ', kurtosis(Se_ica[0,:]))
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se_ica)
    SDR_ica_imp[0, pic_set] = np.mean(sdr) - np.mean(sdr_ref)

    
    #####################################
    # Using the three projection method with different non linearity
    print('Here is the third algorithm')
    separation_method = 'TV'
    sigma = 1e-3
    sigma_final = 1e-5
    max_it = 800
    separation_matrix = three_projection_method.three_projection_demix(X, max_it = max_it, method = separation_method, threshold_value = sigma, threshold_final = sigma_final)   
    # get the estimated sources with the separation matrix
    # print(np.dot(separation_matrix, separation_matrix.T))
    Se = np.dot(separation_matrix, X)   
    # print('Kurtosis of the estimation 1 with soft is: ', kurtosis(Se[0,:]))
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se)    
    SDR_soft2_imp[0, pic_set] = np.mean(sdr) - np.mean(sdr_ref)
    
    
plt.figure()
plt.plot(SDR_soft1_imp[0,: ], 'k*')   

plt.plot(SDR_ica_imp[0,: ], 'r*')    

plt.plot(SDR_soft2_imp[0,: ], 'b*')   
 
plt.grid()
plt.show
"""
print(np.mean(SDR_soft1_imp[0,: ]))
print(np.mean(SDR_soft2_imp[0,: ]))
print(np.mean(SDR_ica_imp[0,: ]))
"""
