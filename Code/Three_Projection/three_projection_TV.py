#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:27:51 2020

@author: fangchenfeng
"""

# This is a script for some tries with the three projections
from __future__ import division
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import scipy.linalg as la
import scipy.io as sio
import matplotlib.pyplot as plt
import pywt
import museval.metrics as mmetrics
from rdc import rdc
from bm3d import bm3d
from skimage.restoration import (denoise_wavelet, denoise_tv_chambolle, estimate_sigma, denoise_nl_means)

# load images and convert them
n = 256
image_size = (n, n)


# load the source here
pic_set = 1
img1=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic1.png')
img2=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic2.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)
# Mixing process here
img2_gray_re = np.fliplr(img2_gray)

source1 = np.matrix(img1_gray)
source1 = source1.flatten('F') #column wise

source2 = np.matrix(img2_gray)
source2 = source2.flatten('F') #column wise

source1 = source1 - np.mean(source1)
source2 = source2 - np.mean(source2)

#source1 = source1/np.linalg.norm(source1)
#source2 = source2/np.linalg.norm(source2)

print("rdc = ", rdc(source1.T,source2.T))
source = np.stack((source1, source2))

print('Covariance matrix is: ')
print(np.matmul(source,source.T))

# randomly generated mixing matrix
#np.random.seed(0)
#mixing_matrix = np.random.randn(2,2)
#mixing_matrix = np.array([[0.36, 0.66], [0.03, 0.95]])
mixing_matrix = np.array([[1, 0.5], [0.5, 1]])
print('Mixing matrix is: ')
print(mixing_matrix)


# X = source * mixing_matrix - The mixed images

X = np.matmul(mixing_matrix, source)


def data_projection(X,S):
    """
    This functions does the data projection with the equation X = AS
    """
    A = np.dot(S, X.T)
    S = np.dot(A, X)
    R = np.dot(A, A.T)
    return np.dot(np.linalg.inv(R),S)

def whiten_projection(S):
    """
    This function does the whitening projection with PCA
    """
    R = np.dot(S,S.T)
    W = la.sqrtm(np.linalg.inv(R))
    return np.dot(W,S)

def soft_proximal(S,value):
    """
    This function does the soft-thresholding
    """
    return pywt.threshold(S, value, 'soft')

def TV_proj(S, lambda_this):
    """
    This function does the TV projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # TV denoising
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def Wavelet_proj(S, lambda_this):
    """
    This function does the Wavelet projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # Wavelet denoising
    S1 = denoise_wavelet(S1, sigma = lambda_this, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
    S2 = denoise_wavelet(S2, sigma = lambda_this, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=False)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def Nonlocal_proj(S, lambda_this):
    """
    This function does the Non local mean projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    patch_kw = dict(patch_size=8,      # 5x5 patches
                patch_distance=7,  # 13x13 search area
                multichannel=False)
    # Nonlocal mean denoising
    S1 = denoise_nl_means(S1, h=1.12*lambda_this, fast_mode=False, sigma = lambda_this, 
                           **patch_kw)
    S2 = denoise_nl_means(S2, h=1.12*lambda_this, fast_mode=False, sigma = lambda_this, 
                           **patch_kw)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def BM3D_proj(S, lambda_this):
    """
    This function does the BM3D projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # BM3D denoising
    S1 = bm3d(S1, lambda_this)
    S2 = bm3d(S2, lambda_this)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def non_linear1(S):
    """
    This function use x^3 as a nonlinear
    """
    return np.power(S,3)

def sigmoid_proj(S):
    
    S1 = S[0,:]
    #S2 = S[1,:]
    
    S[0,:] = 1 / (1 + np.exp(-S1))
    return S

# Here begins the algorithm
# whitening processing. It's important
R = np.dot(X, X.T)
W = la.sqrtm(np.linalg.inv(R))
X = np.dot(W, X)

(sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X))
# mix = [[0.6992, 0.7275], [0.4784, 0.5548]] #or use the matrix from the paper
print("Reference SDR is: ", sdr_ref)
print("Reference SIR is: ", sir_ref)

lambda_max = 0.02
#lambda_final = lambda_max
lambda_final = 0.00002
max_it = 50
lambda_v = np.logspace(np.log10(lambda_max),np.log10(lambda_final),max_it)

#Se = np.random.randn(2, n*n) 
Se = X  
cost_it = np.zeros((1,max_it)) 
SDR_it = np.zeros((2, max_it)) 
SIR_it = np.zeros((2, max_it)) 
SAR_it = np.zeros((2, max_it)) 

for it in np.arange(max_it):
    print(it)
    # we performe three projections
    # Se = whiten_projection(soft_proximal(data_projection(X, Se),lambda_v[it]))
    # Se = TV_proj(data_projection(X,Se), lambda_v[it])
    Se = whiten_projection(Wavelet_proj(data_projection(X,Se), lambda_v[it]))
    # Se = whiten_projection(non_linear1(data_projection(X,Se)))
    # Se = whiten_projection(data_projection(X,Se))
    
    cost_it[0,it] = np.linalg.norm(X - np.dot(np.dot(X,Se.T), Se),ord = 'fro')
    
    Se_inv = np.dot(np.linalg.inv(np.dot(X, Se.T)),X)
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se_inv)
    
    SDR_it[:, it] = np.squeeze(sdr)
    
print(np.dot(Se,Se.T))
"""
plt.figure()
plt.plot(cost_it[0,:])
plt.title('Cost for iterations')
plt.show
"""
plt.figure()
plt.plot(np.mean(SDR_it, axis = 0))
plt.title('SDR for iterations')
plt.grid()
plt.show

s1 = Se[0,:]
s1 = np.reshape(s1, (n,n))

s2 = Se[1,:]
s2 = np.reshape(s2, (n,n))

"""
plt.figure()
plt.imshow(s1.T, cmap='gray')
plt.title("Estimated source 1 with Sparse")
plt.show

plt.figure()
plt.imshow(s2.T, cmap='gray')
plt.title("Estimated source 2 with Sparse")
plt.show()
"""
(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se)
print('********')
print(sdr)
print(sir)
print(sar)

print("RDC of the estimation is: ", rdc(Se[0,:].T,Se[1,:].T))

Se_inv = np.dot(np.linalg.inv(np.dot(X, Se.T)),X)
(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se_inv)
print('********')
print(sdr)
print(sir)
print(sar)

# The idea is now to say that I can use all kinds of thresholding operators
# The thresholding operator along with the decorrelation (PCA) can be seen as an independent projection operator
