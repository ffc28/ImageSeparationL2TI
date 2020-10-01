#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:08:13 2020

@author: fangchenfeng
"""
from __future__ import division
import numpy as np
import scipy.linalg as la
import pywt
import math
from bm3d import bm3d
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.restoration import (denoise_bilateral, denoise_wavelet, denoise_tv_chambolle, estimate_sigma, denoise_nl_means)

def data_projection(X,S):
    """
    This functions does the data projection with the equation X = AS
    """
    A = np.dot(S, X.T)
    S = np.dot(A, X)
    R = np.dot(A, A.T)
    return np.dot(np.linalg.inv(R),S)
    # return S

def whiten_projection(S):
    """
    This function does the whitening projection with PCA
    """
    R = np.dot(S, S.T)
    W = la.sqrtm(np.linalg.inv(R))
    return np.dot(W, S)

def col_norm_proj(A):
    """
    Here we normalise each column of the input matrix A

    """
    A[:,0] = A[:,0]/np.linalg.norm(A[:,0])
    A[:,1] = A[:,1]/np.linalg.norm(A[:,1])
    
    return A


def get_demix(X, S):
    
    return np.dot(S, X.T)


def get_demix_it(X, S):
    """
    I can use sub iteration here
    """
    W_init = np.dot(S, X.T)
    W = np.dot(S, X.T)
    max_it = 50
    alpha = 0.5 # the learning rate
    for it in np.arange(max_it):
        # update W
        W = whiten_projection((1-alpha)*W + alpha*W_init)
    
    #return np.dot(S, X.T)
    return W


def soft_proximal(S, value):
    """
    This function does the soft-thresholding
    """
    return pywt.threshold(S, value, 'soft')

def hard_proximal(S, value):
    """
    This function does the hard-thresholding
    """
    return pywt.threshold(S, value, 'hard')

def garrote_proximal(S, value):
    """
    This function does the garrote-thresholding
    """
    return pywt.threshold(S, value, 'garrote')


def TV_proj(S, image_size, lambda_this):
    """
    This function does the TV projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # TV denoising
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def TV_proj_flip(S, image_size, lambda_this):
    """
    This function does the TV projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # TV denoising
    S1 = S1.T
    S2 = S2.T
    S2 = np.fliplr(S2)
    
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
    
    S2 = np.fliplr(S2)
    S1 = S1.T
    S2 = S2.T
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def Wavelet_proj(S, image_size, lambda_this):
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
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def Nonlocal_proj(S, image_size, lambda_this):
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
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def BM3D_proj(S, image_size, lambda_this):
    """
    This function does the BM3D projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # BM3D denoising
    S1 = bm3d(S1, lambda_this)
    S2 = bm3d(S2, lambda_this)
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def Bilateral_proj(S, image_size, lambda_this):
    
    """
    This function does the Non local mean projection
    """    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    sm1 = S1.min() + 0.5
    sm2 = S2.min() + 0.5
    
    S1 = S1 + sm1
    S2 = S2 + sm2
    # Nonlocal mean denoising
    S1 = denoise_bilateral(S1, sigma_spatial=lambda_this, multichannel=False)
    S2 = denoise_bilateral(S2, sigma_spatial=lambda_this, multichannel=False)
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1])) - sm1
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1])) - sm2
     
    return S


def non_linear1(S):
    """
    This function use x^3 as a nonlinear
    """
    return np.power(S, 3)

def non_linear2(S):
    """
    This function use sign(x)*x^4 as a nonlinear
    """
    return np.multiply(np.sign(S), np.square(S))


def three_projection_separation(X, max_it = 50, method = 'cube', threshold_value = 2e-3, stop_critera = 1e-8):
    """
    Method of three projection to do the separation
    The size of X should be N_channels * N_samples (2 * T)
    """
    T = X.shape[1] # the number of samples 
    image_size = (np.int(math.sqrt(T)), np.int(math.sqrt(T)))
    # X should already be whitened
    # Initialisation
    Se = np.copy(X)
    Se_old = np.copy(Se)
    
    if method == 'cube':
        # Cube non-linearity
        
        for it in np.arange(max_it):
            Se = whiten_projection(non_linear1(data_projection(X,Se)))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Cube convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)
            
    if method == 'neg-square':
        # Negative square non-linearity
        
        for it in np.arange(max_it):
            Se = whiten_projection(non_linear2(data_projection(X,Se)))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Neg-square convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)    
            
    if method == 'soft':
        # soft thresholding non-linearity
        
        for it in np.arange(max_it):
            Se = whiten_projection(soft_proximal(data_projection(X,Se), threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Soft convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)         
            
            
    if method == 'hard':
        # hard thresholding non-linearity
        
        for it in np.arange(max_it):
            Se = whiten_projection(hard_proximal(data_projection(X,Se), threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Hard convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)  
            
            
    if method == 'garrote':
        # garrote thresholding non-linearity
        
        for it in np.arange(max_it):
            Se = whiten_projection(garrote_proximal(data_projection(X,Se), threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Garrote convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)         
            
        
    if method == 'TV':
        # TV denoising nonlinear filter
        
        for it in np.arange(max_it):
            Se = whiten_projection(TV_proj(data_projection(X,Se), image_size, threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('TV convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)    
        
        
    if method == 'wavelet':
        # Wavelet denoising nonlinear filter
        
        for it in np.arange(max_it):
            Se = whiten_projection(Wavelet_proj(data_projection(X,Se), image_size, threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Wavelet convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)     
        
        
        
    if method == 'nonlocal':
        # Nonlocal mean denoising nonlinear filter
        
        for it in np.arange(max_it):
            Se = whiten_projection(Nonlocal_proj(data_projection(X,Se), image_size, threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Nonlocal convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)  
            
            
    if method == 'bm3d':
        # BM3D denoising nonlinear filter
        
        for it in np.arange(max_it):
            Se = whiten_projection(BM3D_proj(data_projection(X,Se), image_size, threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('BM3D convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)     
            
            
    if method == 'bilateral':
        # BM3D denoising nonlinear filter
        
        for it in np.arange(max_it):
            Se = whiten_projection(Bilateral_proj(data_projection(X,Se), image_size, threshold_value))
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Bilateral convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)        
        
     # Get the separation matrix   
    W = np.linalg.inv(np.dot(X, Se.T)) # separation matrix
    
    return W
    
    
    
def three_projection_demix(X, max_it = 50, method = 'cube', threshold_value = 2e-3, threshold_final = 2e-4, stop_critera = 1e-8):
    """
    Method of three projection to do the separation
    The size of X should be N_channels * N_samples (2 * T)
    """
    T = X.shape[1] # the number of samples 
    image_size = (np.int(math.sqrt(T)), np.int(math.sqrt(T)))
    # X should already be whitened
    # Initialisation
    Se = np.copy(X)
    Se_old = np.copy(Se)   
    thresh_v = np.logspace(np.log10(threshold_value), np.log10(threshold_final), max_it)
    
    if method == 'cube':
        # Cube non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = non_linear1(Se)
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. whiten the demixing matrix
            W = whiten_projection(W)
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Cube demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)  
            
    if method == 'soft':
        # Soft non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = soft_proximal(Se, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
           # 3. whiten the demixing matrix
            if it<350:
                W = whiten_projection(W)
            else:
                W = col_norm_proj(W)
                
            # 4. Get the new Se
            Se = np.dot(W, X)
            
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Soft demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)         
            
            
    
    if method == 'TV':
        # TV non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = TV_proj(Se, image_size, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. whiten the demixing matrix
            #if it<350:
            W = whiten_projection(W)
            #else:
            #W = col_norm_proj(W)
                
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('TV demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)  
            
            
    if method == 'TV_flip':
        # TV non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = TV_proj_flip(Se, image_size, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. whiten the demixing matrix
            #if it<350:
            W = whiten_projection(W)
            #else:
            #W = col_norm_proj(W)
                
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('TV demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)          
            
    if method == 'TV_norm':
        # TV non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = TV_proj(Se, image_size, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. Normalise the demixing matrix
            W = np.linalg.inv(W)
            W = col_norm_proj(W)    
            W = np.linalg.inv(W)
            # print(W)
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('TV norm demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)          
            
    if method == 'wavelet':
        # Wavelet non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = Wavelet_proj(Se, image_size, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. whiten the demixing matrix
            W = whiten_projection(W)
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Wavelet demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)      
                      
            
    if method == 'bm3d':
        # Wavelet non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = BM3D_proj(Se, image_size, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. whiten the demixing matrix
            W = whiten_projection(W)
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('BM3D demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)   
            
            
            
    if method == 'Nonlocal':
        # Wavelet non-linearity
        
        for it in np.arange(max_it):
            # 1. denoising
            Se = Nonlocal_proj(Se, image_size, thresh_v[it])
            # 2. demixing matrix
            W = get_demix(X, Se)
            # 3. whiten the demixing matrix
            W = whiten_projection(W)
            # 4. Get the new Se
            Se = np.dot(W, X)
            if np.linalg.norm(Se - Se_old, ord = 'fro') < stop_critera:
                print('Nonlocal demix convergence reached')
                print('The real number of iteration is', it)
                break
            Se_old = np.copy(Se)  
      
    
    return W
    
    
def import_image(pic_set, mix_matrix):
    """
    This function import the images using the number of pic_set
    The output is of size 2*T
    """
    img1=mpimg.imread('./images_hard/set'+ str(pic_set + 1) + '_pic1.png')
    img2=mpimg.imread('./images_hard/set'+ str(pic_set + 1) + '_pic2.png')
    
    img1_gray = rgb2gray(img1) # the value is between 0 and 1
    img2_gray = rgb2gray(img2) # transform it from color to gray
    
    # flip the second image
    img2_gray = np.fliplr(img2_gray)

    # We mix them here    
    source1 = np.matrix(img1_gray)
    source1 = source1.flatten('F') #column wise
    
    source2 = np.matrix(img2_gray)
    source2 = source2.flatten('F') #column wise
    
    source1 = source1 - np.mean(source1)
    source2 = source2 - np.mean(source2)
    
    source1 = source1/np.linalg.norm(source1)
    source2 = source2/np.linalg.norm(source2)

    source = np.stack((source1, source2))
    X = np.matmul(mix_matrix, source)

    return X, np.asarray(source)

    
    
    
    