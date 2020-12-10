#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:43:07 2020

@author: fangchenfeng
"""
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgba2rgb
import numpy as np
import scipy.linalg as la
import pywt
from skimage.restoration import (denoise_wavelet, denoise_tv_chambolle, denoise_nl_means)

np.random.seed(1)
# import the color image
pic_set = 3
# convert from rgba to rgb
img1=rgba2rgb(mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png'))
img2=rgba2rgb(mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png'))

# I get only the R channel
img1 = np.squeeze(img1[:,:,1])
img2 = np.squeeze(img2[:,:,1])

# show the two original color images
plt.figure()
plt.subplot(121)
plt.imshow(img1, cmap = 'gray')
plt.title('Original image1 R')
plt.axis('off')

plt.subplot(122)
plt.imshow(img2, cmap = 'gray')
plt.title('Original image2 R')
plt.axis('off')

image_size = img1.shape

def whiten_projection(S):
    """
    This function does the whitening projection with PCA
    """
    R = np.dot(S, S.T)
    W = la.sqrtm(np.linalg.inv(R))
    return np.dot(W, S)

def image_to_vec(im1, im2):
    """
    The inputs im1 and im2 are single channel images
    """
    
    S = np.zeros((2, image_size[0]*image_size[1]))
    S[0,:] = np.reshape(im1, [1, -1])
    S[1,:] = np.reshape(im2, [1, -1])
    
    return S

def vec_to_image(S):
    """
    The size of S should be 2 times T
    """
    im1 = np.reshape(S[0,:], image_size)
    im2 = np.reshape(S[1,:], image_size)
    
    return im1, im2

def one_channel_mix(im1, im2, A):

    # channel R
    S = image_to_vec(im1, im2)
    X = np.dot(A, S)
    x1, x2 = vec_to_image(X)
   
    return x1, x2


def soft_proximal(S, value):
    """
    This function does the soft-thresholding
    """
    return pywt.threshold(S, value, 'soft')

def positive_proj(S):
    
    S = np.where(S<0, 0, S)
    S = np.where(S>1, 0, S)
    
    return S

def wavelet_proj(S, value):
    """
    Wavelet denoising of a single channel
    """
    se1, se2 = vec_to_image(S)
    se1 = denoise_wavelet(se1, sigma = lambda_this, multichannel = False, convert2ycbcr = False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma = True)
    
    se2 = denoise_wavelet(se2, sigma = lambda_this, multichannel = False, convert2ycbcr = False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma = True)
    
    S = image_to_vec(se1, se2)
    
    return S

def sum_to_one_proj(A):
    
    A = np.where(np.isnan(A), 0, A)

    A[:,0] = A[:,0]/(A[0,0] + A[1,0])        
  
    A[:,1] = A[:,1]/(A[1,1] + A[0,1])
    
    return A

def col_norm_proj(A):
    """
    Here we normalise each column of the input matrix A

    """
    A[:,0] = A[:,0]/np.linalg.norm(A[:,0])
    A[:,1] = A[:,1]/np.linalg.norm(A[:,1])
    
    return A
    

def decorrelation_s(S):
    
    # I get rid of the mean value first
    m1 = np.mean(S[0,:])
    m2 = np.mean(S[1,:])
    
    S[0,:] = S[0,:] - m1
    S[1,:] = S[1,:] - m2
    
    e1 = np.linalg.norm(S[0,:])
    e2 = np.linalg.norm(S[1,:])
    
    S = whiten_projection(S)
    # get the mean and norm back
    S[0,:] = S[0,:]*e1
    S[1,:] = S[1,:]*e2
    
    S[0,:] = S[0,:] + m1
    S[1,:] = S[1,:] + m2
    
    return S


mixing_matrix = np.array([[0.7, 0.3], [0.3, 0.7]]) # In this case, 

# img1 = img1 - np.mean(img1)
# img2 = img2 - np.mean(img2)

x1, x2 = one_channel_mix(img1, img2, mixing_matrix)

plt.figure()
plt.subplot(121)
plt.imshow(x1, cmap = 'gray')
plt.title('Mixed image1 R')
plt.axis('off')

plt.subplot(122)
plt.imshow(x2, cmap = 'gray')
plt.title('Mixed image2 R')
plt.axis('off')

# We do the separation here
max_it = 500
se1 = np.copy(x1)
se2 = np.copy(x2)
A = np.identity(2)

lambda_max = 2e-1
lambda_final = 1e-4
lambda_v = np.logspace(np.log10(lambda_max), np.log10(lambda_final), max_it)
S = image_to_vec(se1, se2)
X = image_to_vec(x1, x2)
for it in np.arange(max_it):
    lambda_this = lambda_v[it]
    # The gradient of S
    grad_s = -np.dot(A.T, X - np.dot(A, S))
    # The Lipschitz constant
    Ls = np.linalg.norm(np.dot(A.T, A), ord = 2)
    # Gradient descent
    S = S - grad_s/Ls
    # S = np.dot(A.T, X)
    
    # Proximal
    S = soft_proximal(S, lambda_this)
    # S = wavelet_proj(S, lambda_this)
    # decorrelation projection
    S = decorrelation_s(S)
    # zero one constrain
    S = positive_proj(S)
    # update A
    grad_A = -np.dot(X - np.dot(A, S), S.T)
    # Lischitz constant
    La = np.linalg.norm(np.dot(S, S.T), ord = 2)
    # Gradient descent
    A = A - grad_A/La
    # Proximal
    A = sum_to_one_proj(A)
    #A = col_norm_proj(A)
    

    
print(mixing_matrix)
print(A)    
se1, se2 = vec_to_image(S) 

plt.figure()
plt.subplot(121)
plt.imshow(se1, cmap = 'gray')
plt.title('Estimated image1 R')
plt.axis('off')

plt.subplot(122)
plt.imshow(se2, cmap = 'gray')
plt.title('Estimated image2 R')
plt.axis('off')





