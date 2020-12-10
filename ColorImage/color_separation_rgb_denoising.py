#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:50:47 2020

@author: fangchenfeng
"""
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgba2rgb
import numpy as np
import pywt
from skimage.restoration import (denoise_wavelet, denoise_tv_chambolle, denoise_nl_means)
# import the color image
pic_set = 7
# convert from rgba to rgb
img1=rgba2rgb(mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png'))
img2=rgba2rgb(mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png'))

# show the two original color images
plt.figure()
plt.subplot(121)
plt.imshow(img1)
plt.title('Original Color image 1')
plt.axis('off')

plt.subplot(122)
plt.imshow(img2)
plt.title('Original Color image 2')
plt.axis('off')

image_size = np.squeeze(img1[:,:,0]).shape
color_image_size = img1.shape

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

def rgb_to_twoimage(R, G, B):
    
    im1 = np.zeros(color_image_size)
    im2 = np.zeros(color_image_size)
    
    im1[:,:,0] = np.reshape(R[0,:], image_size)
    im2[:,:,0] = np.reshape(R[1,:], image_size)
    
    im1[:,:,1] = np.reshape(G[0,:], image_size)
    im2[:,:,1] = np.reshape(G[1,:], image_size)
    
    im1[:,:,2] = np.reshape(B[0,:], image_size)
    im2[:,:,2] = np.reshape(B[1,:], image_size)
    
    return im1, im2

def twoimage_to_rgb(im1, im2):
    
    R = image_to_vec(im1[:,:,0], im2[:,:,0])   
    G = image_to_vec(im1[:,:,1], im2[:,:,1])   
    B = image_to_vec(im1[:,:,2], im2[:,:,2])    

    return R, G, B       


def three_channel_mix(im1, im2, A):
    """
    This function does the mixing process of the color image im1 and im2
    A is the mixing matrix of size 2 times 2. We have the same mixing matrix for all three channels (RGB)
    The output is two mixed color images of zero mean in each channel
    """
    # channel R
    Xr = np.dot(A, image_to_vec(im1[:,:,0], im2[:,:,0]))
    # channel G
    Xg = np.dot(A, image_to_vec(im1[:,:,1], im2[:,:,1]))
    # channel B
    Xb = np.dot(A, image_to_vec(im1[:,:,2], im2[:,:,2]))
    
    x1, x2 = rgb_to_twoimage(Xr, Xg, Xb)
    
    return x1, x2

def sum_to_one_proj(A):
    
    A = np.where(np.isnan(A), 0, A)

    A[:,0] = A[:,0]/(A[0,0] + A[1,0])       
    A[:,1] = A[:,1]/(A[1,1] + A[0,1])
    
    return A


def positive_proj(S):
    
    S = np.where(S<0, 0, S)
    S = np.where(S>1, 0, S)
    
    return S

def zero_one_proj(s):
    s = np.where(s<1, s, 0)
    s = np.where(s>0 ,s, 0)

    return s

def rgb_to_matrix(R, G, B):
    
    A = np.zeros((2, 2, 3))
    A[:,:,0] = R
    A[:,:,1] = G
    A[:,:,2] = B
    
    return A


def matrix_to_rgb(A):
    
    R = np.squeeze(A[:,:,0])
    G = np.squeeze(A[:,:,1])
    B = np.squeeze(A[:,:,2])
    
    return R, G, B

def grad_des_s(X, S, A):
    # Lipcshitz constant
    L = np.linalg.norm(np.dot(A.T, A), ord = 2)
    grad = -np.dot(A.T, X - np.dot(A, S))
    S = S - grad/L
    
    return S

def grad_des_A(X, S, A):
    # Lischitz constant
    La = np.linalg.norm(np.dot(S, S.T), ord = 2)
    grad_A = -np.dot(X - np.dot(A, S), S.T)
    # Gradient descent
    A = A - grad_A/La
    
    return A

def soft_proximal(S, value):
    """
    This function does the soft-thresholding
    """
    return pywt.threshold(S, value, 'soft')


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

def wavelet_color(im1, im2, value):
    
    im1 = denoise_wavelet(im1, sigma = lambda_this, multichannel = True, convert2ycbcr = True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma = True)
    
    im2 = denoise_wavelet(im2, sigma = lambda_this, multichannel = True, convert2ycbcr = True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma = True)
    
    return im1, im2

   
    
def norm_center_vec(R):
    
    R[:,0] = R[:,0] = np.mean(R[:,0])
    R[:,0] = R[:,0]/np.linalg.norm(R[:,0])
    R[:,1] = R[:,1] = np.mean(R[:,1])
    R[:,1] = R[:,1]/np.linalg.norm(R[:,1])
    
    return R
    
def get_perm(X):
    """
    X is the covariance matrix of size 2 times 2. If the digonal is big, we dont't permute
    """
    if (X[0,0] + X[1, 1]) > (X[0,1] + X[1, 0]):
        perm = (0, 1) # we dont permute
    else:
        perm = (1, 0)
        
    return perm    

def put_in_order(S1, S2, A1, A2):
    # First get the covariance matrix
    S1_norm = norm_center_vec(np.copy(S1)) 
    S2_norm = norm_center_vec(np.copy(S2)) 
    
    R = np.abs(np.dot(S1_norm, S2_norm.T))
    perm = get_perm(R)
    
    # permute the second source and the second matrix
    S2 = S2[perm, :]
    A2 = A2[:, perm]
    
    return S1, S2, A1, A2
    
   
def rgb_permute(Sr, Sg, Sb, Ar, Ag, Ab):
    """
    I will try here to use correlation to get the right permutation
    The permutation should be applied to S and A
    """
    # I first permute R and G
    Sr, Sg, Ar, Ag = put_in_order(Sr, Sg, Ar, Ag)
    
    # I then permute G and B
    Sg, Sb, Ag, Ab = put_in_order(Sg, Sb, Ag, Ab)
    
    return Sr, Sg, Sb, Ar, Ag, Ab


def prox_sep(Sr, Sg, Sb, lambda_this):
    # proximal
    Sr = soft_proximal(Sr, lambda_this)
    Sg = soft_proximal(Sg, lambda_this)
    Sb = soft_proximal(Sb, lambda_this)
    
    # wavelet projection
    #Sr = wavelet_proj(Sr, lambda_this)
    #Sg = wavelet_proj(Sg, lambda_this)
    #Sb = wavelet_proj(Sb, lambda_this)
    
    # positive projection
    Sr = positive_proj(Sr)
    Sg = positive_proj(Sg)
    Sb = positive_proj(Sb)
    
    return Sr, Sg, Sb

def prox_joint(Sr, Sg, Sb, Ar, Ag, Ab, lambda_this):
    
    # I fix the permutation problem
    Sr, Sg, Sb, Ar, Ag, Ab = rgb_permute(Sr, Sg, Sb, Ar, Ag, Ab)
    # Get the two color images
    se1, se2 = rgb_to_twoimage(Sr, Sg, Sb)
    # Use RGB or YCC color wavelet denoising
    se1, se2 = wavelet_color(se1, se2, lambda_this)
    # positive projection
    se1 = positive_proj(se1)
    se2 = positive_proj(se2)

    # Get back to RGB components
    Sr, Sg, Sb = twoimage_to_rgb(se1, se2)    
    
    return Sr, Sg, Sb, Ar, Ag, Ab

# now we mix the two images in each channels
mixing_matrix = np.array([[0.7, 0.3], [0.3, 0.7]]) # In this case, a whitening process can do the separation

x1, x2 = three_channel_mix(img1, img2, mixing_matrix)    

# show the color mixtures 
plt.figure()
plt.subplot(121)
plt.imshow(x1)
plt.title('Mixed Color image 1')
plt.axis('off')

plt.subplot(122)
plt.imshow(x2)
plt.title('Mixed Color image 2')
plt.axis('off')

# Now I do the separation. I put or not a constraint of decorrelation of each channel (RGB)
# For the mixing matrix, I use a constraint of sum-to-one
max_it = 200

se1 = np.copy(x1)
se2 = np.copy(x2) # initialisation
Sr, Sg, Sb = twoimage_to_rgb(se1, se2)
Xr, Xg, Xb = twoimage_to_rgb(x1, x2)

AA = np.random.rand(2,2,3) # initialisation of the three mixing matrix
Ar, Ag, Ab = matrix_to_rgb(AA)

lambda_this = 3e-3
for it in np.arange(max_it):
    # Gradient descend  R, G, B
    Sr = grad_des_s(Xr, Sr, Ar)
    Sg = grad_des_s(Xg, Sg, Ag)
    Sb = grad_des_s(Xb, Sb, Ab)
    
    
    Sr, Sg, Sb = prox_sep(Sr, Sg, Sb, lambda_this)    
    #Sr, Sg, Sb, Ar, Ag, Ab = prox_joint(Sr, Sg, Sb, Ar, Ag, Ab, lambda_this)
    
    # update A
    Ar = grad_des_A(Xr, Sr, Ar)
    Ag = grad_des_A(Xg, Sg, Ag)
    Ab = grad_des_A(Xb, Sb, Ab)
    
    # positive projection
    Ar = positive_proj(Ar)
    Ag = positive_proj(Ag)
    Ab = positive_proj(Ab)
    
    # sum_to_one projection
    Ar = sum_to_one_proj(Ar)
    Ag = sum_to_one_proj(Ag)
    Ab = sum_to_one_proj(Ab)
    
# Fix the permutation problem at the end
Sr, Sg, Sb, Ar, Ag, Ab = rgb_permute(Sr, Sg, Sb, Ar, Ag, Ab)
    
se1, se2 = rgb_to_twoimage(Sr, Sg, Sb)
    
# show the color estimated
plt.figure()
plt.subplot(121)
plt.imshow(se1)
plt.title('Estimated Color image 1')
plt.axis('off')

plt.subplot(122)
plt.imshow(se2)
plt.title('Estimated Color image 2')
plt.axis('off')  
    













