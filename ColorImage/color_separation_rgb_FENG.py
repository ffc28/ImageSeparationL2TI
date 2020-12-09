#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:00:02 2020

@author: fangchenfeng
"""
# some studies about color image mixing and illustration

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgba2rgb
import numpy as np
import scipy.linalg as la
# import the color image
pic_set = 5
# convert from rgba to rgb
img1=rgba2rgb(mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic1.png'))
img2=rgba2rgb(mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic2.png'))

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

def whiten_projection(S):
    """
    This function does the whitening projection with PCA
    """
    R = np.dot(S, S.T)
    W = la.sqrtm(np.linalg.inv(R))
    return np.dot(W, S), W

def color_image_whiten(im1, im2):
    """
    This function does the whitening processing for each channel (RGB)
    and returns the three whiteing matrices
    The input images are zero mean
    """
    temp = np.copy(np.squeeze(im1[:,:,0]))
    image_size = temp.shape
    
    R = np.zeros((2, temp.size))
    R[0,:] = np.reshape(im1[:,:,0], (1, -1))
    R[1,:] = np.reshape(im2[:,:,0], (1, -1))
    
    R, Wr = whiten_projection(R)
    
    G = np.zeros((2, temp.size))
    G[0,:] = np.reshape(im1[:,:,1], (1, -1))
    G[1,:] = np.reshape(im2[:,:,1], (1, -1))
    
    G, Wg = whiten_projection(G)
    
    B = np.zeros((2, temp.size))
    B[0,:] = np.reshape(im1[:,:,2], (1, -1))
    B[1,:] = np.reshape(im2[:,:,2], (1, -1))
    
    B, Wb = whiten_projection(B)
    
    im1[:,:,0] = np.reshape(np.squeeze(R[0,:]),image_size)
    im2[:,:,0] = np.reshape(R[1,:],image_size)
    
    im1[:,:,1] = np.reshape(G[0,:],image_size)
    im2[:,:,1] = np.reshape(G[1,:],image_size)
    
    im1[:,:,2] = np.reshape(B[0,:],image_size)
    im2[:,:,2] = np.reshape(B[1,:],image_size)
    
    return im1, im2, Wr, Wg, Wb

def mix_scale_norm(im1, im2, Wr, Wg, Wb):
    """
    This function normalise the first and the last element of the mixing matrix
    """
    # Channel R
    Ar = np.linalg.inv(Wr)
    im1[:,:,0] = (Ar[0,0] + Ar[1,0])*im1[:,:,0]
    im2[:,:,0] = (Ar[1,1] + Ar[0,1])*im2[:,:,0]
    
    Ar[:,0] = Ar[:,0]/(Ar[0,0] + Ar[1,0])
    Ar[:,1] = Ar[:,1]/(Ar[1,1] + Ar[0,1])
    
    Wr = np.linalg.inv(Ar)
    
    # Channel G
    Ag = np.linalg.inv(Wg)
    im1[:,:,1] = (Ag[0,0] + Ag[1,0])*im1[:,:,1]
    im2[:,:,1] = (Ag[1,1] + Ag[0,1])*im2[:,:,1]
    
    Ag[:,0] = Ag[:,0]/(Ag[0,0] + Ag[1,0])
    Ag[:,1] = Ag[:,1]/(Ag[1,1] + Ag[0,1])
    
    Wg = np.linalg.inv(Ag)
    
    # Channel B
    Ab = np.linalg.inv(Wb)
    im1[:,:,2] = (Ab[0,0] + Ab[1,0])*im1[:,:,2]
    im2[:,:,2] = (Ab[1,1] + Ab[0,1])*im2[:,:,2]
    
    Ab[:,0] = Ab[:,0]/(Ab[0,0] + Ab[1,0])
    Ab[:,1] = Ab[:,1]/(Ab[1,1] + Ab[0,1])
    
    Wb = np.linalg.inv(Ab)
    
    return im1, im2, Wr, Wg, Wb

def demix_mean(m1, m2, Wr, Wg, Wb):
    """
    This function does the demixing process for the mean values
    and returns the mean values after the demixing
    """
    # R channel
    R = np.zeros((2,1))
    R[0,:] = m1[0,:]
    R[1,:] = m2[0,:]
    
    R = np.dot(Wr, R)
    
    # G channel
    G = np.zeros((2,1))
    G[0,:] = m1[1,:]
    G[1,:] = m2[1,:]
    
    G = np.dot(Wg, G)
    
    # B channel
    B = np.zeros((2,1))
    B[0,:] = m1[2,:]
    B[1,:] = m2[2,:]
    
    B = np.dot(Wb, B)
    
    m1[0,:] = R[0,:]
    m1[1,:] = G[0,:]
    m1[2,:] = B[0,:]
    
    m2[0,:] = R[1,:]
    m2[1,:] = G[1,:]
    m2[2,:] = B[1,:]
    
    return m1, m2

def remove_mean(im):
    """
    This function just ries to remove the mean vlaues of each channel
    The input im should be rgb color image
    This function also returns the three mean values in the order of RGB in a vector form
    """
    R = np.mean(im[:,:,0])
    G = np.mean(im[:,:,1])
    B = np.mean(im[:,:,2])
    
    v = np.zeros((3,1))
    
    v[0,:] = R
    v[1,:] = G  
    v[2,:] = B

    im[:,:,0] = im[:,:,0] - R
    im[:,:,1] = im[:,:,1] - G
    im[:,:,2] = im[:,:,2] - B
    
    return im, v

def add_mean(im, v):
    """
    This function tries to add mean values to the three channels 
    The inpurt im should be color image with three channels RGB of zero mean
    v is a vector that containes the three mean values
    """
    im[:,:,0] = im[:,:,0] + v[0,:]
    im[:,:,1] = im[:,:,1] + v[1,:]
    im[:,:,2] = im[:,:,2] + v[2,:]
    
    return im

def three_channel_mix(im1, im2, A):
    """
    This function does the mixing process of the color image im1 and im2
    A is the mixing matrix of size 2 times 2. We have the same mixing matrix for all three channels (RGB)
    The output is two mixed color images of zero mean in each channel
    """
    x1 = np.copy(im1)
    x2 = np.copy(im2)
    # channel R
    x1[:,:,0] = A[0,0]*im1[:,:,0] + A[0,1]*im2[:,:,0]
    x2[:,:,0] = A[1,0]*im1[:,:,0] + A[1,1]*im2[:,:,0]
    # channel G
    x1[:,:,1] = A[0,0]*im1[:,:,1] + A[0,1]*im2[:,:,1]
    x2[:,:,1] = A[1,0]*im1[:,:,1] + A[1,1]*im2[:,:,1]
    # channel B
    x1[:,:,2] = A[0,0]*im1[:,:,2] + A[0,1]*im2[:,:,2]
    x2[:,:,2] = A[1,0]*im1[:,:,2] + A[1,1]*im2[:,:,2]
    
    return x1, x2

    
    

# remove the mean value of each channel (RGB). However, we keep the scale of the three channel unchanged
#img1, v1 = remove_mean(img1)
#img2, v2 = remove_mean(img2)

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
    
# Here we try to do the separation, normally a whitening can do the work
    
# 1. First we get rid of the mean values in the mixtues
x1_zero, mx1 = remove_mean(x1)
x2_zero, mx2 = remove_mean(x2)    
    
s1_zero, s2_zero, Wr, Wg, Wb = color_image_whiten(x1_zero, x2_zero)   
# We apply the same demixing matrix to the mean values of each channel
s1_zero, s2_zero, Wr, Wg, Wb = mix_scale_norm(s1_zero, s2_zero, Wr, Wg, Wb)

mse1, mse2 = demix_mean(mx1, mx2, Wr, Wg, Wb)
# add the mean value back to the estimated color images
se1 = add_mean(s1_zero, mse1)
se2 = add_mean(s2_zero, mse2)


# show the color separation
plt.figure()
plt.subplot(121)
plt.imshow(se1)
plt.title('Estimated Color image 1')
plt.axis('off')

plt.subplot(122)
plt.imshow(se2)
plt.title('Estimated Color image 2')
plt.axis('off')

    
    