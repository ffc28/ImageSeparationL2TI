#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:02:35 2020

@author: fangchenfeng
"""

# This is a script for trying the mixture of image containing documents
from __future__ import division
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors
import scipy.linalg as la
import numpy as np
from scipy.linalg import sqrtm
import museval.metrics as mmetrics
from rdc import rdc
from scipy.fftpack import dct, idct

from sklearn.decomposition import FastICA

from skimage import data
from skimage.color import rgb2gray

n = 256

# load images and convert them
pic_set = 3

img1=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic1.png')
img2=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic2.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)

# flip
img2_gray = np.fliplr(img2_gray)
"""
plt.figure()
plt.imshow(img1_gray, cmap='gray')
plt.title("Ground truth 1")
plt.show

plt.figure()
plt.imshow(img2_gray, cmap='gray')
plt.title("Ground truth 2")
plt.show
"""
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
#mixing_matrix = np.array([[0.7, 0.6], [0.68, -0.57]])
mixing_matrix = np.array([[1, 0.7], [0.02, 1]])
# mixing_matrix = np.array([[1, 0.3], [0.5, 1]])

X = np.dot(mixing_matrix, source)

R = np.dot(X, X.T)
W = la.sqrtm(np.linalg.inv(R))
X_whiten = np.dot(W, X)
(sdr_ref, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X_whiten))

print("Reference SDR is: ")
print(sdr_ref)

# DCT of the observation
X1 = np.reshape(X_whiten[0,:], (n,n))
X2 = np.reshape(X_whiten[1,:], (n,n))

X_dct1 = dct(X1)
X_dct2 = dct(X2)

X_dct = np.stack((X_dct1.flatten(), X_dct2.flatten()))

# try the sparsity separation
lambda_max = 0.1
lambda_final = 1e-2
max_it = 2000

# random initialisation
A = np.random.rand(2,2)
S_dct = X_dct

(S_dct, A) = utils.sparsity_sep(X_dct, A, S_dct, max_it, lambda_max, lambda_final, stop_criteria = 1e-6)  

s1_dct = np.reshape(S_dct[0,:], (n,n))
s2_dct = np.reshape(S_dct[1,:], (n,n))

s1 = idct(s1_dct)
s2 = idct(s2_dct)

"""
plt.figure()
plt.imshow(s1.T, cmap='gray')
plt.title("estimated 1")
plt.show

plt.figure()
plt.imshow(s2.T, cmap='gray')
plt.title("estimated 2")
plt.show
"""
Sest = np.stack((s1.flatten(), s2.flatten()))
#S[0,:] = S[0,:]+ms[:,0]
#S[1,:] = S[1,:]+ms[:,1] 
(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(Sest))    

#print(np.matmul(S,S.T))
print("SDR of the DCT-based method is: ")
print(sdr)

print('The SDR improvement is: ', np.mean(sdr) - np.mean(sdr_ref))
