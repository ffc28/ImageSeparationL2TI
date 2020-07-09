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
import numpy as np
from scipy.linalg import sqrtm
import museval.metrics as mmetrics
from rdc import rdc

from sklearn.decomposition import FastICA

from skimage import data
from skimage.color import rgb2gray

n = 256

# load images and convert them
pic_set = 8

img1=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')
img2=mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)

# flip
img2_gray_re = np.fliplr(img2_gray)
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

# create 2x(255*255) source matrix

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
np.random.seed(0)
#mixing_matrix = np.random.rand(2,2)
mixing_matrix = np.array([[1, 0.5], [0.5, 1]])

# X = source * mixing_matrix - The mixed images

X = np.matmul(source.T, mixing_matrix)
(sdr_ref, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X.T))
# mix = [[0.6992, 0.7275], [0.4784, 0.5548]] #or use the matrix from the paper
# X = np.matmul(source.T, mix)

X1 = X[:,0]
mx1 = np.mean(X1)
X1 = np.reshape(X1, (n,n))

#print(X1.min(), X1.max(), X1.mean())

X2 = X[:,1]
mx2 = np.mean(X2)
X2 = np.reshape(X2, (n,n))

mx = np.array([mx1, mx2])
#print(X2.min(), X2.max(), X2.mean())

#Show the mixed images
"""
plt.figure()
plt.imshow(X1.T, cmap='gray')
plt.title("Mixed image 1")
plt.show

plt.figure()
plt.imshow(X2.T, cmap='gray')
plt.title("Mixed image 2")
plt.show
"""
# FastICA algorithm
#X[:,0] = X[:,0] - mx1
#X[:,1] = X[:,1] - mx2
print("mean value of the observations are: ", mx1, mx2)
# ICA does whitening pre-processing automatiquely
ica = FastICA(n_components=2, fun = 'cube', max_iter = 2000)
source_estimated = ica.fit_transform(X)
mixing_estimated = ica.mixing_

#print(mx)
ms = np.dot(np.linalg.inv(mixing_estimated),mx)
#print(ms)

#source_estimated[:,0] = source_estimated[:,0]+ms[0]
#source_estimated[:,1] = source_estimated[:,1]+ms[1]

print("mixing matrix estimated by ICA is: ") 
print(mixing_estimated)

(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(source_estimated.T))

print("Reference SDR is: ")
print(sdr_ref)
print("SDR of ICA is: ")
print(sdr)
# Reshape the estimated source images (from vectors into matrices)
s1 = source_estimated[:,0]
s1 = np.reshape(s1, (n,n))

#print(s1.min(), s1.max(), s1.mean())

s2 = source_estimated[:,1]
s2 = np.reshape(s2, (n,n))

#print(s2.min(), s2.max(), s2.mean())

# Show estimated sources
"""
plt.figure()
plt.imshow(-s1.T, cmap='gray')
plt.title("Estimated source 1 with ICA")
plt.show

plt.figure()
plt.imshow(-s2.T, cmap='gray')
plt.title("Estimated source 2 with ICA")
plt.show()
"""
#print(np.matmul(source_estimated.T,source_estimated))

# try the sparsity separation
lambda_max = 0.002
lambda_final = 0.0001
max_it = 2000

# whitening pre-processing
"""
X = X.T
R = np.dot(X, X.T)
W = np.linalg.inv(sqrtm(R))
X = np.dot(W, X)
X = X.T
"""
A = np.random.rand(2,2)
S = X.T

(S, A) = utils.sparsity_sep(X, A, S, max_it, lambda_max, lambda_final)
ms = np.dot(np.linalg.inv(A),mx)   
 
#S = np.dot(np.linalg.inv(A), X.T)
#S[0,:] = S[0,:]+ms[:,0]
#S[1,:] = S[1,:]+ms[:,1] 
(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(S))    

s1 = S[0,:]
s1 = np.reshape(s1, (n,n))

#print(s1.min(), s1.max(), s1.mean())

s2 = S[1,:]
s2 = np.reshape(s2, (n,n))

#print(s2.min(), s2.max(), s2.mean())

# Show estimated sources
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

#print(np.matmul(S,S.T))
print("SDR of sparsity-based method is: ")
print(sdr)
