#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:02:35 2020

@author: fangchenfeng
"""

# This is a script for trying the mixture of image containing documents
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from munkres import Munkres

from sklearn.decomposition import FastICA
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

from skimage.color import rgb2gray
import museval.metrics as mmetrics
from time import time
from skimage.util import view_as_windows, montage
from patchify import patchify, unpatchify

n = 256
m = 64
image_size = (n, n)
patch_size = (m, m)
step = 32 # <m for overlapping patches

# load images and convert them
img1=mpimg.imread('pic1.png')
img2=mpimg.imread('pic2.png')

img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

img1_gray_re = img1_gray[0:n,0:n]
img2_gray_re = img2_gray[0:n,0:n]

img2_gray_re = np.fliplr(img2_gray_re)

plt.figure()
plt.imshow(img1_gray_re, cmap='gray')
plt.title("Ground truth 1")
plt.show

plt.figure()
plt.imshow(img2_gray_re, cmap='gray')
plt.title("Ground truth 2")
plt.show


# We mix the images here
# create 2x(256*256) source matrix

source1 = np.matrix(img1_gray_re)
source1 = source1.flatten('F') #column wise

source2 = np.matrix(img2_gray_re)
source2 = source2.flatten('F') #column wise

source1 = source1 - np.mean(source1)
source2 = source2 - np.mean(source2)

source1 = source1/np.linalg.norm(source1)
source2 = source2/np.linalg.norm(source2)

# Extract reference patches

ref_patches1 = patchify(np.reshape(source1, image_size).T, patch_size, step)
initial_size1 = ref_patches1.shape
ref_patches1 = ref_patches1.reshape((-1, m, m))

ref_patches2 = patchify(np.reshape(source2, image_size).T, patch_size, step)
initial_size2 = ref_patches2.shape
ref_patches2 = ref_patches2.reshape((-1, m, m))

source = np.stack((source1, source2))

mixing_matrix = [[1, 0.5],[0.5, 1]] 
# X = source * mixing_matrix - The mixed images

X = np.matmul(mixing_matrix, source)
(sdr_ref, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X))
print("***")
print(sdr_ref)
print("***")

# reconstructing the mixed images
X1 = X[0,:]
X1 = np.reshape(X1, (n,n))

X2 = X[1,:]
X2 = np.reshape(X2, (n,n))

#extracting patches from the mixed images
mix_patches1 = patchify(X1.T, patch_size, step)
mix_patches1 = mix_patches1.reshape((-1, m, m))

mix_patches2 = patchify(X2.T, patch_size, step)
mix_patches2 = mix_patches2.reshape((-1, m, m))

# FastICA algorithm on the patches
estimated_patches1 = []
estimated_patches2 = []
for mixp1, mixp2, refp1, refp2 in zip(mix_patches1, mix_patches2, ref_patches1, ref_patches2):
    # remove the mean of the two mixtures
    mixp1 = mixp1 - np.mean(mixp1)
    mixp2 = mixp2 - np.mean(mixp2)
    # get the mean value of the original patches
    m1 = np.mean(refp1)
    m2 = np.mean(refp2)
    n1 = np.linalg.norm(refp1, 'fro')
    n2 = np.linalg.norm(refp2, 'fro')
    
    mix_p = np.stack((mixp1.flatten(), mixp2.flatten()))

    ref_p = np.stack((refp1.flatten(), refp2.flatten()))
    
    ica = FastICA(n_components=2, fun = 'cube', max_iter = 20000)#, whiten=False)#, fun='cube')
    source_estimated = ica.fit_transform(mix_p.T)
    mixing_estimated = ica.mixing_

    # Prepare for the permutation here
    source_estimated = source_estimated.T
    # remove the original mean   
    ref_p[0,:] = ref_p[0,:] - np.mean(ref_p[0,:])
    ref_p[1,:] = ref_p[1,:] - np.mean(ref_p[1,:])   
    
    (sdr_ref, sir, sar, perm) = mmetrics.bss_eval_sources(ref_p, mix_p)
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(ref_p, source_estimated)
    print(sdr_ref)
    print(sdr)
    print(perm)

    # permute here  
    source_estimated = source_estimated[perm,:]
    mixing_estimated = mixing_estimated[:,perm]

    # change the sign here
    if mixing_estimated[0,0]<0:
        source_estimated[0,:] = - source_estimated[0,:]
        
    if mixing_estimated[1,1]<0:
        source_estimated[1,:] = - source_estimated[1,:]
    
    source_estimated[0,:] = n1*source_estimated[0,:]+m1
    source_estimated[1,:] = n2*source_estimated[1,:]+m2
    
    if (np.mean(sdr) - np.mean(sdr_ref)) < -100:
        print("Using original patch")
        source_estimated[0,:] = refp1.flatten()
        source_estimated[1,:] = refp2.flatten()
    # get back to images
    
    print("******")
    s1 = source_estimated[0,:]
    s1 = np.reshape(s1, patch_size)

    s2 = source_estimated[1,:]
    s2 = np.reshape(s2, patch_size)
    
    estimated_patches1.append(s1)
    estimated_patches2.append(s2)

# reconstruct the estimated sources

estimated_patches1 = np.reshape(estimated_patches1, initial_size1)
estimated_patches2 = np.reshape(estimated_patches2, initial_size2)

es1 = unpatchify(np.asarray(estimated_patches1), image_size)
es2 = unpatchify(np.asarray(estimated_patches2), image_size)

# Show estimated sources

plt.figure()
plt.imshow(es1, cmap='gray')
plt.title("Estimated source 1")
plt.show

plt.figure()
plt.imshow(es2, cmap='gray')
plt.title("Estimated source 2")
plt.show()

est1 = es1.flatten('F') #column wise
est2 = es2.flatten('F')

est = np.stack((est1, est2))

(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(est))
print(sdr)