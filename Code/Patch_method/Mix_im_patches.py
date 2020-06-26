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

from sklearn.decomposition import FastICA

from skimage.color import rgb2gray
import mir_eval
from time import time
from skimage.util import view_as_windows, montage

n = 256
m = 64
patch_size = (m, m)
step = m # <m for overlapping patches

# load images and convert them
img1=mpimg.imread('flower1.jpg')
img2=mpimg.imread('flower2.jpg')

img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

img1_gray_re = img1_gray[0:n,0:n]
img2_gray_re = img2_gray[0:n,0:n]

# show the ground truth images
plt.figure()
plt.imshow(img1_gray_re, cmap='gray')
plt.title("Ground truth 1")
plt.show

plt.figure()
plt.imshow(img2_gray_re, cmap='gray')
plt.title("Ground truth 2")
plt.show


# Extract reference patches

ref_patches1 = view_as_windows(img1_gray_re, patch_size, step) #extract patches
ref_patches1 = ref_patches1.reshape((-1, m, m))

ref_patches2 = view_as_windows(img2_gray_re, patch_size, step) #extract patches
ref_patches2 = ref_patches2.reshape((-1, m, m))


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

source = np.stack((source1, source2))

mixing_matrix = [[1, 0.5],[0.5, 1]] 
# randomly generated mixing matrix
#np.random.seed(0)
#np.random.rand(2,2)
print("mixing_matrix = ", mixing_matrix)


# X = source * mixing_matrix - The mixed images

X = np.matmul(mixing_matrix, source)

# reconstructing the mixed images

X1 = X[0,:]
X1 = np.reshape(X1, (n,n))

X2 = X[1,:]
X2 = np.reshape(X2, (n,n))

#Show the mixed images

plt.figure()
plt.imshow(X1.T, cmap='gray')
plt.title("Mixed image 1")
plt.show

plt.figure()
plt.imshow(X2.T, cmap='gray')
plt.title("Mixed image 2")
plt.show

#extracting patches from the mixed images

mix_patches1 = view_as_windows(X1.T, patch_size, step) #extract patches
mix_patches1 = mix_patches1.reshape((-1, m, m))


mix_patches2 = view_as_windows(X2.T, patch_size, step) #extract patches
mix_patches2 = mix_patches2.reshape((-1, m, m))
print(np.linalg.norm(mix_patches1))

# FastICA algorithm on the patches
estimated_patches1 = []
estimated_patches2 = []
import pdb; pdb.set_trace()

for p1, p2, p3, p4 in zip(mix_patches1, mix_patches2, ref_patches1, ref_patches2):
    p1 = p1.flatten()
    p2 = p2.flatten()

    mean1 = np.mean(p1)
    mean2 = np.mean(p2)
    
    p1 = p1 - mean1
    p2 = p2 - mean2

    mix_p = np.stack((p1, p2))

    p3 = p3.flatten()/np.linalg.norm(p3)
    p4 = p4.flatten()/np.linalg.norm(p4)
    
    mean3 = np.mean(p3)
    mean4 = np.mean(p4)
    #print("p3 norm = ", np.linalg.norm(p3))
    #print("p4 norm = ", np.linalg.norm(p4))
    #print("mean3, 4 =", mean3, mean4)
    ref_p = np.stack((p3, p4))
    
    ica = FastICA(n_components=2)#, whiten=False)#, fun='cube')
    source_estimated = ica.fit_transform(mix_p.T)
    mixing_estimated = ica.mixing_

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(ref_p, source_estimated.T)

    s1 = source_estimated[:,0]
    #print("s1 norm = ", np.linalg.norm(s1))
    s1 = np.reshape(s1, patch_size)

    s2 = source_estimated[:,1]
    s2 = np.reshape(s2, patch_size)

    #check permutation ambiguity
    if (np.array_equal(perm, [0, 1])):
        s2 = s2 + mean3
        s1 = s1 + mean4

        estimated_patches1.append(s2)
        estimated_patches2.append(s1)
    else:
        s2 = s2 + mean4
        s1 = s1 + mean3 

        estimated_patches1.append(s1)
        estimated_patches2.append(s2)

# reconstruct the estimated sources

es1 = montage(estimated_patches1)
es2 = montage(estimated_patches2)

# Show estimated sources

plt.figure()
plt.imshow(es1, cmap='gray')
plt.title("Estimated source 1")
plt.show

plt.figure()
plt.imshow(es2, cmap='gray')
plt.title("Estimated source 2")
plt.show()


#estimated_sources = source_estimated.T
#reference_sources = np.array(source)

#(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)

#print("bss_eval_sources sdr, sir, sar, perm =", sdr, sir, sar, perm)
