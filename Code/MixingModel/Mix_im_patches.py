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

# We mix them here
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

# reconstructing the images

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
m = 64
patch_size = (m, m)
step = m # <m for overlapping patches

patches1 = view_as_windows(X1.T, patch_size, step) #extract patches
patches1 = patches1.reshape((-1, m, m))

patches2 = view_as_windows(X2.T, patch_size, step) #extract patches
patches2 = patches2.reshape((-1, m, m))

# FastICA algorithm on the patches
estimated_patches1 = []
estimated_patches2 = []
for p1, p2 in zip(patches1, patches2):
    p1 = p1.flatten()
    p2 = p2.flatten()

    p = np.stack((p1, p2))
    
    ica = FastICA(n_components=2)#, whiten=False)#, fun='cube')
    source_estimated = ica.fit_transform(p.T)
    mixing_estimated = ica.mixing_

    s1 = source_estimated[:,0]
    s1 = np.reshape(s1, patch_size)
    estimated_patches1.append(s1)

    s2 = source_estimated[:,0]
    s2 = np.reshape(s2, patch_size)
    estimated_patches2.append(s2)


print(np.shape(estimated_patches1))

print("mixing matrix estimated = ", mixing_estimated)
"""
#reshaping
X1 = X[0,:]
X1 = np.reshape(X1, (int(n/m),n*m))

reconstruct_patches = view_as_windows(X1, (1,m*m), step = (1,m*m)) #extract patches

X1 = [np.reshape(element, (int(n/m), m, m)) for element in reconstruct_patches]
X1 = np.asarray(X1)
X1 = np.reshape(X1, (-1, m, m))
X1 = montage(X1)

X2 = X[1,:]
X2 = np.reshape(X2, (n,n))



# FastICA algorithm

ica = FastICA(n_components=2)#, fun='cube')
source_estimated = ica.fit_transform(X.T)
mixing_estimated = ica.mixing_

print("mixing matrix estimated = ", mixing_estimated)

# Reshape the estimated source images (from vectors into matrices)
s1 = source_estimated[:,0]
s1 = np.reshape(s1, (n,n))

#print(s1.min(), s1.max(), s1.mean())

s2 = source_estimated[:,1]
s2 = np.reshape(s2, (n,n))

#print(s2.min(), s2.max(), s2.mean())
"""




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
