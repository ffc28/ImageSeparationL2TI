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

from skimage import data
from skimage.color import rgb2gray

n = 255

# load images and convert them
img1=mpimg.imread('pic1.png')
img2=mpimg.imread('pic2.png')

img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

img1_gray_re = img1_gray[0:n,0:n]
img2_gray_re = img2_gray[0:n,0:n]

#print(img1_gray_re.min(), img1_gray_re.max())
#print(img2_gray_re.min(), img2_gray_re.max())

#print(img1_gray_re.shape)
#print(img2_gray_re.shape)

plt.figure()
plt.imshow(img1_gray_re, cmap='gray')
plt.title("Ground truth 1")
plt.show

plt.figure()
plt.imshow(img2_gray_re, cmap='gray')
plt.title("Ground truth 2")
plt.show

# We mix them here

# create 2x(255*255) source matrix

source1 = np.matrix(img1_gray_re)
source1 = source1.flatten('F') #column wise

source2 = np.matrix(img2_gray_re)
source2 = source2.flatten('F') #column wise

source = np.stack((source1, source2))

# randomly generated mixing matrix
np.random.seed(0)
mixing_matrix = np.random.rand(2,2)
print("mixing_matrix = ", mixing_matrix)


# X = source * mixing_matrix - The mixed images

X = np.matmul(source.T, mixing_matrix)

# mix = [[0.6992, 0.7275], [0.4784, 0.5548]] #or use the matrix from the paper
# X = np.matmul(source.T, mix)

X1 = X[:,0]
X1 = np.reshape(X1, (n,n))

#print(X1.min(), X1.max(), X1.mean())

X2 = X[:,0]
X2 = np.reshape(X2, (n,n))

#print(X2.min(), X2.max(), X2.mean())


#Show the mixed images

plt.figure()
plt.imshow(X1.T, cmap='gray')
plt.title("Mixed image 1")
plt.show

plt.figure()
plt.imshow(X2.T, cmap='gray')
plt.title("Mixed image 2")
plt.show

# FastICA algorithm

ica = FastICA(n_components=2)
source_estimated = ica.fit_transform(X)
mixing_estimated = ica.mixing_

print("mixing matrix estimated = ", mixing_estimated)

# Reshape the estimated source images (from vectors into matrices)
s1 = source_estimated[:,0]
s1 = np.reshape(s1, (n,n))

#print(s1.min(), s1.max(), s1.mean())

s2 = source_estimated[:,1]
s2 = np.reshape(s2, (n,n))

#print(s2.min(), s2.max(), s2.mean())

# Show estimated sources

plt.figure()
plt.imshow(s1.T, cmap='gray')
plt.title("Estimated source 1")
plt.show

plt.figure()
plt.imshow(s2.T, cmap='gray')
plt.title("Estimated source 2")
plt.show()

