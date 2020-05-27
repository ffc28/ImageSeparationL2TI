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

from skimage import data
from skimage.color import rgb2gray

# load images and convert them
img1=mpimg.imread('pic1.png')
img2=mpimg.imread('pic2.png')

img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

img1_gray_re = img1_gray[0:255,0:255]
img2_gray_re = img2_gray[0:255,0:255]

print(img1_gray_re.shape)
print(img2_gray_re.shape)

plt.figure()
plt.imshow(img1_gray_re, cmap='gray')
plt.show

plt.figure()
plt.imshow(img2_gray_re, cmap='gray')
plt.show

# We mix them here