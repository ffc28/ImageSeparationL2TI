#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:00:10 2020

@author: fangchenfeng
"""

# This is a script for trying the mixture of image containing documents
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import museval.metrics as mmetrics
from rdc import rdc

from skimage import data
from skimage.color import rgb2gray

n = 256

# load images and convert them
img1=mpimg.imread('29.jpg')
img2=mpimg.imread('29.jpg')


img1_re = img1[550:550+n,200:200+n,:]
img2_re = img2[450:450+n,200:200+n,:]

plt.figure()
plt.imshow(img1_re)
plt.title("Ground truth 1")
plt.show

plt.figure()
plt.imshow(img2_re)
plt.title("Ground truth 2")
plt.show

mpimg.imsave('./images/set6_pic1.png', img1_re)
mpimg.imsave('./images/set6_pic2.png', img2_re)

