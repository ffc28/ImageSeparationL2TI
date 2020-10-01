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

n = 256

# load images and convert them
img = mpimg.imread('f96a6f8603075d38_1024x1024.jpg')

print(img.shape)
start_row = 130
start_col = 250
img_re = img[start_row : start_row + n, start_col : start_col + n, :]

plt.figure()
plt.imshow(img_re)
plt.title("Ground truth")
plt.show


mpimg.imsave('./train_cat/set5_pic.png', img_re)
