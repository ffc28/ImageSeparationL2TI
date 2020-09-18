#Trying to use the dictionary learning
from __future__ import division
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np
import scipy.linalg as la
from skimage.color import rgb2gray
from skimage.util import view_as_windows
from sklearn.feature_extraction.image import extract_patches_2d, PatchExtractor
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from patchify import patchify, unpatchify
import museval.metrics as mmetrics
from rdc import rdc
from math import floor
from scipy.stats import kurtosis
from skimage.restoration import (denoise_wavelet, denoise_tv_chambolle, estimate_sigma, denoise_nl_means)

np.random.seed(1)
# load images and convert them
n = 256
m = 8 # Check other sizes
image_size = (n, n)
patch_size = (m, m)
step = 4

print('Learning the dictionary for recto images...')
patches_recto = []
for pic_set in np.arange(4): # I'm using all the images for the learning
    
    img_train = mpimg.imread('./train_building/set'+ str(pic_set + 1) + '_pic.png')
    img_train_gray = rgb2gray(img_train) # the value is between 0 and 1

#    Extract reference patches from the image

    patches = patchify(img_train_gray, patch_size, step)
    initial_patch_size = patches.shape
    patches = patches.reshape(-1, patch_size[0] * patch_size[1])
    
    patches_recto.append(patches)

# Change the size of patches
patches_recto = np.asarray(patches_recto)
patches_recto = patches_recto.reshape(-1, m*m)
# Do the normalisation here
patches_recto -= np.mean(patches_recto, axis=0) # remove the mean
patches_recto /= np.std(patches_recto, axis=0) # normalise each patch

dico_recto = MiniBatchDictionaryLearning(n_components = 100, alpha = 0.7, n_iter = 400) #TODO:check with different parameters
V_recto = dico_recto.fit(patches_recto).components_
"""
# plot the dictionary
plt.figure(figsize=(8, 6))
for i, comp in enumerate(V_recto[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Recto dictionary learned from patches')
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
"""

print('Learning the dictionary for verso images...')
patches_verso = []
for pic_set in np.arange(4):
    
    img_train = mpimg.imread('./train_nature/set'+ str(pic_set + 1) + '_pic.png')
    img_train_gray = rgb2gray(img_train) # the value is between 0 and 1
#    Extract reference patches from the image

    patches = patchify(img_train_gray, patch_size, step)
    initial_patch_size = patches.shape
    patches = patches.reshape(-1, patch_size[0] * patch_size[1])
    
    patches_verso.append(patches)

# Change the size of patches
patches_verso = np.asarray(patches_verso)
patches_verso = patches_verso.reshape(-1, m*m)
# Do the normalisation here
patches_verso -= np.mean(patches_verso, axis=0) # remove the mean
patches_verso /= np.std(patches_verso, axis=0) # normalise each patch

dico_verso = MiniBatchDictionaryLearning(n_components = 100, alpha = 0.7, n_iter = 400) #TODO:check with different parameters
V_verso = dico_verso.fit(patches_verso).components_
"""
# plot the dictionary
plt.figure(figsize=(8, 6))
for i, comp in enumerate(V_verso[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Verso dictionary learned from patches')
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
"""

print('The dictionary learning is done here')
#################################################################
#################################################################
def Dic_proj_recto(data, n_coef):
    """
    The dictionary projection method
    """
    data = patchify(data, patch_size, step)
    data = data.reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept 
    
    dico_recto.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
    code = dico_recto.transform(data)

    patch = np.dot(code, V_recto)
    patch += intercept

    patch = np.reshape(patch, initial_patch_size)
    # if we use threshold then we have this
    # patch -= patch.min()
    # patch /= patch.max()

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re

def Dic_proj_verso(data, n_coef):
    """
    The dictionary projection method
    """
    data = patchify(data, patch_size, step)
    data = data.reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept 

    dico_verso.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
    code = dico_verso.transform(data)

    patch = np.dot(code, V_verso)
    patch += intercept
    
    patch = np.reshape(patch, initial_patch_size)
    # if we use threshold then we have this
    # patch -= patch.min()
    # patch /= patch.max()

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re


## load the source here
pic_set = 5
img1=mpimg.imread('./train_building/set'+ str(pic_set) + '_pic.png')
img2=mpimg.imread('./train_nature/set'+ str(pic_set) + '_pic.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)

n_coef = 30
img1_gray_re = Dic_proj_recto(img1_gray, n_coef)
img1_diff = np.linalg.norm(img1_gray - img1_gray_re, ord = 'fro')
print('Recto image with Recto dict, the difference is: ', img1_diff)

plt.figure()
plt.subplot(121)
plt.imshow(img1_gray_re, cmap='gray')

plt.subplot(122)
plt.imshow(np.abs(img1_gray_re - img1_gray), cmap='gray')
plt.title("Recto image with Recto dict")
plt.show

####

img1_gray_re = Dic_proj_verso(img1_gray, n_coef)
img1_diff = np.linalg.norm(img1_gray - img1_gray_re, ord = 'fro')
print('Recto image with Verso dict, the difference is: ', img1_diff)

plt.figure()
plt.subplot(121)
plt.imshow(img1_gray_re, cmap='gray')

plt.subplot(122)
plt.imshow(np.abs(img1_gray_re - img1_gray), cmap='gray')
plt.title("Recto image with Verso dict")
plt.show

####

img2_gray_re = Dic_proj_recto(img2_gray, n_coef)
img2_diff = np.linalg.norm(img2_gray - img2_gray_re, ord = 'fro')
print('Verso image with Recto dict, the difference is: ', img2_diff)

plt.figure()
plt.subplot(121)
plt.imshow(img2_gray_re, cmap='gray')

plt.subplot(122)
plt.imshow(np.abs(img2_gray_re - img2_gray), cmap='gray')
plt.title("Verso image with Recto dict")
plt.show

####

img2_gray_re = Dic_proj_verso(img2_gray, n_coef)
img2_diff = np.linalg.norm(img2_gray - img2_gray_re, ord = 'fro')
print('Verso image with Verso dict, the difference is: ', img2_diff)

plt.figure()
plt.subplot(121)
plt.imshow(img2_gray_re, cmap='gray')

plt.subplot(122)
plt.imshow(np.abs(img2_gray_re - img2_gray), cmap='gray')
plt.title("Verso image with Verso dict")
plt.show