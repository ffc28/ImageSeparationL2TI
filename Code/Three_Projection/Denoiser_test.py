#Trying to use the dictionary learning
from __future__ import division
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np
import scipy.linalg as la
import pywt
from bm3d import bm3d
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
dict_components = 100

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

dico_recto = MiniBatchDictionaryLearning(n_components = dict_components, alpha = 0.7, n_iter = 400) #TODO:check with different parameters
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

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re


def Dic_proj_single(S1, S2, n_coeff):

    
    S1 = Dic_proj_recto(S1, n_coeff)
    S2 = Dic_proj_recto(S2, n_coeff)
    
    return S1, S2
    

def TV_proj(S1, S2, lambda_this):
    """
    This function does the TV projection
    """
    # TV denoising
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
    
     
    return S1, S2


def Wavelet_proj(S1, S2, lambda_this):
    """
    This function does the Wavelet projection
    """
    # Wavelet denoising
    S1 = denoise_wavelet(S1, sigma = lambda_this, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
    S2 = denoise_wavelet(S2, sigma = lambda_this, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=False)
     
    return S1, S2

def Nonlocal_proj(S1, S2, lambda_this):
    """
    This function does the Non local mean projection
    """
    patch_kw = dict(patch_size=8,      # 5x5 patches
                patch_distance=7,  # 13x13 search area
                multichannel=False)
    # Nonlocal mean denoising
    S1 = denoise_nl_means(S1, h=1.12*lambda_this, fast_mode=False, sigma = lambda_this, 
                           **patch_kw)
    S2 = denoise_nl_means(S2, h=1.12*lambda_this, fast_mode=False, sigma = lambda_this, 
                           **patch_kw)
     
    return S1, S2

def BM3D_proj(S1, S2, lambda_this):
    """
    This function does the BM3D projection
    """
    # BM3D denoising
    S1 = bm3d(S1, lambda_this)
    S2 = bm3d(S2, lambda_this)
     
    return S1, S2

def Image_diff(S, Se):
    
    return np.linalg.norm(S-Se, ord = 'fro')

## load the source here
pic_set = 4
img1=mpimg.imread('./train_building/set'+ str(pic_set) + '_pic.png')
img2=mpimg.imread('./train_cat/set'+ str(pic_set) + '_pic.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)

source1 = np.matrix(img1_gray)
source1 = source1.flatten('F') #column wise

source2 = np.matrix(img2_gray)
source2 = source2.flatten('F') #column wise

source1 = source1 - np.mean(source1)
source2 = source2 - np.mean(source2)

# I think I can do this
source1 = source1/np.linalg.norm(source1)
source2 = source2/np.linalg.norm(source2)

X = source1 + source2
X = X/np.linalg.norm(X)

img1 = np.reshape(source1, image_size).T
img2 = np.reshape(source2, image_size).T
X = np.reshape(X, image_size).T

# Different denoiser
sigma = 5e-3
#img1_de, X_de = Wavelet_proj(img1, X, sigma)
img1_de, X_de = BM3D_proj(img1, X, sigma)
img2_de, X_de = BM3D_proj(img2, X, sigma)
#img1_de, X_de = BM3D_proj(img1, X, sigma)

plt.figure()
plt.subplot(221)
plt.imshow(img2, cmap='gray')
plt.title('Source original')

plt.subplot(222)
plt.imshow(img2_de, cmap='gray')
plt.title('Source denoised')

plt.subplot(223)
plt.imshow(X, cmap='gray')
plt.title('Mixture original')

plt.subplot(224)
plt.imshow(X_de, cmap='gray')
plt.title('Mixture denoised')

print('The difference of source1 image is: ', Image_diff(img1, img1_de))
print('The difference of source2 image is: ', Image_diff(img2, img2_de))
print('The difference of mixture image is: ', Image_diff(X, X_de))
