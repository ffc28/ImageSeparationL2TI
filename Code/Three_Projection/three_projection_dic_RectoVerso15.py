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
for pic_set in np.arange(30): # I'm using all the images for the learning
    
    img_train = mpimg.imread('./train_images/set'+ str(pic_set + 1) + '_pic.png')
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

print('Learning the dictionary for verso images...')
patches_verso = []
for pic_set in np.arange(30):
    
    img_train = mpimg.imread('./train_images/set'+ str(pic_set + 1) + '_pic.png')
    img_train_gray = rgb2gray(img_train) # the value is between 0 and 1
    img_train_gray = np.fliplr(img_train_gray) # We do the flip here
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

dico_verso = MiniBatchDictionaryLearning(n_components = dict_components, alpha = 0.7, n_iter = 400) #TODO:check with different parameters
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
def Dic_proj_recto(data, n_coef, alpha):
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

def Dic_proj_verso(data, n_coef, alpha):
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


def Dic_proj_double(S, n_coeff, alpha):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = S1.T
    S2 = S2.T
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(S1, cmap='gray')
    plt.title("Estimated Source Before")
    plt.show
    """
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_verso(S2, n_coeff, alpha)
    
    S1 = S1.T
    S2 = S2.T
    """
    plt.subplot(122)
    plt.imshow(S1, cmap='gray')
    plt.title("Estimated Source after")
    plt.show
    """
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S


def Dic_proj_single(S, n_coeff, alpha):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = S1.T
    S2 = S2.T
    
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_recto(S2, n_coeff, alpha)
    
    S1 = S1.T
    S2 = S2.T
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def Dic_proj_single_flip(S, n_coeff, alpha):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = S1.T
    S2 = S2.T
    S2 = np.fliplr(S2)
    
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_recto(S2, n_coeff, alpha)
    
    S2 = np.fliplr(S2)
    S1 = S1.T
    S2 = S2.T
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S
    

def data_projection(X,S):
    """
    This functions does the data projection with the equation X = AS
    """
    A = np.dot(S, X.T)
    S = np.dot(A, X)
    R = np.dot(A, A.T)
    return np.dot(np.linalg.inv(R),S)

def get_demix(X, S):
    
    return np.dot(S, X.T)

def whiten_projection(S):
    """
    This function does the whitening projection with PCA
    """
    R = np.dot(S, S.T)
    W = la.sqrtm(np.linalg.inv(R))
    return np.dot(W, S)

def col_norm_proj(A):
    """
    Here we normalise each column of the input matrix A

    """
    A[:,0] = A[:,0]/np.linalg.norm(A[:,0])
    A[:,1] = A[:,1]/np.linalg.norm(A[:,1])
    
    return A

def TV_proj(S, lambda_this):
    """
    This function does the TV projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # TV denoising
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S


def Wavelet_proj(S, lambda_this):
    """
    This function does the Wavelet projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # Wavelet denoising
    S1 = denoise_wavelet(S1, sigma = lambda_this, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
    S2 = denoise_wavelet(S2, sigma = lambda_this, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=False)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def Nonlocal_proj(S, lambda_this):
    """
    This function does the Non local mean projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    patch_kw = dict(patch_size=8,      # 5x5 patches
                patch_distance=7,  # 13x13 search area
                multichannel=False)
    # Nonlocal mean denoising
    S1 = denoise_nl_means(S1, h=1.12*lambda_this, fast_mode=False, sigma = lambda_this, 
                           **patch_kw)
    S2 = denoise_nl_means(S2, h=1.12*lambda_this, fast_mode=False, sigma = lambda_this, 
                           **patch_kw)
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def BM3D_proj(S, lambda_this):
    """
    This function does the BM3D projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # BM3D denoising
    S1 = bm3d(S1, lambda_this)
    S2 = bm3d(S2, lambda_this)
        
    S[0,:] = np.reshape(S1, (1, image_size[0]*image_size[1]))
    S[1,:] = np.reshape(S2, (1, image_size[0]*image_size[1]))
     
    return S

def soft_proximal(S, value):
    """
    This function does the soft-thresholding
    """
    return pywt.threshold(S, value, 'soft')

def dist_diag(A):
    """
    This function calculates the distance of a matrix from a diagnal matrix
    """
    A = col_norm_proj(A)
    ratio =(np.abs(A[0, 1]) + np.abs(A[1, 0]))/(np.abs(A[0,0]) + np.abs(A[1,1]))
    if np.abs(A[0, 0]) < np.abs(A[0, 1]):
        ratio = 1/ratio
    
    return ratio

def show_image(S):
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = S1.T
    S2 = S2.T
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(S1, cmap='gray')
    plt.title("Source 1")
    plt.show
    
    plt.subplot(122)
    plt.imshow(S2, cmap='gray')
    plt.title("Source 2")
    plt.show

## load the source here
pic_set = 2
img1=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic1.png')
img2=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic2.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)
# Mixing process here
#img2_gray = np.fliplr(img2_gray)

source1 = np.matrix(img1_gray)
source1 = source1.flatten('F') #column wise

source2 = np.matrix(img2_gray)
source2 = source2.flatten('F') #column wise

source1 = source1 - np.mean(source1)
source2 = source2 - np.mean(source2)

# I think I can do this
source1 = source1/np.linalg.norm(source1)
source2 = source2/np.linalg.norm(source2)

print("rdc = ", rdc(source1.T,source2.T))
source = np.stack((source1, source2))

### Now I projection the sources into the dictionaries ###
# source = Dic_proj_single_flip(source, 2, 1e-6)  # Now the sources are really composed of two atoms

k1 = kurtosis(np.squeeze(np.asarray(source[0,:])))
k2 = kurtosis(np.squeeze(np.asarray(source[1,:])))
print('Kuortosis of the original sources is: ', np.abs(k1) + np.abs(k2))

print('Covariance matrix is: ')
print(np.matmul(source, source.T))

mixing_matrix = np.array([[0.7, 0.6], [0.68, -0.57]])
#mixing_matrix = np.array([[1, 0.7], [0.02, 1]])
# mixing_matrix = np.array([[0.6, 0.4], [0.3, 0.7]])
print("condition number is: ", np.linalg.cond(mixing_matrix))
mixing_matrix_unitary = whiten_projection(mixing_matrix)
print("The unitary mixing matrix is: ")
print(mixing_matrix_unitary)
# mixing_matrix = np.array([[0.8488177, 0.17889592], [0.05436321, 0.36153845]])
# mixing_matrix = np.array([[1, 0.9], [0.02, 1]])

# X = source * mixing_matrix - The mixed images

X = np.dot(mixing_matrix, source)
"""
x1 = X[0,:]
x1 = np.reshape(x1, (n,n))
plt.figure()
plt.imshow(x1.T, cmap='gray')
plt.title("Mixture1 before whitening")
plt.show
"""
# Here begins the algorithm
# whitening processing. It's important

R = np.dot(X, X.T)
W = la.sqrtm(np.linalg.inv(R))
X = np.dot(W, X)

mixing_new = np.dot(W, mixing_matrix)
print(mixing_new)
x1 = X[0,:]
x1 = np.reshape(x1, (n,n))
"""
plt.figure()
plt.imshow(x1.T, cmap='gray')
plt.title("Mixture1 after whitening")
plt.show
"""
# mixing_matrix_norm = np.dot(W, mixing_matrix)
# mixing_matrix_norm[:,0] = mixing_matrix_norm[:,0]/np.linalg.norm(mixing_matrix_norm[:,0])
# mixing_matrix_norm[:,1] = mixing_matrix_norm[:,1]/np.linalg.norm(mixing_matrix_norm[:,1])
(sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X))
# mix = [[0.6992, 0.7275], [0.4784, 0.5548]] #or use the matrix from the paper
print('The mean value of the reference SDR is: ', np.mean(sdr_ref))
print('The permutation is: ', perm)

max_it = 50
#Se = np.random.randn(2, n*n) 
Se = np.copy(X)  
SDR_it = np.zeros((2, max_it))
dist_it = np.zeros((1, max_it))

num_coeff_begin = 1
num_coeff_final = 5
num_coeff_v = np.floor(np.linspace(num_coeff_begin, num_coeff_final, max_it))
sigma = 1e-1
sigma_final = 1e-1
sigma_v = np.logspace(np.log10(sigma), np.log10(sigma_final), max_it)
Se_old = np.copy(Se)
WW_old = np.random.randn(2, 2)
for it in np.arange(max_it):
    # print(it)  
    # we performe three projections
    # Se = whiten_projection(soft_proximal(data_projection(X, Se),lambda_v[it]))
    # Se = whiten_projection(Dic_proj_single(data_projection(X,Se), num_coeff_v[it]))
    # 1. denoising (single or double dictionary)
    # Se = Dic_proj_double(Se, num_coeff_v[it], sigma_v[it])
    #Se = Dic_proj_single(Se, num_coeff_v[it], sigma_v[it])
    #Se = Dic_proj_single_flip(Se, num_coeff_v[it], sigma_v[it])
    #Se = Nonlocal_proj(Se, sigma_v[it]) # For nonlocal sigma = 1e-3
    #Se = BM3D_proj(Se, sigma_v[it]) # For BM3D sigma = 1e-2 and sigma_finam = 1e-4
    Se = TV_proj(Se, sigma_v[it])
    # 2. get demixing matrix
    WW = get_demix(X, Se)
    # 3. whiten the demix matrix
    WW = whiten_projection(WW)
    #WW = np.linalg.inv(WW)
    #WW = col_norm_proj(WW)
    #WW = np.linalg.inv(WW)
    # 4. get the new source
    Se = np.dot(WW, X)
    # print(WW)

    # cost_it[0,it] = np.linalg.norm(X - np.dot(np.dot(X, Se.T), Se), ord = 'fro')
    if np.linalg.norm(Se - Se_old, ord = 'fro') < 1e-8:
        print('Dict demix convergence reached because of the source')
        print('The real number of iteration is', it)
        break
    
    if np.linalg.norm(WW - WW_old, ord = 'fro') < 1e-8:
        print('Dict demix convergence reached because of the mixing matrix')
        print('The real number of iteration is', it)
        break
    
    Se_old = np.copy(Se) 
    WW_old = np.copy(WW)
    # matrix evaluation
    estimated_mix = np.dot(WW, mixing_new)
    dist_it[0, it] = dist_diag(estimated_mix)
    
    #if math.floor(it/50)*50 == it:
        #(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se)
        # print(sdr)
        #SDR_it[0, it] = sdr[0]
        #SDR_it[1, it] = sdr[1]
     #   show_image(Se)
    # 
    #(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se_inv)

    # SDR_it[:, it] = np.squeeze(sdr)
# Se = np.dot(WW, X)    
(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se)

# Se = Dic_proj_single(Se, num_coeff_v[it], sigma)
print('The mean value of the SDR is: ', np.mean(sdr))
print('The SDR improvement is: ', np.mean(sdr) - np.mean(sdr_ref))
print('The final distance to a diagonal matrix is: ', dist_diag(estimated_mix))
"""
plt.figure()
plt.subplot(211)
plt.plot(cost_it[0,:])
plt.title('Cost for iterations')
plt.grid()
plt.show

plt.subplot(212)
plt.plot(np.mean(SDR_it, axis = 0))
plt.title('SDR for iterations Dictionary learning')
plt.grid()
plt.show

s1 = Se[0,:]
s1 = np.reshape(s1, (n,n))

s2 = Se[1,:]
s2 = np.reshape(s2, (n,n))

plt.figure()
plt.imshow(s1.T, cmap='gray')
plt.title("Estimated source 1 with Sparse")
plt.show

plt.figure()
plt.imshow(s2.T, cmap='gray')
plt.title("Estimated source 2 with Sparse")
plt.show()




plt.subplot(221)
plt.plot(np.mean(SDR_it, axis = 0))
plt.title('Average SDR for iterations')
#plt.xlabel('iterations')
plt.ylabel('SDR (dB)')
plt.grid()
"""
plt.figure()
plt.plot(dist_it[0,:])
plt.title('Average matrix measure for iterations')
plt.xlabel('iterations')
plt.grid()
plt.show
"""
# plt.figure()
plt.subplot(223)
plt.plot(SDR_it[0,:])
plt.title('SDR1')
plt.ylabel('SDR (dB)')
plt.grid()

plt.subplot(224)
plt.plot(SDR_it[1,:])
plt.title('SDR2')
plt.ylabel('SDR (dB)')
plt.grid()
"""