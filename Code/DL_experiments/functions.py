""" useful functions """

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import log10, sqrt 
from patchify import patchify, unpatchify
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.restoration import (denoise_wavelet, denoise_tv_chambolle, estimate_sigma, denoise_nl_means)
from time import time 
from sewar.full_ref import msssim
from skimage.metrics import structural_similarity as ssim
from bm3d import bm3d
from sewar.full_ref import vifp
import museval.metrics as mmetrics


initial_patch_size =[]
all_patches = []

def vif(sources, estimated, image_size, permutation):
    estimated_copy = np.copy(estimated)
    sources_copy = np.copy(sources)
    estimated1, estimated2 = unflatten(estimated_copy, image_size)
    source1, source2 = unflatten(sources_copy, image_size)
    if permutation:
        vif1 = vifp(source1, estimated2)
        vif2 = vifp(source2, estimated1)
    else:
        vif1 = vifp(source1, estimated1)
        vif2 = vifp(source2, estimated2)
    return vif1, vif2
    

def normalizing_for_ssim(input_data, matrix, permutation):
    data = np.copy(input_data)
    if permutation:
        data[0,: ] = data[0,: ]/ matrix[0,1]
        data[1,: ] = data[1,: ]/ matrix[1,0]
    else:
        data[0,: ] = data[0,: ]/ matrix[0,0]
        data[1,: ] = data[1,: ]/ matrix[1,1]
    return data


def SSIM(original_stacked, estimated_stacked, image_size, permutation, X_normalized, B):
    C = np.dot(estimated_stacked, X_normalized.T)
    D = np.dot(C, B)
    #(sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(original_stacked), estimated_stacked)
    #if np.array_equal(perm, [[1],[0]]):
    #    permutation = True
    source1, source2 = unflatten(original_stacked, image_size)
    estimated_stacked_normalized = normalizing_for_ssim(estimated_stacked, D, permutation)
    estimated1, estimated2 = unflatten(estimated_stacked_normalized, image_size)
    
    if permutation:
        ssim1 = ssim(source1, estimated2, data_range = estimated2.max()- estimated2.min())
        ssim2 = ssim(source2, estimated1, data_range = estimated1.max()- estimated1.min())
    else:
        ssim1 = ssim(source1, estimated1, data_range = estimated1.max()- estimated1.min())
        ssim2 = ssim(source2, estimated2, data_range = estimated2.max()- estimated2.min())
    return ssim1, ssim2

"""def SSIM(original_stacked, estimated_stacked, image_size):
    estimated1, estimated2 = unflatten(estimated_stacked, image_size)
    source1, source2 = unflatten(original_stacked, image_size)
    ssim1 = ssim(source1, estimated1, data_range = estimated1.max()- estimated1.min())
    ssim2 = ssim(source2, estimated2, data_range = estimated2.max()- estimated2.min())
    return ssim1, ssim2"""

def MSSSIM(original_stacked, estimated_stacked, image_size):
    estimated1, estimated2 = unflatten(estimated_stacked, image_size)
    source1, source2 = unflatten(original_stacked, image_size)
    msssim1 = msssim(source1, estimated1, MAX = 1)
    msssim2 = msssim(source2, estimated2, MAX = 1)
    return msssim1, msssim2


def PSNR(original, compressed): 
    """
    Function that computes the PSNR of an image

    Parameters
    ----------
    original: matrix, the ground truth image
    compressed: matrix, the 'distorted' image
    Return
    -----------
    psnr:  float, the PSNR of the image
    """

    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 1 #255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def SNR(a, axis=0, ddof=0):
    """ Function that computes the SNR of image """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def col_norm_proj(A):
    """
    Here we normalise each column of the input matrix A
    """
    A[:,0] = A[:,0]/np.linalg.norm(A[:,0])
    A[:,1] = A[:,1]/np.linalg.norm(A[:,1])
    
    return A

def dist_diag(A):
    """
    This function calculates the distance of a matrix from a diagnal matrix
    """
    A = col_norm_proj(A)
    ratio =(np.abs(A[0, 1]) + np.abs(A[1, 0]))/(np.abs(A[0,0]) + np.abs(A[1,1]))
    if np.abs(A[0, 0]) < np.abs(A[0, 1]):
        ratio = 1/ratio
    
    return ratio

def unflatten(stacked_images, image_size):
    """
    Function that returns to original size the two flattened and stacked images

    Parameters
    ----------
    stacked_images: (2, int*int) array, of the two flattened and stacked images
    image_size: (int, int), the original size of the images before being flattened
    Return
    -----------
    image1: n x n array, the first image reconstructed
    image2: n x n array, the second image reconstructed
    """
    image1 = np.reshape(stacked_images[0, :], image_size)
    image2 = np.reshape(stacked_images[1, :], image_size)
    return image1, image2

def flatten(image1, image2):
    """
    Function that flattens the images row-wise and stackes them on top of each other
    Parameters
    ----------
    image1: n x n array, the first image 
    image2: n x n array, the second image 
    Return
    -----------
    stacked_images: (2, int*int) array, of the two flattened and stacked images

    """
    assert np.shape(image1) == np.shape(image2), 'the shape of images must match'
    stacked_images = np.stack((image1.flatten(), image2.flatten()))
    return stacked_images



def learn_dictionary(patch_size, step, plot_dictionary = False, *args):
    """
    Function that normalizes the patches, learns a dictionary on them and plots it
    
    Parameters
    ----------
    patch_size: (int, int), the size of the patches to be extracted from the images
    step: int, the step of the moving patches, overlap of patches = patch_size - step
    plot_dictionary: boolean, False by default, plots the dictionary if True

    Return
    -----------
    dico: a dictionary (a set of atoms) that can best be used to represent data using a sparse code
    V: array, [n_components, n_features], the components of the fitted data
    """

    argCount = len(args)
    assert argCount > 0, 'no image to extract the patches from'
    
    global initial_patch_size, all_patches
    print(f'Extracting reference patches from {argCount} images...')
    t0 = time()
    for image in args:
        patches = patchify(image, patch_size, step)
        initial_patch_size = patches.shape
        patches = patches.reshape(-1, patch_size[0] * patch_size[1])
        all_patches.append(patches) 
    dt = time() - t0 
    print('done in %.2fs.' % dt)
    #return all_patches

    all_patches = np.reshape(all_patches, (-1, patch_size[0]*patch_size[1]))
    all_patches -= np.mean(all_patches, axis=0) # remove the mean
    all_patches /= np.std(all_patches, axis=0) # normalize each patch

    print('Learning the dictionary...')
    t0 = time()
    dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=400)
    V = dico.fit(all_patches).components_
    dt = time() - t0
    print('done in %.2fs.' % dt)

    if plot_dictionary == True:
        # plotting the dictionary
        plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(V[:100]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                    interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('Dictionary learned from patches')
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    return dico, V

def dictionary_projection(image, dico, V, n_coef, patch_size, step):
    """
    Function that does the dictionary projection on the input image
    
    Parameters
    ----------
    image: numpy.matrix, the distorted image
    dico: a dictionary (a set of atoms) that can best be used to represent data using a sparse code
    V: array, [n_components, n_features], the components of the fitted data
    n_coef = int, the number of non zero atoms
    patch_size: (int, int), the size of the patches to be extracted from the image
    step: int, the step of the moving patches, overlap of patches = patch_size - step
    
    Return
    -----------
    reconstructed_image: numpy.matrix, the estimated image reconstructed from the learned patches
    """

    global initial_patch_size

    data = patchify(image, patch_size, step)
    data = data.reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept 

    dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
    code = dico.transform(data)

    patch = np.dot(code, V)
    patch += intercept
    
    patch = np.reshape(patch, initial_patch_size)
    reconstructed_image = unpatchify(np.asarray(patch), np.shape(image))

    return reconstructed_image




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
    
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_verso(S2, n_coeff, alpha)
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def Dic_proj_single(S, n_coeff, alpha):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_recto(S2, n_coeff, alpha)
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S
    

def data_projection(X,S):
    """
    This functions does the data projection with the equation X = AS
    """
    A = np.dot(S, X.T)
    S = np.dot(A,X)
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
    return np.dot(W, S), W

"""def TV_proj(S, image_size, lambda_this):
    
    #This function does the TV projection
    
    S1, S2 = unflatten(S, image_size)
    
    # TV denoising
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
     
    return flatten(S1, S2)"""

n = 256
image_size = (n, n)
def TV_proj(S, lambda_this):
    #This function does the TV projection
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # TV denoising
    S1 = denoise_tv_chambolle(S1, weight = lambda_this, multichannel=False)
    S2 = denoise_tv_chambolle(S2, weight = lambda_this, multichannel=False)
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S

def bm3d_proj(S, lambda_this):
    """
    This function does the TV projection
    """
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    # TV denoising
    S1 = bm3d(S1, lambda_this)
    S2 = bm3d(S2, lambda_this)
        
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
        
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S