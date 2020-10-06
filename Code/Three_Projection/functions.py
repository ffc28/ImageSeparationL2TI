""" useful functions """

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import log10, sqrt 
from patchify import patchify, unpatchify
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.restoration import (denoise_wavelet, denoise_tv_chambolle, estimate_sigma, denoise_nl_means)
from time import time 

initial_patch_size =[]
all_patches = []


def PSNR(original, compressed): 
    """ Function that computes the PSNR of image """
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
    return np.dot(W, S)

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


