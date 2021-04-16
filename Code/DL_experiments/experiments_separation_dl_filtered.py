""" Separation experiments for DL on pre-filtered images """

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv 
from mixing_models import load_images, linear_mixture
from functions import Nonlocal_proj, normalizing_for_ssim, bm3d_proj, get_demix, SSIM, MSSSIM, TV_proj, flatten, unflatten,whiten_projection, learn_dictionary, dictionary_projection, col_norm_proj, dist_diag
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
import museval.metrics as mmetrics
import matplotlib.image as mpimg
from skimage.color import rgb2gray, rgba2rgb
from pytorch_msssim import ms_ssim
from skimage.metrics import structural_similarity as ssim
import os
from time import time
from patchify import patchify, unpatchify
from sklearn.decomposition import MiniBatchDictionaryLearning


#f = open("separation_experiment_filtered_dl_single_set3_atoms1_theta45.txt","w+")
#f = open("theta_experiment_double_dl_filtered_atom1_set3_degree25.txt","w+")

# parameters
n = 256
m = 8 
image_size = (n, n)
patch_size = (m, m)
step = 4
sigma = 0.005
num_coeff = 2
max_it = 100
permutation = False
t = np.linspace(1, max_it, int(max_it/5))
d = 25

method = 'single_dictionary_learning_filtered'
#f.write(f'\nsigma = {sigma}, degrees = {d}, iterations = {max_it}\n method = {method}')
#f.write(f'\n\nLearned on 15 images from images, tested on images hard set 3')

patches_recto =[]
patches_verso = []

# Extract reference patches from the first images
print('Extracting reference patches...')

for i in range(15):
    image1, image2 = load_images(f'./train_images_bm3d_v/set{i+1}_pic1.png', f'./train_images_bm3d_v/set{i+1}_pic2.png', 256, show_images= False)
    patches1 = patchify(image1, patch_size, step)
    initial_patch_size = patches1.shape
    patches1 = patches1.reshape(-1, patch_size[0] * patch_size[1])
    patches_recto.append(patches1)

    patches2 = patchify(image2, patch_size, step)
    patches2 = patches2.reshape(-1, patch_size[0] * patch_size[1])
    patches_recto.append(patches2)

patches_recto = np.reshape(patches_recto, (-1, m*m))
patches_recto -= np.mean(patches_recto, axis=0) # remove the mean
patches_recto /= np.std(patches_recto, axis=0) # normalize each patch

print('Learning the recto dictionary...')
dico_recto = MiniBatchDictionaryLearning(n_components=256, alpha=2)
V_recto = dico_recto.fit(patches_recto).components_

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
    #plt.figure()
    #plt.subplot(131)
    #plt.imshow(S2, cmap = 'gray')
    #plt.title('Estimated Source before')
    if permutation == True:
        S1 = np.fliplr(S1)
        S1 = Dic_proj_recto(S1, n_coeff, alpha)
        S1 = np.fliplr(S1)
        S2 = Dic_proj_recto(S2, n_coeff, alpha)

    else:
        S1 = Dic_proj_recto(S1, n_coeff, alpha)
        S2 = np.fliplr(S2)
        S2 = Dic_proj_recto(S2, n_coeff, alpha)
        S2 = np.fliplr(S2)
    #S2 = np.fliplr(S2)
    #plt.subplot(132)
    #plt.imshow(S2, cmap = 'gray')
    #plt.title('Estimated Source flipped')
    #S2 = Dic_proj_recto(S2, n_coeff, alpha)
    #S2 = np.fliplr(S2)
    #plt.subplot(133)
    #plt.imshow(S2, cmap = 'gray')
    #plt.title('Estimated Source after')
    #plt.show()
    
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


i = 4
path1 = f'../../../images_hard/set{i+1}_pic1.png'
path2 = f'../../../images_hard/set{i+1}_pic2.png'
source1, source2 = load_images(path1, path2, n = 256, show_images = False)
#f.write(f'sources = {os.path.splitext(os.path.basename(path1))[0]}, {os.path.splitext(os.path.basename(path2))[0]}')

plt.figure(figsize = (6,3))
plt.suptitle(f'Original sources, set{i+1}')
plt.subplot(121)
plt.imshow(source1, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(source2, cmap='gray')
plt.axis('off')
plt.tight_layout()
#plt.savefig('images_sources.pdf') 
plt.show

#plt.title(f'The mean ssim\nusing {method}')
#plt.title('The SSIM of the estimated source 1\nusing TV')

#degrees = [45, 135]
#f.write(f'\nsigma = {sigma}, iterations = {max_it}\n method = {method}')
#f.write(f'\ndegree = {d}')

#mixing_matrix = [[0.6992, 0.7275], [0.4784, 0.5548]]
#mixing_matrix = [[0.3, 0.7], [0.7, 0.3]]
#mixing_matrix = [[0.6, 0.4], [0.3, 0.7]]
theta = np.radians(d)
mixing_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

print('mixing matrix = ', mixing_matrix)
#f.write(f'\nthe mixing matrix = {mixing_matrix}')
sources, mixtures = linear_mixture(source1, source2, mixing_matrix = mixing_matrix, show_images = False)

mix1, mix2 = unflatten(mixtures, image_size)
plt.figure()
plt.suptitle(f'Mixed sources, set{i+1}\ntheta = {d}')
plt.subplot(121)
plt.imshow(mix1, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(mix2, cmap='gray')
plt.axis('off')
plt.tight_layout()
#plt.savefig('images_mixed.pdf')
plt.show

# Whitening process
X, W = whiten_projection(mixtures)
B = np.dot(W, mixing_matrix)

(sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(sources), np.asarray(X))
print('The mean value of the reference SDR is: ', np.mean(sdr_ref), perm)
if np.array_equal(perm, [[1],[0]]):
    permutation = True
#f.write(f'\nreference sdr (after the whitening) = {sdr_ref})\npermutation = {perm}')

X_normalized = normalizing_for_ssim(X, B, permutation) #used for evaluation of SSIM

sdr_estimated_source1 = []
sdr_estimated_source2 = []
ssim_estimated_source1 = []
ssim_estimated_source2 = []
msssim_estimated_source1 = []
msssim_estimated_source2 = []
estimated_matrix = []
mean_sdr = []
mean_ssim = []
mean_msssim = []

Se = np.copy(X) 
Se_old = np.copy(Se)

for it in np.arange(max_it):
    # 1. denoising
    Se = Dic_proj_double(Se_old, num_coeff, sigma)
    # 2. get demixing matrix
    WW = get_demix(X, Se)
    # 3. whiten the demix matrix
    WW, W = whiten_projection(WW)
    # 4. get the new source
    Se = np.dot(WW, X)

    if np.linalg.norm(Se - Se_old, ord = 'fro') < 1e-6:
        print('Dict demix convergence reached at iteration', it)
        #break
    
    if it%5 == 0:
        print(it)
        (sdr_temporary_i, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(sources), Se)
        print("perm = ", perm)
        sdr_estimated_source1.append(sdr_temporary_i[0])
        sdr_estimated_source2.append(sdr_temporary_i[1])
        mean_sdr.append(np.mean(sdr_temporary_i))

        ssim1, ssim2 = SSIM(sources, Se, image_size, permutation, X_normalized, B)
        ssim_estimated_source1.append(ssim1)
        ssim_estimated_source2.append(ssim2)
        current_mean_ssim = (ssim1 + ssim2)/2
        print('mean ssim: ', current_mean_ssim)
        mean_ssim.append(current_mean_ssim)
        
        print("min, max of source1 = ", sources[0].min(), sources[0].max())
        print("min, max of estimated1 = ", Se[0].min(), Se[0].max())
        ##msssim1, msssim2 = MSSSIM(sources, Se, image_size)
        #msssim_estimated_source1.append(msssim1)
        #msssim_estimated_source2.append(msssim2)

        demixing_matrix = col_norm_proj(get_demix(Se_old, Se))
        demixing_matrix = np.dot(WW, B)
        dr = dist_diag(demixing_matrix)
        print('DR = ', dr)
        estimated_matrix.append(dr)
    Se_old = Se
#f.write(f'\n\nmean_sdr = {mean_sdr}\n\nsdr of the source1 = {np.asarray(sdr_estimated_source1)}\n\nsdr of source2 = {np.asarray(sdr_estimated_source2)}')
#f.write(f'\n\nmean_ssim = {mean_ssim}\n\nssim of the source1 = {ssim_estimated_source1}\n\nssim of source2 = {ssim_estimated_source2}')
#f.write(f'\n\nmatrix_evaluation = {estimated_matrix}')
#f.write('\n************************************************\n')

print('ssim = ', ssim1, ssim2)
print('DR = ', dr)

i1,i2 = unflatten(Se, image_size)
plt.figure()
plt.suptitle(f"Estimated sources with Double DL\nSet {i+1}, Mean SSIM = {round(current_mean_ssim, 3)}")
plt.subplot(121)
plt.imshow(i1, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(i2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'estimated_sources_dl_filtered_theta45_set{i+1}.png') 
plt.show()
"""
plt.figure()
#plt.plot(t, mean_sdr, c= 'r', label = 'Mean SDR')
plt.plot(t, mean_ssim, c = 'blue', label = 'Mean SSIM')
plt.plot(t, estimated_matrix, c = 'g', label = 'Matrix Evaluation')
plt.xlabel('iterations')
plt.legend()
plt.show()
"""