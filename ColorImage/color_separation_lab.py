""" Experiments for determining the best separation """

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv 
from mixing_models import load_images, linear_mixture
from functions import vif, Nonlocal_proj, normalizing_for_ssim, bm3d_proj, get_demix, SSIM, MSSSIM, TV_proj, flatten, unflatten,whiten_projection, learn_dictionary, dictionary_projection, col_norm_proj, dist_diag
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
import museval.metrics as mmetrics
import matplotlib.image as mpimg
from skimage.color import rgb2gray, lab2rgb, rgba2rgb, rgb2lab, rgb2lab
from pytorch_msssim import ms_ssim
from skimage.metrics import structural_similarity as ssim
import os
from time import time
from matplotlib import cm

# parameters
n = 256
m = 8 
image_size = (n, n)
patch_size = (m, m)
step = 4
sigma = 0.003
num_coeff = 2
max_it = 100
permutation = False
t = np.linspace(1, max_it, int(max_it/5))
d = 30
method = 'TV'

path1 = '../data/images_hard/set5_pic1.png'
path2 = '../data/images_hard/set5_pic2.png'

image1 = mpimg.imread(path1)
image2 = mpimg.imread(path2)

image1 = rgba2rgb(image1)
image2 = rgba2rgb(image2)

print('channel R of source, min, max = ', image1.min(), image1.max())
print('channel G of source, min, max = ', image1.min(), image1.max())
print('channel B of source, min, max = ', image1.min(), image1.max())

source1_lab = rgb2lab(image1)
source2_lab = rgb2lab(image2)

print('channel L of source1, min, max = ', source1_lab[:,:,0].min(), source1_lab[:,:,0].max())
print('channel A of source1, min, max = ', source1_lab[:,:,1].min(), source1_lab[:,:,1].max())
print('channel B of source1, min, max = ', source1_lab[:,:,2].min(), source1_lab[:,:,2].max())

plt.figure()
plt.subplot(121)
plt.imshow(source1_lab)
plt.axis('off')
plt.subplot(122)
plt.imshow(source2_lab)
plt.axis('off')
plt.savefig('sources_lab_set5.png')
plt.show()

plt.figure()
#plt.suptitle('Source 1 in lab')
plt.subplot(231)
plt.imshow(source1_lab[:,:,0], cmap = 'gray')
plt.title('Channel L')
plt.ylabel('Source 1', rotation  = 90)
plt.axis('off')

plt.subplot(232)
plt.imshow(source1_lab[:,:,1], cmap = 'RdYlGn')
plt.title('Channel A')
plt.axis('off')

plt.subplot(233)
plt.imshow(source1_lab[:,:,2])
plt.title('Channel B')
plt.axis('off')

plt.subplot(234)
plt.imshow(source2_lab[:,:,0], cmap = 'gray')
plt.ylabel('Source 2', rotation  = 90)
plt.axis('off')

plt.subplot(235)
plt.imshow(source2_lab[:,:,1], cmap = 'RdYlGn')
plt.axis('off')

plt.subplot(236)
plt.imshow(source2_lab[:,:,2])
plt.axis('off')
plt.tight_layout()
plt.savefig("sources_lab_channels_set5.png")
plt.show()

theta = np.radians(d)
mixing_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

source_channel_1, mixture_channel_1 = linear_mixture(image1[:,:,0], image2[:,:,0], mixing_matrix, False)
source_channel_2, mixture_channel_2 = linear_mixture(image1[:,:,1], image2[:,:,1], mixing_matrix, False)
source_channel_3, mixture_channel_3 = linear_mixture(image1[:,:,2], image2[:,:,2], mixing_matrix, False)

print('channel R of mixture, min, max = ', mixture_channel_1.min(), mixture_channel_1.max())
print('channel G of mixture, min, max = ', mixture_channel_2.min(), mixture_channel_2.max())
print('channel B of mixture, min, max = ', mixture_channel_3.min(), mixture_channel_3.max())

mixture1 = np.zeros_like(image1)
mixture2 = np.zeros_like(image2)
mixture1[:,:,0], mixture2[:,:,0] =  unflatten(mixture_channel_1, image_size)
mixture1[:,:,1], mixture2[:,:,1] =  unflatten(mixture_channel_2, image_size)
mixture1[:,:,2], mixture2[:,:,2] =  unflatten(mixture_channel_3, image_size)

plt.figure()
plt.suptitle('The mixtures')
plt.subplot(121)
plt.imshow(mixture1)
plt.axis('off')

plt.subplot(122)
plt.imshow(mixture2)
plt.axis('off')
plt.tight_layout()
#plt.savefig("mixtures_set5_theta30.png")
plt.show

plt.figure()
plt.suptitle('The mixtures in RGB')
plt.subplot(131)
plt.imshow(mixture1[:,:,0], cmap = 'gray')
plt.title('Channel R')
plt.axis('off')

plt.subplot(132)
plt.imshow(mixture1[:,:,1], cmap = 'gray')
plt.title('Channel G')
plt.axis('off')

plt.subplot(133)
plt.imshow(mixture1[:,:,2], cmap = 'gray')
plt.title('Channel B')
plt.axis('off')
plt.tight_layout()
#plt.savefig("mixture1_rgb_set5_theta30.png")
plt.show()

# CONVERT TO lab
mixture1_lab = rgb2lab(mixture1)
mixture2_lab = rgb2lab(mixture2)

print('channel H of mixture, min, max = ', mixture1_lab[:,:,0].min(), mixture1_lab[:,:,0].max())
print('channel S of mixture, min, max = ', mixture1_lab[:,:,1].min(), mixture1_lab[:,:,1].max())
print('channel V of mixture, min, max = ', mixture1_lab[:,:,2].min(), mixture1_lab[:,:,2].max())

plt.figure()
plt.suptitle('The mixtures in lab')
plt.subplot(231)
plt.imshow(mixture1_lab[:,:,0], cmap = 'gray')
plt.title('Channel H')
plt.ylabel('Source 1', rotation  = 90)
plt.axis('off')

plt.subplot(232)
plt.imshow(mixture1_lab[:,:,1], cmap = 'RdYlGn')
plt.title('Channel S')
plt.axis('off')

plt.subplot(233)
plt.imshow(mixture1_lab[:,:,2])
plt.title('Channel V')
plt.axis('off')

plt.subplot(234)
plt.imshow(mixture2_lab[:,:,0], cmap = 'gray')
plt.ylabel('Source 2', rotation  = 90)
plt.axis('off')

plt.subplot(235)
plt.imshow(mixture2_lab[:,:,1], cmap = 'RdYlGn')
plt.axis('off')

plt.subplot(236)
plt.imshow(mixture2_lab[:,:,2])
plt.axis('off')
plt.tight_layout()
#plt.savefig("mixtures_lab_set5_theta30.png")
plt.show()

"""X1, W1 = whiten_projection(mixture_channel_1)
B1 = np.dot(W1, mixing_matrix)
X2, W2 = whiten_projection(mixture_channel_2)
B2 = np.dot(W2, mixing_matrix)
X3, W3 = whiten_projection(mixture_channel_3)
B3 = np.dot(W3, mixing_matrix)"""
stacked_channel_l_mixtures = flatten(mixture1_lab[:,:,0], mixture2_lab[:,:,0])
stacked_channel_a_mixtures = flatten(mixture1_lab[:,:,1], mixture2_lab[:,:,1])
stacked_channel_b_mixtures = flatten(mixture1_lab[:,:,2], mixture2_lab[:,:,2])

stacked_channel_l_source = flatten(source1_lab[:,:,0], source2_lab[:,:,0])
stacked_channel_a_source = flatten(source1_lab[:,:,1], source2_lab[:,:,1])
stacked_channel_b_source = flatten(source1_lab[:,:,2], source2_lab[:,:,2])

mixture_channel_list_lab = [stacked_channel_l_mixtures, stacked_channel_a_mixtures, stacked_channel_b_mixtures]
source_channel_list_lab = [stacked_channel_l_source, stacked_channel_a_source, stacked_channel_b_source]
for i in np.arange(3):
    X_i, W_i = whiten_projection(mixture_channel_list_lab[i])
    B_i = np.dot(W_i, mixing_matrix)
    (sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(source_channel_list_lab[i], X_i)
    print('The mean value of the reference SDR is: ', np.mean(sdr_ref), perm)
    if np.array_equal(perm, [[1],[0]]):
        permutation = True
    
    X_normalized = normalizing_for_ssim(X_i, B_i, permutation) #used for evaluation of SSIM
    Se = np.copy(X_i) 
    Se_old = np.copy(Se)


    ssim_estimated_source1 = []
    ssim_estimated_source2 = []
    estimated_matrix = []
    mean_ssim = []

    for it in np.arange(max_it):
        # 1. denoising
        if method == 'TV':
            Se = TV_proj(Se_old, sigma)
        if method == 'bm3d':
            Se = bm3d_proj(Se_old, sigma)
        if method == 'non-local':
            Se = Nonlocal_proj(Se_old, sigma)

        # 2. get demixing matrix
        WW = get_demix(X_i, Se)
        # 3. whiten the demix matrix
        WW, W = whiten_projection(WW)
        # 4. get the new source
        Se = np.dot(WW, X_i)

        if np.linalg.norm(Se - Se_old, ord = 'fro') < 1e-6:
            print('Dict demix convergence reached at iteration', it)
            #break
        
        if it%5 == 0:
            print(it)
            (sdr_temporary_i, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source_channel_list_lab[i]), Se)

            ssim1, ssim2 = SSIM(source_channel_list_lab[i], Se, image_size, permutation, X_normalized, B_i)
            ssim_estimated_source1.append(ssim1)
            ssim_estimated_source2.append(ssim2)
            mean_ssim.append((ssim1 + ssim2)/2)

            demixing_matrix = col_norm_proj(get_demix(Se_old, Se))
            demixing_matrix = np.dot(WW, B_i)
            estimated_matrix.append(dist_diag(demixing_matrix))
        Se_old = Se
    plt.figure()
    plt.title(f"Channel {i+1}")
    plt.plot(t, mean_ssim, c = 'blue', label = 'Mean SSIM')
    plt.plot(t, estimated_matrix, c = 'red', label = 'Diagonal Ratio')
    plt.xlabel('Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"color_lab_channel_{i+1}_set5_{method}.pdf")

    if i==0:
        estimated_channel_l = Se 
    if i==1:
        estimated_channel_a = Se
    if i==2:
        estimated_channel_b = Se

estimated_image1 = np.zeros_like(image1)
estimated_image2 = np.zeros_like(image2)

estimated_image1[:,:,0], estimated_image2[:,:,0] =  unflatten(estimated_channel_l, image_size)
estimated_image1[:,:,1], estimated_image2[:,:,1] =  unflatten(estimated_channel_a, image_size)
estimated_image1[:,:,2], estimated_image2[:,:,2] =  unflatten(estimated_channel_b, image_size)

plt.figure()
plt.suptitle("Estimated source 1")
plt.subplot(131)
plt.imshow(estimated_image1[:,:,0], cmap='gray')
plt.title('Channel L')
plt.axis('off')

plt.subplot(132)
plt.imshow(estimated_image1[:,:,1], cmap='RdYlGn')
plt.title('Channel A')
plt.axis('off')

plt.subplot(133)
plt.imshow(estimated_image1[:,:,2])
plt.title('Channel B')
plt.axis('off')
plt.tight_layout()
plt.savefig("estimated_source1_tv_set5_theta30_channels_lab.png")
plt.show

plt.figure()
plt.suptitle("Estimated source 2")
plt.subplot(131)
plt.imshow(estimated_image2[:,:,0], cmap='gray')
plt.title('Channel L')
plt.axis('off')

plt.subplot(132)
plt.imshow(estimated_image2[:,:,1], cmap='RdYlGn')
plt.title('Channel A')
plt.axis('off')

plt.subplot(133)
plt.imshow(estimated_image2[:,:,2])
plt.title('Channel B')
plt.axis('off')
plt.tight_layout()
plt.savefig("estimated_source2_tv_set5_theta30_channels_lab.png")
plt.show()