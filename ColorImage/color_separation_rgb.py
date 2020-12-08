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
from skimage.color import rgb2gray, rgba2rgb
from pytorch_msssim import ms_ssim
from skimage.metrics import structural_similarity as ssim
import os
from time import time


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

print("the dimensions are: ", np.shape(image1), np.shape(image2))

plt.figure()
plt.subplot(121)
plt.imshow(image1)
plt.axis('off')

plt.subplot(122)
plt.imshow(image2)
plt.axis('off')
plt.tight_layout()
plt.show

image1 = rgba2rgb(image1)
image2 = rgba2rgb(image2)



print('channel R of estimated source1, min, max = ', image1.min(), image1.max())
print('channel G of estimated source1, min, max = ', image1.min(), image1.max())
print('channel B of estimated source1, min, max = ', image1.min(), image1.max())
print('***************************')
print('channel R of estimated source2, min, max = ', image2.min(), image2.max())
print('channel G of estimated source2, min, max = ', image2.min(), image2.max())
print('channel B of estimated source2, min, max = ', image2.min(), image2.max())


plt.figure()
plt.subplot(231)
plt.imshow(image1[:,:,0], cmap = 'Reds_r')
plt.title('Channel R')
plt.axis('off')
plt.ylabel('Source 1', rotation  = 90)

plt.subplot(232)
plt.imshow(image1[:,:,1], cmap = 'Greens_r')
plt.title('Channel G')
plt.axis('off')

plt.subplot(233)
plt.imshow(image1[:,:,2], cmap = 'Blues_r')
plt.title('Channel B')
plt.axis('off')

plt.subplot(234)
plt.imshow(image2[:,:,0], cmap = 'Reds_r')
plt.ylabel('Source 2', rotation  = 90)
plt.axis('off')

plt.subplot(235)
plt.imshow(image2[:,:,1], cmap = 'Greens_r')
plt.axis('off')

plt.subplot(236)
plt.imshow(image2[:,:,2], cmap = 'Blues_r')
plt.axis('off')
plt.tight_layout()
plt.savefig("sources_rgb_channels_set5.png")
plt.show()

"""
plt.figure()
plt.hist(image1[:,:,0].flatten(), bins = 20, color = 'red', alpha = 0.5, label = 'channel R')
plt.hist(image1[:,:,1].flatten(), bins = 20, color = 'green', alpha = 0.5, label = 'channel G')
plt.hist(image1[:,:,2].flatten(), bins = 20, color = 'blue', alpha = 0.5, label = 'channel B')
plt.legend()
plt.show()"""

theta = np.radians(d)
mixing_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

for i in np.arange(3):
    print("Channel ", i+1)
    source_channel_i, mixture_channel_i = linear_mixture(image1[:,:,i], image2[:,:,i], mixing_matrix, False)
    if i==0:
        mixture_channel_1 = mixture_channel_i   
    if i==1:
        mixture_channel_2 = mixture_channel_i   
    if i==2:
        mixture_channel_3 = mixture_channel_i   

    X_i, W_i = whiten_projection(mixture_channel_i)
    B_i = np.dot(W_i, mixing_matrix)
    (sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(source_channel_i, X_i)
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
            (sdr_temporary_i, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source_channel_i), Se)

            ssim1, ssim2 = SSIM(source_channel_i, Se, image_size, permutation, X_normalized, B_i)
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
    plt.savefig(f"color_channel_{i+1}_set5_{method}.png")

    if i==0:
        estimated_channel_1 = Se 
    if i==1:
        estimated_channel_2 = Se
    if i==2:
        estimated_channel_3 = Se

estimated_image1 = np.zeros_like(image1)
estimated_image2 = np.zeros_like(image2)

estimated_image1[:,:,0], estimated_image2[:,:,0] =  unflatten(estimated_channel_1, image_size)
estimated_image1[:,:,1], estimated_image2[:,:,1] =  unflatten(estimated_channel_3, image_size)
estimated_image1[:,:,2], estimated_image2[:,:,2] =  unflatten(estimated_channel_3, image_size)

plt.figure()
plt.subplot(121)
plt.imshow(estimated_image1, vmin = -0.012249913, vmax = 0.006739224)
plt.axis('off')
plt.subplot(122)
plt.imshow(estimated_image2, vmin = -0.012249913, vmax = 0.006739224)
plt.axis('off')
plt.tight_layout()
plt.savefig('estimated_rgb_set5.png')


print('channel R of estimated source1, min, max = ', estimated_image1.min(), estimated_image1.max())
print('channel G of estimated source1, min, max = ', estimated_image1.min(), estimated_image1.max())
print('channel B of estimated source1, min, max = ', estimated_image1.min(), estimated_image1.max())
print('***************************')
print('channel R of estimated source2, min, max = ', estimated_image2.min(), estimated_image2.max())
print('channel G of estimated source2, min, max = ', estimated_image2.min(), estimated_image2.max())
print('channel B of estimated source2, min, max = ', estimated_image2.min(), estimated_image2.max())



plt.figure(figsize=(9,3))
plt.suptitle("Estimated source 1")
plt.subplot(131)
plt.imshow(estimated_image1[:,:,0], cmap='Reds_r')
plt.title('Channel R')
plt.axis('off')

plt.subplot(132)
plt.imshow(estimated_image1[:,:,1], cmap='Greens_r')
plt.title('Channel G')
plt.axis('off')

plt.subplot(133)
plt.imshow(estimated_image1[:,:,2], cmap='Blues_r')
plt.title('Channel B')
plt.axis('off')
plt.tight_layout()
plt.savefig("estimated_source1_tv_set5_channels_rgb.png")
plt.show

plt.figure(figsize=(9,3))
plt.suptitle("Estimated source 2")
plt.subplot(131)
plt.imshow(estimated_image2[:,:,0], cmap='Reds_r')
plt.title('Channel R')
plt.axis('off')

plt.subplot(132)
plt.imshow(estimated_image2[:,:,1], cmap='Greens_r')
plt.title('Channel G')
plt.axis('off')

plt.subplot(133)
plt.imshow(estimated_image2[:,:,2], cmap='Blues_r')
plt.title('Channel B')
plt.axis('off')
plt.tight_layout()
plt.savefig("estimated_source2_tv_set5_channels_rgb.png")
plt.show()