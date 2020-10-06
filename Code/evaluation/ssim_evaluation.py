""" SSIM behaviour """

import matplotlib.pyplot as plt
import numpy as np
from mixing_models import load_images, linear_mixture
from functions import TV_proj, flatten, unflatten,whiten_projection, learn_dictionary, dictionary_projection
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle

n = 256
m = 8 
image_size = (n, n)
patch_size = (m, m)
step = 4
n_coef = 2

# 2 images
source1, source2 = load_images('../../images_hard/set2_pic1.png', '../../images_hard/set2_pic2.png', n = 256, show_images = False)
print('SSIM of the clean image with itself =', ssim(source1, source1))

# add gaussian noise
sigma = 0.1
noisy_image1 = random_noise(source1, var = sigma**2) #var = 0.01 by default
noisy_image2 = random_noise(source2, var = sigma**2) #var = 0.01 by default

print('\nSSIM of source 1 with its noisy one =', ssim(source1, noisy_image1, data_range=noisy_image1.max() - noisy_image1.min()))
print('SSIM of source 2 with its noisy one =', ssim(source2, noisy_image2, data_range=noisy_image2.max() - noisy_image2.min()))

# tv projection
#estimated_image1 = denoise_tv_chambolle(noisy_image1, weight = 0.01, multichannel=False)
#estimated_image2 = denoise_tv_chambolle(noisy_image2, weight = 0.01, multichannel=False)

""" Dictionary part:"""
# mix the sources
sources, mixtures = linear_mixture(source1, source2, show_images = False)

mixture1, mixture2 = unflatten(mixtures, image_size)
print('\nSSIM of source 1 with its mixture =', ssim(source1, mixture1,  data_range=mixture1.max() - mixture1.min()))
print('SSIM of source 2 with its mixture =', ssim(source2, mixture2, data_range=mixture2.max() - mixture2.min()))

# whiten the mixtures
X = whiten_projection(mixtures)
mixture1, mixture2 = unflatten(X, image_size)
print('\nSSIM of source 1 with its whitened mixture =', ssim(source1, mixture1, data_range=mixture1.max() - mixture1.min()))
print('SSIM of source 2 with its whitened mixture =', ssim(source2, mixture2, data_range=mixture2.max() - mixture2.min()))

# Load the images to learn the dictionary
image1, image2 = load_images('../../images_hard/set1_pic1.png', '../../images_hard/set1_pic2.png', n = 256, show_images = False)
image3, image4 = load_images('../../images_hard/set3_pic1.png', '../../images_hard/set3_pic2.png', n = 256, show_images = False)

# Learn the dictionary
dico, V = learn_dictionary(patch_size, step, False, image1, image2, image3, image4)

# dictionary projection
estimated_image1 = dictionary_projection(mixture1, dico, V, n_coef, patch_size, step)
estimated_image2 = dictionary_projection(mixture2, dico, V, n_coef, patch_size, step)
print('\nSSIM of source 1 with its estimation =', ssim(source1, estimated_image1, data_range=estimated_image1.max() - estimated_image1.min()))
print('SSIM of source 2 with its estimation =', ssim(source2, estimated_image2, data_range=estimated_image2.max() - estimated_image2.min()))

plt.figure()
plt.imshow(estimated_image1, cmap='gray')
plt.title('Estimated source 1')
plt.show

plt.figure()
plt.imshow(estimated_image2, cmap='gray')
plt.title('Estimated source 2')
plt.show()