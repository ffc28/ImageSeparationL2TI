""" Experiments for determining the best separation """

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv 
from mixing_models import load_images, linear_mixture

from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
import museval.metrics as mmetrics
import matplotlib.image as mpimg
from skimage.color import rgb2gray, rgba2rgb, rgb2hsv, hsv2rgb
from skimage.metrics import structural_similarity as ssim

path = './images_hard/set1_pic2.png'
path2 = './images_hard/set5_pic2.png'

image_rgba= mpimg.imread(path)

image_rgb = rgba2rgb(image_rgba)

# plot the rgb in each channel
plt.figure()
plt.subplot(221)
plt.imshow(image_rgb[:,:,0], cmap = 'gray')
plt.title('Channel R')
plt.axis('off')

plt.subplot(222)
plt.imshow(image_rgb[:,:,1], cmap = 'gray')
plt.title('Channel G')
plt.axis('off')

plt.subplot(223)
plt.imshow(image_rgb[:,:,2], cmap = 'gray')
plt.title('Channel B')
plt.axis('off')

plt.subplot(224)
plt.imshow(image_rgb)
plt.title('Image in color')
plt.axis('off')

plt.subplots_adjust(wspace = 0.05)
plt.show

print('channel R of source, min, max = ', image_rgb.min(), image_rgb.max())
print('channel G of source, min, max = ', image_rgb.min(), image_rgb.max())
print('channel B of source, min, max = ', image_rgb.min(), image_rgb.max())

image_hsv = rgb2hsv(image_rgb)


print('channel H of source1, min, max = ', image_hsv[:,:,0].min(), image_hsv[:,:,0].max())
print('channel S of source1, min, max = ', image_hsv[:,:,1].min(), image_hsv[:,:,1].max())
print('channel V of source1, min, max = ', image_hsv[:,:,2].min(), image_hsv[:,:,2].max())

plt.figure()
plt.subplot(221)
plt.imshow(np.squeeze(image_hsv[:,:,0]), cmap = 'gray', vmin = image_hsv[:,:,0].min(), vmax = image_hsv[:,:,0].max())
plt.title('Channel H')
plt.axis('off')

plt.subplot(222)
plt.imshow(image_hsv[:,:,1], cmap = 'gray')
plt.title('Channel S')
plt.axis('off')

plt.subplot(223)
plt.imshow(image_hsv[:,:,2], cmap = 'gray')
plt.title('Channel V')
plt.axis('off')

plt.subplot(224)
plt.imshow(image_rgb)
plt.title('Image in color')
plt.axis('off')

plt.subplots_adjust(wspace = 0.05)
plt.show()

#### Smooth the H channel

image_hsv[:,:,0] = np.mean(np.squeeze(image_hsv[:,:,0]))
# Go back to the color image
image_new = hsv2rgb(image_hsv)
# plot the new image
plt.figure()
plt.subplot(221)
plt.imshow(np.squeeze(image_hsv[:,:,0]), cmap = 'gray')
plt.title('Channel H')
plt.axis('off')

plt.subplot(222)
plt.imshow(image_hsv[:,:,1], cmap = 'gray')
plt.title('Channel S')
plt.axis('off')

plt.subplot(223)
plt.imshow(image_hsv[:,:,2], cmap = 'gray')
plt.title('Channel V')
plt.axis('off')

plt.subplot(224)
plt.imshow(image_new)
plt.title('Image in color')
plt.axis('off')

plt.subplots_adjust(wspace = 0.05)
plt.show()

