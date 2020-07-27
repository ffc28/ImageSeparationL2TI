#Trying to use the dictionary learning

from time import time
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

# load images and convert them
n = 256
np.random.seed(2)

m = 8 # Check other sizes
image_size = (n, n)
patch_size = (m, m)
step = 4
# Extract reference patches from the first image
print('Extracting reference patches...')
t0 = time()

pic_set = 1
img_train1=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')
img_train_gray1 = rgb2gray(img_train1) # the value is between 0 and 1

patches1 = patchify(img_train_gray1, patch_size, step)
initial_patch_size = patches1.shape
patches1 = patches1.reshape(-1, patch_size[0] * patch_size[1])

img_train2=mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png')
img_train_gray2 = rgb2gray(img_train2) # the value is between 0 and 1
patches2 = patchify(img_train_gray2, patch_size, step)
patches2 = patches2.reshape(-1, patch_size[0] * patch_size[1])

pic_set = 2
img_train3=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')
img_train_gray3 = rgb2gray(img_train3) # the value is between 0 and 1
patches3 = patchify(img_train_gray3, patch_size, step)
patches3 = patches3.reshape(-1, patch_size[0] * patch_size[1])

img_train4=mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png')
img_train_gray4 = rgb2gray(img_train4) # the value is between 0 and 1
patches4 = patchify(img_train_gray4, patch_size, step)
patches4 = patches4.reshape(-1, patch_size[0] * patch_size[1])

#patches = patches1
patches = np.concatenate((patches1, patches2, patches3, patches4), axis = 0)

patches -= np.mean(patches, axis=0) # remove the mean
patches /= np.std(patches, axis=0) # normalise each patch
print('done in %.2fs.' % (time() - t0))

# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=400) #TODO:check with different parameters
V = dico.fit(patches).components_
dt = time() - t0
print('done in %.2fs.' % dt)

# plot the dictionary
plt.figure(figsize=(8, 6))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from patches\n' + 'Train time %.1fs on %d patches' % (dt, len(patches)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

######################################
## load the source here
pic_set = 1
img1=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic1.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1

print('Extracting noisy patches... ')
t0 = time()
data = patchify(img1_gray, patch_size, step)
data = data.reshape(-1, patch_size[0] * patch_size[1])
intercept = np.mean(data, axis=0)
data -= intercept 

n_coef = 3
dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
code = dico.transform(data)

patch = np.dot(code, V)
patch += intercept
patch = np.reshape(patch, initial_patch_size)

im_re = unpatchify(np.asarray(patch), image_size)
print('done in %.2fs.' % (time() - t0))

difference = img1_gray - im_re
print('Difference', np.sqrt(np.sum(difference ** 2)))

##############################################

img1_gray_flip = np.fliplr(img1_gray)

print('Extracting noisy patches... ')
t0 = time()
data = patchify(img1_gray_flip, patch_size, step)
data = data.reshape(-1, patch_size[0] * patch_size[1])
intercept = np.mean(data, axis=0)
data -= intercept 

n_coef = 3
dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
code = dico.transform(data)

patch = np.dot(code, V)
patch += intercept
patch = np.reshape(patch, initial_patch_size)

im_re_flip = unpatchify(np.asarray(patch), image_size)
print('done in %.2fs.' % (time() - t0))

difference_flip = img1_gray_flip - im_re_flip
print('Difference', np.sqrt(np.sum(difference_flip ** 2)))
