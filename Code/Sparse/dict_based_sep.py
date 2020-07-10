#Trying to use the dictionary learning

from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np
from skimage.color import rgb2gray
from skimage.util import view_as_windows
from sklearn.feature_extraction.image import extract_patches_2d, PatchExtractor
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from patchify import patchify, unpatchify

# load images and convert them
n = 256
pic_set = 6

img1=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')

pic_set = 6
img2=mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)
"""
# show images
plt.figure()
plt.imshow(img1_gray, cmap='gray')
plt.title("Ground truth 1")
"""
plt.figure()
plt.imshow(img2_gray, cmap='gray')
plt.title("Ground truth 2")
plt.show

# Extract reference patches from the first image
print('Extracting reference patches...')
t0 = time()
m = 8 # Check other sizes
image_size = (n, n)
patch_size = (m, m)
step = 4

patches = patchify(img1_gray, patch_size, step)
initial_patch_size = patches.shape
patches = patches.reshape(-1, patch_size[0] * patch_size[1])

patches -= np.mean(patches, axis=0) # remove the mean
patches /= np.std(patches, axis=0) # normalise each patch
print('done in %.2fs.' % (time() - t0))
print("The size of the patch is: ")
print(patches.shape)
# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=0.5, n_iter=400) #TODO:check with different parameters
V = dico.fit(patches).components_
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure(figsize=(8, 6))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from patches\n' + 'Train time %.1fs on %d patches' % (dt, len(patches)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


# Extract noisy patches and reconstruct them using the dictionary
print('Extracting noisy patches... ')
t0 = time()
data = patchify(img2_gray, patch_size, step)
data = data.reshape(-1, patch_size[0] * patch_size[1])
print(data.shape)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

t0 = time()
dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = 50)
code = dico.transform(data)
print("code shape =", code.shape)
patch = np.dot(code, V)
patch += intercept
patch = np.reshape(patch, initial_patch_size)
print(patch.shape)

im_re = unpatchify(np.asarray(patch), image_size)
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure()
plt.imshow(im_re, cmap='gray')
plt.title("Estimated source 2")
plt.show

diff = img2_gray - im_re
print(np.sqrt(np.sum(diff ** 2)))