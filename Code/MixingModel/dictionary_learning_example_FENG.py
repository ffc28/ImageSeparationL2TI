#Trying to use the dictionary learning

from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np
from skimage.color import rgb2gray
from sklearn.feature_extraction.image import extract_patches_2d, PatchExtractor
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# load the images
img1=mpimg.imread('pic1.png')
img2=mpimg.imread('pic2.png')

# convert them to grayscale
img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

# rescale them
n = 256
img1_gray_re = img1_gray[0:n, 0:n]
img2_gray_re = img2_gray[200:200+n, 200:200+n]

# show images
plt.figure()
plt.imshow(img1_gray_re, cmap='gray')
plt.title("Ground truth 1")

plt.figure()
plt.imshow(img2_gray_re, cmap='gray')
plt.title("Ground truth 2")
plt.show

# Extract reference patches from the one image
print('Extracting reference patches...')
t0 = time()
m = 16 # Check other sizes
patch_size = (m, m)

patches = extract_patches_2d(img1_gray_re, patch_size)
patches = patches.reshape(patches.shape[0], -1)
# remove the mean value and do the normalisation
patches -= np.mean(patches, axis=0)
patches /= np.std(patches, axis=0)
print('done in %.2fs.' % (time() - t0))
print(patches.shape)

# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=200, alpha=0.5, n_iter=400) #TODO:check with different parameters
V = dico.fit(patches).components_
dt = time() - t0
print('done in %.2fs.' % dt)

# show the learned dictionary as patches
plt.figure(figsize=(6, 6))
for i, comp in enumerate(V[:100]): # we show the 100 first patches
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from patches\n' + 'Train time %.1fs on %d patches' % (dt, len(patches)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


########################################

def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(6, 3.5))
    
    plt.subplot(1, 3, 1)
    plt.title('Reference')
    plt.imshow(reference, vmin=reference.min(), vmax=reference.max(), cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    
    plt.subplot(1, 3, 2)
    plt.title('Image restored')
    plt.imshow(image, vmin=image.min(), vmax=image.max(), cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    
    plt.subplot(1, 3, 3)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=difference.min(), vmax=difference.max(), cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

# Extract the patches from other image and reconstruct them using the dictionary

print('Extracting patches from the other image ... ')
t0 = time()
data = extract_patches_2d(img2_gray_re, patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

#TODO:try different parameters
transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp', {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp', {'transform_n_nonzero_coefs': 2}),
    ('Orthogonal Matching Pursuit\n3 atoms', 'omp', {'transform_n_nonzero_coefs': 3}),
    ('Orthogonal Matching Pursuit\n4 atoms', 'omp', {'transform_n_nonzero_coefs': 4}),
    ('Orthogonal Matching Pursuit\n5 atoms', 'omp', {'transform_n_nonzero_coefs': 5})]
    #('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = img2_gray_re.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    #print("code shape =", code.shape)
    patch = np.dot(code, V)
    patch += intercept
    patch = patch.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patch -= patch.min()
        patch /= patch.max()
    reconstructions[title][:,:] = reconstruct_from_patches_2d(patch, (n,n))
    dt = time() - t0
    print('done in %.2fs.' % dt)
    show_with_diff(reconstructions[title], img2_gray_re, title + ' (time: %.1fs)' % dt)

plt.show()