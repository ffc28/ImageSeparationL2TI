#Trying to use the dictionary learning

from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np
from skimage.color import rgb2gray
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# load the images
img1=mpimg.imread('pic1.png')
img2=mpimg.imread('pic2.png')

# convert them to grayscale
img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)
#print(img1_gray.shape, img2_gray.shape)

# rescale them
n = 255
img1_gray_re = img1_gray[0:n, 0:n]
img2_gray_re = img2_gray[0:n, 0:n]

"""
# show images
plt.figure()
plt.imshow(img1_gray_re, cmap='gray')
plt.title("Ground truth 1")

plt.figure()
plt.imshow(img2_gray_re, cmap='gray')
plt.title("Ground truth 2")
plt.show"""

# mixing the images
source1 = np.matrix.flatten(img1_gray_re, 'F') #column wise
source2 = np.matrix.flatten(img2_gray_re, 'F') #column wise

source = np.stack((source1, source2)) # 2 x n^2

np.random.seed(0)
mixing_matrix = np.random.rand(2,2) # randomly generated mixing matrix
#mixing_matrix = np.array([[1, 0.5], [0.5, 1]])
print("mixing_matrix = ", mixing_matrix)

X = np.matmul(source.T, mixing_matrix)  # observations

# reconstructiong the mixed images
X1 = X[:,0]
X1 = np.reshape(X1, (n,n))

X2 = X[:,1]
X2 = np.reshape(X2, (n,n))

plt.figure()
plt.imshow(X1.T, cmap='gray')
plt.title("Mixed 1")
plt.show
plt.figure()
plt.imshow(X2.T, cmap='gray')
plt.title("Mixed 2")
plt.show

"""
# downsample for higher speed
X1 = X1[::4, ::4] + X1[1::4, ::4] + X1[::4, 1::4] + X1[1::4, 1::4]
X1 /= 4.0

X2 = X2[::4, ::4] + X2[1::4, ::4] + X2[::4, 1::4] + X2[1::4, 1::4]
X2 /= 4.0
print("downsampled height, width = ", X1.shape)

plt.figure()
plt.imshow(X2.T, cmap='gray')
plt.title("downsampled 2")
plt.show
"""

# Extract reference patches from the original images
print('Extracting reference patches...')
t0 = time()
m = 16 #Check other sizes
patch_size = (m, m)

#TODO: try the other function with multiple images
p1 = extract_patches_2d(img1_gray_re, patch_size)
p2 = extract_patches_2d(img2_gray_re, patch_size)
patches = np.concatenate((p1, p2), axis = 0)
#print("before reshaping patches : ", p1.shape, p2.shape)
patches = patches.reshape(patches.shape[0], -1)
intercept = np.mean(patches, axis=0)
patches -= intercept
print('done in %.2fs.' % (time() - t0))
print(patches.shape)

# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=400) #TODO:check with different parameters
V = dico.fit(patches).components_
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from patches\n' + 'Train time %.1fs on %d patches' % (dt, len(patches)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


########################################

def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

#TODO:try different parameters
transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2}),
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = X1.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(patches)
    patch = np.dot(code, V)

    patch += intercept
    patch = patch.reshape(len(patches), *patch_size)
    if transform_algorithm == 'threshold':
        patch -= patch.min()
        patch /= patch.max()
    reconstructions[title][:,:] = reconstruct_from_patches_2d(patch, (n,n))
    dt = time() - t0
    print('done in %.2fs.' % dt)
    show_with_diff(reconstructions[title], img1_gray_re, title + ' (time: %.1fs)' % dt)

plt.show()