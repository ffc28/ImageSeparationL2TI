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
pic_set = 1
img_train=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')
img_train_gray = rgb2gray(img_train) # the value is between 0 and 1

# Extract reference patches from the first image
print('Extracting reference patches...')
t0 = time()
m = 8 # Check other sizes
image_size = (n, n)
patch_size = (m, m)
step = 4

patches = patchify(img_train_gray, patch_size, step)
initial_patch_size = patches.shape
patches = patches.reshape(-1, patch_size[0] * patch_size[1])

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


## load the source here
pic_set = 6
img1=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')
img2=mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png')

img1_gray = rgb2gray(img1) # the value is between 0 and 1
img2_gray = rgb2gray(img2)
# Mixing process here
img2_gray_re = np.fliplr(img2_gray)

source1 = np.matrix(img1_gray)
source1 = source1.flatten('F') #column wise

source2 = np.matrix(img2_gray)
source2 = source2.flatten('F') #column wise

source1 = source1 - np.mean(source1)
source2 = source2 - np.mean(source2)

#source1 = source1/np.linalg.norm(source1)
#source2 = source2/np.linalg.norm(source2)

print("rdc = ", rdc(source1.T,source2.T))
source = np.stack((source1, source2))

print('Covariance matrix is: ')
print(np.matmul(source,source.T))

# randomly generated mixing matrix
#np.random.seed(0)
mixing_matrix = np.random.rand(2,2)
#mixing_matrix = np.array([[1, 0.5], [0.5, 1]])
print('Mixing matrix is: ')
print(mixing_matrix)

# X = source * mixing_matrix - The mixed images

X = np.matmul(mixing_matrix, source)

def Dic_proj_one(data, n_coef):
    """
    The dictionary projection method
    """
    data = patchify(data, patch_size, step)
    data = data.reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept 

    dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
    code = dico.transform(data)

    patch = np.dot(code, V)
    patch += intercept
    patch = np.reshape(patch, initial_patch_size)

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re

def Dic_proj(S, n_coeff):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = Dic_proj_one(S1, n_coeff)
    S2 = Dic_proj_one(S2, n_coeff)
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S
    

def data_projection(X,S):
    """
    This functions does the data projection with the equation X = AS
    """
    A = np.dot(S, X.T)
    S = np.dot(A,X)
    R = np.dot(A.T, A)
    return np.dot(np.linalg.inv(R),S)

def whiten_projection(S):
    """
    This function does the whitening projection with PCA
    """
    R = np.dot(S,S.T)
    W = la.sqrtm(np.linalg.inv(R))
    return np.dot(W,S)


# Here begins the algorithm
# whitening processing. It's important
R = np.dot(X, X.T)
W = la.sqrtm(np.linalg.inv(R))
X = np.dot(W, X)

(sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X))
# mix = [[0.6992, 0.7275], [0.4784, 0.5548]] #or use the matrix from the paper
print("Reference SDR is: ", sdr_ref)
print("Reference SIR is: ", sir_ref)

max_it = 30
#Se = np.random.randn(2, n*n) 
Se = X  
cost_it = np.zeros((1,max_it)) 
SDR_it = np.zeros((2, max_it)) 
SIR_it = np.zeros((2, max_it)) 
SAR_it = np.zeros((2, max_it)) 

num_coeff = 3
for it in np.arange(max_it):
    print(it)
    # we performe three projections
    # Se = whiten_projection(soft_proximal(data_projection(X, Se),lambda_v[it]))
    # Se = TV_proj(data_projection(X,Se), lambda_v[it])
    Se = whiten_projection(Dic_proj(data_projection(X,Se), num_coeff))
    # Se = whiten_projection(non_linear1(data_projection(X,Se)))
    # Se = whiten_projection(data_projection(X,Se))
    
    cost_it[0,it] = np.linalg.norm(X - np.dot(np.dot(X,Se.T), Se),ord = 'fro')
    
    Se_inv = np.dot(np.linalg.inv(np.dot(X, Se.T)),X)
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), Se_inv)
    
    SDR_it[:, it] = np.squeeze(sdr)
    
print(np.dot(Se,Se.T))
"""
plt.figure()
plt.plot(cost_it[0,:])
plt.title('Cost for iterations')
plt.show
"""
plt.figure()
plt.plot(np.mean(SDR_it, axis = 0))
plt.title('SDR for iterations')
plt.grid()
plt.show




