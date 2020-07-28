#Trying to use the dictionary learning

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
m = 8 # Check other sizes
image_size = (n, n)
patch_size = (m, m)
step = 4

pic_set = 7
img_train1=mpimg.imread('./images/set'+ str(pic_set) + '_pic1.png')
img_train_gray1 = rgb2gray(img_train1) # the value is between 0 and 1

print('Learning the dictionary for recto and verso images...')
# Extract reference patches from the first image

patches1 = patchify(img_train_gray1, patch_size, step)
initial_patch_size = patches1.shape
patches1 = patches1.reshape(-1, patch_size[0] * patch_size[1])

patches_recto = patches1
patches_recto -= np.mean(patches_recto, axis=0) # remove the mean
patches_recto /= np.std(patches_recto, axis=0) # normalise each patch

print('Learning the recto dictionary...')
dico_recto = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=400) #TODO:check with different parameters
V_recto = dico_recto.fit(patches_recto).components_

# plot the dictionary
plt.figure(figsize=(8, 6))
for i, comp in enumerate(V_recto[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Recto dictionary learned from patches')
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

img_train2=mpimg.imread('./images/set'+ str(pic_set) + '_pic2.png')
img_train_gray2 = rgb2gray(img_train2) # the value is between 0 and 1

# Flip to get the verso images
img_train_gray2 = np.fliplr(img_train_gray2)

patches2 = patchify(img_train_gray2, patch_size, step)
patches2 = patches2.reshape(-1, patch_size[0] * patch_size[1])

patches_verso = patches2
patches_verso -= np.mean(patches_verso, axis=0) # remove the mean
patches_verso /= np.std(patches_verso, axis=0) # normalise each patch

print('Learning the verso dictionary...')
dico_verso = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=400) #TODO:check with different parameters
V_verso = dico_verso.fit(patches_verso).components_

# plot the dictionary
plt.figure(figsize=(8, 6))
for i, comp in enumerate(V_verso[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Verso dictionary learned from patches')
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


## load the source here
pic_set = 1
img1=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic1.png')
img2=mpimg.imread('./images_hard/set'+ str(pic_set) + '_pic2.png')

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
#mixing_matrix = np.random.rand(2,2)
#mixing_matrix = np.array([[0.36, 0.66], [0.03, 0.95]])
mixing_matrix = np.array([[1, 0.5], [0.5, 1]])
print('Mixing matrix is: ')
print(mixing_matrix)

# X = source * mixing_matrix - The mixed images

X = np.matmul(mixing_matrix, source)

def Dic_proj_recto(data, n_coef):
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

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re

def Dic_proj_verso(data, n_coef):
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

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re

def Dic_proj(S, n_coeff):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = Dic_proj_recto(S1, n_coeff)
    S2 = Dic_proj_verso(S2, n_coeff)
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S
    

def data_projection(X,S):
    """
    This functions does the data projection with the equation X = AS
    """
    A = np.dot(S, X.T)
    S = np.dot(A,X)
    R = np.dot(A, A.T)
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

max_it = 40
#Se = np.random.randn(2, n*n) 
Se = X  
cost_it = np.zeros((1,max_it)) 
SDR_it = np.zeros((2, max_it)) 
SIR_it = np.zeros((2, max_it)) 
SAR_it = np.zeros((2, max_it)) 

num_coeff_begin = 1
num_coeff_final = 5
num_coeff_v = np.floor(np.linspace(num_coeff_begin, num_coeff_final, max_it))
for it in np.arange(max_it):
    print(it)
    # we performe three projections
    # Se = whiten_projection(soft_proximal(data_projection(X, Se),lambda_v[it]))
    # Se = TV_proj(data_projection(X,Se), lambda_v[it])
    Se = whiten_projection(Dic_proj(data_projection(X,Se), num_coeff_v[it]))
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

print(sdr)