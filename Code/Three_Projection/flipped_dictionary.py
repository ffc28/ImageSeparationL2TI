""" flipped dictionary experiments """

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import museval.metrics as mmetrics
import scipy as sp
import numpy as np
import scipy.linalg as la
from skimage.color import rgb2gray
from sklearn.decomposition import MiniBatchDictionaryLearning
from patchify import patchify, unpatchify
from rdc import rdc
from scipy.stats import kurtosis, skew
from mixing_models import load_images, linear_mixture
from functions import get_demix, whiten_projection

# parameters
n = 256
m = 8
image_size = (n, n)
patch_size = (m, m)
step = 4

patches_recto =[]
patches_verso = []

# Extract reference patches from the images
print('Extracting reference patches...')

for i in range(10):
    image1, image2 = load_images('../images/set'+ str(i+1) + '_pic1.png', '../images/set'+ str(i+1) + '_pic2.png', 256, show_images= False)

    patches1 = patchify(image1, patch_size, step)
    patches2 = patchify(image2, patch_size, step)
    initial_patch_size = patches1.shape

    patches1 = patches1.reshape(-1, patch_size[0] * patch_size[1])
    patches2 = patches2.reshape(-1, patch_size[0] * patch_size[1])
    
    patches_recto.append(patches1)
    patches_recto.append(patches2)


patches_recto = np.reshape(patches_recto, (-1, m*m))
patches_recto -= np.mean(patches_recto, axis=0) # remove the mean
patches_recto /= np.std(patches_recto, axis=0) # normalize each patch

dict_components = 100
# recto dictionary
print('Learning the dictionary...')
dico = MiniBatchDictionaryLearning(n_components=dict_components, alpha=1, n_iter=400)

# fitting the recto patches
V_recto = dico.fit(patches_recto).components_

#verso dictionary = flipped recto dictionary
V_verso = np.reshape(V_recto, (dict_components, m, m))
for i in range(dict_components):
    V_verso[i] = np.fliplr(V_verso[i])
V_verso = np.reshape(V_verso, (dict_components, m*m))


def Dic_proj_recto(data, n_coef, alpha):
    """
    The dictionary projection method
    """
    data = patchify(data, patch_size, step)
    data = data.reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept 
    
    dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
    code = dico.transform(data)

    patch = np.dot(code, V_recto)
    patch += intercept

    patch = np.reshape(patch, initial_patch_size)
    # if we use threshold then we have this
    # patch -= patch.min()
    # patch /= patch.max()

    im_re = unpatchify(np.asarray(patch), image_size)

    return im_re


def Dic_proj_verso(data, n_coef, alpha):
    """
    The dictionary projection method
    """
    data = patchify(data, patch_size, step)
    data = data.reshape(-1, patch_size[0] * patch_size[1])
    intercept = np.mean(data, axis=0)
    data -= intercept 

    dico.set_params(transform_algorithm = 'omp', transform_n_nonzero_coefs = n_coef)
    code = dico.transform(data)

    patch = np.dot(code, V_verso)
    patch += intercept
    
    patch = np.reshape(patch, initial_patch_size)
    # if we use threshold then we have this
    # patch -= patch.min()
    # patch /= patch.max()

    im_re = unpatchify(np.asarray(patch), image_size)
    return im_re

def Dic_proj_double(S, n_coeff, alpha):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_verso(S2, n_coeff, alpha)
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S


def Dic_proj_single(S, n_coeff, alpha):
    
    S1 = np.reshape(S[0,:], image_size)
    S2 = np.reshape(S[1,:], image_size)
    
    S1 = Dic_proj_recto(S1, n_coeff, alpha)
    S2 = Dic_proj_recto(S2, n_coeff, alpha)
    
    S[0,:] = np.reshape(S1, (1, n*n))
    S[1,:] = np.reshape(S2, (1, n*n))
     
    return S


sigma = 2e-3
num_coeff = 2
max_it = 100
all_sdr_single_dictionary = []
all_sdr_double_dictionary = []
improved_sdr = []
reference_sdr = []
sdr_temporary = []

#np.random.seed(15)
#mixing_matrix = np.random.rand(2,2)
mixing_matrix = [[2.5, 0.7], [0.7, 2.5]]
print("mixing_matrix = ", mixing_matrix)
#[[0.5507979  0.70814782] [0.29090474 0.51082761]]

for i in range(5):
    print(i+1)
    # load and mix images
    source1, source2 = load_images('../images_hard/set'+ str(i+1) + '_pic1.png', '../images_hard/set'+ str(i+1) + '_pic2.png', 256, show_images = False)
    sources, mixed_sources = linear_mixture(source1, source2, mixing_matrix= mixing_matrix, show_images = False)#mixing_matrix = mixing_matrix,

    # Here begins the algorithm
    # whitening processing. It's important
    X = whiten_projection(mixed_sources)
    (sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(sources), np.asarray(X))
    print('The mean value of the reference SDR is: ', np.mean(sdr_ref))
    reference_sdr.append(np.mean(sdr_ref))
    
    Se = np.copy(X) 
    Se_old = np.copy(Se)
    for it in np.arange(max_it):
        # 1. denoising
        Se = Dic_proj_double(Se_old, num_coeff, sigma)
        # 2. get demixing matrix
        WW = get_demix(X, Se)
        # 3. whiten the demix matrix
        WW = whiten_projection(WW)
        # 4. get the new source
        Se = np.dot(WW, X)
        
        if np.linalg.norm(Se - Se_old, ord = 'fro') < 1e-6:
            print('Dict demix convergence reached at iteration', it)
            break
            
        if it%5 == 0:
            (sdr_temporary_i, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(sources), Se)
            sdr_temporary.append(np.mean(sdr_temporary_i))


    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(sources), Se)
    print('The mean value of the SDR is: ', np.mean(sdr))
    all_sdr_double_dictionary.append(np.mean(sdr))
    print('The SDR improvement is: ', np.mean(sdr) - np.mean(sdr_ref))
    improved_sdr.append(np.mean(sdr) - np.mean(sdr_ref))

    print('********')

print("\nreference_sdr = ", reference_sdr)
print("\nall_sdr_double_dictionary = ",all_sdr_double_dictionary)
print("\nimproved_sdr = ", improved_sdr)
#print("\ntemporary sdr = ", sdr_temporary)



t = np.linspace(0,20,20)

plt.figure()
plt.plot(t, sdr_temporary[:20], c= 'r', label = 'Set 1')
plt.plot(t, sdr_temporary[20:40], c= 'b', label = 'Set 2')
plt.plot(t, sdr_temporary[40:60], c= 'g', label = 'Set 3')
plt.plot(t, sdr_temporary[60:80], c= 'y', label = 'Set 4')
plt.plot(t, sdr_temporary[80:100], c= 'violet', label = 'Set 5')
plt.title('Temporary SDR in double dictionary learning')
plt.xlabel('iterations')
plt.ylabel('SDR')
plt.legend()
plt.show()
