

# This is a script for trying the mixture of image containing documents
from __future__ import division
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors
import numpy as np
import museval.metrics as mmetrics
from rdc import rdc

from sklearn.decomposition import FastICA

from skimage import data
from skimage.color import rgb2gray

size_v = np.array([20, 40, 60, 100, 150, 200, 250, 270, 300, 350, 370, 400, 450, 500, 600])

SDR_improv = np.zeros([1,len(size_v)])
rdr_it = np.zeros([1,len(size_v)])

for it in np.arange(len(size_v)):
    
    n = size_v[it]
    
    # load images and convert them
    img1=mpimg.imread('pic1.png')
    img2=mpimg.imread('im2.png')
    
    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)
    
    img1_gray_re = img1_gray[50:n+50,70:n+70]
    img2_gray_re = img2_gray[20:n+20,0:n]
    # flip img2
    
    img2_gray_re = np.fliplr(img2_gray_re)
    
    # We mix them here
    
    source1 = np.matrix(img1_gray_re)
    source1 = source1.flatten('F') #column wise
    
    source2 = np.matrix(img2_gray_re)
    source2 = source2.flatten('F') #column wise
    
    source1 = source1 - np.mean(source1)
    source2 = source2 - np.mean(source2)
    
    source1 = source1/np.linalg.norm(source1)
    source2 = source2/np.linalg.norm(source2)
    
    rdr_this = rdc(source1.T,source2.T)
    source = np.stack((source1, source2))
    
    # randomly generated mixing matrix
    np.random.seed(0)
    #mixing_matrix = np.random.rand(2,2)
    mixing_matrix = np.array([[1, 0.5], [0.5, 1]])

    
    X = np.matmul(source.T, mixing_matrix)
    
    X1 = X[:,0]
    mx1 = np.mean(X1)
    X1 = np.reshape(X1, (n,n))
    
    #print(X1.min(), X1.max(), X1.mean())
    
    X2 = X[:,1]
    mx2 = np.mean(X2)
    X2 = np.reshape(X2, (n,n))
    
    mx = np.array([mx1, mx2])
    #print(X2.min(), X2.max(), X2.mean())

    
    # FastICA algorithm
    #X[:,0] = X[:,0] - mx1
    #X[:,1] = X[:,1] - mx2
    
    ica = FastICA(n_components=2, fun = 'cube')
    source_estimated = ica.fit_transform(X)
    mixing_estimated = ica.mixing_
    
    ms = np.dot(np.linalg.inv(mixing_estimated),mx)
    
    #source_estimated[:,0] = source_estimated[:,0]+ms[0]
    #source_estimated[:,1] = source_estimated[:,1]+ms[1]
    
    #print("mixing matrix estimated = ", mixing_estimated)
    
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(source_estimated.T))
    (sdr_ref, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(source), np.asarray(X.T))
    
    SDR_improv[0, it] = np.mean(sdr) - np.mean(sdr_ref)
    rdr_it[0, it] = rdr_this
    
plt.figure
plt.subplot(211)
plt.plot(size_v, SDR_improv[0,:],'-*')
plt.xlabel('Image size')
plt.ylabel('SDR improvement (dB)')
plt.grid()
plt.show

plt.subplot(212)
plt.plot(size_v, 1-rdr_it[0,:],'-*')
plt.xlabel('Image size')
plt.ylabel('One minus rdr measure')
plt.grid()
plt.show