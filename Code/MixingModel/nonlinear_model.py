# Mixing two images in non-linear way

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from rdc import rdc

def nonlinear_mixture(image1, image2, q1, q2, show_images = True):
    """
    Function that mixes to images nonlinearly
    
    Parameters
    ----------
    image1: the first image to be mixed
    image2: the second image to be mixed
    q1: float, interference level of the first image
    q2: float, interference level of the second image
    show_images: boolean, True by default, plots the mixed images if True

    Return
    -----------
    S: the input images, flattened and stacked 
    X: the mixed signals, flattened and stacked
    """

    # mixing the images
    source1 = np.matrix(image1)
    source1 = source1.flatten('F') #column wise

    source2 = np.matrix(image2)
    source2 = source2.flatten('F') #column wise

    S = np.stack((source1, source2))

    rdc_images = rdc(source1.T, source2.T)
    print("RDC =", rdc_images)
    
    #X1 = np.multiply(source1, np.power((source2/255),q2)) 
    #X2 = np.multiply(source2, np.power((source1/255),q1))

    X1 = np.multiply(source1 , (np.exp(-q2 * (1 - source2)))) # if 0<intensities<255 then divide by 255 
    X2 = np.multiply(source2 , (np.exp(-q1 * (1 - source1))))

    X = np.stack((X1, X2))
    
    # Show mixed images
    if (show_images == True):
        X1 = np.reshape(X1, (n,n))
        X2 = np.reshape(X2, (n,n))

        plt.figure()
        plt.imshow(X1.T, cmap='gray')
        plt.title("Mixed image 1")
        plt.show

        plt.figure()
        plt.imshow(X2.T, cmap='gray')
        plt.title("Mixed image 2")
        plt.show()
    return S, X
    