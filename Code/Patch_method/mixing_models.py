# Mixing two images in non-linear way

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from rdc import rdc


def load_images(image1, image2, n, x1=0, y1=0, x2=0, y2=0, show_images = True):
    """
    Function that loads, converts, rescale and show the images 
    
    Parameters
    ----------
    image1: the first image to be loaded
    image2: the second image to be loaded
    n: int, the width, length of the image
    x1: int, 0 by default, shifts image1 along the x-axis
    y1: int, 0 by default, shifts image1 along the y-axis
    x2: int, 0 by default, shifts image2 along the x-axis
    y2: int, 0 by default, shifts image2 along the y-axis
    show_images: boolean, True by default, plots the images if True

    Return
    -----------
    img1_gray_re: the first image
    img2_gray_re: the second image
    """
    # load images and convert them

    img1=mpimg.imread(image1)
    img2=mpimg.imread(image2)

    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)

    img1_gray_re = img1_gray[y1:y1+n, x1:x1+n]
    img2_gray_re = img2_gray[y2:y2+n, x2:x2+n]

    if show_images == True:
        plt.figure()
        plt.imshow(img1_gray_re, cmap='gray')
        plt.title("Ground truth 1")
        plt.show

        plt.figure()
        plt.imshow(img2_gray_re, cmap='gray')
        plt.title("Ground truth 2")
        plt.show()
    return img1_gray_re, img2_gray_re



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
    assert image1.shape == image2.shape
    n = image1.shape

    # mixing the images
    source1 = np.matrix(image1)
    source1 = source1.flatten('F') #column wise

    source2 = np.matrix(image2)
    source2 = source2.flatten('F') #column wise

    S = np.stack((source1, source2))

    #X1 = np.multiply(source1, np.power((source2/255),q2)) 
    #X2 = np.multiply(source2, np.power((source1/255),q1))

    X1 = np.multiply(source1 , (np.exp(-q2 * (1 - source2)))) # if 0<intensities<255 then divide by 255 
    X2 = np.multiply(source2 , (np.exp(-q1 * (1 - source1))))
    
    #rdc_images = rdc(X1, X2)
    #print("RDC =", rdc_images)
    
    X = np.stack((X1, X2))

    
    # Show mixed images
    if (show_images == True):
        X1 = np.reshape(X1, n)
        X2 = np.reshape(X2, n)

        plt.figure()
        plt.imshow(X1.T, cmap='gray')
        plt.title("Mixed image 1")
        plt.show

        plt.figure()
        plt.imshow(X2.T, cmap='gray')
        plt.title("Mixed image 2")
        plt.show()
    return S, X
    
def linear_mixture(image1, image2, mixing_matrix = [[1, 0.5],[0.5, 1]], show_images = True):
    """
    Function that mixes to images linearly
    
    Parameters
    ----------
    image1: the first image to be mixed
    image2: the second image to be mixed
    mixing_matrix: the matrix to which the images will be multiplied
    show_images: boolean, True by default, plots the mixed images if True

    Return
    -----------
    S: the input images, flattened and stacked 
    X: the mixed signals, flattened and stacked
    """
    assert np.shape(mixing_matrix) == (2,2)

    n = image1.shape

    source1 = image1.flatten('F') #column wise
    source2 = image2.flatten('F') #column wise

    source1 = source1 - np.mean(source1)
    source2 = source2 - np.mean(source2)

    source1 = source1/np.linalg.norm(source1)
    source2 = source2/np.linalg.norm(source2)

    S = np.stack((source1, source2))

    X = np.matmul(mixing_matrix, S)

    X1 = X[0,:]
    X2 = X[1,:]

    rdc_images = rdc(X1, X2)
    print("RDC =", rdc_images)
    
    if show_images == True:
        X1 = np.reshape(X1, n)
        X2 = np.reshape(X2, n)

        plt.figure()
        plt.imshow(X1.T, cmap='gray')
        plt.title("Mixed image 1")
        plt.show

        plt.figure()
        plt.imshow(X2.T, cmap='gray')
        plt.title("Mixed image 2")
        plt.show()
    return S.T, X
