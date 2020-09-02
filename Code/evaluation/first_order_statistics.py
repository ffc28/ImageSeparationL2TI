""" first and second order statistics """

import matplotlib.pyplot as plt
import numpy as np
import museval.metrics as mmetrics

import radiomics
from radiomics import featureextractor

from scipy.stats import kurtosis, skew
from mixing_models import linear_mixture, load_images
from sklearn.decomposition import FastICA
from rdc import rdc


def mean_kurtosis(S):
    return (np.abs(kurtosis(S[0, :])) + np.abs(kurtosis(S[1, :])))/2

def mean_skewness(S):
    return (np.abs(skew(S[0, :])) + np.abs(skew(S[1, :])))/2

n = 256

np.random.seed(15)
mixing_matrix = np.random.rand(2,2) 
print(mixing_matrix)
i = 0


all_kurtosis_sources = []
all_skewness_sources = []
all_kurtosis_mixtures = []
all_skewness_mixtures = []
all_kurtosis_estimated =[]
all_skewness_estimated = []
all_rdc = []
all_sdr = []
all_sdr_improved = []

for i in range(15):
    print("Set = ",  i+1)
    source1, source2 = load_images('./images/set'+ str(i+1) + '_pic1.png', './images/set'+ str(i+1) + '_pic2.png', n, show_images=False)
    S, X = linear_mixture(source1, source2, mixing_matrix = mixing_matrix, show_images=False)
    
    #Reference SDR
    (sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(S), np.asarray(X))
    print('The mean value of the reference SDR is: ', np.mean(sdr_ref))

    #RDC of sources
    rdc_i = rdc(S[0, :], S[1, :])
    all_rdc.append(rdc_i)
    print("rdc of sources = ", rdc_i)

    #Kurtosis of sources
    kurtosis_i = mean_kurtosis(S)
    all_kurtosis_sources.append(kurtosis_i)
    print("mean kurtosis of sources = ", kurtosis_i)
    print("kurtosis of sources  = ", kurtosis(S[0, :]), kurtosis(S[1, :]))

    #Skewness of sources
    skewness_i = mean_skewness(S)
    all_skewness_sources.append(skewness_i)
    print("mean skewness of sources = ", skewness_i)
    print("skewness of sources = ", skew(S[0, :]), skew(S[1, :]))

    #Kurtosis of mixtures
    kurtosis_i = mean_kurtosis(X)
    all_kurtosis_mixtures.append(kurtosis_i)
    print("mean kurtosis of mixtures = ", kurtosis_i)
    print("kurtosis of mixtures  = ", kurtosis(X[0, :]), kurtosis(X[1, :]))

    #Skewness of mixtures
    skewness_i = mean_skewness(X)
    all_skewness_mixtures.append(skewness_i)
    print("mean skewness of mixtures = ", skewness_i)
    print("skewness of mixtures = ", skew(X[0, :]), skew(X[1, :]))


    #ica separation
    ica = FastICA(n_components=2, fun = 'cube', max_iter = 20000)
    estimated_source = ica.fit_transform(X.T)
    mixing_estimated = ica.mixing_

    #SDR of separation
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(S), np.asarray(estimated_source.T))
    print("SDR of ICA = ", sdr, np.mean(sdr))
    all_sdr.append(np.mean(sdr))
    print("SDR_improved = ", np.mean(sdr) - np.mean(sdr_ref))
    all_sdr_improved.append(np.mean(sdr) - np.mean(sdr_ref))

    #Kurtosis of estimated sources
    kurtosis_i = mean_kurtosis(estimated_source.T)
    all_kurtosis_estimated.append(kurtosis_i)
    print("mean kurtosis of estimated sources = ", kurtosis_i)
    print("kurtosis of estimated sources  = ", kurtosis(estimated_source.T[0, :]), kurtosis(estimated_source.T[1, :]))

    #Skewness of estimated sources
    skewness_i = mean_skewness(estimated_source.T)
    all_skewness_estimated.append(skewness_i)
    print("mean skewness of estimated sources = ", skewness_i)
    print("skewness of estimated sources = ", skew(estimated_source.T[0, :]), skew(estimated_source.T[1, :]))



    print("*******")


for i in range(5):
    print("Set = ",  i+16)
    source1, source2 = load_images('./images_hard/set'+ str(i+1) + '_pic1.png', './images_hard/set'+ str(i+1) + '_pic2.png', n, show_images=False)
    S, X = linear_mixture(source1, source2, mixing_matrix = mixing_matrix, show_images=False)
    
    #Reference SDR
    (sdr_ref, sir_ref, sar, perm) = mmetrics.bss_eval_sources(np.asarray(S), np.asarray(X))
    print('The mean value of the reference SDR is: ', np.mean(sdr_ref))

    #RDC of sources
    rdc_i = rdc(S[0, :], S[1, :])
    all_rdc.append(rdc_i)
    print("rdc of sources = ", rdc_i)

    #Kurtosis of sources
    kurtosis_i = mean_kurtosis(S)
    all_kurtosis_sources.append(kurtosis_i)
    print("mean kurtosis of sources = ", kurtosis_i)
    print("kurtosis of sources  = ", kurtosis(S[0, :]), kurtosis(S[1, :]))

    #Skewness of sources
    skewness_i = mean_skewness(S)
    all_skewness_sources.append(skewness_i)
    print("mean skewness of sources = ", skewness_i)
    print("skewness of sources = ", skew(S[0, :]), skew(S[1, :]))

    #Kurtosis of mixtures
    kurtosis_i = mean_kurtosis(X)
    all_kurtosis_mixtures.append(kurtosis_i)
    print("mean kurtosis of mixtures = ", kurtosis_i)
    print("kurtosis of mixtures  = ", kurtosis(X[0, :]), kurtosis(X[1, :]))

    #Skewness of mixtures
    skewness_i = mean_skewness(X)
    all_skewness_mixtures.append(skewness_i)
    print("mean skewness of mixtures = ", skewness_i)
    print("skewness of mixtures = ", skew(X[0, :]), skew(X[1, :]))

    #ica separation
    ica = FastICA(n_components=2, fun = 'cube', max_iter = 20000)
    estimated_source = ica.fit_transform(X.T)
    mixing_estimated = ica.mixing_

    #SDR of separation
    (sdr, sir, sar, perm) = mmetrics.bss_eval_sources(np.asarray(S), np.asarray(estimated_source.T))
    print("SDR of ICA = ", sdr, np.mean(sdr))
    all_sdr.append(np.mean(sdr))
    print("SDR_improved = ", np.mean(sdr) - np.mean(sdr_ref))
    all_sdr_improved.append(np.mean(sdr) - np.mean(sdr_ref))

    #Kurtosis of estimated sources
    kurtosis_i = mean_kurtosis(estimated_source.T)
    all_kurtosis_estimated.append(kurtosis_i)
    print("mean kurtosis of estimated sources = ", kurtosis_i)
    print("kurtosis of estimated sources  = ", kurtosis(estimated_source.T[0, :]), kurtosis(estimated_source.T[1, :]))

    #Skewness of estimated sources
    skewness_i = mean_skewness(estimated_source.T)
    all_skewness_estimated.append(skewness_i)
    print("mean skewness of estimated sources = ", skewness_i)
    print("skewness of estimated sources = ", skew(estimated_source.T[0, :]), skew(estimated_source.T[1, :]))



    print("*******")


print("\nall_rdc = ", all_rdc) 
print("\n all_sdr = ", all_sdr)

print("\n all_kurtosis_sources = ", all_kurtosis_sources)
print("\nall_kurtosis_mixtures = ", all_kurtosis_mixtures)
print("\nall_kurtosis_estimated = ", all_kurtosis_estimated)

print("\nall_skewness_sources = ", all_skewness_sources)
print("\nall_skewness_estimated = ", all_skewness_estimated)
print("\nall_skewness_mixtures = ", all_skewness_mixtures)


print("\nall_sdr_improved =", all_sdr_improved)
