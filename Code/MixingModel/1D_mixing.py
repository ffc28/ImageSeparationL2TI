from scipy import signal, special
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import mir_eval
from rdc import rdc

# the signals
t = np.linspace(-np.pi, np.pi, 1000)
sine = np.sin(4*t)  #for np.sin(3*t), rdc is higher => bad performance of ICA
sawtooth = signal.sawtooth(2 * np.pi * t)

#plotting
plt.figure()
plt.plot(t, sawtooth, label='sawtooth')
plt.plot(t, sine, label = 'sine')
plt.legend()
plt.show

#test the independence
print("rdc = ", rdc(sine,sawtooth))

# mixing the reference sources
mixing_matrix = [[1, 0.5], [0.5, 1]]
reference_sources = np.c_[sawtooth, sine]
#sources /= sources.std(axis = 0) #standardize it
M = np.dot(reference_sources, np.transpose(mixing_matrix))

#plotting the mixtures
plt.figure()
plt.plot(t, M[:,0], label = 'mixed signal1')
plt.plot(t, M[:,1], label = 'mixed signal2')
plt.legend()
plt.show

# Compute ICA
ica = FastICA(n_components=2)
estimated_sources = ica.fit_transform(M)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
print("estimated mixing_matrix = ", A_)

#plotting estimated sources
plt.figure()
plt.plot(t,np.transpose(estimated_sources)[0], color = 'r', label = 'source1')
plt.plot(t,np.transpose(estimated_sources)[1], color = 'g', label = 'source2')
plt.title("the estimated sources")
plt.legend()
plt.show()

#evaluating the performance
(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources.T, estimated_sources.T)
print("sdr, sir, sar, perm =", sdr, sir, sar, perm)