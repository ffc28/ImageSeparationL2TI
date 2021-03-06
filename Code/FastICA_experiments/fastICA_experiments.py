"""Mean SSIM of FastICA"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mixing_models import load_images, linear_mixture
from sklearn.decomposition import FastICA
from skimage.metrics import structural_similarity as ssim
from functions import unflatten

n = 256
image_size = (n, n)
max_it = 100
d = 45
i=4

source1, source2 = load_images(f'../images_hard/set{i+1}_pic1.png', f'../images_hard/set{i+1}_pic2.png', 256, show_images= False)
theta = np.radians(d)
mixing_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

sources, mixtures = linear_mixture(source1, source2, mixing_matrix = mixing_matrix, show_images = False)

mix1, mix2 = unflatten(mixtures, image_size)
plt.figure()
plt.suptitle(f'Mixtures, theta = {d}')
plt.subplot(121)
plt.imshow(mix1, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(mix2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show

ica = FastICA(n_components=2, fun = 'cube', max_iter = 20000)
estimated_sources = ica.fit_transform(mixtures.T)
mixing_estimated = ica.mixing_

estimated1, estimated2 = unflatten(estimated_sources.T, image_size)

ssim1 = ssim(source1, estimated1, data_range = estimated1.max()- estimated1.min())
ssim2 = ssim(source2, estimated2, data_range = estimated2.max()- estimated2.min())

mean_ssim = (ssim1 + ssim2)/2 
print(mean_ssim)

# Show estimated sources

plt.figure(figsize = (6,4))
plt.suptitle(f"Estimated sources with FastICA\nSet {i+1}, Mean SSIM = {round(mean_ssim, 4)}")
plt.subplot(121)
plt.imshow(estimated1, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(estimated2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./images/estimated_sources_ica_theta{d}_set{i+1}.png')
plt.show()