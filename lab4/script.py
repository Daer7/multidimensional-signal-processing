import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters


image = data.chelsea()
noise = np.random.normal(0.0, 100.0, size=(*image.shape, 1000))

image_noisy = image[:, :, :, np.newaxis] + noise
image_noisy -= np.min(image_noisy)
image_noisy /= np.max(image_noisy)
image_mean = np.mean(image_noisy, axis=-1)

fig, ax = plt.subplots(2, 1)

print(image_mean.shape)
ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(image_mean)
ax[1].set_title("After noise reduction")
plt.tight_layout()
plt.savefig("foo.png")
