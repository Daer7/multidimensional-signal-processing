import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters


image = data.chelsea()
img_g = image[:, :, 0]  # green channel

fig, ax = plt.subplots(8, 2, figsize=(10, 15))

baseline = np.linspace(0, 255, 256)
img_id = baseline[img_g]
ax[0, 0].imshow(img_id)
ax[0, 1].plot(baseline)

negation = 255 - baseline
neg_image = negation[img_g]
ax[1, 0].imshow(neg_image)
ax[1, 1].plot(negation)

step = np.zeros(256)
step[128:160] = 255
step_img = step[img_g]
ax[2, 0].imshow(step_img)
ax[2, 1].plot(step)

sine1 = np.sin(baseline * 2 * np.pi / 255)
sine1_img = sine1[img_g]
ax[3, 1].plot(sine1)
ax[3, 0].imshow(sine1_img)

sine2 = np.sin(baseline * 4 * np.pi / 255)
sine2_img = sine2[img_g]
ax[4, 0].imshow(sine2_img)
ax[4, 1].plot(sine2)

sine3 = np.sin(baseline * 6 * np.pi / 255)
sine3_img = sine3[img_g]
ax[5, 0].imshow(sine3_img)
ax[5, 1].plot(sine3)

gamma1 = baseline ** (1 / 3)
gamma1 -= np.min(gamma1)
gamma1 /= np.max(gamma1)
gamma1_img = gamma1[img_g]
ax[6, 0].imshow(gamma1_img)
ax[6, 1].plot(gamma1)

gamma2 = baseline ** 5
gamma2 -= np.min(gamma2)
gamma2 /= np.max(gamma2)
gamma2_img = gamma2[img_g]
ax[7, 0].imshow(gamma2_img)
ax[7, 1].plot(gamma2)


plt.suptitle("Image transformations")
plt.tight_layout()
plt.savefig("foo2.png")
