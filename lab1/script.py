import numpy as np
import matplotlib.pyplot as plt

# LAB
# M = np.zeros((16, 16))
# M[1:8, 1:8] = 1

# fig, ax = plt.subplots(2, 3, figsize=(10, 5))

# res = 4
# sampler = np.linspace(0, 4*np.pi, res)  # probkowanie przedzialu
# reflectance = np.sin(sampler)  # probkowanie sinusa przedzialem samplera

# # mnozenie regula broadcastingu
# # https://numpy.org/doc/stable/user/basics.broadcasting.html
# img = reflectance[np.newaxis, :] * reflectance[:, np.newaxis]

# # normalizacja i przyciecie obrazu do [0,1]
# img -= np.min(img)
# img /= np.max(img)
# img = np.clip(img, 0, 1)

# depth = 2  # kwantyzacja na tylu bitach
# dmin, dmax = (0, np.power(2, depth) - 1)  # zakres skali szarosci
# # kwantyzacja obrazu [0, 1] do dyskretnych wartosci [dmin, dmax]
# digital_image = np.rint(img * dmax)

# # ax[0, 0].imshow(M, cmap='binary', vmin=0, vmax=16)
# ax[0, 0].plot(sampler)
# ax[0, 0].set_title("sampling space")

# ax[0, 1].plot(reflectance)
# ax[0, 1].set_title("reflectance space")

# ax[0, 2].plot(sampler, reflectance)
# ax[0, 2].set_title("sampled sine")

# ax[1, 0].imshow(img, cmap='binary')
# ax[1, 0].set_title('%.3f - %.3f' % (np.min(img), np.max(img)))

# ax[1, 1].imshow(digital_image, cmap='binary')
# ax[1, 1].set_title('Obraz w %i-el. skali szarości' %
#                    (np.power(2, depth)))

# plt.tight_layout()
# plt.savefig('foo.png')

# ACTUAL TASK

resolutions = [4, 8, 16, 32, 256]
depths = [2, 4, 8]
fig, ax = plt.subplots(5, 5, figsize=(10, 10))

for i, res in enumerate(resolutions):
    sampler = np.linspace(0, 4*np.pi, res)
    reflectance = np.sin(sampler)

    ax[i, 0].plot(sampler, reflectance)
    ax[i, 0].set_title('Sampled sine')

    img = reflectance[:, np.newaxis] * reflectance[np.newaxis, :]
    img -= np.min(img)
    img /= np.max(img)
    np.clip(img, 0, 1)

    ax[i, 1].imshow(img, cmap='binary')
    ax[i, 1].set_title('Sampled image')

    for j, depth in enumerate(depths):
        dmin, dmax = (0, np.power(2, depth) - 1)
        digital_img = np.rint(img * dmax)
        ax[i, j + 2].imshow(digital_img, cmap='binary')
        ax[i, j + 2].set_title('%.3f -- %.3f' %
                               (np.min(digital_img), np.max(digital_img)))

plt.tight_layout()
plt.savefig('task.png')
