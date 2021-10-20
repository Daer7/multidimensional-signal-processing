import numpy as np
import matplotlib.pyplot as plt

depth = 2
drange = (-1, 1)
resolution = 256
n_iter = 256
N = np.power(2, depth) - 1

prober = np.linspace(0, 8*np.pi, resolution)
prober = np.sin(prober)

perfect_image = prober[:, np.newaxis] * prober[np.newaxis, :]

n_matrix = np.zeros(perfect_image.shape)
o_matrix = np.zeros(perfect_image.shape)

for i in range(n_iter):
    noise = np.random.normal(0, 1, perfect_image.shape)
    print(np.max(noise))
    n_image = perfect_image + noise
    o_image = np.copy(perfect_image)

    n_image -= np.min(n_image)

    n_image /= np.max(n_image)
    n_image = np.clip(n_image, 0, 1)
    n_dimg = np.rint(n_image * N)

    o_image -= np.min(o_image)
    o_image /= np.max(o_image)
    o_image = np.clip(o_image, 0, 1)
    o_dimg = np.rint(o_image * N)

    n_matrix += n_dimg
    o_matrix += o_dimg

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0, 0].imshow(perfect_image, cmap='binary')
ax[1, 0].imshow(noise, cmap='binary')

ax[0, 1].imshow(o_dimg, cmap='binary')
ax[1, 1].imshow(n_dimg, cmap='binary')

ax[0, 2].imshow(o_matrix, cmap='binary')
ax[1, 2].imshow(n_matrix, cmap='binary')

print(np.min(o_matrix), np.max(o_matrix), np.min(n_matrix), np.max(n_matrix))

plt.tight_layout()
plt.savefig('task.png')

# Lepiej odwzorowany jest obraz w wersji zaszumionej (mimo niewielkiej głębi obrazu).
