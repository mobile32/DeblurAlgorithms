import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, restoration

path = "C:/Users/mobil/PycharmProjects/Piao/Content/btcqr.png"
img = data.load(path)

qr = color.rgb2gray(img)

from scipy.signal import convolve2d as conv2
psf = np.ones((5, 5)) / 25
qr = conv2(qr, psf, 'same')
qr += 0.1 * qr.std() * np.random.standard_normal(qr.shape)

deconvolved, _ = restoration.unsupervised_wiener(qr, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

plt.gray()

ax[0].imshow(qr, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()