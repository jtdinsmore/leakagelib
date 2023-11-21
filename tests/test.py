import matplotlib.pyplot as plt
import numpy as np
import sys, time
sys.path.append("../..")
from leakagelib import funcs, PSF, Source

plt.style.use("root")

def test_zoom():
    image = np.zeros((27,27)).astype(float)
    image[13,13] = 1
    image[1,13] = 1
    image[8,16] = 1
    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    axs[0].imshow(image)
    axs[1].imshow(funcs.integrate_zoom(image, 0.7))
    axs[2].imshow(funcs.integrate_zoom(image, 0.4))
    fig.savefig("figs/zoom.png")

def test_vectorized_zoom():
    image = np.random.random(size=(127,127))

    start = time.time()
    a = funcs.integrate_zoom_unvectorized(image, 0.7)
    print(time.time() - start)
    start = time.time()
    b = funcs.integrate_zoom(image, 0.7)
    print(time.time() - start)
    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    axs[0].imshow(image)
    axs[1].imshow(a)
    axs[2].imshow(b)
    fig.savefig("figs/zoom_vectorized.png")
    print(np.max(np.abs(a - b)))

def psf_zoom():
    det = 2
    pixel_sizes = [5,4,3,2,1]
    fig, axs = plt.subplots(figsize=(4,4*len(pixel_sizes)), nrows=len(pixel_sizes))
    for (ax, pixel_size) in zip(axs, pixel_sizes):
        n_pixels = int(80 / pixel_size)
        if n_pixels % 2 == 0:
            n_pixels += 1
        source = Source.delta(False, n_pixels, pixel_size)
        psf = PSF.sky_cal(det, source, 0)
        ax.imshow(psf.psf)
    fig.savefig("figs/psf.png")

if __name__ == "__main__":
    test_zoom()
    test_vectorized_zoom()
    psf_zoom()