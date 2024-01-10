import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.signal import convolve
from .funcs import *
from .settings import *
from ixpeobssim.irf import load_psf

GROUND_BLUR = 2#2.45# Arcsec
OBSSIM_BLUR = 0#1.8# Arcsec

class PSF:
    def rotate(image, rotation_deg):
        '''Rotate an image by an angle `rotation_deg` in degrees'''
        return rotate(np.flip(image, axis=1), -(rotation_deg + 90), cval=0, reshape=False)

    def anti_rotate(image, rotation_deg):
        '''Rotate an image backwards by an angle `rotation_deg` in degrees'''
        return np.flip(rotate(image, (rotation_deg + 90), cval=0, reshape=False), axis=1)

    def sky_cal(detector_index, source, rotation, psf_origin="merge-nn", clip=True):
        '''Load the sky-calibrated PSFs.
        ## Arguments
        - detector_index: 0 for DU1, 1 for DU2, and 2 for DU3
        - source: a Source object used to set the width of the PSF
        - rotation: angle by which to rotate the PSF to get it into the detector's frame
        - psf_origin: name of the sky calibrated PSF to use

        ## Returns
        The PSF of the detector.
        '''
        with fits.open(f'{LEAKAGE_DATA_DIRECTORY}/sky-psfs/{psf_origin}/PSF_MMA{detector_index+1}.fits') as hdul:
            initial_pixel_width = hdul[1].header["PIXANG"]
            psf = hdul[1].data

        return PSF(psf, initial_pixel_width, 0, source, rotation, detector_index, clip)

    def ground_cal(detector_index, source, rotation, ground_blur=GROUND_BLUR, clip=True):
        '''Load the ground-calibrated PSFs.
        ## Arguments
        - detector_index: 0 for DU1, 1 for DU2, and 2 for DU3
        - source: a Source object used to set the width of the PSF
        - rotation: angle by which to rotate the PSF to get it into the detector's frame
        - ground_blur: amount by which to blur the ground PSFs. Default is a value manually tuned to match leakage patterns

        ## Returns
        The PSF of the detector.
        '''
        with fits.open(f'../../data/ground_cal_psfs/PSF_MMA{detector_index+1}.fits') as hdul:
            initial_pixel_width = hdul[1].header["PIXANG"]
            psf = hdul[1].data

        return PSF(psf, initial_pixel_width, ground_blur, source, rotation, detector_index, clip)

    def obssim(detector_index, source, rotation, obssim_blur=OBSSIM_BLUR, clip=True):
        '''Load the IXPEobssim PSFs, which at the time of publication were symmetric.
        ## Arguments
        - detector_index: 0 for DU1, 1 for DU2, and 2 for DU3
        - source: a Source object used to set the width of the PSF
        - rotation: angle by which to rotate the PSF to get it into the detector's frame
        - ground_blur: amount by which to blur the ground PSFs. Default is a value manually tuned to match leakage patterns

        ## Returns
        The PSF of the detector.
        '''

        psf_ixpe = load_psf('ixpe:obssim:v11',du_id=detector_index+1)
        initial_pixel_width = 1
        line = np.arange(len(psf_ixpe.x), dtype=float) * initial_pixel_width
        line -= line[-1]/2
        xs, ys = np.meshgrid(line, line)
        psf = psf_ixpe(np.sqrt(xs**2 + ys**2))

        return PSF(psf, initial_pixel_width, obssim_blur, source, rotation, detector_index, clip)
        
    def __init__(self, image, current_pixel_width, blur_width, source, rotation, detector_index, clip=False):
        '''Get the PSF for a given image. Please don't use this function unless necessary. Use the `sky-cal`, `ground-cal`, and `obssim` functions to load specific PSFs.
        '''
        # Set information about the PSF
        self.pixel_width = current_pixel_width
        self.det = detector_index
        
        # Rotate to the ra-dec frame
        self.psf = image
        if rotation != 0:
            self.psf = PSF.rotate(self.psf, rotation * 180 / np.pi)
        
        # Crop some of PSF that won't be used
        self.pixel_centers = np.arange(len(self.psf), dtype=float) * current_pixel_width
        self.pixel_centers -= self.pixel_centers[-1] / 2

        if clip:
            clip_size = int(len(source.pixel_centers) * source.pixel_size / current_pixel_width * 3)
            self.clip(clip_size)

        # Blur
        xs, ys = np.meshgrid(self.pixel_centers, self.pixel_centers)
        self.blur = blur_width
        if blur_width is not None and blur_width > 0:
            dist2 = xs * xs + ys * ys

            blur = np.exp(-(dist2) / (2 * blur_width**2)) # Gaussian
            # blur = 1 / (1 + dist2/blur_width**2) # Lorentzian

            blur /= np.sum(blur)
            self.psf = convolve(self.psf, blur, mode="same")

        # Compute derivatives
        self.compute_kernels(source.use_nn)

        # Zoom
        zoom_ratio = current_pixel_width / source.pixel_size
        self.psf = super_zoom(self.psf, zoom_ratio, force_odd=True)
        renormalize = 1 / np.sum(self.psf)
        self.psf *= renormalize
        self.d_zs = super_zoom(self.d_zs, zoom_ratio, force_odd=True) * renormalize
        self.d_qs = super_zoom(self.d_qs, zoom_ratio, force_odd=True) * renormalize
        self.d_us = super_zoom(self.d_us, zoom_ratio, force_odd=True) * renormalize
        self.d_zk = super_zoom(self.d_zk, zoom_ratio, force_odd=True) * renormalize
        self.d_qk = super_zoom(self.d_qk, zoom_ratio, force_odd=True) * renormalize
        self.d_uk = super_zoom(self.d_uk, zoom_ratio, force_odd=True) * renormalize
        self.d_xk = super_zoom(self.d_xk, zoom_ratio, force_odd=True) * renormalize
        self.d_yk = super_zoom(self.d_yk, zoom_ratio, force_odd=True) * renormalize

        self.pixel_centers = np.linspace(self.pixel_centers[0], self.pixel_centers[-1], len(self.psf))
        self.pixel_width = self.pixel_centers[1] - self.pixel_centers[0]

        # Now that derivatives are computed, clip tighter.
        if clip:
            self.clip(len(source.pixel_centers) * 2)
        
        assert(self.psf.shape[0] % 2 == 1)

    def clip(self, new_width):
        '''Clip the PSF to a new width in pixels'''
        if new_width % 2 == 0:
            new_width += 1 # Force to be odd
        if new_width >= self.psf.shape[0]:
            return
        
        middle = self.psf.shape[0]//2
        self.psf = self.psf[middle-new_width//2:middle+new_width//2+1,
        middle-new_width//2:middle+new_width//2+1]

        try:
            self.d_zs = self.d_zs[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_qs = self.d_qs[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_us = self.d_us[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_zk = self.d_zk[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_qk = self.d_qk[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_uk = self.d_uk[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_xk = self.d_xk[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
            self.d_yk = self.d_yk[middle-new_width//2:middle+new_width//2+1,middle-new_width//2:middle+new_width//2+1]
        
            # Keep the PSF normalized
            self.d_zs /= np.sum(self.psf)
            self.d_qs /= np.sum(self.psf)
            self.d_us /= np.sum(self.psf)
            self.d_zk /= np.sum(self.psf)
            self.d_qk /= np.sum(self.psf)
            self.d_uk /= np.sum(self.psf)
            self.d_xk /= np.sum(self.psf)
            self.d_yk /= np.sum(self.psf)
        except:
            pass

        self.psf /= np.sum(self.psf)
        self.pixel_centers = self.pixel_centers[middle-new_width//2:middle+new_width//2+1]

    def compute_kernels(self, use_nn):
        '''Compute derivatives of the PSF.'''
        self.d_zs = convolve(self.psf, KERNEL_ZS / self.pixel_width**2, mode="same") / np.nansum(self.psf)
        self.d_qs = convolve(self.psf, KERNEL_QS / self.pixel_width**2, mode="same") / np.nansum(self.psf)
        self.d_us = convolve(self.psf, KERNEL_US / self.pixel_width**2, mode="same") / np.nansum(self.psf)
        self.d_zk = convolve(self.psf, KERNEL_ZK / self.pixel_width**4, mode="same") / np.nansum(self.psf)
        self.d_qk = convolve(self.psf, KERNEL_QK / self.pixel_width**4, mode="same") / np.nansum(self.psf)
        self.d_uk = convolve(self.psf, KERNEL_UK / self.pixel_width**4, mode="same") / np.nansum(self.psf)
        self.d_xk = convolve(self.psf, KERNEL_XK / self.pixel_width**4, mode="same") / np.nansum(self.psf)
        self.d_yk = convolve(self.psf, KERNEL_YK / self.pixel_width**4, mode="same") / np.nansum(self.psf)
        self.psf = self.psf / np.nansum(self.psf)

    def save(self, directory, header=None):
        '''Save this PSF to a new directory'''
        blank = fits.PrimaryHDU([])
        if header is None:
            header = fits.Header();
        header['PIXANG'] = (self.pixel_width, "Angular size of each pixel in arcsec")
        hdu = fits.ImageHDU(self.psf, header=header)
        hdul = fits.HDUList([blank, hdu])
        hdul.writeto(f"{directory}/PSF_MMA{self.det+1}.fits", overwrite=True)