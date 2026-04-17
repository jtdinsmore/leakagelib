import numpy as np
import logging
from astropy.io import fits
from .spectrum import EnergyDependence
from .funcs import integrate_zoom, _convolve

WARN = True

logger = logging.getLogger("leakagelib")

def _pad_image(image, num_pixels):
    '''
    Zero-pad an image

    Parameters
    ----------
    image : array-like
        Image to be padded

    num_pixels : int
        Number of images to add on each side. They will be zero-valued.

    Returns
    -------
    array-like
        Image
    '''
    # Pad the image
    if (len(image) + num_pixels) % 2 == 1:
        # Pad one row
        initial_width = image.shape[0]
        image = np.concatenate((image, np.zeros((1, initial_width))), axis=0)
        image = np.concatenate((image, np.zeros((initial_width+1, 1))), axis=1)
    if len(image) != num_pixels:
        initial_width = image.shape[0]
        add_halfwidth = (num_pixels - len(image)) // 2
        image = np.concatenate((
            np.zeros((add_halfwidth, initial_width)),
            image,
            np.zeros((add_halfwidth, initial_width))
        ), axis=0)
        image = np.concatenate((
            np.zeros((initial_width + 2 * add_halfwidth, add_halfwidth)),
            image,
            np.zeros((initial_width + 2 * add_halfwidth, add_halfwidth))
        ), axis=1)
    return image

def _process_file(file_name, num_pixels, target_pixel_size, hdui=0, rescale=False):
    """
    Load a FITS file, zoom to the correct scale, and return the resulting image
    with the number of pixels and pixel size in arcseconds.

    Parameters
    ----------
    file_name : str
        Name of the file to load.
    num_pixels : int
        Number of pixels to use in the output image.
    target_pixel_size : float or None
        Width of each pixel in arcseconds for the returned image. None implies use the source pixel size.
    hdui : int
        HDU index of the images to load.

    Returns
    -------
    tuple of (list of ndarray, int, float)
        - images: list of loaded and scaled images
        - num_pixels: width of the image(s)
        - target_pixel_size: width of each pixel in arcseconds
    """

    with fits.open(file_name) as hdul:
        image = hdul[hdui].data
        if image.shape[0] != image.shape[1]:
            raise Exception("LeakageLib only deals with square images")
        
        source_pixel_size = hdul[hdui].header["CDELT2"] * 3600 # Arcsec
        if target_pixel_size is None:
            target_pixel_size = source_pixel_size
        elif np.abs(target_pixel_size - source_pixel_size) > 1e-5:
            zoom_ratio = source_pixel_size / target_pixel_size
            image = integrate_zoom(hdul[hdui].data, zoom_ratio, force_odd=True)
            if rescale:
                image *= zoom_ratio**2
    image = np.transpose(image, (1,2,0))

    # Change the image size to match requirements
    if num_pixels is None:
        num_pixels = image.shape[0]
    else:
        if len(image) > num_pixels:
            middle = len(image) // 2
            add = 0 if num_pixels % 2 == 0 else 1
            image = image[
                middle - num_pixels // 2:middle + num_pixels // 2 + add,
                middle - num_pixels // 2:middle + num_pixels // 2 + add
            ]
        elif len(image) < num_pixels:
            logger.warning("Zero padding the source image")
            if len(image.shape) == 2:
                image = _pad_image(image, num_pixels)
            elif len(image.shape) == 3:
                new_image = []
                for last_index in range(image.shape[-1]):
                    new_image.append(_pad_image(image[:,:,last_index], num_pixels))
                image = np.transpose(new_image, (1,2,0))
    
    return image, num_pixels, target_pixel_size

class Source:
    """
    Load a Source object from a 2D array. If you want to load a point source, use :meth:`Source.delta`. To load a Gaussian-shaped source, use :meth:`Source.gaussian`. To load a uniform source, use :meth:`Source.uniform`.

    Parameters
    ----------
    image : ndarray
        Input flux image.
    source_size : int
        Number of pixels in the output Source object.
    pixel_size : float
        Width of each pixel in arcseconds.
    hduis : int
        HDU index to take the source out of. (Default: 0)
    """
    
    def load_file(file_name, num_pixels=None, target_pixel_size=None, hdui=0):
        """
        Load a FITS file, zoom to the correct scale, and return the resulting image
        with the number of p
        ixels and pixel size in arcseconds.

        Parameters
        ----------
        file_name : str
            Name of the file to load.
        num_pixels : int, optional
            Number of pixels to use in the output image. If not provided, the number will be loaded from the file
        target_pixel_size : float, optional
            Width of each pixel in arcseconds for the returned image. If not provided, the number will be loaded from the file
        hduis : list of int
            HDU indices of the images to load.

        Returns
        -------
        tuple of (list of ndarray, int, float)
            - images: list of loaded and scaled images
            - num_pixels: width of the image(s)
            - target_pixel_size: width of each pixel in arcseconds
        """
        image, num_pixels, target_pixel_size = _process_file(file_name, num_pixels, target_pixel_size, source_pixel_size, hdui)
        if len(image.shape) != 2:
            raise Exception("The source image must be dimension 2")
        assert(image.shape == (num_pixels, num_pixels))

        return Source(image, target_pixel_size)

    def delta(num_pixels, pixel_size):
        """
        Create a Source object representing a point source.

        Parameters
        ----------
        num_pixels : int
            Number of pixels in the output image. Odd integer recommended.
        pixel_size : float
            Width of each pixel in arcseconds.
        """

        image = np.zeros((num_pixels, num_pixels))
        if num_pixels % 2 == 0:
            image[num_pixels//2, num_pixels//2] = 0.25
            image[num_pixels//2, num_pixels//2+1] = 0.25
            image[num_pixels//2+1, num_pixels//2+1] = 0.25
            image[num_pixels//2+1, num_pixels//2] = 0.25
        else:
            image[num_pixels//2, num_pixels//2] = 1
        return Source(image, pixel_size, is_point_source=True)
    
    def uniform(num_pixels, pixel_size):
        """
        Create a Source object representing a uniform background.

        Parameters
        ----------
        num_pixels : int
            Number of pixels in the output image. Odd integer recommended.
        pixel_size : float
            Width of each pixel in arcseconds.
        """

        image = np.ones((num_pixels, num_pixels))
        image /= np.sum(image)
        return Source(image, pixel_size, is_uniform=True)
    
    def gaussian(num_pixels, pixel_size, sigma):
        """
        Create a Gaussian-shaped Source object.

        Parameters
        ----------
        num_pixels : int
            Number of pixels in the output image. Odd integer recommended.
        pixel_size : float
            Width of each pixel in arcseconds.
        sigma : float
            Standard deviation of the Gaussian in arcseconds.
        """

        line = np.arange(num_pixels).astype(float) * pixel_size
        line -= line[-1] / 2
        dist2 = np.sum(np.array(np.meshgrid(line, line))**2, axis=0)
        gaussian = np.exp(-dist2 / (2 * sigma**2))
        gaussian /= np.sum(gaussian)

        return Source(gaussian, pixel_size)
    
    def __init__(self, image, pixel_size, is_point_source=False, is_uniform=False):
        if len(image.shape) != 2 or image.shape[0] != image.shape[1]:
            raise Exception("Source image must be two dimensional and square")
        if pixel_size > 5:
            logger.warning(f"Leakage predictions generally perform poorly for pixel sizes larger than 5 arcsec ({pixel_size} was provided). Predictions may be more reliable if smaller bins are used and rebinned later.")
        # Spreads contain the convolution with a psf
        self.source = image
        self.source_size = len(image)
        self.pixel_size = pixel_size
        self.file_name = None
        self.psr_coord = None
        self.fit_rois = None

        if self.source.shape != (self.source_size, self.source_size):
            raise Exception(f"Your source image must be square")

        self.pixel_centers = np.arange(len(image), dtype=float) * self.pixel_size # Arcsec
        self.pixel_centers -= np.max(self.pixel_centers) / 2

        self.q_map = np.zeros_like(image)
        self.u_map = np.zeros_like(image)
        self.is_point_source = is_point_source
        self.is_uniform = is_uniform

    def polarize_file(self, file_names):
        '''
        Add a source polarization to the incoming photons. 
        
        Parameters
        ----------
        filenames : (str, str)
            A list of files containing normalized q and u maps.
        '''
        self.q_map = _process_file(file_names[0], self.source_size, self.pixel_size)[0]
        self.u_map = _process_file(file_names[1], self.source_size, self.pixel_size)[0]

    def polarize_array(self, qu_map):
        '''
        Add a source polarization to the incoming photons. The provided array must have shape (2 ,i, j), where the last axis contains the q and u coordinates of the polarization. This automatically calls invalidate_source_polarization
        '''
        self.q_map = qu_map[0]
        self.u_map = qu_map[1]

    def polarize_net(self, stokes):
        '''
        Add uniform polarization to the entire image. This automatically calls invalidate_source_polarization
        '''
        q, u = stokes
        self.q_map = q
        self.u_map = u

    def convolve_psf(self, psf):
        '''
        Convolve this source image with the PSF and return the image. The provided PSF must be constructed with this source object.
        '''
        return _convolve(self.source, psf.psf)

    def compute_leakage(self, psf, spectrum, energy_dependence=None, normalize=False):
        """
        Get the Q and U maps for this source (unnormalized by default), given the PSF and spectrum.

        Parameters
        ----------
        psf : PSF
            PSF for the detector. Sky-calibrated PSFs are recommended.
        spectrum : Spectrum
            Spectrum of the data, obtainable from an IXPEData object.
        energy_dependence : callable, optional
            Function specifying the energy dependence of sigma_perp and sigma_parallel.
            Defaults to simulation-measured dependences for NN or Mom depending on this Source.
        normalize : bool, optional
            If True, return normalized Stokes q and u maps. Default is False.

        Returns
        -------
        tuple (array-like, array-like, array-like)
            Three images, i, q, u, of leakage patterns.
        """

        if energy_dependence is None:
            energy_dependence = EnergyDependence.default(self.use_nn)
        params = energy_dependence.get_params(spectrum)

        if WARN and params["sigma_minus"] < 0:
            logger.warning("Sigma perp should not be bigger than sigma parallel squared.")

        if not self.store_info or self.d_i_i[psf.det-1] is None:
            self._prepare_psf(psf)
        if not self.store_info or self.d_i_q[psf.det-1] is None:
            self._prepare_source_polarization(psf)

        # See the comment in spectrum.py that k_cross is = -k_minus / 4 if k_both=0.

        i = (
            + self.d_i_i[psf.det-1]
            + params["sigma_plus"] * self.d_zs_i[psf.det-1]
            + params["k_plus"] * self.d_zk_i[psf.det-1] 
            + params["mu_sigma_minus"] * (self.d_qs_q[psf.det-1] + self.d_us_u[psf.det-1]) / 2
            + params["mu_k_minus"] * (self.d_qk_q[psf.det-1] + self.d_uk_u[psf.det-1]) / 2
        )
        q = (
            + params["mu"] * self.d_i_q[psf.det-1]
            + params["sigma_minus"] * self.d_qs_i[psf.det-1]
            + params["k_minus"] * self.d_qk_i[psf.det-1]
            + params["mu_sigma_plus"] * self.d_zs_q[psf.det-1]
            + params["mu_k_plus"] * self.d_zk_q[psf.det-1]
            + params["mu_k_cross"] * (self.d_xk_q[psf.det-1] + self.d_yk_u[psf.det-1]) / 2
        )
        u = (
            + params["mu"] * self.d_i_u[psf.det-1]
            + params["sigma_minus"] * self.d_us_i[psf.det-1]
            + params["k_minus"] * self.d_uk_i[psf.det-1]
            + params["mu_sigma_plus"] * self.d_zs_u[psf.det-1]
            + params["mu_k_plus"] * self.d_zk_u[psf.det-1]
            + params["mu_k_cross"] * (self.d_yk_q[psf.det-1] - self.d_xk_u[psf.det-1]) / 2
        )

        q[np.isnan(q)] = 0
        u[np.isnan(u)] = 0

        if normalize:
            q /= i
            u /= i

        return (i, q, u)

    def divide_by_mu(self, q, u, spectrum):
        '''
        Divide by the detector modulation factor to get the "true" Q and U images. You can pass in either normalized or unnormalized q and u. The spectrum is used to compute the average polarization weight.
        '''
        one_over_mu = spectrum.get_avg_one_over_mu(self.use_nn)
        return q * one_over_mu, u * one_over_mu