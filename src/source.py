import numpy as np
from astropy.io import fits
from scipy.signal import convolve
from .spectrum import EnergyDependence
from .funcs import integrate_zoom

WARN = True

def pad_image(image, num_pixels):
    '''Zero-pad an image'''
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

def _process_file(file_name, num_pixels, target_pixel_size, source_pixel_size, hduis=[1], rescale=False):
    '''Load a file (fits or npy), zoom to the correct scale, and return the resulting image with the number of pixels and pixel size in arcsec.
    ## Arguments
    - file_name: Name of the file
    - num_pixels: number of pixels to use
    - target_pixel_size: width of each pixel in the returned image in arcsec (None implies use the source pixel size)
    - source_pixel_size: width of each pixel in the current image in arcsec (None implies read it from the file)
    - hduis: HDU indices of the images to load
    
    ## Returns
    - image: list of images
    - num_pixels: width of the image(s)
    - target_pixel_size: width of each pixel (arcsec)'''

    if file_name.endswith(".fits"):
        image = []
        with fits.open(file_name) as hdul:
            for hduli in hduis:
                this_source_pixel_size=hdul[hduli].header["CDELT2"] * 3600 # Arcsec
                if source_pixel_size is None:
                    source_pixel_size = this_source_pixel_size
                if target_pixel_size is None:
                    target_pixel_size = source_pixel_size
                if np.abs(target_pixel_size - this_source_pixel_size) > 1e-5:
                    zoom_ratio = this_source_pixel_size / target_pixel_size
                    new_image = integrate_zoom(hdul[hduli].data, zoom_ratio, force_odd=True)
                    if rescale:
                        new_image *= zoom_ratio**2
                else:
                    new_image = hdul[hduli].data
                image.append(new_image)
        image = np.transpose(image, (1,2,0))

    elif file_name.endswith(".npy"):
        if source_pixel_size is None:
            raise Exception("A source pixel size must be passed in if loading an npy file.")

        if target_pixel_size is None:
            target_pixel_size = source_pixel_size
            
        image = np.load(file_name)
        if len(image.shape) == 2:
            image = image.reshape((image.shape[0], image.shape[1], 1))
        for i in image.shape[-1]:
            if np.abs(target_pixel_size - source_pixel_size) > 1e-5:
                zoom_ratio = source_pixel_size / target_pixel_size
                new_image = integrate_zoom(image[:,:,i], zoom_ratio, force_odd=True)
                if rescale:
                    new_image *= zoom_ratio**2
            else:
                new_image = image[:,:,i]

            image[:,:,i] = new_image

    if image.shape[-1] == 1:
        image = image.reshape((image.shape[0], image.shape[1]))

    if num_pixels is None:
        num_pixels = image.shape[0]

    assert(image.shape[0] == image.shape[1])
    if len(image) > num_pixels:
        # Shrink the image
        middle = len(image) // 2
        add = 0 if num_pixels % 2 == 0 else 1
        image = image[
            middle - num_pixels // 2:middle + num_pixels // 2 + add,
            middle - num_pixels // 2:middle + num_pixels // 2 + add
        ]
    elif len(image) < num_pixels:
        print("WARNING: zero padding the source image")
        if len(image.shape) == 2:
            image = pad_image(image, num_pixels)
        elif len(image.shape) == 3:
            new_image = []
            for last_index in range(image.shape[-1]):
                new_image.append(pad_image(image[:,:,last_index], num_pixels))
            image = np.transpose(new_image, (1,2,0))
    
    return image, num_pixels, target_pixel_size

class Source:
    def load_file(file_name, use_nn, num_pixels=None, target_pixel_size=None, source_pixel_size=None, hduis=[1]):
        '''Creates a Source object from file which contains a well-resolved prediction for where the source's flux is coming from. E.g. a Chandra image.
        
        # Arguments:
            - `file_name`: path to a FITS or NPY source file
            - `use_nn`: True if you will later choose to run your results with NN-reconstructed data. False for moments-reconstructed data
            - `num_pixels`: number of pixels to use throughout the leakage prediction pipeline. Must be an integer; an odd integer is recommended. If None, it is taken to be the number of pixels in the image
            - `target_pixel_size`: size of the output image pixels in arcsec. If None, it is taken to be the same as source_pixel_size
            - `source_pixel_size`: size of the input image pixels in arcsec. If None, its value is obtained from the fits file header. 
        '''
        image, num_pixels, target_pixel_size = _process_file(file_name, num_pixels, target_pixel_size, source_pixel_size, hduis)
        if len(image.shape) != 2:
            raise Exception("The source image must be dimension 2")
        assert(image.shape == (num_pixels, num_pixels))

        return Source(image, use_nn, num_pixels, target_pixel_size)

    def delta(use_nn, num_pixels, pixel_size, store_info=False):
        '''Creates a Source object representing a point source.

        # Arguments:
            - `use_nn`: True if you will later choose to run your results with NN-reconstructed data. False for moments-reconstructed data
            - `num_pixels`: number of pixels in the output image. An integer is required and an odd integer is recommended.
            - `pixel_size`: width of each pixel in arcseconds.
            - `store_info`: Set to true to store info from the source and PSF to speed up some computations. This forces you to manually call `invalidate_psf` and `invalidate_source_polarization` if you use it.
        '''
        image = np.zeros((num_pixels, num_pixels))
        if num_pixels % 2 == 0:
            image[num_pixels//2, num_pixels//2] = 0.25
            image[num_pixels//2, num_pixels//2] = 0.25
            image[num_pixels//2+1, num_pixels//2+1] = 0.25
            image[num_pixels//2+1, num_pixels//2] = 0.25
        else:
            image[num_pixels//2, num_pixels//2] = 1
        return Source(image, use_nn, num_pixels, pixel_size, store_info, is_point_source=True)
    
    def gaussian(use_nn, num_pixels, pixel_size, sigma, store_info=False):
        '''Creates a Gaussian-shaped Source object

        # Arguments:
            - `use_nn`: True if you will later choose to run your results with NN-reconstructed data. False for moments-reconstructed data
            - `num_pixels`: number of pixels in the output image. An integer is required and an odd integer is recommended.
            - `pixel_size`: width of each pixel in arcseconds.
            - `sigma`: standard deviation of the Gaussian in arcsec
            - `store_info`: Set to true to store info from the source and PSF to speed up some computations. This forces you to manually call `invalidate_psf` and `invalidate_source_polarization` if you use it.
        '''

        line = np.arange(num_pixels).astype(float) * pixel_size
        line -= line[-1] / 2
        dist2 = np.sum(np.array(np.meshgrid(line, line))**2, axis=0)
        gaussian = np.exp(-dist2 / (2 * sigma**2))
        gaussian /= np.sum(gaussian)

        return Source(gaussian, use_nn, num_pixels, pixel_size, store_info)
    
    def no_image(use_nn):
        """Create an empty source for use in initializing a dataset which will not be binned"""
        source = Source(np.array([[0]]), use_nn, 1, 1)
        source.has_image = False
        return source

    def __init__(self, image, use_nn, source_size, pixel_size, store_info=False, is_point_source=False):
        '''Loads a Source object from a 2d array.

        ARGUMENTS:
            - `image`: image of the incomming flux
            - `source_size`: number of pixels in the output image. An integer is required.
            - `pixel_size`: width of each pixel in arcseconds
            - `store_info`: Set to true to store info from the source and PSF to speed up some computations. This forces you to manually call `invalidate_psf` and `invalidate_source_polarization` if you use it.
        '''
        if len(image.shape) != 2 or image.shape[0] != image.shape[1]:
            print("Source image must be two dimensional and square")
        if pixel_size > 5:
            print(f"Leakage predictions generally perform poorly for pixel sizes larger than 5 arcsec ({pixel_size} was provided). Predictions may be more reliable if smaller bins are used and rebinned later.")
        # Spreads contain the convolution with a psf
        self.source = image
        self.source_size = source_size
        self.pixel_size = pixel_size
        self.file_name = None
        self.psr_coord = None
        self.store_info = store_info
        self.use_nn = use_nn

        if self.source.shape != (source_size, source_size):
            raise Exception("Your source image must have shape (source_size, source_size)")

        self.pixel_centers = np.arange(len(image), dtype=float) * self.pixel_size # Arcsec
        self.pixel_centers -= np.max(self.pixel_centers) / 2

        self.q_map = np.zeros_like(image)
        self.u_map = np.zeros_like(image)
        self.is_point_source = is_point_source
        self.has_image = True
        self.invalidate_psf()
        self.invalidate_source_polarization()

    def invalidate_psf(self):
        '''Invalidate the stored data concerning the source flux.'''
        self.d_i_i = [None, None, None]
        self.d_zs_i = [None, None, None]
        self.d_qs_i = [None, None, None]
        self.d_us_i = [None, None, None]
        self.d_zk_i = [None, None, None]
        self.d_qk_i = [None, None, None]
        self.d_uk_i = [None, None, None]
        self.d_xk_i = [None, None, None]
        self.d_yk_i = [None, None, None]

    def invalidate_source_polarization(self):
        '''Invalidate the stored data concerning the source polarization.'''
        self.d_i_q  = [None, None, None]
        self.d_i_u  = [None, None, None]
        self.d_zs_q = [None, None, None]
        self.d_zk_q = [None, None, None]
        self.d_xk_q = [None, None, None]
        self.d_yk_q = [None, None, None]
        self.d_zs_u = [None, None, None]
        self.d_zk_u = [None, None, None]
        self.d_xk_u = [None, None, None]
        self.d_yk_u = [None, None, None]
        self.d_qs_q = [None, None, None]
        self.d_qk_q = [None, None, None]
        self.d_us_u = [None, None, None]
        self.d_uk_u = [None, None, None]

    def polarize_file(self, file_name, source_pixel_size=None):
        '''Add a source polarization to the incoming photons. The provided file must either be a fits file with Q in hdul[1] and U in hdul[2], or a numpy array of shape (i, j, 2), where the last axis contains the q and u coordinates of the polarization.'''

        image = _process_file(file_name, self.source_size, self.pixel_size, None, [1,2], rescale=True)[0]
        
        assert(image.shape == (self.source_size, self.source_size, 2))
        self.q_map = image[:,:,0]
        self.u_map = image[:,:,1]
        self.invalidate_source_polarization()

    def polarize_array(self, qu_map):
        '''Add a source polarization to the incoming photons. The provided array must have shape (2 ,i, j), where the last axis contains the q and u coordinates of the polarization.'''
        self.q_map = qu_map[0]
        self.u_map = qu_map[1]
        self.invalidate_source_polarization()

    def polarize_net(self, stokes):
        '''Add uniform polarization to the entire image'''
        q, u = stokes
        self.q_map = q
        self.u_map = u
        self.invalidate_source_polarization()

    def _prepare_psf(self, psf):
        '''Prepare the leakage maps for the given source'''
        self.d_i_i[psf.det-1] = convolve(self.source, psf.psf, mode="same")

        self.d_zs_i[psf.det-1] = convolve(self.source, psf.d_zs, mode="same")
        self.d_qs_i[psf.det-1] = convolve(self.source, psf.d_qs, mode="same")
        self.d_us_i[psf.det-1] = convolve(self.source, psf.d_us, mode="same")

        self.d_zk_i[psf.det-1] = convolve(self.source, psf.d_zk, mode="same")
        self.d_qk_i[psf.det-1] = convolve(self.source, psf.d_qk, mode="same")
        self.d_uk_i[psf.det-1] = convolve(self.source, psf.d_uk, mode="same")
        self.d_xk_i[psf.det-1] = convolve(self.source, psf.d_xk, mode="same")
        self.d_yk_i[psf.det-1] = convolve(self.source, psf.d_yk, mode="same")


    def _prepare_source_polarization(self, psf):
        '''Prepare the leakage maps for the given source polarization'''

        if self.is_point_source:
            q_src, u_src = np.mean(self.q_map), np.mean(self.u_map)
            self.d_i_q[psf.det-1] = self.d_i_i[psf.det-1] * q_src
            self.d_i_u[psf.det-1] = self.d_i_i[psf.det-1] * u_src

            self.d_zs_q[psf.det-1] = self.d_zs_i[psf.det-1] * q_src
            self.d_zk_q[psf.det-1] = self.d_zk_i[psf.det-1] * q_src
            self.d_xk_q[psf.det-1] = self.d_xk_i[psf.det-1] * q_src
            self.d_yk_q[psf.det-1] = self.d_yk_i[psf.det-1] * q_src

            self.d_zs_u[psf.det-1] = self.d_zs_i[psf.det-1] * u_src
            self.d_zk_u[psf.det-1] = self.d_zk_i[psf.det-1] * u_src
            self.d_xk_u[psf.det-1] = self.d_xk_i[psf.det-1] * u_src
            self.d_yk_u[psf.det-1] = self.d_yk_i[psf.det-1] * u_src

            self.d_qs_q[psf.det-1] = self.d_qs_i[psf.det-1] * q_src
            self.d_qk_q[psf.det-1] = self.d_qk_i[psf.det-1] * q_src

            self.d_us_u[psf.det-1] = self.d_us_i[psf.det-1] * u_src
            self.d_uk_u[psf.det-1] = self.d_uk_i[psf.det-1] * u_src
        else:
            self.d_i_q[psf.det-1] = convolve(self.source * self.q_map, psf.psf, mode="same")
            self.d_i_u[psf.det-1] = convolve(self.source * self.u_map, psf.psf, mode="same")
            
            self.d_zs_q[psf.det-1] = convolve(self.source * self.q_map, psf.d_zs, mode="same")
            self.d_zk_q[psf.det-1] = convolve(self.source * self.q_map, psf.d_zk, mode="same")
            self.d_xk_q[psf.det-1] = convolve(self.source * self.q_map, psf.d_xk, mode="same")
            self.d_yk_q[psf.det-1] = convolve(self.source * self.q_map, psf.d_yk, mode="same")
            
            self.d_zs_u[psf.det-1] = convolve(self.source * self.u_map, psf.d_zs, mode="same")
            self.d_zk_u[psf.det-1] = convolve(self.source * self.u_map, psf.d_zk, mode="same")
            self.d_xk_u[psf.det-1] = convolve(self.source * self.u_map, psf.d_xk, mode="same")
            self.d_yk_u[psf.det-1] = convolve(self.source * self.u_map, psf.d_yk, mode="same")
            
            self.d_qs_q[psf.det-1] = convolve(self.source * self.q_map, psf.d_qs, mode="same")
            self.d_qk_q[psf.det-1] = convolve(self.source * self.q_map, psf.d_qk, mode="same")
            
            self.d_us_u[psf.det-1] = convolve(self.source * self.u_map, psf.d_us, mode="same")
            self.d_uk_u[psf.det-1] = convolve(self.source * self.u_map, psf.d_uk, mode="same")

    def compute_leakage(self, psf, spectrum, energy_dependence=None, normalize=False):
        '''Get the Q and U maps for this source (unnormalized by default), given the provided PSF and spectrum. Note: these are _detection_ predictions, so you will have to divide by mu (use source.divide_by_mu) to compare to true polarizations
        ARGUMENTS:
        - psf: the psf for the detector to be used. Sky-calibrated PSFs recommended
        - spectrum: the spectrum of the data. Can be obtained with (IXPEData object).spectrum.
        - energy_dependence: optional argument for the energy dependence of sigma perp and parallel. Default is to use the simulation-measured dependences for either NN or Mom depending on the configuration of this Source object.
        - normalize: Set to True to return the normalized Stokes q and u maps. Default is False.
        '''
        if energy_dependence is None:
            energy_dependence = EnergyDependence.default(self.use_nn)
        params = energy_dependence.get_params(spectrum)

        if WARN and params["sigma_minus"] < 0:
            print("WARNING: sigma perp should not be bigger than sigma parallel squared.")

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
        '''Divide by the detector modulation factor to get the "true" Q and U images. You can pass in either normalized or unnormalized q and u. The spectrum is used to compute the average polarization weight.'''
        one_over_mu = spectrum.get_avg_one_over_mu(self.use_nn)
        return q * one_over_mu, u * one_over_mu