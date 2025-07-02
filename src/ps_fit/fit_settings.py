import numpy as np
import warnings
from ..source import Source

def get_num_pixels(data):
    max_radius = np.sqrt(np.max(data.evt_xs**2 + data.evt_ys**2))
    n_pixels = int(np.ceil(2 * max_radius / 2.9729)) + 1
    if n_pixels % 2 == 0:
        n_pixels += 1
    return n_pixels

class FitSettings:
    def __init__(self, datas):
        """
        Create a fit settings object for use in fitting the data sets listed in "datas"
        """
        self.datas = datas
        self.sources = []
        self.names = []
        self.detectors = []
        self.fixed_qu = []
        self.fixed_flux = []
        self.particles = []
        self.guess_qu = []
        self.guess_f = []
        self.roi = None
        self.fixed_blur = 0

    def finalize(self):
        """
        This function performs some final actions which get the FitSettings object ready to be
        fitted. In particular, it makes sure at least one of the fluxes is fixed so that
        normalization is not ambiguous, and it sets store_info in each source object if possible
        """

        # Fix a flux
        if np.all([self.fixed_flux[i] is None for i in range(len(self.names))]):
            name = self.names[0]
            warnings.warn(f"All of your sources had variable flux. The fitter requires"\
            f" one of the fluxes to be fixed. Fixing the flux associated with source {name} to 1.")
            self.fix_flux(name, 1)

        # Make a particle source
        need_particle_source = False
        for data in self.datas:
            if np.any(data.evt_bg_probs > 0):
                need_particle_source = True
                break
        if need_particle_source:
            self.add_background("pbkg", override_checks=True)
            self.particles[-1] = True
            self.fix_qu("pbkg", (0,0))
            self.fix_flux("pbkg", 1)
        else:
            warnings.warn("None of your data sets have nonzero bg_probs, so background particle" \
            " weighting will not be done.")
        
        # Set store info
        store_info = True
        # Do not store info if the PSF might be blurred
        if self.fixed_blur is None:
            store_info = False

        for source in self.sources:
            source.store_info = store_info
            source.fit_roi = np.copy(self.roi)
            source.invalidate_psf()
            source.invalidate_source_polarization()
            source.invalidate_event_data()

        common_pixel_size = None
        common_source_dimensions = None

        # Print warnings if any of the sources have weird properties
        for (name, source) in zip(self.names, self.sources):
            if np.abs(source.pixel_size - 2.9729) > 1e-4:
                warnings.warn(f"The source {name} had pixel width not equal to 2.9729. 2.9279 is" \
                " the pixel width of the sky calibrated PSF, and analysis will be most accurate if" \
                " you use that pixel width for your sources too.")
            if not source.has_image:
                raise Exception(f"All sources must have an image.")
            if common_pixel_size is not None:
                if source.pixel_size != common_pixel_size or source.source.shape != common_source_dimensions:
                    warnings.warn(f"Your sources do not all have the same pixel size or dimensions."\
                    " This will incorrectly apply different ROIs to different sources, or"\
                    " maybe just crash. Make your source images all on the same scale to avoid this.")
            common_pixel_size = source.pixel_size
            common_source_dimensions = source.source.shape
        
        if self.roi is None:
            raise Exception("You did not provide an ROI. Provide an ROI so that the background PDF"\
            " can be normalized")
        
    def check_name(self, name):
        if name in self.names:
            raise Exception(f"The name {name} is not unique. Please pass another name.")
        if name == "pbkg":
            raise Exception("The name pbkg is reserved")


    def add_point_source(self, name="src", det=(1,2,3,)):
        """
        Add a point source to the fit.
        # Arguments:
        * name (optional): a string representing the name of the source. Default: src
        * det (optional): A tuple containing the detector numbers the source should apply to.
        Default is (1,2,3)
        """
        self.check_name(name)
        use_nn = self.datas[0].use_nn

        if len(self.sources) == 0:
            num_pixels = get_num_pixels(self.datas[0])
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size

        source = Source.delta(use_nn, num_pixels, pixel_width)
        self.sources.append(source)
        self.names.append(name)
        self.detectors.append(det)
        self.fixed_qu.append(None)
        self.fixed_flux.append(None)
        self.guess_qu.append((None, None))
        self.guess_f.append(None)
        self.particles.append(False)
    
    def add_background(self, name="bkg", det=(1,2,3), override_checks=False):
        """
        Add a uniform, polarized background to the fit.
        # Arguments:
        * name: a string representing the name of the source. Default: bkg
        * det (optional): A tuple containing the detector numbers the source should apply to.
        Default is (1,2,3)
        # Notes
        * To add an independent background to each detector, add three background sources and pass 
        det=1, det=2, det=3 for each. Make sure to pass different names.
        * The shape of the background image is set to be the same shape as the first source object
        you added to this fit settings. If you have not yet added an object, the pixel width is
        set to the PSF native size of 2.9729 arcsec and the number of pixels is chosen to be the
        largest radius of events in the data set.
        """
        if not override_checks:
            self.check_name(name)
        use_nn = self.datas[0].use_nn
        
        if len(self.sources) == 0:
            num_pixels = get_num_pixels(self.datas[0])
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size

        source = Source.uniform(use_nn, num_pixels, pixel_width)
        self.sources.append(source)
        self.names.append(name)
        self.detectors.append(det)
        self.fixed_qu.append(None)
        self.fixed_flux.append(None)
        self.guess_qu.append((None, None))
        self.guess_f.append(None)
        self.particles.append(False)
    
    def add_source(self, source, name, det=(1,2,3,)):
        """
        Add an extended soruce to the fit.
        # Arguments:
        * source: a Source object containing the flux map of the source
        * name: a string representing the name of the source
        * det (optional): A tuple containing the detector numbers the source should apply to.
        Default is (1,2,3)
        """
        self.check_name(name)
        self.sources.append(source)
        self.names.append(name)
        self.detectors.append(det)
        self.fixed_qu.append(None)
        self.fixed_flux.append(None)
        self.guess_qu.append((None, None))
        self.guess_f.append(None)
        self.particles.append(False)

    def fix_qu(self, source_name, qu):
        """
        Fix the Q and U of a given source
        # Arguments:
        * source_name: a string referring to the name of the source
        * qu: a 2-tuple containing what Stokes coefficients to fix. Pass None to free the polarization.
        """
        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.fixed_qu[index] = qu

    def set_initial_qu(self, source_name, qu):
        """
        Set the initial guess for the Q and U of a source
        # Arguments:
        * source_name: a string referring to the name of the source
        * qu: a 2-tuple containing what Stokes coefficients to fix. Pass None to free the polarization.
        """
        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.guess_qu[index] = qu

    def fix_flux(self, source_name, flux):
        """
        Fix the flux of a source. Flux units are arbitrary and do not need to be cts / s
        
        Note: the fitter will normalize flux so that the correct number of
        counts in the image is reproduced. Therefore, fixing the flux of one source does nothing.
        Fixing the flux of multiple sources fixes their relative luminosities.

        Note: the fitter assumes the true flux is between 0 and 100. 

        # Arguments:
        * source_name: a string referring to the name of the source
        * flux: the flux of the osurce
        """
        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.fixed_flux[index] = flux

    def set_initial_flux(self, source_name, flux):
        """
        Set the initial guess for the flux of a source
        # Arguments:
        * source_name: a string referring to the name of the source
        * flux: the flux of the osurce
        """
        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.guess_f[index] = flux

    def fit_psf_sigma(self):
        """
        Fit a blur parameter, which is a Gaussian with sigma equal to blur convolved with the PSF to better match the image.
        """
        self.fixed_blur = None

    def fix_psf_sigma(self, psf_sigma):
        """
        Fix the psf blur sigma
        """
        self.fixed_blur = psf_sigma

    def apply_roi(self, roi_image):
        """
        If you have cut the data to a certain region, this function provides that region of interest
        (ROI) to the fitter for use in normalizing the position PDFs. This is necessary whenever
        fitting with an extended object, including background. I.e. it's basically always necessary.

        # Assumptions
        * All source objects have the same image size and pixel scale
        * The roi_image object is has the same dimensions as the source objects
        * The data sets you will fit to have been cut to this roi.
        """
        self.roi = roi_image

    def apply_circular_roi(self, radius):
        """
        Provide a circular ROI, centered on the origin (radius in arcsec). This function makes the
        same assumption as `apply_roi`.
        """
        if len(self.sources) == 0:
            raise Exception("You cannot apply a circular ROI until you have added at least one source. The function needs to know the dimensions of your source images")
        xs, ys = np.meshgrid(self.sources[0].pixel_centers, self.sources[0].pixel_centers)
        roi_image = xs**2 + ys**2 < radius**2
        self.apply_roi(roi_image)

    def get_n_sources(self):
        return len(self.sources)