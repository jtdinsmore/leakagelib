import numpy as np
from scipy.interpolate import interp1d
import warnings
from astropy.io import fits
from . import spectral_weights
from ..source import Source
from ..settings import LEAKAGE_DATA_DIRECTORY
from ..ixpe_data import IXPE_PIXEL_SIZE

USE_SPECTRAL_MUS = False # Set to True to account for the fact that the finite detector energy
# resolution means that the mu corresponding to the measured energy is different from the true mu,
# which depends on the true energy. Accounting for this will lower mu on average.

def _get_num_pixels(datas):
    max_radius = 0
    for data in datas:
        max_radius = max(max_radius, np.sqrt(np.max(data.evt_xs**2 + data.evt_ys**2)))
    n_pixels = int(np.ceil(2 * max_radius / 2.9729)) + 5 # Inflate the radius by 5 pixels just for safety
    if n_pixels % 2 == 0:
        n_pixels += 1
    return n_pixels

def _get_nn_bkg_spectrum():
    e_centers, spectrum = np.load(f"{LEAKAGE_DATA_DIRECTORY}/bkg-spec/nn_bkg_spectrum.npy")
    return interp1d(e_centers, spectrum, fill_value=0, bounds_error=False)

class FitSettings:
    """
    An object that keeps track of the meta-parameters of a fit.
    """
    def __init__(self, datas):
        """
        Create a fit settings object for use in fitting the data sets listed in "datas"
        """
        self.datas = datas
        self.sources = []
        self.names = []
        self.detectors = []
        self.obs_ids = []
        self.fixed_qu = []
        self.fixed_flux = []
        self.particles = []
        self.guess_qu = []
        self.guess_f = []
        self.spectral_weights = []
        self.spectral_mus = []
        self.temporal_weights = []
        self.sweeps = []
        self.extra_param_names = []
        self.extra_param_data = []
        self.model_fns = []
        self.roi = None
        self.fixed_blur = 0

    def _finalize(self):
        """
        This function performs some final actions which get the FitSettings object ready to be
        fitted. In particular, it makes sure at least one of the fluxes is fixed so that
        normalization is not ambiguous, and it sets store_info in each source object if possible
        """

        # Check if there are background weights
        bg_weights = False
        for data in self.datas:
            if np.any(data.evt_bg_chars > 0):
                bg_weights = True
            
        # Check if there is a particle source and an energy-weighted source
        spectral_weights = False
        particle_source_names = []
        for (name, particle, weights) in zip(self.names, self.particles, self.spectral_weights):
            if particle: 
                particle_source_names.append(name)
            if weights is not None:
                spectral_weights = True

        # Add the particle spectrum
        if spectral_weights:
            if self.datas[0].use_nn_energies:
                spectrum = _get_nn_bkg_spectrum()
            else:
                spectrum = lambda e: (e**-1.87)
            for name in particle_source_names:
                self.set_spectrum(name, spectrum, use_rmf=False)

        # Warn if no particle source added
        if len(particle_source_names) == 0 and bg_weights:
            warnings.warn("Your data set has background characters assigned, but you did not add a "\
                          "particle component. Particles will not be modeled. Was this intentional?")

        # Warn if no particle source added
        if len(particle_source_names) > 0 and not bg_weights:
            raise Exception("You added a particle source, but there are no particle weights in this" \
            " data set. Please remove the particle source or use a data set with weights.")

        # Fix a flux
        if np.all([self.fixed_flux[i] is None for i in range(len(self.names))]):
            name = self.names[0]
            warnings.warn(f"All of your sources had variable flux. The fitter requires"\
            f" one of the fluxes to be fixed. Fixing the flux associated with source {name} to 1.")
            self.fix_flux(name, 1)
        
        # Set store info
        store_info = True
        # Do not store info if the PSF might be blurred
        if self.fixed_blur is None:
            store_info = False

        for source in self.sources:
            source.store_info = store_info
            source._apply_roi(self._vignette())
            source.invalidate_psf()
            source.invalidate_source_polarization()
            source.invalidate_event_data()

        common_pixel_size = None
        common_source_dimensions = None

        # Check to make sure the data doesn't have any duplicates
        keys = []
        for data in self.datas:
            key = (data.obs_id, data.det)
            if key in keys:
                raise Exception("Two of your data sets have identical detectors and observation numbers. If this is not a mistake, you should manually edit one of the observation numbers to make them distinct.")
            keys.append(key)

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
        
    def _check_name(self, name):
        if name in self.names:
            raise Exception(f"The name {name} is not unique. Please pass another name.")

    def _check_source_dim(self, source):
        if len(self.sources) == 0: return
        standard_text = "LeakageLib requires each source to have the same dimensions and pixel size."\
        " If you are creating a source using your own Source object, add that source to the FitSettings" \
        " first. Make all `add_background` or `add_point_source` calls afterwards. They should use the" \
        " same image properties as the source you created."
        if not self.sources[0].source.shape == source.source.shape:
            raise Exception(f"This source does not have the same image size as previous source(s). {standard_text}")
        if not self.sources[0].pixel_size == source.pixel_size:
            raise Exception(f"This source does not have the same pixel size as previous source(s). {standard_text}")
        
    def _vignette(self):
        """
        Get the vignetted ROI image for each detector

        Returns
        -------
            list of array-like
        A list of vignetted ROIs, one per detector.
        """
        output = {}
        xs, ys = np.meshgrid(self.sources[0].pixel_centers, self.sources[0].pixel_centers)
        for data in self.datas:
            key = (data.obs_id, data.det)
            if data.expmap is None:
                warnings.warn(f"Data set {data.obs_id} DU {data.det} had no exposure map loaded. Please load an exposure map if you are fitting to events in the vignetted portion")
                output[key] = np.copy(self.roi)
            else:
                with fits.open(data.filename) as hdul:
                    colx = hdul[1].columns["X"]
                    coly = hdul[1].columns["Y"]
                stretch = np.cos(coly.coord_ref_value * np.pi / 180)
                ras = ((xs - data.offsets[0]) / IXPE_PIXEL_SIZE - colx.coord_ref_point) / stretch * colx.coord_inc + colx.coord_ref_value
                decs = ((ys - data.offsets[1]) / IXPE_PIXEL_SIZE - coly.coord_ref_point) * coly.coord_inc + coly.coord_ref_value
                
                exposures = data.expmap((ras, decs))
                output[key] = self.roi * exposures
        return output

    def add_point_source(self, name="src", det=(1,2,3,), obs_ids=None):
        """
        Add a point source to the fit.

        Parameters
        ----------
        name : str, optional
            Name of the source. Default is "src".
        det : tuple of int, optional
            Detectors the source should apply to.
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is (1, 2, 3).
        """

        use_nn = self.datas[0].use_nn

        if len(self.sources) == 0:
            num_pixels = _get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size

        source = Source.delta(use_nn, num_pixels, pixel_width)
        self.add_source(source, name, det, obs_ids)
    
    def add_background(self, name="bkg", det=(1,2,3), obs_ids=None):
        """
        Add a uniform, polarized background to the fit.

        Parameters
        ----------
        name : str, optional
            Name of the source. Default is "bkg".
        det : tuple of int, optional
            Detectors the source should apply to. Default is (1, 2, 3).
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).

        Notes
        -----
        - To add an independent background to each detector, add three background sources and pass det=1, det=2, det=3 for each, with different names.
        - The shape of the background image is set to the same shape as the first source object added to the fit settings. If no source exists, pixel width is set to the PSF native size of 2.9729 arcsec and number of pixels is the largest radius of events in the data set. 
        """

        use_nn = self.datas[0].use_nn
        if len(self.sources) == 0:
            num_pixels = _get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size
        source = Source.uniform(use_nn, num_pixels, pixel_width)
        self.add_source(source, name, det, obs_ids)
    
    def add_particle_background(self, name="pbkg", det=(1,2,3), obs_ids=None):
        """
        Add a uniform particle background component to the fit.

        Parameters
        ----------
        name : str, optional
            Name of the source. Default is "pbkg".
        det : tuple of int, optional
            Detectors the source should apply to. Default is (1, 2, 3).
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).
        """

        use_nn = self.datas[0].use_nn
        if len(self.sources) == 0:
            num_pixels = _get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size
        source = Source.uniform(use_nn, num_pixels, pixel_width)
        self.add_particle_source(source, name, det, obs_ids)
    
    def add_particle_source(self, source, name, det=(1,2,3), obs_ids=None):
        """
        Add a particle source component to the fit.

        Parameters
        ----------
        source : Source
            Source object containing the flux map.
        name : str
            Name of the source.
        det : tuple of int, optional
            Detectors the source should apply to.
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).
        """

        self.add_source(source, name, det, obs_ids)
        self.particles[-1] = True
    
    def add_source(self, source, name, det=(1,2,3,), obs_ids=None):
        """
        Add an extended source to the fit.

        Parameters
        ----------
        source : Source
            Source object containing the flux map.
        name : str
            Name of the source.
        det : tuple of int, optional
            Detectors the source should apply to.
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is (1, 2, 3).
        """

        self._check_name(name)
        self._check_source_dim(source)
        self.sources.append(source)
        self.names.append(name)
        self.detectors.append(det)
        self.obs_ids.append(obs_ids)
        self.fixed_qu.append(None)
        self.fixed_flux.append(None)
        self.guess_qu.append((None, None))
        self.guess_f.append(None)
        self.particles.append(False)
        self.spectral_weights.append(None)
        self.spectral_mus.append(None)
        self.temporal_weights.append(None)
        self.model_fns.append(None)
        self.sweeps.append(None)
        
    def set_spectrum(self, source_name, spectrum, use_rmf=True, duty_cycle=None):
        """
        Set a spectrum for a source. Weights are assigned by running the spectrum
        function on all event energies.

        Parameters
        ----------
        source_name : str
            Name of the source to assign spectral weights.
        spectrum : callable
            Function taking an energy scalar or array and returning a weight scalar or array.
        duty_cycle : callable, optional
            Fraction of data to distribute over energy. Default is uniform over the data range.
            If a contiguous energy cut was applied, the default may be used.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)

        if use_rmf:
            rmf = spectral_weights.RMF()
        else:
            rmf = spectral_weights.RMF.delta()
        convolved_spec = rmf.convolve_spectrum(spectrum)

        weights = []
        mus = []
        max_energy = -np.inf
        min_energy = np.inf
        for data in self.datas:
            convolved_spec_mu = rmf.convolve_spectrum_mu(spectrum, data.use_nn)
            max_energy = max(np.max(data.evt_energies), max_energy)
            min_energy = min(np.min(data.evt_energies), min_energy)
            these_weights = convolved_spec(data.evt_energies)
            these_mus = convolved_spec_mu(data.evt_energies) / these_weights

            weights.append(these_weights)

            if USE_SPECTRAL_MUS:
                mus.append(these_mus)
            else:
                mus.append(data.evt_mus)

        # Compute normalization
        energies = np.linspace(min_energy, max_energy, 1000)
        if duty_cycle is None:
            duty_cycle = lambda e: np.ones_like(e)
        integral = np.sum(convolved_spec(energies) * duty_cycle(energies))
        integral_constant = np.sum(duty_cycle(energies)) # Settings weights equal to one would give this value

        # Normalize the spectrum so that the spectrum normalization is equal to that of weights=1
        multiplier = integral_constant / integral        
        for i in range(len(weights)):
            weights[i] *= multiplier
        
        self.spectral_weights[index] = weights
        self.spectral_mus[index] = mus
        
    def set_lightcurve(self, source_name, lightcurve, duty_cycle=None):
        """
        Set a light curve for a source. Weights are assigned by running the light curve
        function on all event times.

        Parameters
        ----------
        source_name : str
            Name of the source to assign time weights.
        lightcurve : callable
            Function taking a time scalar or array and returning a weight scalar or array.
        duty_cycle : callable, optional
            Fraction of data to distribute over time. Default is uniform over the data range.
            If a contiguous or no time cut was applied, the default may be used.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)

        weights = []
        max_time = -np.inf
        min_time = np.inf
        for data in self.datas:
            max_time = max(np.max(data.evt_times), max_time)
            min_time = min(np.min(data.evt_times), min_time)
            these_weights = lightcurve(data.evt_times)
            weights.append(these_weights)

        # Compute normalization
        times = np.linspace(min_time, max_time, 1000)
        if duty_cycle is None:
            duty_cycle = lambda t: np.ones_like(t)
        integral = np.sum(lightcurve(times) * duty_cycle(times))
        integral_constant = np.sum(duty_cycle(times)) # Settings weights equal to one would give this value

        # Normalize the lightcurve so that the lightcurve normalization is equal to that of weights=1
        multiplier = integral_constant / integral        
        for i in range(len(weights)):
            weights[i] *= multiplier

        self.temporal_weights[index] = weights
        
    def set_sweep(self, source_name, sweep):
        """
        Set a polarization sweep model for a source.

        Parameters
        ----------
        source_name : str
            Name of the source to apply the sweep model.
        sweep : callable
            Function taking event time(s) and returning (q, u) for a normalized polarization model.
            A fit will determine a global PA offset and PD.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        if self.model_fns[index] is not None:
            raise Exception("You cannot set a sweep for a source that you have already set a polarization model function for")

        sweeps = []
        for data in self.datas:
            sweeps.append(sweep(data.evt_times))

        self.sweeps[index] = sweeps
        
    def set_model_fn(self, source_name, model_fn):
        """
        Set a polarization model for a source with fittable parameters.

        Parameters
        ----------
        source_name : str
            Name of the source to apply the polarization model.
        model_fn : callable
            Function returning (q, u) from a polarization model. It takes three arguments:
            event time, FitData object, and parameter array. Use FitData.param_to_value
            to access parameter values. Additional parameters can be added with
            FitSettings.add_param.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        if self.sweeps[index] is not None:
            raise Exception("You cannot set a model function for a source that you have already set a polarization sweep for")
        
        self.model_fns[index] = model_fn
        self.fix_qu(source_name, (0, 0)) # Turn off the q and u values from this source, since this function overwrites them

    def add_param(self, name, initial_value=0, bounds=(None, None), num_diff_step=1e-3):
        """
        Add a parameter for the fitter. Only use after set_model_fn has been called.

        Parameters
        ----------
        name : str
            Parameter name.
        initial_value : float, optional
            Initial value for the fitter. Default is 0.
        bounds : tuple of (float, float) or None, optional
            Bounds for the parameter. Default is no bounds.
        num_diff_step : float, optional
            Step size used for numerical computation of uncertainties.
        """

        if name == "q" or name == "u" or name == "f" or name == "sigma" or name in self.extra_param_names:
            raise Exception(f"The name {name} cannot be used twice. Parameter names `q`, `u`, `f`, and `sigma` are forbidden.")
        self.extra_param_names.append(name)
        self.extra_param_data.append((initial_value, bounds, num_diff_step))

    def fix_qu(self, source_name, qu):
        """
        Fix the Q and U values of a source.

        Parameters
        ----------
        source_name : str
            Name of the source.
        qu : tuple of float or None
            Tuple of Stokes coefficients to fix. Pass None to free the polarization.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.fixed_qu[index] = qu

    def set_initial_qu(self, source_name, qu):
        """
        Set the initial guess for Q and U of a source.

        Parameters
        ----------
        source_name : str
            Name of the source.
        qu : tuple of float or None
            Initial guess of Stokes coefficients. Pass None to leave free.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.guess_qu[index] = qu

    def fix_flux(self, source_name, flux):
        """
        Fix the flux of a source. Flux units are arbitrary.

        Parameters
        ----------
        source_name : str
            Name of the source.
        flux : float
            Flux of the source.

        Notes
        -----
        Fixing a single source is necessary to set the flux scale. Fixing multiple sources fixes relative luminosities. The fitter assumes the true flux is between 0 and 100, so you should set your fluxes accordingly.
        """

        if not source_name in self.names:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        index = self.names.index(source_name)
        self.fixed_flux[index] = flux

    def set_initial_flux(self, source_name, flux):
        """
        Set the initial guess for the flux of a source.

        Parameters
        ----------
        source_name : str
            Name of the source.
        flux : float
            Initial flux value.
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
        Provide the region of interest (ROI) to the fitter after data has been cut.
        
        This function does NOT cut events; it only tells the fitter to expect events cut to this ROI. You should cut the events yourself.

        Notes
        -----
        The roi_image must have the same dimensions as source objects.
        """
        if len(self.sources) > 0:
            if self.sources[0].source.shape != roi_image.shape:
                raise Exception("The ROI image must have the same size as the source images. You can access the source dimensions and pixel size with FitSettings.sources[0].source.shape, and FitSettings.sources[0].pixel_size.")
        self.roi = roi_image

    def apply_circular_roi(self, radius):
        """
        Provide a circular ROI centered on the origin (radius in arcseconds).
        
        This function does NOT cut events; it only tells the fitter to expect events cut to this radius. You should cut the events yourself.
        """
        if len(self.sources) == 0:
            raise Exception("You cannot apply a circular ROI until you have added at least one source. The function needs to know the dimensions of your source images")
        
        # Make a subsampled grid
        original_dim = len(self.sources[0].source)
        subsamples = 8
        subsample_edges = np.arange(original_dim*subsamples+1).astype(float) * self.sources[0].pixel_size/subsamples
        subsample_centers = (subsample_edges[1:] + subsample_edges[:-1])/2
        subsample_centers -= np.mean(subsample_centers)

        # Make the ROI for this grid
        xs, ys = np.meshgrid(subsample_centers, subsample_centers)
        roi_image = xs**2 + ys**2 < radius**2

        # Re-sum it into the original dimensions
        resummed = roi_image.reshape(original_dim, subsamples, original_dim, subsamples).mean(axis=(1, 3))

        self.apply_roi(resummed)

    def get_n_sources(self):
        """
        Returns the number of sources that have been added to the fitter.
        """
        return len(self.sources)