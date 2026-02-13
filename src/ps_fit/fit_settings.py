import numpy as np
from scipy.interpolate import interp1d
import warnings
from . import spectral_weights
from ..source import Source
from ..settings import LEAKAGE_DATA_DIRECTORY

USE_SPECTRAL_MUS = False # Set to True to account for the fact that the finite detector energy
# resolution means that the mu corresponding to the measured energy is different from the true mu,
# which depends on the true energy. Accounting for this will lower mu on average.

def get_num_pixels(datas):
    max_radius = 0
    for data in datas:
        max_radius = max(max_radius, np.sqrt(np.max(data.evt_xs**2 + data.evt_ys**2)))
    n_pixels = int(np.ceil(2 * max_radius / 2.9729)) + 5 # Inflate the radius by 5 pixels just for safety
    if n_pixels % 2 == 0:
        n_pixels += 1
    return n_pixels

def get_nn_bkg_spectrum():
    e_centers, spectrum = np.load(f"{LEAKAGE_DATA_DIRECTORY}/bkg-spec/nn_bkg_spectrum.npy")
    return interp1d(e_centers, spectrum, fill_value=0, bounds_error=False)

class FitSettings:
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

    def finalize(self):
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
                spectrum = get_nn_bkg_spectrum()
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
            source.apply_roi(self.roi)
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
        
    def check_name(self, name):
        if name in self.names:
            raise Exception(f"The name {name} is not unique. Please pass another name.")


    def add_point_source(self, name="src", det=(1,2,3,), obs_ids=None):
        """
        Add a point source to the fit.
        # Arguments:
        * name (optional): a string representing the name of the source. Default: src
        * det (optional): A tuple containing the detector numbers the source should apply to.
        * obs_ids (optional): A tuple containing the observation IDs the source should apply to.
        Default is (1,2,3)
        """
        use_nn = self.datas[0].use_nn

        if len(self.sources) == 0:
            num_pixels = get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size

        source = Source.delta(use_nn, num_pixels, pixel_width)
        self.add_source(source, name, det, obs_ids)
    
    def add_background(self, name="bkg", det=(1,2,3), obs_ids=None):
        """
        Add a uniform, polarized background to the fit.
        # Arguments:
        * name: a string representing the name of the source. Default: bkg
        * det (optional): A tuple containing the detector numbers the source should apply to.
        Default is (1,2,3)
        * obs_ids (optional): A tuple containing the observation IDs the source should apply to.
        Default is None, meaning all observations
        # Notes
        * To add an independent background to each detector, add three background sources and pass 
        det=1, det=2, det=3 for each. Make sure to pass different names.
        * The shape of the background image is set to be the same shape as the first source object
        you added to this fit settings. If you have not yet added an object, the pixel width is
        set to the PSF native size of 2.9729 arcsec and the number of pixels is chosen to be the
        largest radius of events in the data set.
        """
        use_nn = self.datas[0].use_nn
        if len(self.sources) == 0:
            num_pixels = get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size
        source = Source.uniform(use_nn, num_pixels, pixel_width)
        self.add_source(source, name, det, obs_ids)
    
    def add_particle_background(self, name="pbkg", det=(1,2,3), obs_ids=None):
        """
        Add a uniform particle background component to the fit.
        # Arguments:
        * name: a string representing the name of the source. Default: pbkg
        * det (optional): A tuple containing the detector numbers the source should apply to.
        Default is (1,2,3)
        * obs_ids (optional): A tuple containing the observation IDs the source should apply to.
        Default is None, meaning all observations
        """
        use_nn = self.datas[0].use_nn
        if len(self.sources) == 0:
            num_pixels = get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = self.sources[0].source.shape[0]
            pixel_width = self.sources[0].pixel_size
        source = Source.uniform(use_nn, num_pixels, pixel_width)
        self.add_particle_source(source, name, det, obs_ids)
    
    def add_particle_source(self, source, name, det=(1,2,3), obs_ids=None):
        """
        Add a particle source component to the fit.
        # Arguments:
        * source: a Source object containing the flux map of the source
        * name: a string representing the name of the source
        * det (optional): A tuple containing the detector numbers the source should apply to.
        * obs_ids (optional): A tuple containing the observation IDs the source should apply to.
        Default is None, meaning all observations
        """
        self.add_source(source, name, det, obs_ids)
        self.particles[-1] = True
    
    def add_source(self, source, name, det=(1,2,3,), obs_ids=None):
        """
        Add an extended soruce to the fit.
        # Arguments:
        * source: a Source object containing the flux map of the source
        * name: a string representing the name of the source
        * det (optional): A tuple containing the detector numbers the source should apply to.
        * obs_ids (optional): A tuple containing the observation IDs the source should apply to.
        Default is (1,2,3)
        """
        self.check_name(name)
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
        Set a spectrum for the source. Weights will be assigned by running the spectrum function on all event energies
        # Arguments
        * source_name: the source name to assign spectral weights to
        * spectrum: A function that takes in a scalar (energy) and returns a scalar (weight). Must be able to take a numpy array as input
        * duty_cycle (option): Range over which the data is distributed. Default is uniformly distributed over the range of the data. If you did a contiguous energy cut, then you may leave the default option.  Otherwise, you should pass a function of energy which is equal to the fraction of data you cut at that energy

        Assigns spectral weights to each event.
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
        Set a lightcurve for the source. Weights will be assigned by running the lightcurve function on all event times
        # Arguments
        * source_name: the source name to assign spectral weights to
        * lightcurve: A function that takes in a scalar (time) and returns a scalar (weight). Must be able to take a numpy array as input.
        * duty_cycle (option): Range over which the data is distributed. Default is uniformly distributed over the range of the data. If you did a contiguous time cut or no time cut, then you may leave the default option. Otherwise, you should pass a function of time which is equal to the fraction of data you cut at that time.

        Assigns time weights to each event.
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
        Set a polarization sweep model for the source.
        # Arguments
        * source_name: the name of the source to apply the sweep model to
        * sweep: a function which takes in event time and returns (q, u) for a polarization model (normalized Stokes coefficient). A fit will then be run to determine a global PA offset and PD.
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
        Set a polarization model for the source with fittable parameters. Set the fittable parameters by calling FitSettings.add_param.
        # Arguments
        * source_name: the name of the source to apply the polarization model to
        * model_fn: a function that returns (q, u) from a polarization model (normalized Stokes coefficient). It takes three arguments: event time, a FitData object, and an array of parameters. You should get the value of a parameter by calling `FitData.param_to_value(param_array, parameter_name)`. This will return the value of the parameter named "parameter_name". To add a parameter other than the default q, u, use `FitSettings.add_param`
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
        Add a parameter for the fitter. You should only use this function if you have used `set_model_fn` to set a polarization model for a source, and now you need to add parameters for that model.
        # Arguments
        * name: Parameter name
        * initial_value: Initial value for the fitter. Default: 0
        * bounds: Bounds for the parameter. Default: no bounds.
        * num_diff_step: Step size to use when numerically computing the uncertainties
        """
        if name == "q" or name == "u" or name == "f" or name == "sigma" or name in self.extra_param_names:
            raise Exception(f"The name {name} cannot be used twice. Parameter names `q`, `u`, `f`, and `sigma` are forbidden.")
        self.extra_param_names.append(name)
        self.extra_param_data.append((initial_value, bounds, num_diff_step))

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
        return len(self.sources)