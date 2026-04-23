import numpy as np
import logging
from astropy.io import fits
from . import spectral_weights
from ..source import Source
from ..ixpe_data import IXPE_PIXEL_SIZE
from ixpeobssim.irf import load_vign, load_arf

USE_SPECTRAL_MUS = False # Set to True to account for the fact that the finite detector energy
# resolution means that the mu corresponding to the measured energy is different from the true mu,
# which depends on the true energy. Accounting for this will lower mu on average.

logger = logging.getLogger("leakagelib")

def _get_num_pixels(datas):
    max_radius = 0
    for data in datas:
        max_radius = max(max_radius, np.sqrt(np.max(data.evt_xs**2 + data.evt_ys**2)))
    n_pixels = int(np.ceil(2 * max_radius / 2.9729)) + 5 # Inflate the radius by 5 pixels just for safety
    if n_pixels % 2 == 0:
        n_pixels += 1
    return n_pixels

class FitSettings:
    """
    An object that keeps track of the meta-parameters of a fit.
    """
    def __init__(self, datas):
        """
        Create a fit settings object for use in fitting the data sets listed in "datas"
        """
        max_radius = 0
        all_obs_ids = []
        all_detectors = {}
        for data in datas:
            center = 300*IXPE_PIXEL_SIZE
            off_axis_arcmin = np.sqrt((data.evt_xs - data.offsets[0] - center)**2 + (data.evt_ys - data.offsets[1] - center)**2) / 60
            max_radius = max(max_radius, np.max(off_axis_arcmin))
            all_obs_ids.append(data.obs_id)
            if data.obs_id not in all_detectors:
                all_detectors[data.obs_id] = []
            all_detectors[data.obs_id].append(data.det)
        self.all_detectors = all_detectors
        self.all_obs_ids = np.unique(all_obs_ids)

        self.pixel_centers = None
        self.vignette_radial_bins = np.arange(0, max_radius, 0.05) # Radial bins to be used to get the vignetting map
        self.spatial_weight = True
        self.fixed_blur = 0
        self.roi = None
        self.datas = datas

        self.sources = {}
        self.detectors = {}
        self.obs_ids = {}
        self.fixed_quf = {}
        self.particles = {}
        self.guess_quf = {}
        self.model_fns = {}
        self.spectral_weights = {}
        self.spectral_mus = {}
        self.temporal_weights = {}
        self.sweeps = {}

        self.spectral_vignettes = {}
        self.extra_params = {}

    def _check_source_dim(self, source):
        if len(self.sources) == 0: return
        standard_text = "LeakageLib requires each source to have the same dimensions and pixel size."\
        " If you are creating a source using your own Source object, add that source to the FitSettings" \
        " first. Make all `add_background` or `add_point_source` calls afterwards. They should use the" \
        " same image properties as the source you created."

        if len(self.pixel_centers) != len(source.pixel_centers):
            raise Exception(f"This source does not have the same image size as previous source(s). {standard_text}")
        if np.any(np.abs(self.pixel_centers - source.pixel_centers) > 1e-4):
            raise Exception(f"This source does not have the same pixel size as previous source(s). {standard_text}")
        
    def add_point_source(self, name="src", det=None, obs_ids=None):
        """
        Add a point source to the fit.

        Parameters
        ----------
        name : str, optional
            Name of the source. Default is "src".
        det : tuple of int, optional
            Detectors the source should apply to. Default is None (all detectors)
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).
        """

        if self.pixel_centers is None:
            num_pixels = _get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = len(self.pixel_centers)
            pixel_width = self.pixel_centers[1] - self.pixel_centers[0]

        source = Source.delta(num_pixels, pixel_width)
        if self.pixel_centers is None:
            self.pixel_centers = source.pixel_centers
        self.add_source(source, name, det, obs_ids)
    
    def add_background(self, name="bkg", det=None, obs_ids=None):
        """
        Add a uniform, polarized background to the fit.

        Parameters
        ----------
        name : str, optional
            Name of the source. Default is "bkg".
        det : tuple of int, optional
            Detectors the source should apply to. Default is None (all detectors)
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).

        Notes
        -----
        - To add an independent background to each detector, add three background sources and pass det=1, det=2, det=3 for each, with different names.
        - The shape of the background image is set to the same shape as the first source object added to the fit settings. If no source exists, pixel width is set to the PSF native size of 2.9729 arcsec and number of pixels is the largest radius of events in the data set. 
        """

        if self.pixel_centers is None:
            num_pixels = _get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = len(self.pixel_centers)
            pixel_width = self.pixel_centers[1] - self.pixel_centers[0]

        source = Source.uniform(num_pixels, pixel_width)
        self.add_source(source, name, det, obs_ids)
    
    def add_particle_background(self, name="pbkg", det=None, obs_ids=None):
        """
        Add a uniform particle background component to the fit.

        Parameters
        ----------
        name : str, optional
            Name of the source. Default is "pbkg".
        det : tuple of int, optional
            Detectors the source should apply to. Default is None (all detectors)
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).
        """

        if len(self.sources) == 0:
            num_pixels = _get_num_pixels(self.datas)
            pixel_width = 2.9729
        else:
            num_pixels = len(self.pixel_centers)
            pixel_width = self.pixel_centers[1] - self.pixel_centers[0]
        source = Source.uniform(num_pixels, pixel_width)
        self.add_source(source, name, det, obs_ids)
        self.particles[name] = True
    
    def add_source(self, source, name, det=None, obs_ids=None):
        """
        Add an extended source to the fit.

        Parameters
        ----------
        source : Source
            Source object containing the flux map.
        name : str
            Name of the source.
        det : tuple of int, optional
            Detectors the source should apply to. Default is None (all detectors)
        obs_ids : tuple of int, optional
            Observation IDs the source should apply to. Default is None (all observations).
        """

        real_obs_ids = np.copy(self.all_obs_ids) if obs_ids is None else obs_ids
        detectors = []
        for o in real_obs_ids:
            for d in self.all_detectors[o]:
                detectors.append(d)
        detectors = np.unique(detectors)

        self._check_source_dim(source)
        self.sources[name] = source
        self.detectors[name] = detectors
        self.obs_ids[name] = real_obs_ids
        self.fixed_quf[name] = [None, None, None]
        self.guess_quf[name] = [0, 0, 1]
        self.particles[name] = False
        self.spectral_weights[name] = None
        self.spectral_mus[name] = None
        self.spectral_vignettes[name] = None
        self.temporal_weights[name] = None
        self.model_fns[name] = None
        self.sweeps[name] = None
        if self.pixel_centers is None:
            self.pixel_centers = source.pixel_centers
        
    def set_spectrum(self, source_name, spectrum, use_arf=True, use_rmf=True, duty_cycle=None, irf_name=None):
        """
        Set a spectrum for a source. Weights are assigned by running the spectrum
        function on all event energies.

        Parameters
        ----------
        source_name : str
            Name of the source to assign spectral weights.
        spectrum : callable
            Function taking an energy scalar or array and returning a weight scalar or array.
        use_arf : bool
            Set to False to not apply IXPE's ARF. When applied, the ARF of detector 1 is used (the ARF difference is quite small between detectors) Default: True. The ARF is loaded by IXPEobssim.
        use_rmf : bool
            Set to False to not apply IXPE's RMF.
        irf_name : str, optional
            IXPEobssim's name for the ARF to be used. IXPEobssim's default will be used when not provided.
        duty_cycle : callable, optional
            Fraction of data to distribute over energy. Default is uniform over the data range.
            If a contiguous energy cut was applied, the default may be used.
        """

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")

        if use_arf:
            if irf_name is None:
                arf = load_arf()
            else:
                arf = load_arf(irf_name=irf_name)
            arf_spectrum = lambda e: spectrum(e) * arf(e)
        else:
            arf_spectrum = lambda e: spectrum(e)

        if use_rmf:
            rmf = spectral_weights.RMF()
        else:
            rmf = spectral_weights.RMF.delta()
        convolved_spec = rmf.convolve_spectrum(arf_spectrum)

        weights = []
        mus = []
        max_energy = -np.inf
        min_energy = np.inf
        for data in self.datas:
            max_energy = max(np.max(data.evt_energies), max_energy)
            min_energy = min(np.min(data.evt_energies), min_energy)
            these_weights = convolved_spec(data.evt_energies)
            weights.append(these_weights)

            if USE_SPECTRAL_MUS:
                convolved_spec_mu = rmf.convolve_spectrum_mu(arf_spectrum, data.use_nn)
                these_mus = convolved_spec_mu(data.evt_energies) / these_weights
                mus.append(these_mus)
            else:
                mus.append(data.evt_mus)

        # Compute normalization
        energies = np.linspace(min_energy, max_energy, 100)
        if duty_cycle is None:
            duty_cycle = lambda e: np.ones_like(e)
        spectrum_array = convolved_spec(energies) * duty_cycle(energies)
        integral = np.sum(spectrum_array)
        integral_constant = np.sum(duty_cycle(energies)) # Settings weights equal to one would give this value

        # Normalize the spectrum so that the spectrum normalization is equal to that of weights=1
        multiplier = integral_constant / integral        
        for i in range(len(weights)):
            weights[i] *= multiplier

        # Get vignetting function
        vign_function = load_vign()
        vignettes = []
        for radius in self.vignette_radial_bins:
            vignettes.append(np.sum(vign_function(energies, radius) * spectrum_array / integral))

        self.spectral_vignettes[source_name] = vignettes
        self.spectral_weights[source_name] = weights
        self.spectral_mus[source_name] = mus
        
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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")

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

        self.temporal_weights[source_name] = weights
        
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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        if self.model_fns[source_name] is not None:
            raise Exception("You cannot set a sweep for a source that you have already set a polarization model function for")

        sweeps = []
        for data in self.datas:
            sweeps.append(sweep(data.evt_times))

        self.sweeps[source_name] = sweeps
        
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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        if self.sweeps[source_name] is not None:
            raise Exception("You cannot set a model function for a source that you have already set a polarization sweep for")
        
        self.model_fns[source_name] = model_fn
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

        if name == "q" or name == "u" or name == "f" or name == "sigma":
            raise Exception(f"Parameter names `q`, `u`, `f`, and `sigma` are forbidden.")
        if  name in self.extra_params:
            raise Exception(f"The name {name} cannot be used twice.")
        self.extra_params[name] = (initial_value, bounds, num_diff_step)

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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        self.fixed_quf[source_name][0] = qu[0]
        self.fixed_quf[source_name][1] = qu[1]

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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        self.guess_quf[source_name][0] = qu[0]
        self.guess_quf[source_name][1] = qu[1]

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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        self.fixed_quf[source_name][2] = flux

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

        if not source_name in self.sources:
            raise Exception(f"The source {source_name} is not in the list of sources.")
        self.guess_quf[source_name][2] = flux

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
        
        Notes
        -----
            The roi_image must have the same dimensions as source objects. If a source with a different size is already added to the fit, an error will be thrown. If those pre-loaded sources are point sources or background, solve this problem by simply applying the ROI first. If they are sources you created, then you need to make sure your sources and background have the same shape and pixel sizes.
        """

        if len(self.sources) > 0:
            if len(self.pixel_centers) != roi_image.shape[0]:
                raise Exception("The ROI image must have the same size as the source images. You can access the coordinates of each pixel are stored in FitSettings.pixel_centers.")
        self.roi = roi_image

        # Cut events outside the ROI
        total_cut = 0
        for data_index, data in enumerate(self.datas):
            ix = np.digitize(data.evt_xs, self.pixel_centers) - 1
            iy = np.digitize(data.evt_ys, self.pixel_centers) - 1
            cut_mask = roi_image[ix, iy] < 1e-4
            if np.sum(cut_mask) > 0:
                total_cut += np.sum(cut_mask)
                data.retain(~cut_mask)

            # Remove weights already established
            for name in self.sources.keys():
                if self.spectral_weights[name] is not None:
                    self.spectral_weights[name][data_index] = self.spectral_weights[name][data_index][~cut_mask]
                if self.temporal_weights[name] is not None:
                    self.temporal_weights[name][data_index] = self.temporal_weights[name][data_index][~cut_mask]
                if self.spectral_mus[name] is not None:
                    self.spectral_mus[name][data_index] = self.spectral_mus[name][data_index][~cut_mask]
                if self.sweeps[name] is not None:
                    self.sweeps[name][data_index] = self.sweeps[name][data_index][~cut_mask]

        if total_cut > 0:
            logger.warning(f"{total_cut} events were cut for being outside the region of interest.")

    def apply_circular_roi(self, radius):
        """
        Provide a circular ROI centered on the origin (radius in arcseconds).
        """
        if len(self.sources) == 0:
            pixel_width = 2.9729
            num_pixels = np.ceil(2*radius / 2.9729) + 2
            if num_pixels % 1 == 0:
                num_pixels += 1
            self.pixel_centers = np.arange(num_pixels) * pixel_width
            self.pixel_centers -= np.mean(self.pixel_centers)

        xs, ys = np.meshgrid(self.pixel_centers, self.pixel_centers)
        roi_image = xs**2 + ys**2 < radius**2
        self.apply_roi(roi_image.astype(float))

    def get_n_sources(self):
        """
        Returns the number of sources that have been added to the fitter.
        """
        return len(self.sources)