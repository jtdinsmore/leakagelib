import numpy as np
import logging, copy
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import convolve as scipy_convolve
from ..settings import LEAKAGE_DATA_DIRECTORY
from ..funcs import _convolve
from ..spectrum import EnergyDependence

logger = logging.getLogger("leakagelib")

def _get_nn_bkg_spectrum():
    e_centers, spectrum = np.load(f"{LEAKAGE_DATA_DIRECTORY}/bkg-spec/nn_bkg_spectrum.npy")
    return interp1d(e_centers, spectrum, fill_value=0, bounds_error=False)

class FitProperties:
    """
    This class is internal to LeakageLib. When the FitSettings object is finalized, a FitProperties object is generated which contains all the precomputed fit data.
    """

    def __init__(self, fit_settings, psfs=None):
        FitProperties.validate(fit_settings)
        FitProperties.finalize_fit_settings(fit_settings)

        self.datas = fit_settings.datas
        self.finalize_roi(fit_settings)

        # Get all the combos
        psfs = # TODO

        self.combos = []
        for i, source in enumerate(fit_settings.sources):
            for det in fit_settings.detectors[i]:
                for obs_id in fit_settings.obs_ids[i]:
                    for data in fit_settings.datas:
                        if data.obs_id == obs_id and data.det == det: break
                    combo = DataPSFSourceCombo(source, psfs[det-1], data.use_nn)
                    combo.prepare_fit(fit_settings)
                    self.combos.append(combo)

    def finalize_roi(self, fit_settings):
        """
        Get the ROI image for each detector, exposure map and vignetting corrected.

        Parameters
        ----------
        source_index : int
            The source index to be vignetted. The source's spectrum is used to weight vignetting. If None, vignetting is not applied

        Returns
        -------
            list of array-like
        A dictionary of vignetted ROIs, one per data set. The dictionary is indexed by the standard key
        """
        output = {}
        for data in self.datas:
            image = np.copy(fit_settings.roi)
            key = (data.obs_id, data.det)

            # Load the coords of the roi map
            xs, ys = np.meshgrid(self.sources[0].pixel_centers, self.sources[0].pixel_centers)
            with fits.open(data.filename) as hdul:
                colx = hdul[1].columns["X"]
                coly = hdul[1].columns["Y"]
            stretch = np.cos(coly.coord_ref_value * np.pi / 180)
            ras = ((xs - data.offsets[0]) / IXPE_PIXEL_SIZE - colx.coord_ref_point) / stretch * colx.coord_inc + colx.coord_ref_value
            decs = ((ys - data.offsets[1]) / IXPE_PIXEL_SIZE - coly.coord_ref_point) * coly.coord_inc + coly.coord_ref_value
            center = 300*IXPE_PIXEL_SIZE
            off_axis_arcmin = np.sqrt((xs - data.offsets[0] - center)**2 + (ys - data.offsets[1] - center)**2) / 60

            # Apply exposure map
            if data.expmap is None:
                logger.warning(f"Data set {data.obs_id} DU {data.det} had no exposure map loaded. Please load an exposure map if you are fitting to events in the vignetted portion")
                output[key] = np.copy(self.roi)
            else:
                image *= data.expmap((ras, decs))

            # Apply vignetting
            if source_index is not None and self.spectral_vignettes[source_index] is not None:
                image *= np.interp(off_axis_arcmin, self.vignette_radial_bins, self.spectral_vignettes[source_index])

            output[key] = image
        return output

    def finalize_fit_settings(fit_settings):
        # Set particle spectra
        if np.any([w is not None for w in fit_settings.spectral_weights]):
            for (name, particle, weights) in zip(fit_settings.names, fit_settings.particles, fit_settings.spectral_weights):
                if weights is not None: continue
                if particle:
                    # Apply spectral weights
                    if fit_settings.datas[0].use_nn_energies:
                        spectrum = _get_nn_bkg_spectrum()
                    else:
                        spectrum = lambda e: (e**-1.6)
                    fit_settings.set_spectrum(name, spectrum, use_rmf=False)
                else:
                    logger.warning("The source {name} has no spectrum assigned, but spectra were assigned to other sources. This is equivalent to assuming that {name} has a flat spectrum. Is that intentional?")
        
        # Fix a flux
        if np.all([fit_settings.fixed_flux[i] is None for i in range(len(fit_settings.names))]):
            name = fit_settings.names[0]
            logger.warning(f"All of your sources had variable flux. The fitter requires"\
            f" one of the fluxes to be fixed. Fixing the flux associated with source {name} to 1.")
            fit_settings.fix_flux(name, 1)

    def validate(fit_settings):
        # Check that an ROI was provided
        if fit_settings.roi is None:
            raise Exception("You did not provide an ROI. Provide an ROI so that the background PDF"\
            " can be normalized")
        
        # Check if there are background weights
        bg_weights = False
        for data in fit_settings.datas:
            if np.any(data.evt_bg_chars > 0):
                bg_weights = True
            
        # Check if there is a particle source and an energy-weighted source
        spectral_weights = False
        includes_particles = False
        for (particle, weights) in zip(fit_settings.particles, fit_settings.spectral_weights):
            if particle: 
                includes_particles = True
            if weights is not None:
                spectral_weights = True

        # Warn if no particle source added
        if not includes_particles and bg_weights:
            logger.warning("Your data set has background characters assigned, but you did not add a particle component. Particles will not be modeled. Was this intentional?")

        # Warn if no particle source added
        if includes_particles and not bg_weights:
            raise Exception("You added a particle source, but there are no particle weights in this data set. Please remove the particle source or use a data set with weights.")
        
        # Warn if some sources have spectral weights and some don't
        if spectral_weights:
            for (name, particle, weights) in zip(fit_settings.names, fit_settings.particles, fit_settings.spectral_weights):
                if weights is not None: continue
                if not particle:
                    logger.warning("The source {name} has no spectrum assigned, but spectra were assigned to other sources. This is equivalent to assuming that {name} has a flat spectrum. Is that intentional?")

        # Warn if the sources are different sizes
        common_pixel_size = None
        common_source_dimensions = None
        for (name, source) in zip(fit_settings.names, fit_settings.sources):
            if common_pixel_size is not None:
                if source.pixel_size != common_pixel_size or source.source.shape != common_source_dimensions:
                    raise Exception(f"Your sources do not all have the same pixel size or dimensions. Please make your source images all have the same size with the same pixel scale.")
            common_pixel_size = source.pixel_size
            common_source_dimensions = source.source.shape
        if np.abs(common_pixel_size - 2.9729) > 1e-4:
            logger.warning(f"The source {name} had pixel width not equal to 2.9729. 2.9279 is the pixel width of the sky calibrated PSF, and analysis will be most accurate if you use that pixel width for your sources too.")

class DataPSFSourceCombo:
    """
    Carries preprocessed data for a  specific psf, source, and data set
    """
    def __init__(self, source, psf, use_nn):
        self.source = copy.deepcopy(source)
        self.psf = copy.deepcopy(psf)
        self.fit_prepared = False
        self.use_nn = use_nn
        self.energy_dependence = EnergyDependence.default(self.use_nn)
        
        self._prepare_psf()
        self._prepare_source_polarization()

    def prepare_fit(self, data, fit_settings):
        self.data = data
        source_index = fit_settings.sources.index(self.source)
        data_index = fit_settings.datas.index(self.data)
        self._prepare_event_data()
        self.evt_mus = np.copy(self.data.evt_mus)
        self.clipped_chars = np.clip(self.data.evt_bg_chars, 1e-5, 1-1e-5)

        self.name = fit_settings.names[source_index]
        self.spatial_weight = fit_settings.spatial_weight
        self.particles = fit_settings.particles[source_index]
        self.temporal_weights = fit_settings.temporal_weights[source_index][data_index]
        self.spectral_weights = fit_settings.spectral_weights[source_index][data_index]
        if fit_settings.spectral_mus[source_index] is not None:
            self.evt_mus = fit_settings.spectral_mus[source_index]
        self.sweeps = fit_settings.sweeps[source_index][data_index]
        self.model_fn = fit_settings.model_fns[source_index]

    def get_log_prob(self, q, u):
        probs = np.ones_like(self.data.evt_xs)

        if self.particles:
            # Polarization weights (no need for the 1/2pi)
            probs += 0.5 * (self.data.evt_qs*q + self.data.evt_us*u) # No modulation factor included
            probs *= self.clipped_chars / (1 - self.clipped_chars)
        else:
            # Polarization weights (no need for the 1/2pi)
            probs += 0.5 * self.evt_mus * (self.data.evt_qs*q + self.data.evt_us*u)

        # Spatial weights
        if self.spatial_weight:
            probs *= self.get_event_p_r_given_phi()

        # Phase weights
        if self.temporal_weights is not None:
            probs *= self.temporal_weights

        # Spectral weights
        if self.spectral_weights is not None:
            probs *= self.spectral_weights

        return probs

    def _roi_weighted_mean(self, array):
        return np.sum(array * self.roi) / np.sum(self.roi)

    def _prepare_psf(self):
        if self.is_uniform:
            self.d_i_i = np.copy(self.source.source)
            self.d_zs_i = np.zeros_like(self.source.source)
            self.d_qs_i = np.zeros_like(self.source.source)
            self.d_us_i = np.zeros_like(self.source.source)
            self.d_zk_i = np.zeros_like(self.source.source)
            self.d_qk_i = np.zeros_like(self.source.source)
            self.d_uk_i = np.zeros_like(self.source.source)
            self.d_xk_i = np.zeros_like(self.source.source)
            self.d_yk_i = np.zeros_like(self.source.source)
        else:
            self.d_i_i = _convolve(self.source.source, self.psf.psf)
            self.d_zs_i = _convolve(self.source.source, self.psf.d_zs, fix_edges=False)
            self.d_qs_i = _convolve(self.source.source, self.psf.d_qs, fix_edges=False)
            self.d_us_i = _convolve(self.source.source, self.psf.d_us, fix_edges=False)
            self.d_zk_i = _convolve(self.source.source, self.psf.d_zk, fix_edges=False)
            self.d_qk_i = _convolve(self.source.source, self.psf.d_qk, fix_edges=False)
            self.d_uk_i = _convolve(self.source.source, self.psf.d_uk, fix_edges=False)
            self.d_xk_i = _convolve(self.source.source, self.psf.d_xk, fix_edges=False)
            self.d_yk_i = _convolve(self.source.source, self.psf.d_yk, fix_edges=False)

    def _prepare_source_polarization(self):
        if self.is_point_source:
            q_src, u_src = np.sum(self.source.q_map), np.sum(self.source.u_map)
            self.d_i_q = self.d_i_i * q_src
            self.d_i_u = self.d_i_i * u_src

            self.d_zs_q = self.d_zs_i * q_src
            self.d_zk_q = self.d_zk_i * q_src
            self.d_xk_q = self.d_xk_i * q_src
            self.d_yk_q = self.d_yk_i * q_src

            self.d_zs_u = self.d_zs_i * u_src
            self.d_zk_u = self.d_zk_i * u_src
            self.d_xk_u = self.d_xk_i * u_src
            self.d_yk_u = self.d_yk_i * u_src

            self.d_qs_q = self.d_qs_i * q_src
            self.d_qk_q = self.d_qk_i * q_src

            self.d_us_u = self.d_us_i * u_src
            self.d_uk_u = self.d_uk_i * u_src
        elif self.is_uniform:
            q_src, u_src = np.mean(self.source.q_map), np.mean(self.source.u_map)
            self.d_i_q = self.d_i_i * q_src
            self.d_i_u = self.d_i_i * u_src

            self.d_zs_q = self.d_zs_i * 0
            self.d_zk_q = self.d_zk_i * 0
            self.d_xk_q = self.d_xk_i * 0
            self.d_yk_q = self.d_yk_i * 0

            self.d_zs_u = self.d_zs_i * 0
            self.d_zk_u = self.d_zk_i * 0
            self.d_xk_u = self.d_xk_i * 0
            self.d_yk_u = self.d_yk_i * 0

            self.d_qs_q = self.d_qs_i * 0
            self.d_qk_q = self.d_qk_i * 0

            self.d_us_u = self.d_us_i * 0
            self.d_uk_u = self.d_uk_i * 0
        else:
            self.d_i_q = _convolve(self.source * self.source.q_map, self.psf.psf)
            self.d_i_u = _convolve(self.source * self.source.u_map, self.psf.psf)
            
            self.d_zs_q = _convolve(self.source * self.source.q_map, self.psf.d_zs, fix_edges=False)
            self.d_zk_q = _convolve(self.source * self.source.q_map, self.psf.d_zk, fix_edges=False)
            self.d_xk_q = _convolve(self.source * self.source.q_map, self.psf.d_xk, fix_edges=False)
            self.d_yk_q = _convolve(self.source * self.source.q_map, self.psf.d_yk, fix_edges=False)
            
            self.d_zs_u = _convolve(self.source * self.source.u_map, self.psf.d_zs, fix_edges=False)
            self.d_zk_u = _convolve(self.source * self.source.u_map, self.psf.d_zk, fix_edges=False)
            self.d_xk_u = _convolve(self.source * self.source.u_map, self.psf.d_xk, fix_edges=False)
            self.d_yk_u = _convolve(self.source * self.source.u_map, self.psf.d_yk, fix_edges=False)
            
            self.d_qs_q = _convolve(self.source * self.source.q_map, self.psf.d_qs, fix_edges=False)
            self.d_qk_q = _convolve(self.source * self.source.q_map, self.psf.d_qk, fix_edges=False)
            
            self.d_us_u = _convolve(self.source * self.source.u_map, self.psf.d_us, fix_edges=False)
            self.d_uk_u = _convolve(self.source * self.source.u_map, self.psf.d_uk, fix_edges=False)

    def _prepare_event_data(self):
        poses = (self.data.evt_ys, self.data.evt_xs)
        lines = (self.pixel_centers, self.pixel_centers)
        self.evt_d_i_i = RegularGridInterpolator(lines, self.d_i_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_zs_i = RegularGridInterpolator(lines, self.d_zs_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_zk_i = RegularGridInterpolator(lines, self.d_zk_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_qs_i = RegularGridInterpolator(lines, self.d_qs_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_qk_i = RegularGridInterpolator(lines, self.d_qk_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_us_i = RegularGridInterpolator(lines, self.d_us_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_uk_i = RegularGridInterpolator(lines, self.d_uk_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_xk_i = RegularGridInterpolator(lines, self.d_xk_i, fill_value=0, bounds_error=False)(poses)
        self.evt_d_yk_i = RegularGridInterpolator(lines, self.d_yk_i, fill_value=0, bounds_error=False)(poses)

    def polarize_net(self, qu):
        self.source.polarize_net(qu)
        self._prepare_source_polarization()
        self._prepare_event_data()

    def blur_psf(self, sigma):
        self.psf.blur(sigma)
        self._prepare_psf()
        self._prepare_source_polarization()
        self._prepare_event_data()

    def get_event_p_r_given_phi(self):
        """
        Get the probability for an array of events to have their positions given their polarization.

        Parameters
        ----------
        psf : PSF
            PSF for the detector. Sky-calibrated PSFs recommended.
        data : IXPEData
            IXPEData object containing events. Reads evt_xs, evt_ys, evt_qs, evt_us, evt_energies.
            If the PSF rotation angle is < 1e-5, the antirotated versions will be read.

        Returns
        -------
        array-like
            Probabilities for each event.
        """

        if not self.fit_prepared:
            raise Exception("You cannot call this function unless the DataPSFSourceCombo has been prepared for fitting")

        # Compute leakage parameters
        sigma_para2, sigma_perp2, k_para4 = self.energy_dependence.evaluate(self.data.evt_energies)
        k_perp4 = 3 * sigma_perp2**2
        sigma_plus = sigma_para2 + sigma_perp2
        sigma_minus = sigma_para2 - sigma_perp2
        k_plus = k_para4 + k_perp4
        k_minus = k_para4 - k_perp4
        k_cross = -k_minus / 4

        # Normalize the probabilities by computing the integral over all position.
        # The normalization condition is that the mean over the image is equal to 1
        normalization = (
            self._roi_weighted_mean(self.d_i_i) +

            sigma_plus * self._roi_weighted_mean(self.d_zs_i) +
            k_plus * self._roi_weighted_mean(self.d_zk_i) +

            self.evt_mus/2 * (
                sigma_minus * self._roi_weighted_mean(self.d_qs_q) +
                k_minus * self._roi_weighted_mean(self.d_qk_q) +

                sigma_minus * self._roi_weighted_mean(self.d_us_u) +
                k_minus * self._roi_weighted_mean(self.d_uk_u)
            )
        ) # NB it's guaranteed that all sources have the same size

        out = (
            self.evt_d_i_i +

            sigma_plus * self.evt_d_zs_i +
            k_plus * self.evt_d_zk_i + 

            self.data.evt_qs/2 * (
                sigma_minus * self.evt_d_qs_i + 
                k_minus * self.evt_d_qk_i
            ) +

            self.data.evt_us/2 * (
                sigma_minus * self.evt_d_us_i + 
                k_minus * self.evt_d_uk_i
            ) +

            (self.data.evt_qs**2 - self.data.evt_us**2)/4 * (
                k_cross * self.evt_d_xk_i
            ) + 

            (self.data.evt_qs*self.data.evt_us)/2 * (
                k_cross * self.evt_d_yk_i
            )
        ) / normalization
    
        return out