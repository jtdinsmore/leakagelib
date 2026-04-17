
import numpy as np
import copy
from scipy.interpolate import RegularGridInterpolator
from .ixpe_data import IXPE_PIXEL_SIZE
from .funcs import _convolve
from .spectrum import EnergyDependence

class PSFSourceCombo:
    """
    Carries preprocessed data for a  specific psf, source, and data set
    """
    def __init__(self, source, psf, use_nn=False):
        self.source = copy.deepcopy(source)
        self.psf = copy.deepcopy(psf)
        self.fit_prepared = False
        self.use_nn = use_nn
        self.energy_dependence = EnergyDependence.default(self.use_nn)
        
        self._prepare_psf()
        self._prepare_source_polarization()

    def _prepare_fit(self, data, fit_settings, source_name):
        self.data = data
        self.name = source_name
        for data_index, this_data in enumerate(fit_settings.datas):
            if data.obs_id == this_data.obs_id and data.det == this_data.det: break
        self._prepare_event_data()
        self.clipped_chars = np.clip(self.data.evt_bg_chars, 1e-5, 1-1e-5)

        self.spatial_weight = fit_settings.spatial_weight
        self.particles = fit_settings.particles[source_name]
        self.temporal_weights = fit_settings.temporal_weights[source_name]
        if self.temporal_weights is not None:
            self.temporal_weights = self.temporal_weights[data_index]
        self.spectral_weights = fit_settings.spectral_weights[source_name]
        if self.spectral_weights is not None:
            self.spectral_weights = self.spectral_weights[data_index]
        self.evt_mus = np.copy(self.data.evt_mus)
        if fit_settings.spectral_mus[source_name] is not None:
            self.evt_mus = fit_settings.spectral_mus[source_name][data_index]
        self.sweeps = fit_settings.sweeps[source_name]
        if self.sweeps is not None:
            self.sweeps = self.sweeps[data_index]
        self.model_fn = fit_settings.model_fns[source_name]

        # Prepare ROI
        self.roi = np.copy(fit_settings.roi)
        xs, ys = np.meshgrid(fit_settings.pixel_centers, fit_settings.pixel_centers)

        # Apply exposure map
        if not self.data.expmap is None:
            self.roi *= self.data.expmap((xs - self.data.offsets[0], ys - self.data.offsets[1]))

        # Apply vignetting
        if fit_settings.spectral_vignettes[source_name] is not None:
            center_arcsec = 300 * IXPE_PIXEL_SIZE
            off_axis_arcmin = np.sqrt((xs - self.data.offsets[0] - center_arcsec)**2 + (ys - self.data.offsets[1] - center_arcsec)**2) / 60
            self.roi *= np.interp(off_axis_arcmin, fit_settings.vignette_radial_bins, fit_settings.spectral_vignettes[source_name])

        self.roi /= np.mean(self.roi)
        self.fit_prepared = True
        self.data_key = (self.data.obs_id, self.data.det)

    def get_log_prob(self, q, u):
        probs = np.ones(len(self.data.evt_xs), float)

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
        if self.source.is_uniform:
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
        if self.source.is_point_source:
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
        elif self.source.is_uniform:
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
            self.d_i_q = _convolve(self.source.source * self.source.q_map, self.psf.psf)
            self.d_i_u = _convolve(self.source.source * self.source.u_map, self.psf.psf)
            
            self.d_zs_q = _convolve(self.source.source * self.source.q_map, self.psf.d_zs, fix_edges=False)
            self.d_zk_q = _convolve(self.source.source * self.source.q_map, self.psf.d_zk, fix_edges=False)
            self.d_xk_q = _convolve(self.source.source * self.source.q_map, self.psf.d_xk, fix_edges=False)
            self.d_yk_q = _convolve(self.source.source * self.source.q_map, self.psf.d_yk, fix_edges=False)
            
            self.d_zs_u = _convolve(self.source.source * self.source.u_map, self.psf.d_zs, fix_edges=False)
            self.d_zk_u = _convolve(self.source.source * self.source.u_map, self.psf.d_zk, fix_edges=False)
            self.d_xk_u = _convolve(self.source.source * self.source.u_map, self.psf.d_xk, fix_edges=False)
            self.d_yk_u = _convolve(self.source.source * self.source.u_map, self.psf.d_yk, fix_edges=False)
            
            self.d_qs_q = _convolve(self.source.source * self.source.q_map, self.psf.d_qs, fix_edges=False)
            self.d_qk_q = _convolve(self.source.source * self.source.q_map, self.psf.d_qk, fix_edges=False)
            
            self.d_us_u = _convolve(self.source.source * self.source.u_map, self.psf.d_us, fix_edges=False)
            self.d_uk_u = _convolve(self.source.source * self.source.u_map, self.psf.d_uk, fix_edges=False)

    def _prepare_event_data(self):
        poses = (self.data.evt_ys, self.data.evt_xs)
        lines = (self.source.pixel_centers, self.source.pixel_centers)
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

    def blur_psf(self, sigma):
        self.psf.blur(sigma)
        self._prepare_psf()
        self._prepare_event_data()
        self._prepare_source_polarization()

    def _get_event_p_r_given_phi(self):
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
            raise Exception("You cannot call this function unless the PSFSourceCombo has been prepared for fitting")

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

    def compute_leakage(self, spectrum, normalize=False):
        """
        Get the Q and U maps for this source (unnormalized by default), given the PSF and spectrum.

        Parameters
        ----------
        spectrum : DataSpectrum
            Spectrum of the data. You can create this from an IXPEData object with DataSpectrum.from_data.
        normalize : bool, optional
            If True, return normalized Stokes q and u maps. Default is False.

        Returns
        -------
        tuple (array-like, array-like, array-like)
            Three images, i, q, u, of leakage patterns.
        """

        params = self.energy_dependence.get_params(spectrum)

        i = (
            + self.d_i_i
            + params["sigma_plus"] * self.d_zs_i
            + params["k_plus"] * self.d_zk_i 
            + params["mu_sigma_minus"] * (self.d_qs_q + self.d_us_u) / 2
            + params["mu_k_minus"] * (self.d_qk_q + self.d_uk_u) / 2
        )
        q = (
            + params["mu"] * self.d_i_q
            + params["sigma_minus"] * self.d_qs_i
            + params["k_minus"] * self.d_qk_i
            + params["mu_sigma_plus"] * self.d_zs_q
            + params["mu_k_plus"] * self.d_zk_q
            + params["mu_k_cross"] * (self.d_xk_q + self.d_yk_u) / 2
        )
        u = (
            + params["mu"] * self.d_i_u
            + params["sigma_minus"] * self.d_us_i
            + params["k_minus"] * self.d_uk_i
            + params["mu_sigma_plus"] * self.d_zs_u
            + params["mu_k_plus"] * self.d_zk_u
            + params["mu_k_cross"] * (self.d_yk_q - self.d_xk_u) / 2
        )

        q[np.isnan(q)] = 0
        u[np.isnan(u)] = 0

        if normalize:
            q /= i
            u /= i

        return (i, q, u)