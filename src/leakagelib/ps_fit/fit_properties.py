import numpy as np
import logging
from scipy.interpolate import interp1d
from ..settings import LEAKAGE_DATA_DIRECTORY
from ..psf import PSF
from ..combo import PSFSourceCombo

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

        # Get all the combos
        self.combos = []
        for source_name, source in fit_settings.sources.items():
            for det in fit_settings.detectors[source_name]:
                for obs_id in fit_settings.obs_ids[source_name]:
                    for data_index, data in enumerate(fit_settings.datas):
                        if data.obs_id == obs_id and data.det == det:
                            break

                    if psfs is not None:
                        psf = psfs[data_index]
                    else:
                        sample_source = fit_settings.sources[list(fit_settings.sources.keys())[0]]
                        psf = PSF.sky_cal(det, sample_source, data.rotation)
                        if fit_settings.fixed_blur is not None and fit_settings.fixed_blur != 0:
                            psf.blur(fit_settings.fixed_blur)

                    combo = PSFSourceCombo(source, psf, data.use_nn)
                    combo._prepare_fit(data, fit_settings, source_name)
                    self.combos.append(combo)

        # Copy over some of the other data
        self.guess_quf = fit_settings.guess_quf
        self.fixed_quf = fit_settings.fixed_quf
        self.extra_params = fit_settings.extra_params

    def finalize_fit_settings(fit_settings):
        # Set particle spectra
        if np.any([w is not None for w in fit_settings.spectral_weights]):
            for name in fit_settings.particles:
                if fit_settings.spectral_weights[name] is not None: continue
                if fit_settings.particles[name]:
                    # Apply spectral weights
                    if fit_settings.datas[0].use_nn_energies:
                        spectrum = _get_nn_bkg_spectrum()
                    else:
                        spectrum = lambda e: (e**-1.6)
                    fit_settings.set_spectrum(name, spectrum, use_arf=False, use_rmf=False)
        
        # Fix a flux
        if np.all([quf[2] is None for quf in fit_settings.fixed_quf.values()]):
            name = list(fit_settings.sources.keys())[0]
            logger.warning(f"All of your sources had variable flux. The fitter requires"\
            f" one of the fluxes to be fixed. Fixing the flux associated with source {name} to 1.")
            fit_settings.fix_flux(name, 1)

    def validate(fit_settings):
        # Check that an ROI was provided
        if fit_settings.roi is None:
            raise Exception("You did not provide an ROI. Provide an ROI so that the background PDF can be normalized")
        
        for data in fit_settings.datas:
            if data.expmap is None:
                logger.warning(f"Data set {data.obs_id} DU {data.det} had no exposure map loaded. Please load an exposure map if you are fitting to events in the vignetted portion.")
        
        # Check if there are background weights
        bg_weights = False
        for data in fit_settings.datas:
            if np.any(data.evt_bg_chars > 0):
                bg_weights = True
            
        # Check if there is a particle source and an energy-weighted source
        spectral_weights = np.any([w is not None for w in fit_settings.spectral_weights.values()])
        includes_particles = np.any([v for v in fit_settings.particles.values()])

        # Warn if no particle source added
        if not includes_particles and bg_weights:
            logger.warning("Your data set has background characters assigned, but you did not add a particle component. Particles will not be modeled. Was this intentional?")

        # Warn if no particle source added
        if includes_particles and not bg_weights:
            raise Exception("You added a particle source, but there are no particle weights in this data set. Please remove the particle source or use a data set with weights.")
        
        # Warn if some sources have spectral weights and some don't
        if spectral_weights:
            for name in fit_settings.sources:
                if fit_settings.spectral_weights[name] is not None: continue
                if not fit_settings.particles[name]:
                    logger.warning(f"The source {name} has no spectrum assigned, but spectra were assigned to other sources. This is equivalent to assuming that {name} has a flat spectrum. Is that intentional?")

        # Warn if the sources are different sizes
        common_pixel_size = None
        common_source_dimensions = None
        for source in fit_settings.sources.values():
            if common_pixel_size is not None:
                if np.abs(source.pixel_size - common_pixel_size) > 1e-4:
                    raise Exception(f"Your sources do not all have the same pixel size. Please make your source images all have the same pixel scale.")
                if source.source.shape != common_source_dimensions:
                    raise Exception(f"Your sources do not all have the same dimensions. Please make your source images all have the same size.")
            common_pixel_size = source.pixel_size
            common_source_dimensions = source.source.shape
        if np.abs(common_pixel_size - 2.9729) > 1e-4:
            logger.warning(f"The sources had pixel width not equal to 2.9729. 2.9279 is the pixel width of the sky calibrated PSF, and analysis will be most accurate if you use that pixel width for your sources too.")
