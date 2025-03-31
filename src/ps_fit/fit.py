import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from numdifftools import Hessian
from .fit_settings import FitSettings
from .pcube import get_pcube
from ..psf import PSF
from ..source import Source
from ..spectrum import EnergyDependence

def fit_point_source(datas, fixed_qu=None, fixed_position=None, fixed_blur=0, fixed_bg=None, psfs=None, full_hessian=False):
    """Fit for the Stokes coefficient of a point source. By default, returns the PS q and u (normalized),
    the background fraction, and the PS position offset.

    # Arguments:
    * datas: a list of IXPEData objects. Does not need to be binned; only the events will be used.
    * fixed_qu: set to fix the value of q and u
    * fixed_position: set to fix the PS position
    * fixed_blur: set to fix the PSF blur. Increases runtime if set to None.
    * fixed_bg: set to fix the BG fraction
    * psfs: set to your own list of PSF objects if you wish to alter the PSFs in some way. Set to
        None (default) to use the Dinsmore & Romani 2023 PSFs. Must be a list of three elements
        ordered as DU 1, 2, and 3.
    * full_hessian: set to True to calculate uncertainties for all parameters, not just Q and U

    # Returns:
    * A FitResult object containing the fit statistics summary
    """

    # Check provided values
    if len(datas) == 0:
        raise Exception("Please provide a list of IXPEDatas")

    # Get PCUBE estimate
    pcube = get_pcube(datas)

    # Determine how big an image we need to generate leakage predictions
    pixel_size = 2.9729
    min_x = np.nan
    min_y = np.nan
    max_x = np.nan
    max_y = np.nan
    for data in datas:
        data.antirotate_events()
        min_x = np.nanmin([min_x, np.min(data.evt_xs_antirot)])
        min_y = np.nanmin([min_y, np.min(data.evt_ys_antirot)])
        max_x = np.nanmax([max_x, np.max(data.evt_xs_antirot)])
        max_y = np.nanmax([max_y, np.max(data.evt_ys_antirot)])

    num_pixels = (max(max_x - min_x, max_y - min_y) + 20) / pixel_size # Add 10 arcsec for x, y offsets
    num_pixels = int(np.ceil(num_pixels))
    if num_pixels % 2 == 0:
        num_pixels += 1 # Make odd
    
    source = Source.delta(datas[0].use_nn, num_pixels=num_pixels, pixel_size=pixel_size)

    # Load PSFs
    if psfs is not None:
        if len(psfs) != 3:
            raise Exception("You must pass three PSFs")
        for i in range(3):
            if psfs[i].det != i+1:
                raise Exception(f"The {i}th element of psfs must be represent detector {i+1}")
    else:
        psfs = [
            PSF.sky_cal(det, source, 0) for det in range(1, 4)
        ]

    # Make the fit structure
    fit_settings = FitSettings(fixed_qu, fixed_position, fixed_blur, fixed_bg, len(datas))
    fitter = Fitter(source, datas, psfs, fit_settings, pcube)
    return fitter.fit(full_hessian)

class FitResult:
    def __init__(self, result, hessian, fit_settings):
        """
        Create a FitResult from a scipy.optimize.minimize output (hessian should be the chisquared hessian)
        """
        self.params = {}
        for i, param in enumerate(result.x):
            self.params[fit_settings.index_to_param(i)] = param
        self.fun = result.fun
        self.cov = np.linalg.pinv(hessian/2)
        self.message = result.message
        self.fit_settings = fit_settings

    def get_pd_pa(self):
        q = self.params[("q",None)]
        u = self.params[("u",None)]
        q_index = self.fit_settings.param_to_index("q")
        u_index = self.fit_settings.param_to_index("u")
        q_unc2 = self.cov[q_index,q_index]
        u_unc2 = self.cov[u_index,u_index]
        pd = np.sqrt(q**2 + u**2)
        pa = np.arctan2(u, q)/2
        pd_unc = np.sqrt(q**2 * q_unc2 + u**2 * u_unc2) / pd
        pa_unc = np.sqrt(q**2 * u_unc2 + u**2 * q_unc2) / pd**2 / 2
        return pd, pa, pd_unc, pa_unc

    def __str__(self):
        pd, pa, pd_unc, pa_unc = self.get_pd_pa()
        text = "FitResult:\n"
        for ((name, index), value) in self.params.items():
            param_index = self.fit_settings.param_to_index(name, index)
            if param_index < self.cov.shape[0]:
                unc = np.sqrt(self.cov[param_index, param_index])
            else:
                unc = None
            if index is None:
                text += f"\t{name} = {value} +/- {unc}\n"
            else:
                text += f"\t{name} ({index}) = {value} +/- {unc}\n"
        text += "\nPolarization:\n"
        text += f"\tPD: {pd} +/- {pd_unc}\n"
        text += f"\tPA: {pa*180/np.pi} deg +/- {pa_unc*180/np.pi}\n"
        text += f"fun: {self.fun}\n"
        text += self.message
        return text

    def __repr__(self):
        return str(self)


class Fitter:
    def __init__(self, source, datas, psfs, fit_settings, pcube):
        self.source = source
        self.datas = datas
        self.psfs = psfs
        self.fit_settings = fit_settings
        self.pcube = pcube
        self.energy_dependence = EnergyDependence.default(source.use_nn, use_mu=False)

    def fit(self, full_hessian):
        # Set up initial fit parameters
        x0 = np.zeros(self.fit_settings.length())
        bounds = []
        for _ in range(self.fit_settings.length()):
            bounds.append((None, None))
        
        index = self.fit_settings.param_to_index("q")
        if index is not None:
            q0, u0, _, _ = self.pcube
            x0[index] = q0
            bounds[index] = (-1, 1)
            index = self.fit_settings.param_to_index("u")
            x0[index] = u0
            bounds[index] = (-1, 1)
        index = self.fit_settings.param_to_index("x")
        if index is not None:
            x0[index] = np.mean(self.datas[0].evt_xs)
            bounds[index] = (-10, 10)
            index = self.fit_settings.param_to_index("y")
            x0[index] = np.mean(self.datas[0].evt_ys)
            bounds[index] = (-10, 10)
        index = self.fit_settings.param_to_index("sigma")
        if index is not None:
            x0[index] = 0
            bounds[index] = (0, 10)
        index = self.fit_settings.param_to_index("bg")
        if index is not None:
            x0[index] = 0.01
            bounds[index] = (0, 1)

        def chisq(params):
            return -2 * self.log_prob(params)
        
        # Perform the fit
        results = minimize(chisq, x0, bounds=bounds, method="nelder-mead", options=dict(maxiter=len(x0)*300))

        # Get the uncertainty
        print("Calculating hessian")
        if full_hessian:
            hessian = Hessian(chisq)(results.x)
        else:
            def hess_func(params):
                # This is chisq but where params is only a 2D matrix of the first two parameters
                full_params = [params[0], params[1]] + list(results.x[2:])
                return -2 * self.log_prob(full_params)
            hessian = Hessian(hess_func)((results.x[0], results.x[1]))

        return FitResult(results, hessian, self.fit_settings)


    def log_prob(self, params):
        """
        Get the log posterior of the Fitter (log_like + log_prior).
        This function works by preparing the data for the leakagelib prediction, and then calling
        raw_log_prob which actually performs the prediction.
        """

        # Blur the PSF
        index = self.fit_settings.param_to_index("sigma")
        if index is not None:
            blur = params[index]
            for psf in self.psfs:
                psf.blur # TODO blur the PSF
                raise NotImplementedError()

        return self.raw_log_prob(params)

    def raw_log_prob(self, params):
        """
        Get the log posterior of the Fitter (log_like + log_prior) where params contains q, u, x, y, bg.
        This function will not be called directly by the minimizer.
        """
        q = self.fit_settings.param_to_value(params, "q")
        u = self.fit_settings.param_to_value(params, "u")

        # Prior
        if q**2 + u**2 > 1:
            return -np.inf

        # Likelihood
        log_prob = 0
        for (i, data) in enumerate(self.datas):
            x = self.fit_settings.param_to_value(params, "x", i)
            y = self.fit_settings.param_to_value(params, "y", i)
            bg = self.fit_settings.param_to_value(params, "bg", i)
            psf = self.psfs[data.det-1]
            x_antirot, y_antirot = data.get_antirotation_matrix() @ (x, y)
            q_antirot, u_antirot = data.get_stokes_antirotation_matrix() @ (q, u)

            # Make leakage prediction
            self.source.polarize_net((q_antirot, u_antirot))
            pred_i, pred_q, pred_u = self.source.compute_leakage(psf, data.spectrum, energy_dependence=self.energy_dependence, normalize=True)
            pred_i /= np.mean(pred_i)

            # TODO eventually do the fit in some coarse energy bins

            PLOT = False
            if PLOT and data.det == 1:
                import matplotlib.pyplot as plt
                vmax = 0.15
                fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(ncols=2, nrows=2, sharex=True,sharey=True)
                
                bins = np.linspace(self.source.pixel_centers[0], self.source.pixel_centers[-1], 17)
                ax1.pcolormesh(-self.source.pixel_centers, -self.source.pixel_centers, np.transpose(pred_q), vmin=-vmax, vmax=vmax, cmap="RdBu")
                ax3.pcolormesh(-self.source.pixel_centers, -self.source.pixel_centers, np.transpose(pred_u), vmin=-vmax, vmax=vmax, cmap="RdBu")
                q_bin = np.histogram2d(data.evt_xs_antirot, data.evt_ys_antirot, bins, weights=data.evt_qs_antirot)[0]
                u_bin = np.histogram2d(data.evt_xs_antirot, data.evt_ys_antirot, bins, weights=data.evt_us_antirot)[0]
                c_bin = np.histogram2d(data.evt_xs_antirot, data.evt_ys_antirot, bins)[0].astype(float)
                ax2.pcolormesh(bins, bins, np.transpose(q_bin/c_bin), vmin=-vmax, vmax=vmax, cmap="RdBu")
                ax4.pcolormesh(bins, bins, np.transpose(u_bin/c_bin), vmin=-vmax, vmax=vmax, cmap="RdBu")

                # bins = np.linspace(self.source.pixel_centers[0], self.source.pixel_centers[-1], 50)
                # ax1.pcolormesh(-self.source.pixel_centers, -self.source.pixel_centers, np.transpose(pred_i), vmin=0)
                # # ax2.hist2d(data.evt_xs_antirot, data.evt_ys_antirot, 40, vmin=0)
                # c = np.histogram2d(data.evt_xs_antirot, data.evt_ys_antirot, bins)[0]
                # ax2.pcolormesh(bins, bins, np.transpose(c))
                for ax in fig.axes:
                    ax.set_aspect("equal")
                    # ax1.set_xlim(-50,50)
                    # ax2.set_ylim(-50,50)
                fig.savefig("test.png")
                import time
                time.sleep(0.5)

            # Make interpolation maps
            interp_i = RegularGridInterpolator((-self.source.pixel_centers, -self.source.pixel_centers), pred_i, bounds_error=False)
            interp_q = RegularGridInterpolator((-self.source.pixel_centers, -self.source.pixel_centers), pred_q, bounds_error=False)
            interp_u = RegularGridInterpolator((-self.source.pixel_centers, -self.source.pixel_centers), pred_u, bounds_error=False)

            src_i_prob = interp_i((data.evt_xs_antirot-x_antirot, data.evt_ys_antirot-y_antirot))
            src_q_expected = interp_q((data.evt_xs_antirot-x_antirot, data.evt_ys_antirot-y_antirot))
            src_u_expected = interp_u((data.evt_xs_antirot-x_antirot, data.evt_ys_antirot-y_antirot))
            src_polarization_prob = (1 + data.evt_mus / 2 * (data.evt_qs_antirot * src_q_expected + data.evt_us_antirot * src_u_expected)) / (2 * np.pi)
            total_source_prob = src_i_prob * src_polarization_prob

            bkg_polarization_prob = 1 / (2 * np.pi) # Unpolarized background

            bkg_prob = bg + (1-bg) * data.evt_bg_probs

            total_log_prob = np.log(total_source_prob * (1 - bkg_prob) + bkg_polarization_prob * bkg_prob)
            log_prob += np.nansum(total_log_prob)
        
        return log_prob