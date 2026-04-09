import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy, logging
from .fit_data import FitData
from .fit_result import FitResult, get_hess
from .pcube import get_pcube
from ..psf import PSF

logger = logging.getLogger("leakagelib")

class Fitter:
    """
    Prepare the fitter.

    Parameters
    ----------
    datas : list of IXPEData
        List of :class:`IXPEData` objects. Does not need to be binned; only the events will be used.
    fit_settings : FitSettings
        FitSettings object containing the sources to be fitted.
    psfs : list of PSF, optional
        List of three PSF objects (ordered DU 1, 2, 3) if you wish to alter the PSFs.
        Default is None, which uses the Dinsmore & Romani 2023 PSFs.

    Returns
    -------
    Fitter
        A Fitter object.

    Notes
    -----
    - It is advisable to first center your data and retain it to a large circular aperture.

    """
    def __init__(self, datas, fit_settings, psfs=None):


        # Check provided values
        if len(datas) == 0:
            raise Exception("Please provide a list of IXPEDatas")

        self.datas = datas
        self.fit_settings = copy.deepcopy(fit_settings)
        self.fit_settings._finalize()
        self.fit_data = FitData(self.fit_settings)
        self._load_psfs(psfs)
        self.spatial_weight = True
        
        self._get_pcube_estimate()
        self._get_flux_estimates()

    def __repr__(self):
        out = "FITTED PARAMETERS:\n"
        out += "Source\tParam\n"
        for i in range(self.fit_data.length()):
            param_type, name = self.fit_data.index_to_param(i)
            out += f"{name}:\t{param_type}\n"
        out += '\n'
        out += "FIXED PARAMETERS:\n"
        out += "Source\tParam\tValue\n"
        for source_name in self.fit_settings.names:
            if source_name in self.fit_data.fixed_qu:
                out += f"{source_name}:\tq\t{self.fit_data.fixed_qu[source_name][0]}\n"
                out += f"{source_name}:\tu\t{self.fit_data.fixed_qu[source_name][1]}\n"
            if source_name in self.fit_data.fixed_flux:
                out += f"{source_name}:\tf\t{self.fit_data.fixed_flux[source_name]}\n"
        return out

    def display_sources(self, fig_name=None, data_pixel_size=None):
        """
        Display the sources for debugging purposes

        Parameters
        ----------
        fig_name: str, optional
            Filename of the figure to be saved. Leave blank to return the figure object instead.

        data_pixel_size: float, optional
            Size of the spatial bins in arcseconds to use when displaying the data. Leave blank to use the native PSF pixel size.

        Returns
        -------
            Returns the figure object if fig_name was None, otherwise returns None.
        """

        psfs = [PSF.sky_cal(data.det, self.fit_settings.sources[0], data.rotation) for data in self.datas]

        import matplotlib.pyplot as plt
        n_images = len(self.fit_settings.sources)+1
        fig, axs = plt.subplots(nrows=n_images, ncols=2, figsize=(6,3*n_images), sharex=True, sharey=True)
        for ax_row, source, name in zip(axs, self.fit_settings.sources, self.fit_settings.names):
            image = np.flip(np.log(1+source.source), axis=1)
            convolved_image = np.zeros_like(image)
            for psf in psfs:
                convolved_image += source.convolve_psf(psf)
            convolved_image = np.flip(np.log(1+convolved_image), axis=1)
            ax_row[0].pcolormesh(source.pixel_centers, source.pixel_centers, image, vmin=0, cmap="viridis")
            ax_row[1].pcolormesh(source.pixel_centers, source.pixel_centers, convolved_image, vmin=0, cmap="viridis")
            ax_row[0].set_xlim(source.pixel_centers[-1], source.pixel_centers[0])
            ax_row[0].set_ylim(source.pixel_centers[0], source.pixel_centers[-1])
            ax_row[0].set_title(name)
            ax_row[1].set_title(f"{name} w/ PSF")

        # Show ROI
        roi = np.zeros_like(image)
        for image in self.fit_settings._finalize_roi(None).values():
            roi += image
        axs[-1,0].pcolormesh(source.pixel_centers, source.pixel_centers, np.flip(roi, axis=1), vmin=0, cmap="viridis")
        axs[-1,0].set_aspect("equal")
        axs[-1,0].set_title("ROI")

        axs[-1,0].set_xlabel("x [arcsec]")
        axs[-1,0].set_ylabel("y [arcsec]")

        delta = source.pixel_size if data_pixel_size is None else data_pixel_size
        x_line = np.arange(np.min(-self.datas[0].evt_xs), np.max(-self.datas[0].evt_xs), delta)
        y_line = np.arange(np.min(self.datas[0].evt_ys), np.max(self.datas[0].evt_ys), delta)
        x_centers = (x_line[1:] + x_line[:-1]) / 2
        y_centers = (y_line[1:] + y_line[:-1]) / 2
        image = np.zeros((len(x_line)-1, len(y_line)-1))
        for data in self.datas:
            image += np.histogram2d(-data.evt_xs, data.evt_ys, (x_line, y_line))[0]
        axs[-1,1].pcolormesh(x_centers, y_centers, np.transpose(image), cmap="viridis")
        axs[-1,1].set_title("Data")

        for ax in fig.axes:
            ax.set_aspect("equal")

        if fig_name is None:
            return fig
        else:
            fig.savefig(fig_name)

    def _get_numerical_uncertainty(self, params):
        # Figure out which steps to use when computing the Hessian
        steps = []
        for i in range(len(params)):
            ptype, name = self.fit_data.index_to_param(i)
            if ptype == "q" or ptype == "u":
                steps.append(1e-3)
            elif ptype == "f":
                steps.append(1e-4)
            elif ptype == "sigma":
                steps.append(1e-2)
            elif ptype in self.fit_settings.extra_param_names:
                index = self.fit_settings.extra_param_names.index(ptype)
                steps.append(self.fit_settings.extra_param_data[index][2])
            else:
                raise Exception(f"Coord type {ptype} not recognized")
            
        hessian = get_hess(self.log_prob, params, steps)

        for i in range(len(params)):
            for j in range(len(params)):
                i_ptype, name = self.fit_data.index_to_param(i)
                j_ptype, name = self.fit_data.index_to_param(j)
                i_is_stokes = (i_ptype == "q" or i_ptype == "u")
                j_is_stokes = (j_ptype == "q" or j_ptype == "u")
                if i_is_stokes ^ j_is_stokes:
                    hessian[i,j] = 0

        try:
            cov = np.linalg.pinv(-hessian)
        except:
            logger.warning("Hessian inversion did not converge")
            cov = np.zeros_like(hessian)

        return cov
    
    def plot(self, params=None, n_bins=101):
        """
        Plot the image predicted by the fitter vs the data

        Parameters
        ----------
        filename : string
            Name of the file to save the image to
        params : array-like, optional
            Parameter array to plot. Default: the starting values
        n_bins : int, optional
            Number of spatial bins to use
        """
        if params is None:
            params = self._get_start_params()[0]

        data = self.datas[0]
        psf = self.psfs[0]
        max_r = np.max(np.sqrt(data.evt_xs**2 + data.evt_ys**2))

        evt_probs = np.zeros_like(data.evt_xs)
        for source_index, source in enumerate(self.fit_settings.sources):
            source_name = self.fit_settings.names[source_index]
            if data.det not in self.fit_settings.detectors[source_index]:
                continue
            if self.fit_settings.obs_ids[source_index] is not None and (data.obs_id not in self.fit_settings.obs_ids[source_index]):
                continue

            # Get the parameters and use the prior
            f = self.fit_data.param_to_value(params, "f", source_name)
            source.polarize_net((0, 0))
            evt_probs += source.get_event_p_r_given_phi(psf, data, overwrite_mus=data.evt_mus) * f
            
        mask = data.evt_bg_chars < 0.2

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
        line = np.linspace(-max_r, max_r, n_bins)
        
        counts = np.histogram2d(data.evt_xs[mask], data.evt_ys[mask], (line, line))[0].astype(float)
        pred = np.histogram2d(data.evt_xs[mask], data.evt_ys[mask], (line, line), weights=evt_probs[mask])[0].astype(float)/counts
        counts /= np.nanmax(counts) * 0.005
        image = np.log(1+counts)
        ax1.pcolormesh(line, line, np.transpose(image), vmin=0)
        ax1.set_title(f"Data (DU {data.det})")
        
        pred[~np.isfinite(pred)] = 0
        pred /= np.nanmax(pred) * 0.005
        image = np.log(1+pred)
        ax2.pcolormesh(line, line, np.transpose(image), vmin=0)
        ax2.set_title("Prediction")

        for ax in fig.axes:
            ax.set_aspect("equal")
            ax.set_xlim(line[0], line[-1])
            ax.set_ylim(line[0], line[-1])

        return fig

    def fit(self, method="nelder-mead"):
        """
        Fit analytically

        Parameters
        ----------
        method: str, optional
            `scipy.optimize.minimize` method fit method. Default is Nelder-Mead.
        """
        x0, bounds = self._get_start_params()

        def minus_log_like(params):
            return -self.log_prob(params)
        
        # Perform the fit
        results = minimize(minus_log_like, x0, bounds=bounds, method=method, options=dict(maxiter=len(x0)*300))
        
        # Get the uncertainty
        cov = self._get_numerical_uncertainty(results.x)

        return FitResult(results.x, -results.fun, results.message, cov, self.fit_data, self.fit_settings)

    def fit_mcmc(self, n_walkers=16, n_iter=5_000, burnin=1_000, save_corner="corner.png",
                 save_samples=None, progress=True):
        """
        Fit using an MCMC (via the `emcee` package).

        Parameters
        ----------
        n_walkers : int, optional
            Number of walkers to use for the MCMC. Default is 16.
        n_iter : int, optional
            Number of iterations to run the MCMC for. Default is 5000.
        burnin : int, optional
            Number of points to discard. Default is 1000.
        save_corner : str or None, optional
            File path to save the corner plot. Default is "corner.png". Set to None to skip saving.
            Requires the `corner` package.
        save_samples : str or None, optional
            File path to save the MCMC samples in h5 format. Recommended for advanced analysis.
            Default is None. WARNING: emcee appends to the file if it exists; delete first if you wish to redo a run.
        progress : bool, optional
            Whether to show the MCMC progress bar. Default is True.

        Returns
        -------
        FitData
            Object containing the results of the fit. The covariance matrix in FitData is set to the covariance of the parameters.

        Notes
        -----
        - For detailed manipulation or custom corner plots, use `save_samples`. You can then discard the FitData object and operate on the saved samples.
        """

        import emcee
        x0, bounds = self._get_start_params()
        sigmas = self._get_emcee_sigmas()
        initial_state = [x0 + np.random.randn(len(x0)) * sigmas for _ in range(n_walkers)]

        def log_prob_fn(params):
            for i in range(len(params)):
                if bounds[i][0] is not None and params[i] < bounds[i][0]: return -np.inf
                if bounds[i][1] is not None and params[i] > bounds[i][1]: return -np.inf
            return self.log_prob(params)

        if save_samples is not None:
            backend = emcee.backends.HDFBackend(save_samples)
        else:
            backend = None
        sampler = emcee.EnsembleSampler(n_walkers, len(x0), log_prob_fn, backend=backend)
        sampler.run_mcmc(initial_state, n_iter, progress=progress)

        samples = sampler.get_chain()[:burnin,:,:].reshape(-1, len(x0))
        log_likes = sampler.get_log_prob()[:burnin,:].reshape(-1)

        if save_corner is not None:
            import corner
            labels = []
            for i in range(len(x0)):
                param, name = self.fit_data.index_to_param(i)
                labels.append(f"${param}_\\mathrm{{{name}}}$")
            fig = corner.corner(samples, labels=labels, show_titles=True)
            fig.savefig(save_corner)

        return self._load_samples(samples, log_likes)
    
    def _load_samples(self, samples, log_likes):
        """
        return a FitResult for the provided MCMC samples.
        """

        best_index = np.argmin(log_likes)
        best_param = samples[best_index]
        best_like = log_likes[best_index]
        cov = np.cov(np.transpose(samples))
        return FitResult(best_param, best_like, "MCMC fit", cov, self.fit_data, self.fit_settings)

    def _load_psfs(self, psfs):
        # Load PSFs
        if psfs is not None:
            if len(psfs) != len(self.datas):
                raise Exception("You must pass the same number of PSFs as detectors")
            self.psfs = psfs
        else:
            self.psfs = []
            for data in self.datas:
                psf = PSF.sky_cal(data.det, self.fit_settings.sources[0], data.rotation)
                if self.fit_settings.fixed_blur is not None and self.fit_settings.fixed_blur != 0:
                    psf.blur(self.fit_settings.fixed_blur)
                self.psfs.append(psf)

    def _get_pcube_estimate(self):
        # Get PCUBE estimate
        outer_source = 40 # arcsec
        inner_bg = 80 # arcsec
        outer_bg = 120 # arcsec

        src_datas = copy.deepcopy(self.datas)
        bg_datas = copy.deepcopy(self.datas)
        total_counts = 0
        bg_counts = 0
        data_mask = np.ones(len(src_datas), bool)
        for i in range(len(self.datas)):
            dist2s = self.datas[i].evt_xs**2 + self.datas[i].evt_ys**2
            src_mask = dist2s < outer_source**2
            bg_mask = (dist2s < outer_bg**2) & (dist2s > inner_bg**2)
            if np.sum(src_mask) == 0 or np.sum(bg_mask) == 0:
                data_mask[i] = False
                continue
            src_datas[i].retain(src_mask)
            bg_datas[i].retain(bg_mask)
            total_counts += np.sum(src_mask)
            bg_counts += np.sum(bg_mask)
        src_datas = np.array(src_datas)[data_mask]
        bg_datas = np.array(bg_datas)[data_mask]

        area_ratio = outer_source**2 / (outer_bg**2 - inner_bg**2)

        if bg_counts == 0 or total_counts == 0:
            self.pcube = [0, 0]
            self.bg_frac = 0.5
        else:
            self.pcube = get_pcube(src_datas, (bg_datas, area_ratio))[:2]
            self.bg_frac = bg_counts / total_counts * area_ratio

        # If the PCUBE estimate is unphysical, move the PD to a physical value
        result_pd = np.sqrt(self.pcube[0]**2 + self.pcube[1]**2)
        if result_pd > 1:
            self.pcube /= result_pd * 1.2

    def _get_flux_estimates(self):
        source_probs = np.zeros(len(self.fit_settings.sources))
        for (data, psf) in zip(self.datas, self.psfs):
            evt_probs = np.zeros((len(self.fit_settings.sources), len(data.evt_xs)))
            for i, source in enumerate(self.fit_settings.sources):
                evt_probs[i] = source.get_event_p_r_given_phi(psf, data)
            evt_probs /= np.sum(evt_probs, axis=0)
            source_probs += np.mean(evt_probs, axis=1)

        norm = 1
        for i in range(len(self.fit_settings.sources)):
            if not self.fit_settings.fixed_flux[i]: continue
            norm = self.fit_settings.fixed_flux[i] / source_probs[i]
            break

        self.flux_estimates = {}
        for i, prob in enumerate(source_probs):
            self.flux_estimates[self.fit_settings.names[i]] = prob * norm

    def _get_start_params(self):
        """
        Returns the start parameters of the fit, and the bounds. This function guesses that the
        first source you provided has the polarization of the PCUBE analysis, and that all the
        others are unpolarized.
        """
        x0 = []
        bounds = []
        first_source_name = self.fit_settings.names[0]

        for i in range(self.fit_data.length()):
            param, source_name = self.fit_data.index_to_param(i)
            if source_name is not None:
                source_index = self.fit_settings.names.index(source_name)
            else:
                source_name = None

            if param == "q":
                if source_name == first_source_name:
                    x0.append(self.pcube[0])
                else:
                    x0.append(0)
                if self.fit_settings.guess_qu[source_index][0] is not None:
                    x0[-1] = self.fit_settings.guess_qu[source_index][0]
                bounds.append((-1, 1))

            elif param == "u":
                if source_name == first_source_name:
                    x0.append(self.pcube[1])
                else:
                    x0.append(0)
                if self.fit_settings.guess_qu[source_index][1] is not None:
                    x0[-1] = self.fit_settings.guess_qu[source_index][1]
                bounds.append((-1, 1))

            elif param == "f":
                x0.append(self.flux_estimates[source_name])
                bounds.append((0, 100))
                if self.fit_settings.guess_f[source_index] is not None:
                    x0[-1] = self.fit_settings.guess_f[source_index]

            elif param == "sigma":
                x0.append(10)
                bounds.append((0, 30))

            elif param in self.fit_settings.extra_param_names:
                index = self.fit_settings.extra_param_names.index(param)
                x0.append(self.fit_settings.extra_param_data[index][0])
                bounds.append(self.fit_settings.extra_param_data[index][1])

            else:
                raise Exception(f"Parameter {param} not handled")
            
        return x0, bounds
    
    def _get_emcee_sigmas(self):
        # Returns a list of standard deviations to use to offset walkers
        sigmas = []
        for i in range(self.fit_data.length()):
            param, source_name = self.fit_data.index_to_param(i)
            if param == "q":
                sigmas.append(0.3)
            if param == "u":
                sigmas.append(0.3)
            if param == "f":
                sigmas.append(0.3)

        return sigmas


    def log_prob(self, params, prior=True, return_array=False):
        """
        Get the log posterior of the Fitter (log_like + log_prior).

        Arguments
        ---------
        params : array_like
            List of parameters. If you are calling this function manually, you should order the parameters in the same order as the :attr:`FitResult.parameter_names` attribute of the fit result.

        prior : bool
            set to True to include the priors, which are all finite uniform.

        return_array : bool
            set to True to return the log posteriors of each event as an array, instead of the summed log posterior.
        """

        # Blur the PSF
        index = self.fit_data.param_to_index("sigma")
        if index is not None:
            blur = params[index]
            for psf in self.psfs:
                psf.blur(blur)

        # Get the log prob
        return self._raw_log_prob(params, prior, return_array)

    def _raw_log_prob(self, params, prior, return_array=False):
        """
        Get the log posterior of the Fitter (log_like + log_prior), assuming the PSFs have already 
        been blurred
        """
        if return_array:
            log_prob = []
        else:
            log_prob = 0
        
        for data_index, (data, psf) in enumerate(zip(self.datas, self.psfs)):
            evt_probs = np.zeros_like(data.evt_xs)
            flux_norms = 0
            for source_index, source in enumerate(self.fit_settings.sources):
                source_name = self.fit_settings.names[source_index]
                temporal_weights = self.fit_settings.temporal_weights[source_index]
                spectral_weights = self.fit_settings.spectral_weights[source_index]
                spectral_mus = self.fit_settings.spectral_mus[source_index]
                sweeps = self.fit_settings.sweeps[source_index]
                model_fn = self.fit_settings.model_fns[source_index]
                if data.det not in self.fit_settings.detectors[source_index]:
                    continue
                if self.fit_settings.obs_ids[source_index] is not None and (data.obs_id not in self.fit_settings.obs_ids[source_index]):
                    continue

                # Get the parameters and use the prior
                q = self.fit_data.param_to_value(params, "q", source_name)
                u = self.fit_data.param_to_value(params, "u", source_name)
                f = self.fit_data.param_to_value(params, "f", source_name)
                if spectral_mus is None:
                    mus = data.evt_mus
                else:
                    mus = spectral_mus[data_index]
                if prior:
                    if q**2 + u**2 > 1:
                        return -1e10 * (1 + q**2 + u**2 - 1)
                    if f < 0:
                        return -1e10 * (1 - f)
                    
                if sweeps is not None:
                    # Use the time-dependent PA sweep models
                    new_q = q * sweeps[data_index][0] - u * sweeps[data_index][1]
                    new_u = q * sweeps[data_index][1] + u * sweeps[data_index][0]
                    q = new_q
                    u = new_u
                if model_fn is not None:
                    q, u = model_fn(data.evt_times, self.fit_data, params)
                source.polarize_net((np.mean(q), np.mean(u)))

                probs = np.ones_like(data.evt_xs)
                if self.fit_settings.particles[source_index]:
                    # Polarization weights (no need for the 1/2pi)
                    probs += 0.5 * (data.evt_qs*q + data.evt_us*u) # No modulation factor included
                    clipped_chars = np.clip(data.evt_bg_chars, 1e-5, 1-1e-5)
                    probs *= clipped_chars / (1 - clipped_chars)
                else:
                    # Polarization weights (no need for the 1/2pi)
                    probs += 0.5 * mus * (data.evt_qs*q + data.evt_us*u)

                # Flux weights
                probs *= f
                flux_norms += f

                # Spatial weights
                if self.spatial_weight:
                    probs *= source.get_event_p_r_given_phi(psf, data, overwrite_mus=mus)

                # Phase weights
                if temporal_weights is not None:
                    probs *= temporal_weights[data_index]

                # Spectral weights
                if spectral_weights is not None:
                    probs *= spectral_weights[data_index]
                    
                evt_probs += probs

            if return_array:
                log_prob = np.concatenate([log_prob, np.log(evt_probs / flux_norms)])
            else:
                log_prob += np.sum(np.log(evt_probs / flux_norms))

        if not return_array and not np.isfinite(log_prob):
            problem = None
            if self.fit_data.param_to_index("f", "pbkg") is not None:
                if params[self.fit_data.param_to_index("f", "pbkg")] == 0:
                    if np.any([np.any(data.evt_bg_chars==1) for data in self.datas]):
                        problem = "You have events with bg_prob=1 in your data set, and the particle background flux is zero. This probably caused the problem. Try removing these events"
            if problem is None:
                for name in self.fit_settings.names:
                    if self.fit_data.param_to_index("f", name) is not None and params[self.fit_data.param_to_index("f", name)] == 0:
                        problem = f"Your flux for source {name} is equal to zero."
            if problem is None:
                problem = "Your background source might be at fault - is the ROI you provided correct?"
                    
            param_str = "\n".join([f"{self.fit_data.index_to_param(i)}: {params[i]}" for i in range(len(params))])
            raise Exception(f"The log prob was not finite ({log_prob}) with parameters\n{param_str}\nThis happens when all your sources predict zero flux in a region of parameter space where at least one event was detected.\n\n{problem}")
        
        return log_prob