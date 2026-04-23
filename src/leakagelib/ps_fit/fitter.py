import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
from .fit_data import FitData
from .fit_result import FitResult, get_hess
from .fit_properties import FitProperties
import emcee

logger = logging.getLogger("leakagelib")

class Fitter:
    """
    Prepare the fitter.

    Parameters
    ----------
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
    def __init__(self, fit_settings, psfs=None):
        # Check provided values
        self.fit_props = FitProperties(fit_settings, psfs)
        self.fit_data = FitData(fit_settings)

    def __repr__(self):
        out = "FITTED PARAMETERS:\n"
        out += "Source\tParam\n"
        for i in range(self.fit_data.length()):
            param_type, name = self.fit_data.index_to_param(i)
            out += f"{name}:\t{param_type}\n"
        out += '\n'
        out += "FIXED PARAMETERS:\n"
        out += "Source\tParam\tValue\n"
        for source_name in self.fit_props.guess_quf:
            if source_name in self.fit_data.fixed_qu:
                out += f"{source_name}:\tq\t{self.fit_data.fixed_qu[source_name][0]}\n"
                out += f"{source_name}:\tu\t{self.fit_data.fixed_qu[source_name][1]}\n"
            if source_name in self.fit_data.fixed_flux:
                out += f"{source_name}:\tf\t{self.fit_data.fixed_flux[source_name]}\n"
        return out

    def display_sources(self, data_pixel_size=None):
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

        n_images = len(self.fit_props.guess_quf)+1
        data_key = self.fit_props.combos[0].data_key

        fig, axs = plt.subplots(nrows=n_images, ncols=2, figsize=(6,3*n_images), sharex=True, sharey=True)
        i = 0
        for combo in self.fit_props.combos:
            if combo.data_key != data_key: continue
            ax_row = axs[i]
            image = np.flip(np.log(1+combo.source.source), axis=1)
            convolved_image = combo.source.convolve_psf(combo.psf)
            convolved_image = np.flip(np.log(1+convolved_image), axis=1)
            pixel_centers = combo.source.pixel_centers
            ax_row[0].pcolormesh(pixel_centers, pixel_centers, image, vmin=0, cmap="viridis")
            ax_row[1].pcolormesh(pixel_centers, pixel_centers, convolved_image, vmin=0, cmap="viridis")
            ax_row[0].set_xlim(pixel_centers[-1], pixel_centers[0])
            ax_row[0].set_ylim(pixel_centers[0], pixel_centers[-1])
            ax_row[0].set_title(combo.name)
            ax_row[1].set_title(f"{combo.name} w/ PSF")
            i += 1

        # Show ROI
        axs[-1,0].pcolormesh(pixel_centers, pixel_centers, np.flip(combo.roi, axis=1), vmin=0, cmap="viridis")
        axs[-1,0].set_aspect("equal")
        axs[-1,0].set_title("ROI")

        axs[-1,0].set_xlabel("x [arcsec]")
        axs[-1,0].set_ylabel("y [arcsec]")

        delta = (pixel_centers[1] - pixel_centers[0]) if data_pixel_size is None else data_pixel_size
        x_line = np.arange(np.min(-combo.data.evt_xs), np.max(-combo.data.evt_xs), delta)
        y_line = np.arange(np.min(combo.data.evt_ys), np.max(combo.data.evt_ys), delta)
        x_centers = (x_line[1:] + x_line[:-1]) / 2
        y_centers = (y_line[1:] + y_line[:-1]) / 2
        image = np.histogram2d(-combo.data.evt_xs, combo.data.evt_ys, (x_line, y_line))[0]
        axs[-1,1].pcolormesh(x_centers, y_centers, np.transpose(image), cmap="viridis")
        axs[-1,1].set_title("Data")

        for ax in fig.axes:
            ax.set_aspect("equal")

        return fig
    
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
            elif ptype in self.fit_props.extra_params:
                steps.append(self.fit_props.extra_params[ptype][2])
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
    
    def plot(self, params=None, n_bins=101, det=1):
        """
        Plot the image predicted by the fitter vs the data

        Parameters
        ----------
        params : array-like, optional
            Parameter array to plot. Default: the starting values
        n_bins : int, optional
            Number of spatial bins to use
        det

        Returns
        -------
        The matplotlib Figure
        """
        if params is None:
            params = self._get_start_params()[0]

        evt_probs = None
        for combo in self.fit_props.combos:
            if combo.data.det != det: continue
            if evt_probs is None:
                evt_probs = np.zeros_like(combo.data.evt_xs)

            f = self.fit_data.param_to_value(params, "f", combo.name)
            combo.polarize_net((0, 0))
            evt_probs += combo._get_event_p_r_given_phi() * f

        max_r = np.max(np.sqrt(combo.data.evt_xs**2 + combo.data.evt_ys**2))
        mask = combo.data.evt_bg_chars < 0.2

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
        line = np.linspace(-max_r, max_r, n_bins)
        
        counts = np.histogram2d(combo.data.evt_xs[mask], combo.data.evt_ys[mask], (line, line))[0].astype(float)
        pred = np.histogram2d(combo.data.evt_xs[mask], combo.data.evt_ys[mask], (line, line), weights=evt_probs[mask])[0].astype(float)/counts
        counts /= np.nanmax(counts) * 0.005
        image = np.log(1+counts)
        ax1.pcolormesh(line, line, np.flip(np.transpose(image), axis=1), vmin=0)
        ax1.set_title(f"Data (DU {det})")
        
        pred[~np.isfinite(pred)] = 0
        pred /= np.nanmax(pred) * 0.005
        image = np.log(1+pred)
        ax2.pcolormesh(line, line, np.flip(np.transpose(image), axis=1), vmin=0)
        ax2.set_title("Prediction")

        for ax in fig.axes:
            ax.set_aspect("equal")
            ax.set_xlim(line[-1], line[0])
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

        return FitResult(results.x, -results.fun, results.message, cov, self.fit_data, self.fit_props)

    def fit_mcmc(self, n_walkers=16, n_iter=5_000, save_samples=None, progress=True):
        """
        Fit using an MCMC (via the `emcee` package). You must install emcee and corner to use this package.

        Parameters
        ----------
        n_walkers : int, optional
            Number of walkers to use for the MCMC. Default is 16.
        n_iter : int, optional
            Number of iterations to run the MCMC for. Default is 5000.
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

        x0, bounds = self._get_start_params()
        sigmas = [(b[1] - b[0]) / 6 for b in bounds]
        initial_state = np.array([x0 + np.random.randn(len(x0))*sigmas for _ in range(n_walkers)])
        while True:
            bad_mask = np.zeros(len(initial_state), bool)
            for i in range(len(bounds)):
                bad_mask |= initial_state[:,i] < bounds[i][0]
                bad_mask |= initial_state[:,i] > bounds[i][1]
            if np.sum(bad_mask) == 0:
                break
            initial_state[bad_mask,:] = np.array([x0 + np.random.randn(len(x0))*sigmas for _ in range(np.sum(bad_mask))])
            
        if save_samples is not None:
            backend = emcee.backends.HDFBackend(save_samples)
        else:
            backend = None
        sampler = emcee.EnsembleSampler(n_walkers, len(x0), self.log_prob, backend=backend)
        sampler.run_mcmc(initial_state, n_iter, progress=progress)

        samples = sampler.get_chain().reshape(-1, len(x0))
        log_likes = sampler.get_log_prob().reshape(-1)

        return FitResult.from_samples(samples, log_likes, self.fit_data, self.fit_props)

    def _get_start_params(self):
        """
        Returns the start parameters of the fit, and the bounds. This function guesses that the
        first source you provided has the polarization of the PCUBE analysis, and that all the
        others are unpolarized.
        """
        x0 = []
        bounds = []

        for i in range(self.fit_data.length()):
            param, source_name = self.fit_data.index_to_param(i)
            if param == "q":
                x0.append(self.fit_props.guess_quf[source_name][0])
                bounds.append((-1, 1))
            elif param == "u":
                x0.append(self.fit_props.guess_quf[source_name][1])
                bounds.append((-1, 1))
            elif param == "f":
                x0.append(self.fit_props.guess_quf[source_name][2])
                bounds.append((0, 100))
            elif param == "sigma":
                x0.append(10)
                bounds.append((0, 30))
            elif param in self.fit_props.extra_params:
                x0.append(self.fit_props.extra_params[param][0])
                bounds.append(self.fit_props.extra_params[param][1])

            else:
                raise Exception(f"Parameter {param} not handled")
            
        return x0, bounds
    
    def log_prob(self, params, prior=True, return_array=False):
        """
        Get the log posterior of the Fitter (log_like + log_prior).

        Arguments
        ---------
        params : array_like
            List of parameters. If you are calling this function manually, you should order the parameters in the same order as the :attr:`leakagelib.FitResult.parameter_names` attribute of the fit result.

        prior : bool
            set to True to include the priors, which are all finite uniform.

        return_array : bool
            set to True to return the log posteriors of each event as an array, instead of the summed log posterior.
        """

        # Blur the PSF
        
        index = self.fit_data.param_to_index("sigma")
        if index is not None:
            blur = params[index]
            for combo in self.fit_props.combos:
                combo.blur_psf(blur)

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
        
        evt_probs = {}
        flux_norms = {}
        for combo in self.fit_props.combos:
            q = self.fit_data.param_to_value(params, "q", combo.name)
            u = self.fit_data.param_to_value(params, "u", combo.name)
            f = self.fit_data.param_to_value(params, "f", combo.name)
            if prior:
                if q**2 + u**2 > 1:
                    return -1e8 * (q**2 + u**2)
                if f < 0:
                    return -1e8 * (101 - f)
                
            # Set polarization
            if combo.sweeps is not None:
                # Use the time-dependent PA sweep models
                new_q = q * combo.sweeps[0] - u * combo.sweeps[1]
                new_u = q * combo.sweeps[1] + u * combo.sweeps[0]
                q = new_q
                u = new_u
            if combo.model_fn is not None:
                q, u = combo.model_fn(combo.data.evt_times, self.fit_data, params)
            combo.polarize_net((np.mean(q), np.mean(u)))

            probs = combo.get_log_prob(q, u)
            
            if combo.data_key not in evt_probs:
                evt_probs[combo.data_key] = np.zeros(len(combo.data))
                flux_norms[combo.data_key] = 0
            evt_probs[combo.data_key] += probs * f
            flux_norms[combo.data_key] += f

        for key in evt_probs:
            evt_probs[key] /= flux_norms[key]




        fig, axs = plt.subplots(ncols=2)
        counts = np.histogram2d(combo.data.evt_xs, combo.data.evt_ys, (50,50))[0].astype(float)
        probs = np.histogram2d(combo.data.evt_xs, combo.data.evt_ys, (50,50), weights=evt_probs[combo.data_key])[0] / counts
        axs[0].imshow(counts)
        axs[1].imshow(probs)
        fig.savefig("out.png")




        if return_array:
            log_prob = np.log(np.concatenate(list(evt_probs.values())))
        else:
            log_prob = 0
            for v in evt_probs.values():
                log_prob += np.sum(np.log(v))

        if not return_array and not np.isfinite(log_prob):
            problem = ""
            if self.fit_data.param_to_index("f", "pbkg") is not None:
                if params[self.fit_data.param_to_index("f", "pbkg")] == 0:
                    if np.any([np.any(data.evt_bg_chars==1) for data in self.datas]):
                        problem = "You have events with bg_prob=1 in your data set, and the particle background flux is zero. This probably caused the problem. Try removing these events"
            if problem == "":
                for name in self.fit_props.guess_quf:
                    if self.fit_data.param_to_index("f", name) is not None and params[self.fit_data.param_to_index("f", name)] == 0:
                        problem = f"Your flux for source {name} is equal to zero."
                    
            param_str = "\n".join([f"{self.fit_data.index_to_param(i)}: {params[i]}" for i in range(len(params))])
            raise Exception(f"The log prob was not finite ({log_prob}) with parameters\n{param_str}\nThis happens when all your sources predict zero flux in a region of parameter space where at least one event was detected.\n\n{problem}")
        
        return log_prob