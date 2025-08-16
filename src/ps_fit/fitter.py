import numpy as np
from scipy.optimize import minimize
import copy, warnings
from .fit_data import FitData
from .fit_result import FitResult, get_hess
from .pcube import get_pcube
from ..psf import PSF

class Fitter:
    """A class to fit polarized, extended or point sources to IXPE data"""
    def __init__(self, datas, fit_settings, psfs=None):
        """Prepare the fitter

        # Arguments:
        * datas: a list of IXPEData objects. Does not need to be binned; only the events will be used.
        * fit_settings: a FitSettings object containing the sources to be fitted
        * psfs (optional): set to your own list of PSF objects if you wish to alter the PSFs in some
            way. Set to None (default) to use the Dinsmore & Romani 2023 PSFs. Must be a list of three
            elements ordered as DU 1, 2, and 3.

        # Returns:
        * A Fitter object

        Notes:
        * It is advisable to first center your data and retain it to a large circular aperture.
        """

        # Check provided values
        if len(datas) == 0:
            raise Exception("Please provide a list of IXPEDatas")

        self.datas = datas
        self.fit_settings = copy.deepcopy(fit_settings)
        self.fit_settings.finalize()
        self.fit_data = FitData(self.fit_settings)
        self.load_psfs(psfs)
        self.spatial_weight = True
        
        self.get_pcube_estimate()
        self.get_flux_estimates()

    def __repr__(self):
        out = "PARAMETERS:\n"
        out = "Source:\t Parameter\n"
        for i in range(self.fit_data.length()):
            param, name = self.fit_data.index_to_param(i)
            out += f"{name}:\t{param}\n"
        return out

    def display_sources(self, fig_name):
        import matplotlib.pyplot as plt
        plt.style.use("root")
        fig, axs = plt.subplots(nrows=len(self.fit_settings.sources), sharex=True, sharey=True)
        for ax, source, name in zip(axs, self.fit_settings.sources, self.fit_settings.names):
            image = np.flip(np.log(1+source.source), axis=1)
            ax.pcolormesh(source.pixel_centers, source.pixel_centers, image, vmin=0)
            ax.set_aspect("equal")
            ax.set_title(name)
        axs[-1].set_xlabel("x [arcsec]")
        axs[-1].set_ylabel("y [arcsec]")
        axs[-1].set_xlim(axs[-1].get_xlim()[1], axs[-1].get_xlim()[0])
        fig.savefig(fig_name)

    def get_numerical_uncertainty(self, params):
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
            print("Hessian inversion did not converge")
            cov = np.zeros_like(hessian)

        return cov

    def fit(self, method="nelder-mead"):
        """
        Fit analytically
        """
        x0, bounds = self.get_start_params()

        def minus_log_like(params):
            return -self.log_prob(params)
        
        # Perform the fit
        results = minimize(minus_log_like, x0, bounds=bounds, method=method, options=dict(maxiter=len(x0)*300))
        
        # Get the uncertainty
        cov = self.get_numerical_uncertainty(results.x)

        return FitResult(results.x, -results.fun, results.message, cov, self.fit_data, self.fit_settings)

    def fit_mcmc(self, n_walkers=16, n_iter=5_000, burnin=1_000, save_corner="corner.png",
                 save_samples=None, progress=True):
        """
        Fit using an MCMC (via the emcee package)
        # Arguments
            * n_walkers (Default: 16): Number of walkers to use for the MCMC
            * n_iter (Default: 5_000): Number of iterations to run the MCMC for
            * burnin (Default: 1_000): Number of points to discard
            * save_corner (Default: corner.png): File to which to save the corner plot. Set to None if you do not wish to save the corner plot. Requires the `corner` package
            * save_samples (Default: None): Set to a string to save the samples to an h5 file. This is recommended if you'd like to do any more advanced manipulation of the walkers than this very simple function does. WARNING: By default, emcee appends to the save_samples file if it already exists, instead of overwriting it. If you wish to redo a run, make sure to delete the save_samples file first.
            * progress (Default: True): Set to True to show the MCMC progress bar
        # Returns
            A FitData object containing the results of the fit. The covariance matrix in the FitData object is set to be the covariance of the parameters.
            
        If you'd like to do any detailed manipulation or make your own corner plots, it's recommended to set the "save_samples" argument. This function will then save the samples to a file. You can discard the FitResult this function returns and operate based on that sample file.
        """
        import emcee
        x0, bounds = self.get_start_params()
        sigmas = self.get_emcee_sigmas()
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

        return self.load_samples(samples, log_likes)
    
    def load_samples(self, samples, log_likes):
        """
        return a FitResult for the provided MCMC samples.
        """

        best_index = np.argmin(log_likes)
        best_param = samples[best_index]
        best_like = log_likes[best_index]
        cov = np.cov(np.transpose(samples))
        return FitResult(best_param, best_like, "MCMC fit", cov, self.fit_data, self.fit_settings)

    def load_psfs(self, psfs):
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

    def get_pcube_estimate(self):
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

    def get_flux_estimates(self):
        source_probs = np.zeros(len(self.fit_settings.sources))
        for (data, psf) in zip(self.datas, self.psfs):
            evt_probs = np.zeros((len(self.fit_settings.sources), len(data.evt_xs)))
            for i, source in enumerate(self.fit_settings.sources):
                evt_probs[i] = source.get_event_p_r_given_phi(psf, data)
            evt_probs /= np.sum(evt_probs, axis=0)
            source_probs += np.mean(evt_probs, axis=1)

        for i in range(len(self.fit_settings.sources)):
            if self.fit_settings.particles[i]: continue
            if not self.fit_settings.fixed_flux[i]: continue
            norm = self.fit_settings.fixed_flux[i] / source_probs[i]
            break

        self.flux_estimates = {}
        for i, prob in enumerate(source_probs):
            self.flux_estimates[self.fit_settings.names[i]] = prob * norm

    def get_start_params(self):
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

            else:
                raise Exception(f"Parameter {param} not handled")
            
        return x0, bounds
    
    def get_emcee_sigmas(self):
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
        This function works by preparing the data for the leakagelib prediction, and then calling
        raw_log_prob which actually performs the prediction.
        """

        # Blur the PSF
        index = self.fit_data.param_to_index("sigma")
        if index is not None:
            blur = params[index]
            for psf in self.psfs:
                psf.blur(blur)

        # Get the log prob
        return self.raw_log_prob(params, prior, return_array)

    def raw_log_prob(self, params, prior, return_array=False):
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
                if data.det not in self.fit_settings.detectors[source_index]:
                    continue
                if self.fit_settings.obs_ids[source_index] is not None and (data.obs_id not in self.fit_settings.obs_ids[source_index]):
                    continue

                # Get the parameters and use the prior
                q = self.fit_data.param_to_value(params, "q", source_name)
                u = self.fit_data.param_to_value(params, "u", source_name)
                f = self.fit_data.param_to_value(params, "f", source_name)
                source.polarize_net((q, u))
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
                    new_q = q * sweeps[data_index][0] + u * sweeps[data_index][1]
                    new_u = -q * sweeps[data_index][1] + u * sweeps[data_index][0]
                    q = new_q
                    u = new_u

                # Polarization weights (no need for the 1/2pi)
                probs = 1 + mus/2 * (data.evt_qs*q + data.evt_us*u)

                # Particle weights
                if self.fit_settings.particles[source_index]:
                    clipped_chars = np.clip(data.evt_bg_chars, 1e-5, 1-1e-5)
                    probs *= clipped_chars / (1 - clipped_chars)

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

            # if data.det == 1:
            #     import matplotlib.pyplot  as plt
            #     plt.style.use("root")
            #     line = np.linspace(-50, 50, 42)

            #     fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
            #     counts = np.histogram2d(data.evt_xs, data.evt_ys, (line, line))[0].astype(float)
            #     pred = np.histogram2d(data.evt_xs, data.evt_ys, (line, line), weights=evt_probs)[0].astype(float)/counts
            #     image = (counts)
            #     ax1.pcolormesh(line, line, np.transpose(image))
            #     image = (pred)
            #     image[~np.isfinite(image)] = 0
            #     ax2.pcolormesh(line, line, np.transpose(pred))
            #     ax1.set_aspect("equal")
            #     ax2.set_aspect("equal")
            #     fig.savefig("dbg.png")
            #     plt.close("all")
            #     import time
            #     time.sleep(0.25)

        if not return_array and not np.isfinite(log_prob):
            problem = None
            if self.fit_data.param_to_index("f", "pbkg") is not None:
                if params[self.fit_data.param_to_index("f", "pbkg")] == 0:
                    if np.any([np.any(data.evt_bg_chars==1) for data in self.datas]):
                        problem = "You have events with bg_prob=1 in your data set, and the particle background flux is zero. This probably caused the problem. Try removing these events"
            if problem is None:
                for name in self.fit_settings.names:
                    if self.fit_data.param_to_index("f", name) is not None and params[self.fit_data.param_to_index("f", name)] == 0:
                        problem = f"Your flux for source {name} is equal to zero. Is that a background region?"
            if problem is None:
                problem = "Your background source might be at fault - is the ROI you provided correct?"
                    
            param_str = "\n".join([f"{self.fit_data.index_to_param(i)}: {params[i]}" for i in range(len(params))])
            raise Exception(f"The log prob was not finite ({log_prob}) with parameters\n{param_str}\n This happens when all your sources predict zero flux in a region of parameter space where at least one event was detected.\n\n{problem}")

        return log_prob