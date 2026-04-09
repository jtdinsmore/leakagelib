import numpy as np

def _get_constraint_probs(samples, constraint, fit_data):
    SIGMA = 0.01
    DX = 0.001

    # Returns the chi squared of the constraint function
    f_val = constraint(fit_data, samples)
    f_grad2 = np.zeros_like(f_val)
    for i in range(samples.shape[0]):
        samples[i] += DX
        f_grad2 += ((constraint(fit_data, samples) - f_val) / DX)**2
        samples[i] -= DX
    sample_probs = f_val**2 / f_grad2 / SIGMA**2
    if sample_probs.ndim == 2:
        sample_probs = np.sum(sample_probs, axis=0)
    
    return sample_probs

def get_grad(func, params, steps):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        delta = np.zeros_like(params)
        delta[i] += steps[i]
        grad[i] = (func(params+delta) - func(params-delta)) / (2 * steps[i])
    return grad

def get_hess(func, params, steps):
    hessian = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        delta = np.zeros_like(params)
        delta[i] += steps[i]
        hessian[i] = (get_grad(func, params+delta, steps) - get_grad(func, params-delta, steps)) / (2 * steps[i])
    return hessian

class FitResult:
    def __init__(self, best_params, best_like, message, cov, fit_data, fit_settings):
        """
        Create a `FitResult` from a `scipy.optimize.minimize` output
        """
        self.params = {}
        self.parameter_names = []
        for i, param in enumerate(best_params):
            self.params[fit_data.index_to_param(i)] = param
            self.parameter_names.append(fit_data.index_to_param(i))
        self.fun = best_like
        self.message = message

        # Get uncertainty information
        self.means = best_params
        self.cov = cov
        try:
            self.evals, self.evecs = np.linalg.eigh(self.cov)
        except:
            self.evals, self.evecs = None
        self.sigmas = {}
        for i, sigma2 in enumerate(np.diagonal(self.cov)):
            self.sigmas[fit_data.index_to_param(i)] = np.sqrt(sigma2)

        sigma_array = np.array(list(self.sigmas.values()))
        if np.any(sigma_array) <= 0 or np.any(np.isnan(sigma_array)):
            self.message = "At least one of the parameters is at the boundary. " + self.message

        self.source_names = fit_settings.names
        self.fit_data = fit_data
        self.dof = np.sum([len(data.evt_xs) for data in fit_settings.datas]) - fit_data.length()

    def get_pd_pa(self, tag="src"):
        """
        Get the polarization degree and polarization angle for a source

        Parameters
        ----------
        tag : str, optional
            Name of the source

        Returns
        -------
        tuple (float, float, float, float)
            The PD, EVPA, PD uncertainty, and EVPA uncertainty. Angles are in radians.
        """
        q_index = self.fit_data.param_to_index("q", tag)
        u_index = self.fit_data.param_to_index("u", tag)
        if q_index is None:
            return None, None, None, None
        q = self.params[("q",tag)]
        u = self.params[("u",tag)]
        q_unc2 = self.cov[q_index,q_index]
        u_unc2 = self.cov[u_index,u_index]
        pd = np.sqrt(q**2 + u**2)
        pa = np.arctan2(u, q)/2
        pd_unc = np.sqrt(q**2 * q_unc2 + u**2 * u_unc2) / pd
        pa_unc = np.sqrt(q**2 * u_unc2 + u**2 * q_unc2) / pd**2 / 2
        return pd, pa, pd_unc, pa_unc

    def __str__(self):
        text = "FitResult:\n"
        for ((name, index), value) in self.params.items():
            param_index = self.fit_data.param_to_index(name, index)
            if param_index < self.cov.shape[0]:
                unc = np.sqrt(self.cov[param_index, param_index])
            else:
                unc = None
            if index is None:
                text += f"\t{name} = {value:.4f} +/- {unc:.4f}\n"
            else:
                text += f"\t{name} ({index}) = {value:.4f} +/- {unc:.4f}\n"
        text += "\nPolarization:\n"
        for source_name in self.source_names:
            pd, pa, pd_unc, pa_unc = self.get_pd_pa(source_name)
            if pd is not None:
                sigma = pd / pd_unc
                text += f"\tPD ({source_name}): {pd:.4f} +/- {pd_unc:.4f}\n"
                text += f"\tPA ({source_name}): {pa*180/np.pi:.4f} deg +/- {pa_unc*180/np.pi:.4f}\n"
        text += f"Likelihood {self.fun}, dof {self.dof}\n"
        text += self.message
        return text

    def __repr__(self):
        return str(self)
    
    def sample(self, n_samples):
        if self.evals is None or np.any(self.evals < 0):
            raise Exception("Cannot generate samples from a non-positive definite covariance matrix")
        samples = np.random.randn(n_samples, len(self.evals)) * np.sqrt(self.evals)
        samples = np.einsum("ij,aj->ai", self.evecs, samples)
        samples += self.means
        return samples.transpose()

    def get_constrained_samples(self, constraint, n_samples, n_iterations=100):
        """
        Gets samples that satisfy some constraint
        
        Parameters
        ----------
        constraint: callable
            A function which is zero-valued along the constraint. It should receive a FitData object and array of parameters as arguments, and output a value which is zero-valued along the constraint surface. To enforce multiple constraints, have the function output an array. The function must be numpy compatible; i.e. if an array of shape (P, N) is passed, representing N parameters of dimension P, then f must return an array of size N for 1 constraint, or (C, N) where C is the number of constraints. A uniform prior is automatically enforced.
        n_samples: int
            Number of samples to use
        
        Returns
        -------
            array
        The constrained samples
        """
        # Draw first samples
        samples = self.sample(n_samples)
        sample_probs = _get_constraint_probs(samples, constraint, self.fit_data)

        # Evolve sample distribution
        for _ in range(n_iterations):
            new_samples = self.sample(n_samples)
            new_sample_probs = _get_constraint_probs(new_samples, constraint, self.fit_data)
            
            acceptance_mask = np.random.random(n_samples) < np.exp((sample_probs - new_sample_probs) / 2)
            samples[:,acceptance_mask] = new_samples[:,acceptance_mask]
            sample_probs[acceptance_mask] = new_sample_probs[acceptance_mask]

        return samples

    def get_pdfs(self, params, constraint, bins=50, frac_err=0.01):
        """
        Gets the PDFs of some parameters given constraints.
        
        Parameters
        ----------
        params: callable or list of callable
            A function which takes in the Stokes coefficients and returns the parameter for which a PDF is desired. If a list is provided, multiple PDFs will be returned. Each function should receive a FitData object and array of parameters as arguments, and output a scalar.
        constraint: callable or list of callable
            A function which is zero-valued along the constraint. It should receive a FitData object and array of parameters as arguments, and output a value which is zero-valued along the constraint surface. To enforce multiple constraints, have the function output an array. The function must be numpy compatible; i.e. if an array of shape (P, N) is passed, representing N parameters of dimension P, then f must return an array of size N for 1 constraint, or (C, N) where C is the number of constraints. A uniform prior is automatically enforced.
        bins: int, list of int, array, or list of array.
            If an integer, number of bins to use in the parameter PDF. If an array, edges of the bins for each PDF. Provide a list to set the bins for each variable.
        frac_err: int
            Anticipated fractional error in the average bin. (To be specific, the algorithm will use 1/frac_err**2 samples per bin on average)
        
        Returns
        -------
            (array, array) or (list of array, list of array)
        Returns the bin edges and PDF values of each parameter.
        """

        if type(params) != list:
            params = [params]
        if type(bins) != list:
            bins = [bins]

        # Get PDF bins
        bin_edges = []
        for param, bin_ in zip(params, bins):
            # Get bin edges
            if type(bin_) == int:
                samples = self.sample(10_000)
                param_values = param(self.fit_data, samples)
                bin_edges.append(np.linspace(np.nanpercentile(param_values, 0.5), np.nanpercentile(param_values, 99.5), bin_+1))
            else:
                bin_edges.append(bin_)

        n_samples = int(np.max([len(b) for b in bin_edges]) * 1/frac_err**2)
        samples = self.get_constrained_samples(constraint, n_samples)

        pdfs = []
        for param, bin_edge in zip(params, bin_edges):
            pdfs.append(np.histogram(param(self.fit_data, samples), bin_edge, density=True)[0])

        if len(params) == 1:
            return bin_edges[0], pdfs[0]
        return np.array(bin_edges), np.array(pdfs)

    def get_statistics(self, params, constraint, frac_err=0.005):
        """
        Gets the PDFs of some parameters given constraints.
        
        Parameters
        ----------
        params: callable or list of callable
            A function which takes in the Stokes coefficients and returns the parameter for which a PDF is desired. If a list is provided, multiple PDFs will be returned. Each function should receive a FitData object and array of parameters as arguments, and output a scalar.
        constraint: callable or list of callable
            A function which is zero-valued along the constraint. It should receive a FitData object and array of parameters as arguments, and output a value which is zero-valued along the constraint surface. To enforce multiple constraints, have the function output an array. The function must be numpy compatible; i.e. if an array of shape (P, N) is passed, representing N parameters of dimension P, then f must return an array of size N for 1 constraint, or (C, N) where C is the number of constraints. A uniform prior is automatically enforced.
        frac_err: int
            Anticipated fractional error of the results (To be specific, the algorithm will use 1/frac_err**2 samples)
        
        Returns
        -------
            array, matrix
        The parameter means and covariance matrix subject to the constraint
        """

        if type(params) != list:
            params = [params]

        n_samples = int(1/frac_err**2)
        samples = self.get_constrained_samples(constraint, n_samples)

        parameters = []
        for param in params:
            parameters.append(param(self.fit_data, samples))
        parameters = np.array(parameters)

        means = np.mean(parameters, axis=-1)
        cov = np.cov(parameters)

        if len(params) == 1:
            return means[0], cov
        else:
            return means, cov