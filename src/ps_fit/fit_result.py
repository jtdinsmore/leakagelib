import numpy as np

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
        Create a FitResult from a scipy.optimize.minimize output
        """
        self.params = {}
        self.parameter_names = []
        for i, param in enumerate(best_params):
            self.params[fit_data.index_to_param(i)] = param
            self.parameter_names.append(fit_data.index_to_param(i))
        self.cov = cov
        self.sigmas = {}
        for i, sigma2 in enumerate(np.diagonal(self.cov)):
            self.sigmas[fit_data.index_to_param(i)] = np.sqrt(sigma2)
        self.fun = best_like
        self.message = message
        self.fit_settings = fit_settings
        self.fit_data = fit_data
        self.dof = np.sum([len(data.evt_xs) for data in fit_settings.datas]) - fit_data.length()

    def get_pd_pa(self, tag="src"):
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
        for source_name in self.fit_settings.names:
            pd, pa, pd_unc, pa_unc = self.get_pd_pa(source_name)
            if pd is not None:
                sigma = pd / pd_unc
                text += f"\tPD ({source_name}): {pd:.4f} +/- {pd_unc:.4f} ({sigma:.1f} sig)\n"
                text += f"\tPA ({source_name}): {pa*180/np.pi:.4f} deg +/- {pa_unc*180/np.pi:.4f}\n"
        text += f"Likelihood {self.fun}, dof {self.dof}\n"
        text += self.message
        return text

    def __repr__(self):
        return str(self)