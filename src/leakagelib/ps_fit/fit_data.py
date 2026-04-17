import numpy as np

class FitData:
    """
    A class that handles converting between parameter values and an array that can be fed into a minimizer.
    """
    def __init__(self, fit_settings):
        """
        Create the FitData from a FitSettings object.
        """
        self.qu_indices = {}
        self.flux_indices = {}
        self.fixed_qu = {}
        self.fixed_flux = {}
        qu_index = 0
        flux_index = 0
        for source_name in fit_settings.sources.keys():
            if fit_settings.fixed_quf[source_name][0] is None:
                self.qu_indices[source_name] = qu_index
                qu_index += 1
            else:
                self.fixed_qu[source_name] = (fit_settings.fixed_quf[source_name][0], fit_settings.fixed_quf[source_name][1])

            if fit_settings.fixed_quf[source_name][2] is None:
                self.flux_indices[source_name] = flux_index
                flux_index += 1
            else:
                self.fixed_flux[source_name] = fit_settings.fixed_quf[source_name][2]
        self.num_qu_params = qu_index*2
        self.num_flux_params = flux_index

        for key in self.flux_indices.keys():
            self.flux_indices[key] += self.num_qu_params

        self.fixed_blur = fit_settings.fixed_blur
        self.num_sigma_params = 1 if self.fixed_blur is None else 0

        self.extra_param_names = list(fit_settings.extra_params.keys())

    def param_to_value(self, params, param, source_name=None):
        """
        Return the value of a parameter.

        Parameters
        ----------
        params : list
            List of parameter values.
        param : {'q', 'u', 'f', 'sigma'}
            The parameter whose value will be returned.
        source_name : str
            Name of the source for which the parameter value is requested.

        Returns
        -------
        float
            The value of the requested parameter.
        """

        index = self.param_to_index(param, source_name)
        if index is None:
            if param == "q":
                return self.fixed_qu[source_name][0]
            if param == "u":
                return self.fixed_qu[source_name][1]
            if param == "f":
                return self.fixed_flux[source_name]
            if param == "sigma":
                return self.fixed_blur
            raise Exception(f"Could not recognize parameter {param}")
        else:
            return params[index]

    def param_to_index(self, param, source_name=None):
        """
        Return the array index of a parameter.

        Parameters
        ----------
        param : {'q', 'u', 'f', 'sigma'}
            The parameter whose index will be returned.
        source_name : str
            Name of the source for which the parameter index is requested. This is required for some parameters.

        Returns
        -------
        int or None
            The index of the parameter in the array if it exists; otherwise, `None`.
        """
        if param == "q":
            if source_name not in self.qu_indices: return None
            return self.qu_indices[source_name] * 2 + 0
        if param == "u":
            if source_name not in self.qu_indices: return None
            return self.qu_indices[source_name] * 2 + 1

        if param == "f":
            if source_name not in self.flux_indices: return None
            return self.flux_indices[source_name]
        
        if param == "sigma":
            if self.fixed_blur is not None: return None
            return self.num_qu_params + self.num_flux_params
        
        if param in self.extra_param_names:
            index = self.extra_param_names.index(param)
            return self.num_qu_params + self.num_flux_params + self.num_sigma_params + index
        
        raise Exception(f"Could not recognize parameter {param}")

    def index_to_param(self, index):
        """
        Returns the parameter name and the index of the data set it belongs to. If the parameter is
        global, the index will be None

        Parameters
        ----------
        index : int
            The index of the parameter

        Returns
        -------
        tuple of (str, str)
            A tuple where the first element is the parameter type and the second is the source name. `None` if the parameter is not associated with a source.
        """
        original_index = index
        for key, value in self.qu_indices.items():
            if value * 2 + 0 == index:
                return "q", key
            if value * 2 + 1 == index:
                return "u", key
            
        for key, value in self.flux_indices.items():
            if value == index:
                return "f", key
            
        index -= self.num_qu_params + self.num_flux_params
        if self.fixed_blur is None:
            if index == 0: return "sigma", None
            index -= self.num_sigma_params

        if index < len(self.extra_param_names):
            return (self.extra_param_names[index], None)
        index -= len(self.extra_param_names)

        raise Exception(f"Could not recognize index {original_index}")

    def length(self):
        length = self.num_qu_params + self.num_flux_params + self.num_sigma_params + len(self.extra_param_names)
        return length