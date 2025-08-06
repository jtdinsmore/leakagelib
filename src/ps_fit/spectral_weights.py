import numpy as np
from scipy.interpolate import interp1d
from ..settings import LEAKAGE_DATA_DIRECTORY
from ..modulation import *

class RMF:
    """
    Load an RMF from the RMF saved in leakagelib, which was made from the CALDB IXPE RMF 20240101_v013
    """
    def __init__(self):
        self.rmf_matrix = np.load(f"{LEAKAGE_DATA_DIRECTORY}/rmf/rmf.npy")
        self.es = np.load(f"{LEAKAGE_DATA_DIRECTORY}/rmf/es.npy")
        self.ehats = np.load(f"{LEAKAGE_DATA_DIRECTORY}/rmf/ehats.npy")
    
    def convolve_spectrum(self, spectrum):
        """
        Returns the spectrum convolved with the RMF.
        # Arguments
        * `spectrum`: a function that takes energies in keV and returns the spectrum unit
        """
        digitized_spec = spectrum(self.es)
        convolved_spec = digitized_spec @ self.rmf_matrix
        return interp1d(self.ehats, convolved_spec)
    
    def convolve_spectrum_mu(self, spectrum, use_nn):
        """
        Returns the spectrum times the modulation factor convolved with the RMF.
        # Arguments
        * `spectrum`: a function that takes energies in keV and returns the spectrum unit
        """
        if use_nn:
            modf = get_nn_modf(self.es)
        else:
            modf = get_mom_modf(self.es)
        digitized_spec = spectrum(self.es) * modf
        convolved_spec = digitized_spec @ self.rmf_matrix
        return interp1d(self.ehats, convolved_spec)