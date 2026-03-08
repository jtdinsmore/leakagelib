import numpy as np
from scipy.interpolate import interp1d
from .settings import *
from . import modulation

class Spectrum:
    def __init__(self, bin_centers, counts, weights):
        """
        Returns the spectrum object, which contains information about the distribution
        of counts and weights as functions of energy in a given data set.

        Parameters
        ----------
        bin_centers : array-like
            Centers of the energy bins of the spectrum.
        counts : array-like
            Number of counts in each energy bin.
        weights : array-like or None
            Average value of the weight in each energy bin, or None if the image is unweighted.
        """

        self.bin_centers = bin_centers
        self.counts = counts
        if weights is None:
            weights = np.ones_like(counts)
        self.counts *= weights
        self.counts /= np.sum(self.counts)

    def from_power_law_index(pl_index):
        '''
        Load the spectrum assuming power law-distributed 2-8 keV counts. Unweighted. Useful for demonstration purposes only. In practice, use a more accurate simulation for an IXPE observation.
        '''

        from ixpeobssim.irf import load_arf
        arf = load_arf()
        energies = np.arange(2, 8, 0.04)
        counts = arf(energies) * energies**(-pl_index)
        return Spectrum(energies, counts, None)
    
    def weighted_average(self, array):
        '''
        Get the event fractions in each energy bin
        '''

        return np.sum(self.counts * array)
    
    def _sample_from_interpolator(self, interp):
        return interp(self.bin_centers)
    
    def save(self, f):
        '''
        Save to a file
        '''

        np.save(f, [self.bin_centers, self.counts])

    def load(f):
        '''
        Load from a file
        '''

        bin_centers, counts = np.load(f)
        return Spectrum(bin_centers, counts, None)

    def get_avg_one_over_mu(self, use_nn):
        '''
        Get the average of 1 over the modulation factor.
        '''

        if use_nn:
            mus = modulation.get_nn_modf(self.bin_centers)
        else:
            mus = modulation.get_mom_modf(self.bin_centers)
        return self.weighted_average(1/mus)

    def get_avg_mu(self, use_nn):
        '''
        Get the average of 1 over the modulation factor.
        '''

        if use_nn:
            mus = modulation.get_nn_modf(self.bin_centers)
        else:
            mus = modulation.get_mom_modf(self.bin_centers)
        return self.weighted_average(mus)
    

class EnergyDependence:
    """
    A class that holds interpolators for the leakage parameters sigma_parallel, sigma_perp, and mu as a function of energy. Differences between detectors is not accounted for.
    """
    def __init__(self, energies, sigma_parallel2, sigma_perp2, kurtosis4, mu):
        self.interpolator_parallel = interp1d(energies, sigma_parallel2, bounds_error=False, fill_value=(sigma_parallel2[0], sigma_parallel2[-1]))
        self.interpolator_perp = interp1d(energies, sigma_perp2, bounds_error=False, fill_value=(sigma_perp2[0], sigma_perp2[-1]))
        self.interpolator_kurtosis = interp1d(energies, kurtosis4, bounds_error=False, fill_value=(kurtosis4[0], kurtosis4[-1]))
        self.interpolator_mu = interp1d(energies, mu, bounds_error=False, fill_value=(mu[0], mu[-1]))

    def default(use_nn, use_mu=True):
        """
        Use the default energy dependence, depending on whether the analysis method is NN or Mom.

        Parameters
        ----------
        use_nn : bool
            Set to True to use the neural net Leakage parameters and modulation factor, False for Moments.
        use_mu : bool, optional
            Set to False to ignore the modulation factor. Only use this if you plan to handle the modulation factor event-by-event.
        """

        if use_nn:
            return EnergyDependence.lawrence_nn(use_mu)
        else:
            return EnergyDependence.lawrence_mom(use_mu)
    
    def constant(sigma_parallel, use_nn):
        """
        Assume no sigma_perp, no kurtosis, and constant energy dependence.

        Parameters
        ----------
        use_mu : bool, optional
            Set to False to ignore the modulation factor. Only use this if you plan to handle the modulation factor event-by-event.
        """
        energies = np.array([1., 10.])
        sigma_parallel2 = np.ones_like(energies) * sigma_parallel**2
        sigma_perp2 = np.zeros_like(energies)
        kurtosis4 = np.zeros_like(energies)
        if use_nn:
            mu = modulation.get_nn_modf(energies)
        else:
            mu = modulation.get_mom_modf(energies)
        return EnergyDependence(energies, sigma_parallel2, sigma_perp2, kurtosis4, mu)

    def lawrence_nn(use_mu=True):
        """
        Get the energy-dependent functions for Neural Net data.

        Parameters
        ----------
        use_mu : bool, optional
            Set to False to ignore the modulation factor. Only use this if you plan to handle the modulation factor event-by-event.
        """

        # Generated in vis/get-true-trends
        values = np.load(f"{LEAKAGE_DATA_DIRECTORY}/sigma-tot/sigma-tot.npy")
        energies = values[0]
        sigma_parallels = values[1]
        sigma_perps = values[3]
        kurts = values[5]

        sigma_parallel2 = (NN_SIGMA_PARALLEL_SCALE * sigma_parallels)**2
        sigma_perp2 = sigma_perps**2
        kurtosis4 = (NN_KURT_SCALE * kurts)**4

        mu = modulation.get_nn_modf(energies)
        if not use_mu:
            mu = np.ones_like(mu)

        return EnergyDependence(energies, sigma_parallel2, sigma_perp2, kurtosis4, mu)
    
    def lawrence_mom(use_mu=True):
        """
        Get the energy-dependent functions for Moments data.

        Parameters
        ----------
        use_mu : bool, optional
            Set to False to ignore the modulation factor. Only use this if you plan to handle the modulation factor event-by-event.
        """

        # Generated in vis/get-true-trends
        values = np.load(f"{LEAKAGE_DATA_DIRECTORY}/sigma-tot/sigma-tot.npy")
        energies = values[0]
        sigma_parallels = values[2]
        sigma_perps = values[4]
        kurts = values[6]

        sigma_parallel2 = (MOM_SIGMA_PARALLEL_SCALE * sigma_parallels)**2
        sigma_perp2 = sigma_perps**2
        kurtosis4 = (MOM_KURT_SCALE * kurts)**4

        mu = modulation.get_mom_modf(energies)
        if not use_mu:
            mu = np.ones_like(mu)

        return EnergyDependence(energies, sigma_parallel2, sigma_perp2, kurtosis4, mu)
    
    def evaluate(self, energies):
        """
        Get the leakage parameters for an array of event energies.

        Parameters
        ----------
        energies : array-like
            Array of event energies in keV.

        Returns
        -------
        tuple of (ndarray, ndarray, ndarray)
            A tuple containing `(sigma_parallels, sigma_perps, kurts)`.
        """

        return (
            self.interpolator_parallel(energies),
            self.interpolator_perp(energies),
            self.interpolator_kurtosis(energies)
        )


    def get_params(self, spectrum):
        """
        In some cases, one really wants spectrum-weighted quantities, such as a spectrum-weighted mu, sigma_parallel, mu*sigma_parallel, etc. This function returns these quantities.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum object for the data in question. Either create one directly or use
            the `spectrum` attribute of an IXPEData object.

        Returns
        -------
        dict
            Dictionary of all leakage parameters: `sigma_plus`, `sigma_minus`, `k_plus`,
            `k_minus`, `k_cross`, and `mu_*`.
        """

        sigma_para2_vals = spectrum._sample_from_interpolator(self.interpolator_parallel)
        sigma_perp2_vals = spectrum._sample_from_interpolator(self.interpolator_perp)
        k_parallel4 = spectrum._sample_from_interpolator(self.interpolator_kurtosis)
        mu_vals = spectrum._sample_from_interpolator(self.interpolator_mu)
        mu_vals[np.isnan(mu_vals)] = 0

        k_perp4 = 3 * sigma_perp2_vals**2
        k_both = 0

        sigma_plus = sigma_para2_vals + sigma_perp2_vals
        sigma_minus = sigma_para2_vals - sigma_perp2_vals
        k_plus = k_parallel4 + k_perp4 + 2 * k_both
        k_minus = k_parallel4 - k_perp4
        k_cross = (6 * k_both - k_parallel4 + k_perp4) / 4

        return {
            "sigma_plus": spectrum.weighted_average(sigma_plus),
            "sigma_minus": spectrum.weighted_average(sigma_minus),
            "k_plus": spectrum.weighted_average(k_plus),
            "k_minus": spectrum.weighted_average(k_minus),
            "k_cross": spectrum.weighted_average(k_cross),

            "mu": spectrum.weighted_average(mu_vals),
            "mu_sigma_plus": spectrum.weighted_average(mu_vals * sigma_plus),
            "mu_sigma_minus": spectrum.weighted_average(mu_vals * sigma_minus),
            "mu_k_plus": spectrum.weighted_average(mu_vals * k_plus),
            "mu_k_minus": spectrum.weighted_average(mu_vals * k_minus),
            "mu_k_cross": spectrum.weighted_average(mu_vals * k_cross),
        }