import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .settings import *

PIXEL_SIZE = 2.6
NO_K_PERP = False

def analytical_mu_model():
    '''Weights generally increase with energy, so here we approximate weights as depending quadratically on energy for demonstration purposes. In reality one should either simulate an IXPE detection with weights or use the event weights from a real detection.'''
    mu_energies = np.linspace(1, 10, 100)
    mu_val = 0.75 * (1 - (1 - mu_energies / 8)**2)
    return interp1d(mu_energies, mu_val)

class Spectrum:
    def from_power_law_index(pl_index):
        '''Load the spectrum from a power law. Demonstration purposes only. In practice, use your own simulation for an IXPE observation of a power-law source.'''
        values = pd.read_csv(f"{LEAKAGE_DATA_DIRECTORY}/effective-area/ixpe-eff-area.csv")
        energies = np.array(values["ENERGY"])
        areas = np.array(values["AREA"])
        interpolator = interp1d(energies, areas, bounds_error=False)
        energies = np.arange(2, 8, 0.04)
        counts = interpolator(energies) * energies**(-pl_index)
        mus = analytical_mu_model()(energies)
        return Spectrum(energies, counts, mus)
        
    def __init__(self, bin_centers, counts, mus):
        self.bin_centers = bin_centers
        self.counts = counts
        self.mus = mus
    
    def get_fracs(self, weight):
        '''Get the event fractions in each energy bin'''
        if weight:
            return (self.bin_centers, self.mus * self.counts / np.sum(self.counts))
        else:
            return (self.bin_centers, self.counts / np.sum(self.counts))

    def get_avg_weight(self):
        '''Get the average sigma plus and sigma minus for a given source.'''
        bins, fracs = self.get_fracs(True)
        return np.sum(fracs)
    
    def save(self, f):
        '''Save to a file'''
        np.save(f, [self.bin_centers, self.counts, self.mus])

    def load(f):
        '''Load from a file'''
        bin_centers, counts, mus = np.load(f)
        return Spectrum(bin_centers, counts, mus)
    

class EnergyDependence:
    def default(use_nn):
        if use_nn:
            '''Use the default energy dependence depending on whether the analysis method is NN or Mom'''
            return EnergyDependence.lawrence_nn(use_nn)
        else:
            return EnergyDependence.lawrence_mom(use_nn)

    def lawrence_nn(use_nn):
        '''Get the energy dependence functions for sigma parallel and sigma perp for Neural Net data'''
        # Generated in vis/get-true-trends
        # values = pd.read_csv(f"{DATA_DIRECTORY}/nn-energy-unc/lawrence-nn.csv")
        # energies = np.array(values["ENERGY"])
        # sigma_tots = np.array(values["UNC"])
        values = np.load(f"{LEAKAGE_DATA_DIRECTORY}/sigma-tot/sigma-tot.npy")
        energies = values[0]
        sigma_tots = values[1]
        errors = (sigma_tots * PIXEL_SIZE)**2

        sigma_perp2 = (NN_SIGMA_PERP + (energies - 2) * NN_SIGMA_PERP_SLOPE)**2
        sigma_parallel2 = errors - sigma_perp2
        kurtosis4 = sigma_parallel2**2 * NN_KURTOSIS_RATIO**4

        return EnergyDependence(energies, sigma_parallel2, sigma_perp2, kurtosis4, use_nn)
    
    def lawrence_mom(use_nn):
        '''Get the energy dependence functions for sigma parallel and sigma perp for Moments data'''
        # values = pd.read_csv(f"{DATA_DIRECTORY}/nn-energy-unc/lawrence-mom.csv")
        # energies = np.array(values["ENERGY"])
        # sigma_tots = np.array(values["UNC"])
        # Generated in vis/get-true-trends
        values = np.load(f"{LEAKAGE_DATA_DIRECTORY}/sigma-tot/sigma-tot.npy")
        energies = values[0]
        sigma_tots = values[2]
        errors = (sigma_tots * PIXEL_SIZE)**2
        
        sigma_perp2 = (MOM_SIGMA_PERP + (energies - 2) * MOM_SIGMA_PERP_SLOPE)**2
        sigma_parallel2 = errors - sigma_perp2
        kurtosis4 = sigma_parallel2**2 * MOM_KURTOSIS_RATIO**4

        return EnergyDependence(energies, sigma_parallel2, sigma_perp2, kurtosis4, use_nn)
    
    def constant(sigma_parallel, use_nn):
        '''Get the energy dependence functions for sigma parallel and sigma perp for Moments data'''
        energies = np.array([1., 10.])
        sigma_parallel2 = np.ones_like(energies) * sigma_parallel**2
        sigma_perp2 = np.zeros_like(energies)
        kurtosis4 = np.zeros_like(energies)
        return EnergyDependence(energies, sigma_parallel2, sigma_perp2, kurtosis4, use_nn)

    def __init__(self, energies, sigma_parallel2, sigma_perp2, kurtosis4, use_nn):
        self.interpolator_parallel = interp1d(energies, sigma_parallel2, bounds_error=False, fill_value=(sigma_parallel2[0], sigma_parallel2[-1]))
        self.interpolator_perp = interp1d(energies, sigma_perp2, bounds_error=False, fill_value=(sigma_perp2[0], sigma_perp2[-1]))
        self.interpolator_kurtosis = interp1d(energies, kurtosis4, bounds_error=False, fill_value=(kurtosis4[0], kurtosis4[-1]))

    def get_params(self, spectrum, weighted):
        '''Get the average sigma plus and sigma minus for a given source.
        # Arguments:
            - spectrum: Spectrum object for the data in question. Either create one with the Spectrum object directly or pass in the spectrum attribute of an IXPEData object.
            - weighted: True to weight by mu, False otherwise.
        '''
        bins, fracs = spectrum.get_fracs(weighted)

        sigma_para2 = np.sum(fracs * self(bins)[0])
        sigma_perp2 = np.sum(fracs * self(bins)[1])
        kurtosis4 = np.sum(fracs * self(bins)[2])
        
        k_parallel = kurtosis4
        k_perp = 3 * sigma_perp2**2
        if NO_K_PERP:
            k_perp = 0
        k_both = 0

        sigma_plus = sigma_para2 + sigma_perp2
        sigma_minus = sigma_para2 - sigma_perp2
        k_plus = k_parallel + k_perp + 2 * k_both
        k_minus = k_parallel - k_perp
        k_cross = (6 * k_both - k_parallel + k_perp) / 4 # = -k_minus / 4 if k_both=0# 

        return sigma_plus, sigma_minus, k_plus, k_minus, k_cross

    def __call__(self, energy):
        return [
            self.interpolator_parallel(energy),
            self.interpolator_perp(energy),
            self.interpolator_kurtosis(energy),
        ]