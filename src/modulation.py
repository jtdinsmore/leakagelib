import numpy as np
from scipy.interpolate import interp1d
from .settings import LEAKAGE_DATA_DIRECTORY

def load_generic_modf_file(filename):
    energies, modfs = np.load(filename)
    return interp1d(energies, modfs, bounds_error=False, fill_value=1)


def get_mom_modf(energies):
    '''Returns the modulation factor as a function of energy for moments data.
    These data were generated using the IXPEobssim load_modf function using ixpe:obssim:v12 data, averaged over all three detectors. Credit to the ixpeobssim team for making the code available; we do not import the ixpeobssim function in order to avoid requiring ixpeobssim as a dependency to LeakageLib.'''
    return load_generic_modf_file(f"{LEAKAGE_DATA_DIRECTORY}/modulation/mom.npy")(energies)


def get_nn_modf(energies):
    '''Returns the modulation factor as a function of energy for neural nets data.
    The data were generated by taking the average spectrum of weights across several distributions, using the fact that neural net weights are nearly optimal and optimal weights equal the modulation factor of the instrument.'''
    print("WARNING: The NN modulation factor is experimental and is in active development by the authors. Please contact them (jtd@stanford.edu) for more information.")
    return load_generic_modf_file(f"{LEAKAGE_DATA_DIRECTORY}/modulation/nn.npy")(energies)