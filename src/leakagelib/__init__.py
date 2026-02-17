from .source import Source
from .psf import PSF
from .ixpe_data import IXPEData, IXPE_PIXEL_SIZE
from .ps_fit.fitter import Fitter
from .ps_fit.fit_settings import FitSettings
from .spectrum import EnergyDependence, Spectrum
from .modulation import get_nn_modf, get_mom_modf
from .region import Region
from .ps_fit.pcube import get_pcube
from . import extended, funcs

__all__ = ["Source", "PSF", "IXPEData", "IXPE_PIXEL_SIZE", "EnergyDependence", "Spectrum", "Fitter", "FitSettings", "Region", "get_nn_modf", "get_mom_modf"]

from importlib.metadata import version as get_version
__version__ = get_version("leakagelib")

