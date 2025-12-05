"""Use the PSF to fit for the Stokes coefficients of objects

Jack Dinsmore 2025
"""

from .fitter import Fitter
from .fit_settings import FitSettings
from .pcube import get_pcube

import numpy as np
from scipy.optimize import minimize

def estimate_particle_flux(datas):
    """Estimate the fraction of events which are particles based on the particle character"""
    all_particle_odds = []
    for data in datas:
        bg_chars = np.clip(data.evt_bg_chars, 1e-5, 1-1e-5)
        odds = bg_chars / (1 - bg_chars)
        all_particle_odds = np.concatenate([all_particle_odds, odds])

    def minus_like(params):
        f, = params
        like = np.sum(np.log(f*all_particle_odds + (1 - f)))
        print(f, like)
        return -like
    
    result = minimize(minus_like, x0=[0.5], bounds=[[0, 1]], method="nelder-mead")
    print(result)
    return result.x[0]