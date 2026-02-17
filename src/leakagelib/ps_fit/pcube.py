"""
Do the PCUBE-level PD extraction 
"""

import numpy as np
import copy

def get_pcube(datas, bg_data=None):
    """
    Get the PCUBE-level analysis of polarization degree (PD).

    Parameters
    ----------
    datas : list of IXPEData
        List of IXPEData objects. Does not need to be binned.
    bg_data : tuple (list of IXPEData, float), optional
        Tuple where the first element is a list of IXPEData to use as background, and
        the second element is the area ratio between the source region and the background region.

    Returns
    -------
    tuple of (float, float, float, float)
        A tuple `(q, u, sigma_q, sigma_u)` containing the Stokes parameters and their uncertainties.

    Notes
    -----
    - Uses Kislat 2014 methodology, but applies per-event modulation factor. This function does not give the same errors as the IXPEObssim (Baldini+) PCUBE algorithm; it uses the formulas resulting from linear propagation of uncertainties.
    """

    # https://arxiv.org/pdf/1409.6214

    # Stack the events
    all_q = []
    all_u = []
    all_w = []
    all_mu = []
    for data in datas:
        all_q = np.concatenate([all_q, data.evt_qs])
        all_u = np.concatenate([all_u, data.evt_us])
        all_w = np.concatenate([all_w, data.evt_ws])
        all_mu = np.concatenate([all_mu, data.evt_mus])

    # Stack the background events, if using
    if bg_data is not None:
        bg_data, area_ratio = bg_data
        for data in bg_data:
            all_q = np.concatenate([all_q, data.evt_qs])
            all_u = np.concatenate([all_u, data.evt_us])
            all_w = np.concatenate([all_w, data.evt_ws * -area_ratio])
            all_mu = np.concatenate([all_mu, data.evt_mus])

    # Compute total quantities
    coefficients = all_w / all_mu
    total_q = np.sum(all_q * coefficients)
    total_u = np.sum(all_u * coefficients)
    total_w = np.sum(all_w)
    total_w2 = np.sum(all_w**2)
    total_coefficients2 = np.sum(coefficients**2)

    avg_q = total_q / total_w
    avg_u = total_u / total_w
    sigma_q = np.sqrt(2 * total_coefficients2 - avg_q**2 * total_w2) / total_w
    sigma_u = np.sqrt(2 * total_coefficients2 - avg_u**2 * total_w2) / total_w

    return (avg_q, avg_u, sigma_q, sigma_u)