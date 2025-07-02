"""
Do the PCUBE-level PD extraction 
"""

import numpy as np

def get_pcube(datas, bg_data=(None)):
    """ Get the PCUBE-level analysis of PD
    # Arguments:
    * datas: list of IXPEData (does not need to be binned
    * bg_data (optional) a tuple whose first element is a list of IXPEData to use as background and second element is the ratio in area between the src region and the bg region.
    # Returns
    (q, u, sigma_q, sigma_u)
    
    Uses Kislat 2014 but with per-event modulation factor in the same way as IXPEObssim (Baldini+)
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

def get_optimal(datas, bg_data=(None)):
    """ Get the PCUBE-level analysis of PD
    # Arguments:
    * datas: list of IXPEData (does not need to be binned
    * bg_data (optional) a tuple whose first element is a list of IXPEData to use as background and second element is the ratio in area between the src region and the bg region.
    # Returns
    (q, u, sigma_q, sigma_u)
    
    Uses Kislat 2014 but with per-event modulation factor that is optimal
    """
    # https://arxiv.org/pdf/1409.6214

    # Stack the events
    all_q = []
    all_u = []
    all_w = []
    all_original_w = []
    all_mu = []
    for data in datas:
        all_q = np.concatenate([all_q, data.evt_qs])
        all_u = np.concatenate([all_u, data.evt_us])
        all_w = np.concatenate([all_w, np.ones_like(data.evt_ws)])
        all_original_w = np.concatenate([all_original_w, data.evt_ws])
        all_mu = np.concatenate([all_mu, data.evt_mus])

    # Stack the background events, if using
    if bg_data is not None:
        bg_data, area_ratio = bg_data
        for data in bg_data:
            all_q = np.concatenate([all_q, data.evt_qs])
            all_u = np.concatenate([all_u, data.evt_us])
            all_w = np.concatenate([all_w, np.ones_like(data.evt_ws) * -area_ratio])
            all_original_w = np.concatenate([all_original_w, data.evt_ws])
            all_mu = np.concatenate([all_mu, data.evt_mus])

    # Compute total quantities
    # all_mu = all_original_w
    total_q = np.sum(all_q * all_mu * all_w)
    total_u = np.sum(all_u * all_mu * all_w)
    total_m2 = np.sum(all_mu**2 * all_w**2)

    avg_q = total_q / total_m2
    avg_u = total_u / total_m2
    sigma_q = np.sqrt(2 / total_m2)
    sigma_u = np.sqrt(2 / total_m2)

    return (avg_q, avg_u, sigma_q, sigma_u)