"""
Fit for the polarization of a point source which varies over time.

The example source is pulsed and has a sweeping PA, both at a frequency of 10 Hz. The simulated lightcurve is sin(phase * pi)**2 and the simulated PA sweep is fixed-rate counterclockwise motion, with PD = 50%.
"""
import numpy as np
import sys
sys.path.append("../../..")
import leakagelib

def fit(energy_cut=(2,8)):
    source = leakagelib.source.Source.no_image(False)

    datas = [leakagelib.IXPEData(source, (
        "data/pulse/event_l2/ixpepulse_det1_evt2_v00.fits",
        "data/pulse/hk/ixpepulse_det1_att_v00.fits",
    ), energy_cut=energy_cut, bin=False)]

    for data in datas:
        data.iterative_centroid_center()
        data.retain(np.sqrt(data.evt_xs**2 + data.evt_ys**2) < 280)

        # It's convenient to overwrite the "times" list with a list of phases. This source sweeps with frequency of 10 Hz, so multiplying by 10 gives the phase
        data.evt_times *= 10
        data.evt_times = np.fmod(data.evt_times, 1)

    # Add the source object to the fitter information
    settings = leakagelib.ps_fit.FitSettings(datas)
    settings.add_point_source("src")
    settings.fix_flux("src", 1)

    settings.add_background("bkg")
    settings.fix_qu("bkg", (0, 0))
    settings.set_initial_flux("bkg", 1)

    settings.apply_circular_roi(280)

    from ixpeobssim.irf import load_arf
    arf = load_arf()
    settings.set_spectrum("bkg", lambda e: arf(e) * e**-2.5)
    settings.set_spectrum("src", lambda e: arf(e) * e**-1.5)

    # The lightcurve returns the flux at time t, as reported in the data.evt_times field. Since we replaced the times with phases, in this case the lightcurve is a function of phase. This example had a sinusoidal lightcurve.
    settings.set_lightcurve("src", lambda ph: np.sin(ph * np.pi)**2) # The normalization will not affect the fit
    # The background lightcurve is assumed to be constant

    # Create a constant PA model
    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    result = fitter.fit()
    print("CONSTANT POLARIZATION MODEL")
    print(result)
    print()

    # Try including a sweeping model with unknown PD and unknown phase offset.
    def model_fn(ph, fit_data, param_array):
        pd = fit_data.param_to_value(param_array, "sweep-PD")
        pa = fit_data.param_to_value(param_array, "sweep-PA")
        q = pd * np.cos(2 * (ph * 2*np.pi + pa))
        u = pd * np.sin(2 * (ph * 2*np.pi + pa))
        return q, u
    settings.set_model_fn("src", model_fn)

    # Add the new parameters sweep-PD and sweep-PA
    settings.add_param("sweep-PD", 0.1, [0, 1])
    settings.add_param("sweep-PA", 0, [-np.pi, np.pi])

    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    result = fitter.fit()
    print("SWEEPING POLARIZATION MODEL")
    print(result)
    print()

    print("The constant model detected nothing because the PA averaged to zero. The sweeping model has the correct parameters (PD=50%, no phase offset), and the higher likelihood indicates a better fit.")

if __name__ == "__main__":
    fit()