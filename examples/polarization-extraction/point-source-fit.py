"""
Fit for the polarization of a point source 
"""
import numpy as np
import sys
sys.path.append("../../..")
import leakagelib

def fit(energy_cut=(2,8)):
    source = leakagelib.source.Source.no_image(
        False  # True if you would like to use neural net-reconstructed data in the future, and False if you would like to use Mom-reconstructed data.
    )

    # Load the data.
    # There are three ways to load data
    #   1. IXPEData.load_all_detectors (the easiest). It will search in all directories listed in the DATA_DIRECTORIES variable defined in settings.py for the given obs id, and load all detectors.
    #   2. leakagelib.IXPEData.load_all_detectors_with_path. Provide a path and the function will load all detectors. This will not use the DATA_DIRECTORIES variable
    #   3. leakagelib.IXPEData. Make sure you have downloaded the HK data.
    # This line loads simulated data with Q=0.5, U=0
    datas = [leakagelib.IXPEData(source, (
        "data/ps/event_l2/ixpeps_det1_evt2_v00.fits",
        "data/ps/hk/ixpeps_det1_att_v00.fits",
    ), energy_cut=energy_cut, bin=False)]
    # datas is a list of IXPEData objects. The fitter will simultaneously fit to all objects in the list. You can include as many as you want.

    for data in datas:
        # Center coordinates on the object. This is important because the point source below is assumed to be at coordinates (0, 0)
        data.iterative_centroid_center()

        # Cut to a 280 arcsecond circular aperture. This region should be much larger than the PSF to fully utilize the PSF extraction method, but it should not stray to the edges of the chip where vignetting is important. You can also use retain_region to use a ciao-formated region file in physical coordinates instead.
        data.retain(np.sqrt(data.evt_xs**2 + data.evt_ys**2) < 280)

    # Make a "fit settings" object, which encodes fit components
    settings = leakagelib.ps_fit.FitSettings(datas)

    # Add a point source component
    settings.add_point_source("src")
    settings.fix_flux("src", 1) # Fix the point source flux to be 1. Since the fitter does not measure absolute flux, one source flux may always be set to one without loss of generality.

    # Add a photon background component
    settings.add_background("bkg")
    settings.fix_qu("bkg", (0, 0)) # Set the background to be unpolarized. (Real observations may have polarized photon background)
    settings.set_initial_flux("bkg", 1) # Set the background flux estimate equal to source flux; the fitter will use this as the initial guess

    # If you have particle weights, add a particle background component. 
    # The spectrum is automatically set if you set other components' spectra
    # settings.add_particle_source()

    settings.apply_circular_roi(280) # Tell the fitter how big the fit region is, in arcsec, so that it can normalize the background PDF. It is important that the ROI exactly match the way that you cut the data. If your ROI is not a circle centered on (0, 0), use the apply_roi function with a custom region image.

    # For high precision, model the spectrum of both the source and background. Not doing so can treat the background improperly and lead to bias.
    from ixpeobssim.irf import load_arf
    arf = load_arf()
    settings.set_spectrum("bkg", lambda e: arf(e) * e**-2.5) # An example backgroudn spectrum. The overall normalization does not affect the fit.
    settings.set_spectrum("src", lambda e: arf(e) * e**-1.5) # A Gamma=1.5, unabsorbed powerlaw for the flux)

    # Set up the fitter. Do not modify the settings object after this is done
    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    print(fitter)
    fitter.display_sources("figs/sources.png") # Plot the fit models you're using

    # Fit for the source Q, U, and flux. The true polarization of this simulated data set is Q=0.5, U=0
    # Use fit_mcmc to fit with an MCMC instead. You can also choose the minimizer method (one of scipy.optimize.minimize's options) using the method argument.
    result = fitter.fit()
    print(result)

if __name__ == "__main__":
    fit()