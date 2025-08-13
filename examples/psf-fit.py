"""
Fit for the polarization of a point source 
"""
import numpy as np
import sys
sys.path.append("../..")
import leakagelib

def fit(energy_cut=(2,8)):
    source = leakagelib.source.Source.no_image(
        True  # True if you would like to use neural net-reconstructed data in the future, and False if you would like to use Mom-reconstructed data.
    )

    # Load the data. This uses LeakageLib's in-built data finding system. To use it, edit the DATA_DIRECTORIES variable in settings.py to point to the folder(s) that contains your IXPE data. If you don't want to use this and instead want to point directly to your own files, use leakagelib.IXPEData.load_all_detectors_with_path. If you want to load one specific file, use the constructor leakagelib.IXPEData. Make sure you have downloaded the HK data.
    datas = leakagelib.IXPEData.load_all_detectors(source, "01002401", energy_cut=energy_cut, bin=False)

    # datas is a list of IXPEData objects. The fitter will simultaneously fit to all objects in the list. You can include as many as you want.

    # Cut the data
    for data in datas:
        # Crop out the edges of the image. This region should be much larger than the PSF to fully utilize the PSF extraction method, but it should not stray to the edges of the chip where vignetting is important.
        data.retain_region(f"psf-region.reg") 

        # Center coordinates on the object. This is necessary to do PSF fitting
        data.iterative_centroid_center()

        # If you wish to make further cuts to the data, use the data.retain function and pass in an event mask. You can use the event fields (see documentation of IXPEData) to create these masks, as illustrated below. You can also load the FITS file itself and create a mask directly from the file. IXPEData.filename gives the file's name. For example,
        # data.retain(np.sqrt(data.evt_xs**2 + data.evt_ys**2) < 300) # Keeps data within 300'' of the center.

    # Make a "fit settings" object, which encodes fit components
    settings = leakagelib.ps_fit.FitSettings(datas)
    settings.add_point_source("src") # Point source component
    settings.fix_flux("src", 1) # Fix the point source flux to be 1
    settings.add_background("bkg") # Background component
    settings.fix_qu("bkg", (0, 0)) # Set the background to be unpolarized
    settings.set_initial_flux("bkg", 1) # Set the background flux estimate equal to source flux; the fitter will use this as the initial guess
    settings.apply_circular_roi(80 * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec. If your ROI is not a circle, use the apply_roi function.

    # If you have a non-point source model, you can use the settings.add_source function to fit it.

    # If you wish, you may add spectral models
    settings.set_spectrum("bkg", lambda e: e**-2.5) # Unabsorbed powerlaw, unmodified by the ARF (You must put in the ARF function yourself if you wish to use it. Consider the IXPEobssim load_arf function.)

    # You can also add lightcurves
    settings.set_spectrum("src", lambda t: 1 + np.sin(t)) # Oscillating ligthcurve

    # You can even add polarization sweep models
    settings.set_spectrum("src", (lambda t: np.sin(2*t), lambda t: np.cos(2*t))) # Model the PA as rotating with frequency 1 Hz, so that q and u (provided) oscillate with frequency 2 Hz.

    # Set up the fitter. Do not modify the settings object after this is done
    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    print(fitter)
    fitter.display_sources("sources.png") # Use this line if you'd like to see images of the sources you're fitting

    # Fit for the source Q, U, and flux, assuming the position is fixed at the origin. Use fit_mcmc to fit with an MCMC instead. You can also choose the minimizer method (one of scipy.optimize.minimize's options) using the method argument.
    result = fitter.fit()

    # Display results
    print(result)

if __name__ == "__main__":
    fit()