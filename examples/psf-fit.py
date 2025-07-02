"""
Fit for the polarization of a point source 
"""
import sys
sys.path.append("../..")
import leakagelib

def fit(energy_cut=(2,8)):
    source = leakagelib.source.Source.no_image(
        False  # True if you would like to use neural net-reconstructed data in the future, and False if you would like to use Mom-reconstructed data.
    )

    # Load the data. This uses LeakageLib's in-built data finding system. To use it, edit the DATA_DIRECTORIES variable in settings.py to point to the folder(s) that contains your IXPE data. If you don't want to use this and instead want to point directly to your own files, use leakagelib.IXPEData.load_all_detectors_with_path.
    # Make sure you have downloaded the HK data.
    datas = leakagelib.IXPEData.load_all_detectors(source, "01002401", energy_cut=energy_cut, bin=False)

    # Cut the data
    for data in datas:
        # Crop out the edges of the image. This region should be much larger than the PSF to fully utilize the PSF extraction method, but it should not stray to the edges of the chip where vignetting is important and should not contain external point sources.
        data.cut_region(f"psf-region.reg") 

        # WARNING: If you use an extended source model whose flux extends to the edge of your region cut, there are some technicalities near the edge that you must treat carefully to achieve accurate fits. This caveat does not apply to point sources

        # Center coordinates on the object
        data.centroid_center()

        # If you wish to make further cuts to the data, use the data.cut function and pass in an event mask. You can use the event fields (see documentation of IXPEData) to create these masks, as illustrated below. You can also load the FITS file itself and create a mask directly from the file. IXPEData.filename gives the file's name

        # This example cuts background particles from the data set. If you don't do this, LeakageLib will de-weight the background instead, which is better in principle. That's why these lines are commented out:
        # event_mask = data.bg_probs < 0.5
        # data.cut(event_mask)

    # Make a "fit settings" object, which encodes fit components
    settings = leakagelib.ps_fit.FitSettings()
    settings.add_point_source(datas[0]) # Point source component
    settings.add_background() # Background component
    settings.fix_qu("bkg", (0, 0)) # Set the background to be unpolarized
    settings.add_background(name="pbkg") # Make an additional component for the particles. This is not necessary if you cut all the particles using the lines commented out above
    settings.set_particles("pbkg", True)
    settings.apply_circular_roi(80 * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec.

    fitter = leakagelib.ps_fit.Fitter(datas, settings)
    print(fitter)
    # fitter.display_sources("sources.png") # Use this line if you'd like to see images of the sources you're fitting

    # Fit for the source Q, U, and flux, assuming the position is fixed at the origin. Use fit_mcmc to fit with an MCMC instead. You can also choose the minimizer method (one of scipy.optimize.minimize's options) using the method argument.
    result = fitter.fit()

    print(result)

if __name__ == "__main__":
    fit()