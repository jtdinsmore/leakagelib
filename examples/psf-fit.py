"""
Fit for the polarization of a point source 
"""
import sys
sys.path.append("../..")
import leakagelib

def fit(energy_cut=(2,8)):
    source = leakagelib.source.Source.no_image(
        True  # True if you would like to use neural net-reconstructed data in the future, and False if you would like to use Mom-reconstructed data.
    )

    # Load the data. This uses LeakageLib's in-built data finding system. To use it, edit the DATA_DIRECTORIES variable in settings.py to point to the folder(s) that contains your IXPE data. If you don't want to use this and instead want to point directly to your own files, use leakagelib.IXPEData.load_all_detectors_with_path.
    # Make sure you have downloaded the HK data.
    datas = leakagelib.IXPEData.load_all_detectors(source, "01002401", energy_cut=energy_cut, bin=False)

    # Cut the data
    for data in datas:
        # Crop out the edges of the image. This region should be much larger than the PSF to fully utilize the PSF extraction method, but it should not stray to the edges of the chip where vignetting is important and should not contain external point sources.
        data.cut_region(f"psf-region.reg") 

        # Center coordinates on the object
        data.centroid_center()

        # If you wish to make further cuts to the data, use the data.cut function and pass in an event mask. You can use the event fields (see documentation of IXPEData) to create these masks, as illustrated below. You can also load the FITS file itself and create a mask directly from the file. IXPEData.filename gives the file's name
        # This example cuts background particles from the data set. If you don't do this, LeakageLib will de-weight the background instead, which is better in principle. That's why these lines are commented out.
        # event_mask = data.bg_probs < 0.5
        # data.cut(event_mask)

    # Fit for the source Q, U, and flux, assuming the position is fixed at the origin. (Faster and usually accurate).
    # If you just want error for Q and U and don't want to incorporate error from the other parameters, set full_hessian=False.
    result = leakagelib.ps_fit.fit_point_source(datas, fixed_position=(0, 0), full_hessian=True)

    # Fit for the source Q, U, flux, and position
    # result = leakagelib.ps_fit.fit_point_source(datas)

    print(result)

if __name__ == "__main__":
    fit()